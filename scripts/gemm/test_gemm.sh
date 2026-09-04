#!/usr/bin/env bash
# Run TPU GEMM sequentially on each local JAX chiplet device.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PATH="${REPO_ROOT}/.venv/bin:${PATH}"

# CommPilot second-precision timing markers.
timing_event_at() {
    local timestamp="$1" phase="$2" event="$3"
    shift 3
    printf '[TIMING] timestamp=%s phase=%s event=%s script=%s' \
        "${timestamp}" "${phase}" "${event}" "${BASH_SOURCE[0]##*/}" >&2
    (( $# == 0 )) || printf ' %s' "$@" >&2
    printf '\n' >&2
}
timing_event() { local phase="$1" event="$2"; shift 2; timing_event_at "$(date '+%Y-%m-%dT%H:%M:%S%z')" "${phase}" "${event}" "$@"; }

TIMING_SCRIPT_STARTED_EPOCH="$(date +%s)"
TIMING_SCRIPT_STARTED_AT="$(date '+%Y-%m-%dT%H:%M:%S%z')"
timing_finish() {
    local exit_code="${1:-$?}"
    local finished_epoch
    finished_epoch="$(date +%s)"
    timing_event script end \
        "exit_code=${exit_code}" \
        "duration_seconds=$((finished_epoch - TIMING_SCRIPT_STARTED_EPOCH))"
}
timing_event_at "${TIMING_SCRIPT_STARTED_AT}" script start
trap timing_finish EXIT

die() { printf '[ERROR] %s\n' "$*" >&2; exit 1; }
sanitize_label() { printf '%s' "$1" | tr '[:space:]/' '__' | sed 's/[^A-Za-z0-9._-]/_/g'; }
validate_host_token() { [[ "$1" =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid host or IP token: $1"; }
load_container_runtime_environment() {
    local key value
    [[ -r /proc/1/environ ]] || return 0
    while IFS='=' read -r key value; do
        case "${key}" in
            PATH|LD_LIBRARY_PATH|PYTHONPATH|LIBTPU_INIT_ARGS|XLA_FLAGS|TPU_*|JAX_*)
                [[ -n "${!key+x}" ]] || export "${key}=${value}"
                ;;
        esac
    done < <(tr '\0' '\n' < /proc/1/environ)
}
detect_node_identity() {
    local node_name="${NODE_SN_NAME:-}" host_ip="${NODE_HOST_IP:-}" address interface
    if [[ -z "${node_name}" && -r /proc/1/environ ]]; then
        node_name="$(tr '\0' '\n' </proc/1/environ | awk -F= '$1=="NODE_SN_NAME" {print substr($0,index($0,"=")+1); exit}')"
    fi
    if [[ -n "${node_name}" ]]; then sanitize_label "${node_name}"; return; fi
    if [[ -z "${host_ip}" && -r /proc/1/environ ]]; then
        host_ip="$(tr '\0' '\n' </proc/1/environ | awk -F= '$1=="NODE_HOST_IP" {print substr($0,index($0,"=")+1); exit}')"
    fi
    if [[ -n "${host_ip}" ]]; then sanitize_label "${host_ip}"; return; fi
    for interface in bond0 eth0 "$(ip route show default 2>/dev/null | awk 'NR==1 {print $5}')"; do
        [[ -n "${interface}" ]] || continue
        address="$(ip -4 -o addr show dev "${interface}" scope global 2>/dev/null | awk 'NR==1 {sub(/\/.*/,"",$4); print $4}')"
        [[ -z "${address}" ]] || { sanitize_label "${address}"; return; }
    done
    sanitize_label "$(hostname)"
}
ENTRY="${REPO_ROOT}/src/gemm/test_gemm.py"
SCRIPT_PATH="${REPO_ROOT}/scripts/gemm/test_gemm.sh"
LOG_ROOT="${REPO_ROOT}/logs/gemm"
TEMP_ROOT="${REPO_ROOT}/logs/.tmp"
HOSTFILE=""
MASTER_HOST=""
SSH_PORT=22
SSH_USER="${USER:-$(id -un)}"
RUN_LABEL=""
SUBTREE_NODES_CSV=""
SUBTREE_RUN_TOKEN=""
DTYPES_CSV="fp16,bf16,fp8"
M=32768
N=32768
K=32768
ITERATIONS=5
WARMUP=2
XPROF_TIMING=0
XPROF_MODES_CSV=""
PROFILE=0

use_xprof_for_mode() {
    if [[ -n "${XPROF_MODES_CSV}" ]]; then
        [[ ",${XPROF_MODES_CSV}," == *",$1,"* ]]
    else
        (( XPROF_TIMING == 1 ))
    fi
}

usage() {
    cat <<'EOF'
Usage: test_gemm.sh --hostfile <path> --master <host> [options]
  --hostfile <path>            Complete TPU Slice host list.
  --master <host>              Current Master host/IP; becomes the SSH tree root.
  --ssh-user <name>           SSH login (default: current user).
  --ssh-port <N>               Default: 22.
  --dtype, --gemm-dtype <csv>  FP16/BF16/FP8 list. Default: fp16,bf16,fp8.
  --m <N> --n <N> --k <N>     GEMM dimensions. Default: 32768 each.
  --iterations <N>             Timed iterations. Default: 5.
  --warmup <N>                 Warmup iterations. Default: 2.
  --xprof-timing               Use temporary Xprof traces for timing.
  --xprof-modes <csv>          Use Xprof only for selected fp16,bf16,fp8 values;
                              non-empty selection overrides --xprof-timing.
  --profile                    Retain Xprof traces and HLO dumps.
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --hostfile) [[ $# -ge 2 ]] || die "--hostfile requires a value"; HOSTFILE="$2"; shift 2 ;;
        --master) [[ $# -ge 2 ]] || die "--master requires a value"; MASTER_HOST="$2"; shift 2 ;;
        --ssh-user) [[ $# -ge 2 ]] || die "--ssh-user requires a value"; SSH_USER="$2"; shift 2 ;;
        --ssh-port) [[ $# -ge 2 ]] || die "--ssh-port requires a value"; SSH_PORT="$2"; shift 2 ;;
        --label) [[ $# -ge 2 ]] || die "--label requires a value"; RUN_LABEL="$2"; shift 2 ;;
        --dtype|--gemm-dtype|--gemm_dtype) [[ $# -ge 2 ]] || die "$1 requires a value"; DTYPES_CSV="$2"; shift 2 ;;
        --m) [[ $# -ge 2 ]] || die "--m requires a value"; M="$2"; shift 2 ;;
        --n) [[ $# -ge 2 ]] || die "--n requires a value"; N="$2"; shift 2 ;;
        --k) [[ $# -ge 2 ]] || die "--k requires a value"; K="$2"; shift 2 ;;
        --iterations) [[ $# -ge 2 ]] || die "--iterations requires a value"; ITERATIONS="$2"; shift 2 ;;
        --warmup) [[ $# -ge 2 ]] || die "--warmup requires a value"; WARMUP="$2"; shift 2 ;;
        --xprof-timing) XPROF_TIMING=1; shift ;;
        --xprof-modes) [[ $# -ge 2 ]] || die "--xprof-modes requires a value"; XPROF_MODES_CSV="$2"; shift 2 ;;
        --profile|--dump-hlo) PROFILE=1; shift ;;
        --internal-subtree-nodes) [[ $# -ge 2 ]] || die "--internal-subtree-nodes requires a value"; SUBTREE_NODES_CSV="$2"; shift 2 ;;
        --internal-run-token) [[ $# -ge 2 ]] || die "--internal-run-token requires a value"; SUBTREE_RUN_TOKEN="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done
[[ "${SSH_USER}" =~ ^[A-Za-z_][A-Za-z0-9_.-]*$ ]] || die "invalid --ssh-user"

[[ -z "${XPROF_MODES_CSV}" || "${XPROF_MODES_CSV}" =~ ^(fp16|bf16|fp8)(,(fp16|bf16|fp8))*$ ]] || die "invalid --xprof-modes: ${XPROF_MODES_CSV}"
[[ -f "${ENTRY}" ]] || die "GEMM entry not found"
[[ "${SSH_PORT}" =~ ^[1-9][0-9]*$ ]] && (( 10#${SSH_PORT} <= 65535 )) || die "--ssh-port must be in 1..65535"
for value in "${M}" "${N}" "${K}" "${ITERATIONS}"; do [[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "dimensions and iterations must be positive integers"; done
[[ "${WARMUP}" =~ ^[0-9]+$ ]] || die "--warmup must be a non-negative integer"
IFS=',' read -r -a DTYPES <<< "${DTYPES_CSV}"
(( ${#DTYPES[@]} > 0 )) || die "at least one dtype is required"
for dtype in "${DTYPES[@]}"; do [[ "${dtype}" =~ ^(fp16|bf16|fp8)$ ]] || die "unsupported dtype: ${dtype}"; done
load_container_runtime_environment

run_local_benchmark() {
    local node_identity
    node_identity="$(detect_node_identity)"
    timing_event local_benchmark start "node=${node_identity}" "testcase=gemm"
    printf 'Node identity: %s\n' "${node_identity}"
    printf 'TPU GEMM: m=%s n=%s k=%s warmup=%s iterations=%s dtypes=%s fallback_timing=%s xprof_modes=%s profile=%s local_devices_only=true\n' \
        "${M}" "${N}" "${K}" "${WARMUP}" "${ITERATIONS}" "${DTYPES[*]}" \
        "$([[ ${XPROF_TIMING} -eq 1 ]] && printf xprof || printf cpu)" "${XPROF_MODES_CSV:-unset}" "${PROFILE}"
    local dtype command_output
    local -a command
    for dtype in "${DTYPES[@]}"; do
        printf '\n================================================================================\n'
        printf 'dtype=%s local_device=all\n' "${dtype}"
        printf '================================================================================\n'
        command=(python3 "${ENTRY}" --m "${M}" --n "${N}" --k "${K}" --dtype "${dtype}" \
            --warmup "${WARMUP}" --iteration "${ITERATIONS}")
        if use_xprof_for_mode "${dtype}"; then
            command+=(--xprof-timing)
            printf 'GEMM subtest=%s timing=xprof\n' "${dtype}"
        else
            printf 'GEMM subtest=%s timing=cpu\n' "${dtype}"
        fi
        (( PROFILE == 0 )) || command+=(--profile)
        if ! command_output="$("${command[@]}" 2>&1)"; then
            printf '%s\n' "${command_output}"
            die "GEMM ${dtype} benchmark failed"
        fi
        printf '%s\n' "${command_output}"
        grep -Fq '"testcase":"gemm","metric":"'"${dtype}"'"' <<< "${command_output}" \
            || die "GEMM ${dtype} produced no structured metric"
    done
    timing_event local_benchmark end "node=${node_identity}" "testcase=gemm"
}

build_subtree_command() {
    local nodes_csv="$1" quoted
    local -a command=(bash "${SCRIPT_PATH}" --internal-subtree-nodes "${nodes_csv}" \
        --internal-run-token "${SUBTREE_RUN_TOKEN}" --ssh-user "${SSH_USER}" --ssh-port "${SSH_PORT}" \
        --gemm-dtype "${DTYPES_CSV}" --m "${M}" --n "${N}" --k "${K}" \
        --iterations "${ITERATIONS}" --warmup "${WARMUP}")
    [[ -z "${XPROF_MODES_CSV}" ]] || command+=(--xprof-modes "${XPROF_MODES_CSV}")
    (( XPROF_TIMING == 0 )) || command+=(--xprof-timing)
    (( PROFILE == 0 )) || command+=(--profile)
    printf -v quoted '%q ' "${command[@]}"
    printf 'exec %s' "${quoted}"
}

subtree_execute_node() {
    local output_dir="$1"; shift
    local -a nodes=("$@") child_csvs=() child_hosts=() child_tars=() child_errs=() child_pids=()
    (( ${#nodes[@]} > 0 )) || return 1
    local current="${nodes[0]}"
    local remaining=$((${#nodes[@]} - 1))
    local left_count=$(((remaining + 1) / 2))
    local right_count=$((remaining - left_count))
    local -a left_nodes=() right_nodes=()
    local csv child safe_child remote_command index local_pid failed=0 local_log tar_path err_path
    validate_host_token "${current}"
    (( left_count == 0 )) || { left_nodes=("${nodes[@]:1:left_count}"); child_csvs+=("$(IFS=,; printf '%s' "${left_nodes[*]}")"); }
    (( right_count == 0 )) || { right_nodes=("${nodes[@]:$((left_count + 1)):right_count}"); child_csvs+=("$(IFS=,; printf '%s' "${right_nodes[*]}")"); }
    # Bash 4.2 treats an empty array as unset under nounset.
    for csv in "${child_csvs[@]+"${child_csvs[@]}"}"; do
        child="${csv%%,*}"; validate_host_token "${child}"; safe_child="$(sanitize_label "${child}")"
        tar_path="${output_dir}/.subtree_${safe_child}.tar"
        err_path="${output_dir}/.subtree_${safe_child}.stderr"
        child_tars+=("${tar_path}")
        child_errs+=("${err_path}")
        child_hosts+=("${child}")
        remote_command="$(build_subtree_command "${csv}")"
        ssh -p "${SSH_PORT}" -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
             -o LogLevel=ERROR -o ConnectTimeout=10 \
            -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
            "${SSH_USER}@${child}" "${remote_command}" > "${tar_path}" 2> "${err_path}" &
        child_pids+=("$!")
    done
    local_log="${output_dir}/${SUBTREE_RUN_TOKEN}_$(detect_node_identity).log"
    run_local_benchmark > "${local_log}" 2>&1 & local_pid=$!
    wait "${local_pid}" || failed=1
    for index in "${!child_pids[@]}"; do
        wait "${child_pids[index]}" || { printf '[ERROR] GEMM subtree rooted at %s failed\n' "${child_hosts[index]}" >&2; failed=1; }
        [[ ! -s "${child_tars[index]}" ]] || tar xf "${child_tars[index]}" -C "${output_dir}" || failed=1
        [[ ! -s "${child_errs[index]}" ]] || cat "${child_errs[index]}" >&2
        rm -f "${child_tars[index]}" "${child_errs[index]}"
    done
    return "${failed}"
}

run_subtree() {
    [[ -n "${SUBTREE_RUN_TOKEN}" ]] || die "internal run token is required"
    local -a nodes=(); IFS=',' read -r -a nodes <<< "${SUBTREE_NODES_CSV}"
    local temp_dir status=0
    mkdir -p "${TEMP_ROOT}"; temp_dir="$(mktemp -d "${TEMP_ROOT}/gemm-subtree.XXXXXX")"
    subtree_execute_node "${temp_dir}" "${nodes[@]}" || status=$?
    tar cf - -C "${temp_dir}" .
    rm -rf -- "${temp_dir}"
    return "${status}"
}

run_all_hosts() {
    [[ -n "${HOSTFILE}" && -f "${HOSTFILE}" ]] || die "--hostfile must reference an existing file"
    [[ -n "${MASTER_HOST}" ]] || die "--master is required"; validate_host_token "${MASTER_HOST}"
    local -a raw_hosts=() hosts=("${MASTER_HOST}")
    local host temp_dir status=0 log_count file destination
    mapfile -t raw_hosts < <(awk '!/^[[:space:]]*(#|$)/ {print $1}' "${HOSTFILE}")
    (( ${#raw_hosts[@]} > 0 )) || die "hostfile is empty"
    [[ "$(printf '%s\n' "${raw_hosts[@]}" | sort -u | wc -l | tr -d ' ')" == "${#raw_hosts[@]}" ]] || die "hosts must be unique"
    printf '%s\n' "${raw_hosts[@]}" | grep -Fxq -- "${MASTER_HOST}" || die "master must be present in hostfile"
    for host in "${raw_hosts[@]}"; do validate_host_token "${host}"; [[ "${host}" == "${MASTER_HOST}" ]] || hosts+=("${host}"); done
    SUBTREE_RUN_TOKEN="$(date +%Y%m%d_%H%M%S)_singlenode_gemm"
    [[ -z "${RUN_LABEL}" ]] || SUBTREE_RUN_TOKEN+="_$(sanitize_label "${RUN_LABEL}")"
    mkdir -p "${TEMP_ROOT}"; temp_dir="$(mktemp -d "${TEMP_ROOT}/gemm-root.XXXXXX")"
    subtree_execute_node "${temp_dir}" "${hosts[@]}" || status=$?
    log_count="$(find "${temp_dir}" -maxdepth 1 -type f -name '*.log' | wc -l | tr -d ' ')"
    (( log_count == ${#hosts[@]} )) || { printf '[ERROR] expected %s node logs, collected %s\n' "${#hosts[@]}" "${log_count}" >&2; status=1; }
    mkdir -p "${LOG_ROOT}"; shopt -s nullglob
    for file in "${temp_dir}"/*.log; do destination="${LOG_ROOT}/$(basename "${file}")"; [[ ! -e "${destination}" ]] || destination="${destination%.log}_$RANDOM.log"; mv "${file}" "${destination}"; done
    printf '[INFO] collected %s node logs\n' "${log_count}"
    shopt -u nullglob; rm -rf -- "${temp_dir}"
    return "${status}"
}

if [[ -n "${SUBTREE_NODES_CSV}" ]]; then run_subtree; else run_all_hosts; fi
