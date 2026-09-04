#!/usr/bin/env bash
# Run all host-local TPU PCIe, HBM, and VMEM bandwidth benchmarks.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PATH="${REPO_ROOT}/.venv/bin:${PATH}"

# CommPilot second-precision timing markers.
timing_event_at() { local timestamp="$1" phase="$2" event="$3"; shift 3; printf '[TIMING] timestamp=%s phase=%s event=%s script=%s' "${timestamp}" "${phase}" "${event}" "${BASH_SOURCE[0]##*/}" >&2; (( $# == 0 )) || printf ' %s' "$@" >&2; printf '\n' >&2; }
timing_event() { local phase="$1" event="$2"; shift 2; timing_event_at "$(date '+%Y-%m-%dT%H:%M:%S%z')" "${phase}" "${event}" "$@"; }
TIMING_SCRIPT_STARTED_EPOCH="$(date +%s)"; TIMING_SCRIPT_STARTED_AT="$(date '+%Y-%m-%dT%H:%M:%S%z')"
timing_finish() {
    local exit_code="${1:-$?}"
    local finished_epoch
    finished_epoch="$(date +%s)"
    timing_event script end "exit_code=${exit_code}" "duration_seconds=$((finished_epoch - TIMING_SCRIPT_STARTED_EPOCH))"
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
    if [[ -z "${node_name}" && -r /proc/1/environ ]]; then node_name="$(tr '\0' '\n' </proc/1/environ | awk -F= '$1=="NODE_SN_NAME" {print substr($0,index($0,"=")+1); exit}')"; fi
    if [[ -n "${node_name}" ]]; then sanitize_label "${node_name}"; return; fi
    if [[ -z "${host_ip}" && -r /proc/1/environ ]]; then host_ip="$(tr '\0' '\n' </proc/1/environ | awk -F= '$1=="NODE_HOST_IP" {print substr($0,index($0,"=")+1); exit}')"; fi
    if [[ -n "${host_ip}" ]]; then sanitize_label "${host_ip}"; return; fi
    for interface in bond0 eth0 "$(ip route show default 2>/dev/null | awk 'NR==1 {print $5}')"; do [[ -n "${interface}" ]] || continue; address="$(ip -4 -o addr show dev "${interface}" scope global 2>/dev/null | awk 'NR==1 {sub(/\/.*/,"",$4); print $4}')"; [[ -z "${address}" ]] || { sanitize_label "${address}"; return; }; done
    sanitize_label "$(hostname)"
}
SRC_ROOT="${REPO_ROOT}/src"
SCRIPT_PATH="${REPO_ROOT}/scripts/tpubandwidth/test_tpubandwidth.sh"
LOG_ROOT="${REPO_ROOT}/logs/tpubandwidth"
TEMP_ROOT="${REPO_ROOT}/logs/.tmp"
HOSTFILE=""
MASTER_HOST=""
SSH_PORT=22
SSH_USER="${USER:-$(id -un)}"
RUN_LABEL=""
SUBTREE_NODES_CSV=""
SUBTREE_RUN_TOKEN=""
MODES_CSV="h2d,d2h,hbm_read,hbm_write,hbm_copy,vmem_copy"
PCIE_DATA_SIZE=1GiB
PCIE_MODE=one_to_one
HBM_DATA_SIZE=4GiB
VMEM_DATA_SIZE=4GiB
TARGET_DEVICES=8
ITERATIONS=5
WARMUP=2
XPROF_TIMING=0
XPROF_MODES_CSV=""
PROFILE=0

# A non-empty selection takes precedence over the legacy all-modes flag.
use_xprof_for_mode() {
    if [[ -n "${XPROF_MODES_CSV}" ]]; then
        [[ ",${XPROF_MODES_CSV}," == *",$1,"* ]]
    else
        (( XPROF_TIMING == 1 ))
    fi
}

usage() {
    cat <<'EOF'
Usage: test_tpubandwidth.sh --hostfile <path> --master <host> [options]
  --hostfile <path>        Complete TPU Slice host list.
  --master <host>          Current Master host/IP; becomes the SSH tree root.
  --ssh-user <name>           SSH login (default: current user).
  --ssh-port <N>           Default: 22.
  --mode <csv>             h2d,d2h,hbm_read,hbm_write,hbm_copy,vmem_copy.
  --pcie-data-size <size>  Default: 1GiB.
  --pcie-mode <mode>       one_to_one (default), one_to_many, or both. H2D/D2H only.
  --hbm-data-size <size>   Default: 4GiB.
  --vmem-data-size <size>  Default: 4GiB.
  --target-devices <N>     Local JAX chiplet devices. Default: 8.
  --iterations <N>         Default: 5.
  --warmup <N>             Default: 2.
  --xprof-timing           Use temporary Xprof traces for timing.
  --xprof-modes <csv>      Use Xprof only for selected --mode values; overrides
                           --xprof-timing when non-empty. Other modes use CPU.
  --profile                Retain Xprof traces and HLO dumps.
EOF
}
while (( $# > 0 )); do
    case "$1" in
        --hostfile) [[ $# -ge 2 ]] || die "--hostfile requires a value"; HOSTFILE="$2"; shift 2 ;;
        --master) [[ $# -ge 2 ]] || die "--master requires a value"; MASTER_HOST="$2"; shift 2 ;;
        --ssh-user) [[ $# -ge 2 ]] || die "--ssh-user requires a value"; SSH_USER="$2"; shift 2 ;;
        --ssh-port) [[ $# -ge 2 ]] || die "--ssh-port requires a value"; SSH_PORT="$2"; shift 2 ;;
        --label) [[ $# -ge 2 ]] || die "--label requires a value"; RUN_LABEL="$2"; shift 2 ;;
        --mode) [[ $# -ge 2 ]] || die "--mode requires a value"; MODES_CSV="$2"; shift 2 ;;
        --pcie-data-size) [[ $# -ge 2 ]] || die "--pcie-data-size requires a value"; PCIE_DATA_SIZE="$2"; shift 2 ;;
        --pcie-mode) [[ $# -ge 2 ]] || die "--pcie-mode requires a value"; PCIE_MODE="$2"; shift 2 ;;
        --hbm-data-size) [[ $# -ge 2 ]] || die "--hbm-data-size requires a value"; HBM_DATA_SIZE="$2"; shift 2 ;;
        --vmem-data-size) [[ $# -ge 2 ]] || die "--vmem-data-size requires a value"; VMEM_DATA_SIZE="$2"; shift 2 ;;
        --target-devices) [[ $# -ge 2 ]] || die "--target-devices requires a value"; TARGET_DEVICES="$2"; shift 2 ;;
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
[[ -z "${XPROF_MODES_CSV}" || "${XPROF_MODES_CSV}" =~ ^(h2d|d2h|hbm_read|hbm_write|hbm_copy|vmem_copy)(,(h2d|d2h|hbm_read|hbm_write|hbm_copy|vmem_copy))*$ ]] || die "invalid --xprof-modes: ${XPROF_MODES_CSV}"
[[ "${PCIE_MODE}" =~ ^(one_to_one|one_to_many|both)$ ]] || die "unsupported PCIe mode: ${PCIE_MODE}"
[[ "${TARGET_DEVICES}" =~ ^[1-9][0-9]*$ && "${ITERATIONS}" =~ ^[1-9][0-9]*$ && "${WARMUP}" =~ ^[0-9]+$ ]] || die "device/iteration values are invalid"
[[ "${SSH_PORT}" =~ ^[1-9][0-9]*$ ]] && (( 10#${SSH_PORT} <= 65535 )) || die "--ssh-port must be in 1..65535"
IFS=',' read -r -a MODES <<< "${MODES_CSV}"
for mode in "${MODES[@]}"; do [[ "${mode}" =~ ^(h2d|d2h|hbm_read|hbm_write|hbm_copy|vmem_copy)$ ]] || die "unsupported mode: ${mode}"; done
load_container_runtime_environment
run_local_benchmark() {
    local node_identity
    node_identity="$(detect_node_identity)"
    timing_event local_benchmark start "node=${node_identity}" "testcase=tpubandwidth"
    printf 'Node identity: %s\n' "${node_identity}"
    printf 'TPUbandwidth: modes=%s pcie_mode=%s warmup=%s iterations=%s fallback_timing=%s xprof_modes=%s profile=%s local_devices_only=true\n' \
        "${MODES[*]}" "${PCIE_MODE}" "${WARMUP}" "${ITERATIONS}" \
        "$([[ ${XPROF_TIMING} -eq 1 ]] && printf xprof || printf cpu)" "${XPROF_MODES_CSV:-unset}" "${PROFILE}"
    local mode hbm_mode command_output metric
    local -a command expected_metrics
    for mode in "${MODES[@]}"; do
        timing_event "tpubandwidth.${mode}" start
        case "${mode}" in
            h2d|d2h) command=(python3 "${SRC_ROOT}/pcie/test_pcie.py" "${mode}" --pcie-mode "${PCIE_MODE}" --data-size "${PCIE_DATA_SIZE}" --target-devices "${TARGET_DEVICES}" --warmup "${WARMUP}" --iteration "${ITERATIONS}") ;;
            # Four scratch banks must leave VMEM space for compiler temporaries.
            hbm_read|hbm_write|hbm_copy) hbm_mode="${mode#hbm_}"; command=(python3 "${SRC_ROOT}/memory/test_hbm.py" --mode "${hbm_mode}" --data-size "${HBM_DATA_SIZE}" --block-shape 2048 1024 --dtype float32 --warmup "${WARMUP}" --iteration "${ITERATIONS}") ;;
            vmem_copy) command=(python3 "${SRC_ROOT}/memory/test_vmem.py" --data-size "${VMEM_DATA_SIZE}" --block-shape 2048 1024 --dtype float32 --warmup "${WARMUP}" --iteration "${ITERATIONS}") ;;
        esac
        if use_xprof_for_mode "${mode}"; then
            command+=(--xprof-timing)
            printf 'TPUbandwidth subtest=%s timing=xprof\n' "${mode}"
        else
            printf 'TPUbandwidth subtest=%s timing=cpu\n' "${mode}"
        fi
        (( PROFILE == 0 )) || command+=(--profile)
        if ! command_output="$("${command[@]}" 2>&1)"; then
            printf '%s\n' "${command_output}"
            die "TPUbandwidth ${mode} benchmark failed"
        fi
        printf '%s\n' "${command_output}"
        expected_metrics=("${mode}")
        if [[ "${mode}" == h2d || "${mode}" == d2h ]]; then
            [[ "${PCIE_MODE}" != one_to_many ]] || expected_metrics=()
            [[ "${PCIE_MODE}" == one_to_one ]] || expected_metrics+=("${mode}_one_to_many")
        fi
        for metric in "${expected_metrics[@]}"; do
            grep -Fq '"testcase":"tpubandwidth","metric":"'"${metric}"'"' <<< "${command_output}" \
                || die "TPUbandwidth ${metric} produced no structured metric"
        done
        timing_event "tpubandwidth.${mode}" end
    done
    timing_event local_benchmark end "node=${node_identity}" "testcase=tpubandwidth"
}

build_subtree_command() {
    local nodes_csv="$1" quoted
    local -a command=(bash "${SCRIPT_PATH}" --internal-subtree-nodes "${nodes_csv}" \
        --internal-run-token "${SUBTREE_RUN_TOKEN}" --ssh-user "${SSH_USER}" --ssh-port "${SSH_PORT}" \
        --mode "${MODES_CSV}" --pcie-data-size "${PCIE_DATA_SIZE}" --pcie-mode "${PCIE_MODE}" \
        --hbm-data-size "${HBM_DATA_SIZE}" --vmem-data-size "${VMEM_DATA_SIZE}" \
        --target-devices "${TARGET_DEVICES}" --iterations "${ITERATIONS}" --warmup "${WARMUP}")
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
        tar_path="${output_dir}/.subtree_${safe_child}.tar"; err_path="${output_dir}/.subtree_${safe_child}.stderr"
        child_tars+=("${tar_path}"); child_errs+=("${err_path}"); child_hosts+=("${child}")
        remote_command="$(build_subtree_command "${csv}")"
        ssh -p "${SSH_PORT}" -o BatchMode=yes -o StrictHostKeyChecking=accept-new  \
            -o LogLevel=ERROR -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
            "${SSH_USER}@${child}" "${remote_command}" > "${tar_path}" 2> "${err_path}" &
        child_pids+=("$!")
    done
    local_log="${output_dir}/${SUBTREE_RUN_TOKEN}_$(detect_node_identity).log"
    run_local_benchmark > "${local_log}" 2>&1 & local_pid=$!
    wait "${local_pid}" || failed=1
    for index in "${!child_pids[@]}"; do
        wait "${child_pids[index]}" || { printf '[ERROR] TPUbandwidth subtree rooted at %s failed\n' "${child_hosts[index]}" >&2; failed=1; }
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
    mkdir -p "${TEMP_ROOT}"; temp_dir="$(mktemp -d "${TEMP_ROOT}/tpubandwidth-subtree.XXXXXX")"
    subtree_execute_node "${temp_dir}" "${nodes[@]}" || status=$?
    tar cf - -C "${temp_dir}" .; rm -rf -- "${temp_dir}"; return "${status}"
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
    SUBTREE_RUN_TOKEN="$(date +%Y%m%d_%H%M%S)_singlenode_tpubandwidth"
    [[ -z "${RUN_LABEL}" ]] || SUBTREE_RUN_TOKEN+="_$(sanitize_label "${RUN_LABEL}")"
    mkdir -p "${TEMP_ROOT}"; temp_dir="$(mktemp -d "${TEMP_ROOT}/tpubandwidth-root.XXXXXX")"
    subtree_execute_node "${temp_dir}" "${hosts[@]}" || status=$?
    log_count="$(find "${temp_dir}" -maxdepth 1 -type f -name '*.log' | wc -l | tr -d ' ')"
    (( log_count == ${#hosts[@]} )) || { printf '[ERROR] expected %s node logs, collected %s\n' "${#hosts[@]}" "${log_count}" >&2; status=1; }
    mkdir -p "${LOG_ROOT}"; shopt -s nullglob
    for file in "${temp_dir}"/*.log; do destination="${LOG_ROOT}/$(basename "${file}")"; [[ ! -e "${destination}" ]] || destination="${destination%.log}_$RANDOM.log"; mv "${file}" "${destination}"; done
    printf '[INFO] collected %s node logs\n' "${log_count}"
    shopt -u nullglob; rm -rf -- "${temp_dir}"; return "${status}"
}

if [[ -n "${SUBTREE_NODES_CSV}" ]]; then run_subtree; else run_all_hosts; fi
