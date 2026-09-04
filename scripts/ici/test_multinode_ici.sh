#!/usr/bin/env bash
# Launch one Slice-wide TPU ICI process group through a binary SSH tree.
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
timing_event() {
    local phase="$1" event="$2"
    shift 2
    timing_event_at "$(date '+%Y-%m-%dT%H:%M:%S%z')" "${phase}" "${event}" "$@"
}
TIMING_SCRIPT_STARTED_EPOCH="$(date +%s)"
TIMING_SCRIPT_STARTED_AT="$(date '+%Y-%m-%dT%H:%M:%S%z')"
timing_finish() {
    local exit_code="${1:-$?}" finished_epoch
    finished_epoch="$(date +%s)"
    timing_event script end \
        "exit_code=${exit_code}" \
        "duration_seconds=$((finished_epoch - TIMING_SCRIPT_STARTED_EPOCH))"
}
timing_event_at "${TIMING_SCRIPT_STARTED_AT}" script start
trap timing_finish EXIT

die() { printf '[ERROR] %s\n' "$*" >&2; exit 1; }
sanitize_label() { printf '%s' "$1" | tr '[:space:]/' '__' | sed 's/[^A-Za-z0-9._-]/_/g'; }
detect_node_identity() {
    local node_name="${NODE_SN_NAME:-}" host_ip="${NODE_HOST_IP:-}" interface address
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
validate_host_token() {
    [[ "$1" =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid host or IP token: $1"
}

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

ENTRY="${REPO_ROOT}/src/ici/test_ici.py"
SCRIPT_PATH="${REPO_ROOT}/scripts/ici/test_multinode_ici.sh"
LOG_ROOT="${REPO_ROOT}/logs/ici"
TEMP_ROOT="${REPO_ROOT}/logs/.tmp"
HOSTFILE=""
BLOCK_RANGE=""
SSH_PORT=22
SSH_USER="${USER:-$(id -un)}"
COORDINATOR_PORT=8476
MODES_CSV="p2p,alltoall,alltoall_parallel,allreduce,allreduce_parallel"
P2P_DATA_SIZE=1GiB
P2P_PAIR_MODE=all
ALLTOALL_DATA_SIZE=1GiB
ALLREDUCE_DATA_SIZE=4GiB
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
RUN_LABEL=""
SUBTREE_NODES_CSV=""
SUBTREE_COORDINATOR_ADDRESS=""
SUBTREE_PROCESS_COUNT=""
SUBTREE_RUN_TOKEN=""

SSH_OPTS=(
    -p "${SSH_PORT}"
    -o BatchMode=yes
    -o StrictHostKeyChecking=accept-new
    -o LogLevel=ERROR
    -o ConnectTimeout=10
    -o ServerAliveInterval=30
    -o ServerAliveCountMax=3
)

usage() {
    cat <<'EOF'
Usage: test_multinode_ici.sh --hostfile <path> [options]
  --hostfile <path>             One TPU VM host per line.
  --block-range <range>         TPU chip slices, e.g. 0:2,0:2,2:4.
  --mode <csv>                  p2p,alltoall,alltoall_parallel,allreduce,allreduce_parallel.
  --p2p-data-size <size>        Default: 1GiB.
  --p2p-pair-mode <mode>        all (default) or neighbors (all die links + adjacent core0 links).
  --alltoall-data-size <size>   Default: 1GiB.
  --allreduce-data-size <size>  Default: 4GiB.
  --iterations <N>              Default: 5.
  --warmup <N>                  Default: 2.
  --ssh-user <name>           SSH login (default: current user).
  --ssh-port <N>                Default: 22.
  --xprof-timing                Use temporary Xprof traces for timing.
  --xprof-modes <csv>           Use Xprof only for selected --mode values;
                               non-empty selection overrides --xprof-timing.
  --profile                     Retain Xprof traces and HLO dumps.
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --hostfile) [[ $# -ge 2 ]] || die "--hostfile requires a value"; HOSTFILE="$2"; shift 2 ;;
        --block-range) [[ $# -ge 2 ]] || die "--block-range requires a value"; BLOCK_RANGE="$2"; shift 2 ;;
        --mode) [[ $# -ge 2 ]] || die "--mode requires a value"; MODES_CSV="$2"; shift 2 ;;
        --p2p-data-size) [[ $# -ge 2 ]] || die "--p2p-data-size requires a value"; P2P_DATA_SIZE="$2"; shift 2 ;;
        --p2p-pair-mode) [[ $# -ge 2 ]] || die "--p2p-pair-mode requires a value"; P2P_PAIR_MODE="$2"; shift 2 ;;
        --alltoall-data-size) [[ $# -ge 2 ]] || die "--alltoall-data-size requires a value"; ALLTOALL_DATA_SIZE="$2"; shift 2 ;;
        --allreduce-data-size) [[ $# -ge 2 ]] || die "--allreduce-data-size requires a value"; ALLREDUCE_DATA_SIZE="$2"; shift 2 ;;
        --iterations) [[ $# -ge 2 ]] || die "--iterations requires a value"; ITERATIONS="$2"; shift 2 ;;
        --warmup) [[ $# -ge 2 ]] || die "--warmup requires a value"; WARMUP="$2"; shift 2 ;;
        --ssh-user) [[ $# -ge 2 ]] || die "--ssh-user requires a value"; SSH_USER="$2"; shift 2 ;;
        --ssh-port) [[ $# -ge 2 ]] || die "--ssh-port requires a value"; SSH_PORT="$2"; shift 2 ;;
        --label) [[ $# -ge 2 ]] || die "--label requires a value"; RUN_LABEL="$2"; shift 2 ;;
        --batch-test|--batch_test) shift ;;
        --xprof-timing) XPROF_TIMING=1; shift ;;
        --xprof-modes) [[ $# -ge 2 ]] || die "--xprof-modes requires a value"; XPROF_MODES_CSV="$2"; shift 2 ;;
        --profile|--dump-hlo) PROFILE=1; shift ;;
        --internal-subtree-nodes) [[ $# -ge 2 ]] || die "--internal-subtree-nodes requires a value"; SUBTREE_NODES_CSV="$2"; shift 2 ;;
        --internal-coordinator-address) [[ $# -ge 2 ]] || die "--internal-coordinator-address requires a value"; SUBTREE_COORDINATOR_ADDRESS="$2"; shift 2 ;;
        --internal-process-count) [[ $# -ge 2 ]] || die "--internal-process-count requires a value"; SUBTREE_PROCESS_COUNT="$2"; shift 2 ;;
        --internal-run-token) [[ $# -ge 2 ]] || die "--internal-run-token requires a value"; SUBTREE_RUN_TOKEN="$2"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done
[[ "${SSH_USER}" =~ ^[A-Za-z_][A-Za-z0-9_.-]*$ ]] || die "invalid --ssh-user"

[[ -z "${XPROF_MODES_CSV}" || "${XPROF_MODES_CSV}" =~ ^(p2p|alltoall|alltoall_parallel|allreduce|allreduce_parallel)(,(p2p|alltoall|alltoall_parallel|allreduce|allreduce_parallel))*$ ]] || die "invalid --xprof-modes: ${XPROF_MODES_CSV}"
SSH_OPTS[1]="${SSH_PORT}"
[[ "${P2P_PAIR_MODE}" =~ ^(all|neighbors)$ ]] || die "invalid --p2p-pair-mode: ${P2P_PAIR_MODE}"
[[ -f "${ENTRY}" ]] || die "ICI entry not found"
[[ "${ITERATIONS}" =~ ^[1-9][0-9]*$ && "${WARMUP}" =~ ^[0-9]+$ ]] \
    || die "iteration values are invalid"
[[ "${SSH_PORT}" =~ ^[1-9][0-9]*$ ]] && (( 10#${SSH_PORT} <= 65535 )) \
    || die "--ssh-port must be in 1..65535"
IFS=',' read -r -a MODES <<< "${MODES_CSV}"
for mode in "${MODES[@]}"; do
    [[ "${mode}" =~ ^(p2p|alltoall|alltoall_parallel|allreduce|allreduce_parallel)$ ]] \
        || die "unsupported mode: ${mode}"
done
[[ -z "${BLOCK_RANGE}" || "${BLOCK_RANGE}" =~ ^[0-9]+:[0-9]+,[0-9]+:[0-9]+,[0-9]+:[0-9]+$ ]] \
    || die "--block-range must use x,y,z half-open slices such as 0:2,0:2,2:4"
BLOCK_LABEL="$(sanitize_label "${BLOCK_RANGE:-full_slice}")"
load_container_runtime_environment
# The image also contains the single-node copy of the TPU package. Force this
# launcher to resolve imports from its multi-node payload first.
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

run_rank() {
    local mode="$1" rank="$2" host_label="$3"
    local subcommand data_size node_identity
    local -a command
    node_identity="$(detect_node_identity)"
    timing_event local_benchmark start \
        "node=${node_identity}" "testcase=ici" "mode=${mode}" "process_id=${rank}"
    printf 'Node identity: %s\n' "${node_identity}"
    printf 'ICI local rank: process_id=%s host=%s block_range=%s mode=%s\n' \
        "${rank}" "${host_label}" "${BLOCK_RANGE:-full_slice}" "${mode}"
    case "${mode}" in
        p2p) subcommand=p2p; data_size="${P2P_DATA_SIZE}" ;;
        alltoall) subcommand=a2a; data_size="${ALLTOALL_DATA_SIZE}" ;;
        alltoall_parallel) subcommand=a2a; data_size="${ALLTOALL_DATA_SIZE}" ;;
        allreduce) subcommand=ar; data_size="${ALLREDUCE_DATA_SIZE}" ;;
        allreduce_parallel) subcommand=ar; data_size="${ALLREDUCE_DATA_SIZE}" ;;
    esac
    command=(
        python3 "${ENTRY}" "${subcommand}"
        --runtime-scope slice
        --coordinator-address "${SUBTREE_COORDINATOR_ADDRESS}"
        --process-count "${SUBTREE_PROCESS_COUNT}"
        --process-id "${rank}"
        --data-size "${data_size}"
        --warmup "${WARMUP}"
        --iteration "${ITERATIONS}"
    )
    [[ -z "${BLOCK_RANGE}" ]] || command+=(--block-range "${BLOCK_RANGE}")
    [[ "${mode}" != p2p ]] || command+=(--p2p-pair-mode "${P2P_PAIR_MODE}")
    [[ "${mode}" != alltoall_parallel && "${mode}" != allreduce_parallel ]] \
        || command+=(--parallel)
    if use_xprof_for_mode "${mode}"; then
        command+=(--xprof-timing)
        printf 'ICI subtest=%s timing=xprof\n' "${mode}"
    else
        printf 'ICI subtest=%s timing=cpu\n' "${mode}"
    fi
    (( PROFILE == 0 )) || command+=(--profile)
    # The caller collects failures with ||, which disables Bash errexit inside
    # nested functions. Preserve the pipeline status explicitly.
    local exit_code=0
    "${command[@]}" 2>&1 | sed "s/^/[${host_label}] /" || exit_code=$?
    timing_event local_benchmark end \
        "node=${node_identity}" "testcase=ici" "mode=${mode}" "process_id=${rank}" \
        "exit_code=${exit_code}"
    return "${exit_code}"
}

build_subtree_command() {
    local nodes_csv="$1" mode="$2" quoted_command
    local -a command=(
        bash "${SCRIPT_PATH}"
        --internal-subtree-nodes "${nodes_csv}"
        --internal-coordinator-address "${SUBTREE_COORDINATOR_ADDRESS}"
        --internal-process-count "${SUBTREE_PROCESS_COUNT}"
        --internal-run-token "${SUBTREE_RUN_TOKEN}"
        --mode "${mode}"
        --p2p-data-size "${P2P_DATA_SIZE}"
        --p2p-pair-mode "${P2P_PAIR_MODE}"
        --alltoall-data-size "${ALLTOALL_DATA_SIZE}"
        --allreduce-data-size "${ALLREDUCE_DATA_SIZE}"
        --iterations "${ITERATIONS}"
        --warmup "${WARMUP}"
        --ssh-user "${SSH_USER}" --ssh-port "${SSH_PORT}"
        --batch-test
    )
    [[ -z "${BLOCK_RANGE}" ]] || command+=(--block-range "${BLOCK_RANGE}")
    [[ -z "${XPROF_MODES_CSV}" ]] || command+=(--xprof-modes "${XPROF_MODES_CSV}")
    (( XPROF_TIMING == 0 )) || command+=(--xprof-timing)
    (( PROFILE == 0 )) || command+=(--profile)
    printf -v quoted_command '%q ' "${command[@]}"
    printf '%s' "${quoted_command}"
}

subtree_execute_node() {
    local output_dir="$1" mode="$2"
    shift 2
    local -a nodes=("$@") child_csvs=() child_hosts=() child_pids=() child_tars=() child_errs=()
    local node_count="${#nodes[@]}"
    (( node_count > 0 )) || return 1

    local current="${nodes[0]}"
    local current_rank="${current%%@*}"
    local current_host="${current#*@}"
    [[ "${current_rank}" =~ ^[0-9]+$ && -n "${current_host}" && "${current_host}" != "${current}" ]] \
        || die "invalid internal rank@host entry: ${current}"
    validate_host_token "${current_host}"

    local remaining=$((node_count - 1))
    local left_count=$(((remaining + 1) / 2))
    local right_count=$((remaining - left_count))
    local -a left_nodes=() right_nodes=()
    local csv child_root child_host safe_child remote_command index local_pid failed=0 local_log tar_path err_path
    if (( left_count > 0 )); then
        left_nodes=("${nodes[@]:1:left_count}")
        child_csvs+=("$(IFS=,; printf '%s' "${left_nodes[*]}")")
    fi
    if (( right_count > 0 )); then
        right_nodes=("${nodes[@]:$((left_count + 1)):right_count}")
        child_csvs+=("$(IFS=,; printf '%s' "${right_nodes[*]}")")
    fi

    # Bash 4.2 treats an empty array as unset under nounset.
    for csv in "${child_csvs[@]+"${child_csvs[@]}"}"; do
        child_root="${csv%%,*}"
        child_host="${child_root#*@}"
        validate_host_token "${child_host}"
        remote_command="$(build_subtree_command "${csv}" "${mode}")"
        safe_child="$(sanitize_label "${child_host}")"
        tar_path="${output_dir}/.subtree_${safe_child}.tar"
        err_path="${output_dir}/.subtree_${safe_child}.stderr"
        child_tars+=("${tar_path}")
        child_errs+=("${err_path}")
        ssh "${SSH_OPTS[@]}" "${SSH_USER}@${child_host}" \
            "exec ${remote_command}" \
            > "${tar_path}" 2> "${err_path}" &
        child_hosts+=("${child_host}")
        child_pids+=("$!")
    done

    local_log="${output_dir}/${SUBTREE_RUN_TOKEN}_$(detect_node_identity).log"
    run_rank "${mode}" "${current_rank}" "${current_host}" > "${local_log}" 2>&1 &
    local_pid=$!
    wait "${local_pid}" || failed=1
    for index in "${!child_pids[@]}"; do
        if ! wait "${child_pids[index]}"; then
            printf '[ERROR] ICI subtree rooted at %s failed\n' "${child_hosts[index]}" >&2
            failed=1
        fi
        if [[ -s "${child_tars[index]}" ]]; then
            tar xf "${child_tars[index]}" -C "${output_dir}" || failed=1
        else
            printf '[ERROR] ICI subtree rooted at %s returned no log archive\n' \
                "${child_hosts[index]}" >&2
            failed=1
        fi
        [[ ! -s "${child_errs[index]}" ]] || cat "${child_errs[index]}" >&2
        rm -f "${child_tars[index]}" "${child_errs[index]}"
    done
    return "${failed}"
}

run_subtree() {
    [[ "${SUBTREE_PROCESS_COUNT}" =~ ^[1-9][0-9]*$ ]] \
        || die "invalid internal process count"
    [[ -n "${SUBTREE_COORDINATOR_ADDRESS}" ]] \
        || die "internal coordinator address is required"
    [[ -n "${SUBTREE_RUN_TOKEN}" ]] || die "internal run token is required"
    (( ${#MODES[@]} == 1 )) || die "internal subtree mode requires exactly one ICI mode"
    local -a nodes=()
    IFS=',' read -r -a nodes <<< "${SUBTREE_NODES_CSV}"
    (( ${#nodes[@]} > 0 )) || die "internal subtree node list is empty"
    local temp_dir status=0
    mkdir -p "${TEMP_ROOT}"
    temp_dir="$(mktemp -d "${TEMP_ROOT}/ici-subtree.XXXXXX")"
    subtree_execute_node "${temp_dir}" "${MODES[0]}" "${nodes[@]}" || status=$?
    tar cf - -C "${temp_dir}" .
    rm -rf -- "${temp_dir}"
    return "${status}"
}

run_slice_mode() {
    local mode="$1" index temp_dir status=0 log_count metric_count file destination label_segment=""
    local -a ranked_hosts=()
    for index in "${!HOSTS[@]}"; do
        ranked_hosts+=("${index}@${HOSTS[index]}")
    done
    [[ -z "${RUN_LABEL}" ]] || label_segment="_$(sanitize_label "${RUN_LABEL}")"
    SUBTREE_RUN_TOKEN="$(date +%Y%m%d_%H%M%S)_multinode_ici${label_segment}_${mode}_block_${BLOCK_LABEL}"
    mkdir -p "${TEMP_ROOT}"
    temp_dir="$(mktemp -d "${TEMP_ROOT}/ici-root.XXXXXX")"
    timing_event "ici.${mode}" start "runtime_scope=slice" "hosts=${#HOSTS[@]}"
    subtree_execute_node "${temp_dir}" "${mode}" "${ranked_hosts[@]}" || status=$?
    log_count="$(find "${temp_dir}" -maxdepth 1 -type f -name '*.log' | wc -l | tr -d ' ')"
    (( log_count == ${#HOSTS[@]} )) || {
        printf '[ERROR] expected %s node logs, collected %s\n' "${#HOSTS[@]}" "${log_count}" >&2
        status=1
    }
    shopt -s nullglob
    local -a collected_logs=("${temp_dir}"/*.log)
    if (( ${#collected_logs[@]} > 0 )); then
        if grep -Eq 'Traceback \(most recent call last\)|DEADLINE_EXCEEDED|Terminating process because the JAX distributed service detected fatal errors' "${collected_logs[@]}"; then
            printf '[ERROR] ICI %s logs contain a distributed runtime failure\n' "${mode}" >&2
            status=1
        fi
        metric_count="$(awk 'index($0,"[COMMPILOT_METRIC]"){count++} END{print count+0}' "${collected_logs[@]}")"
    else
        metric_count=0
    fi
    (( metric_count > 0 )) || {
        printf '[ERROR] ICI %s produced no structured metric records\n' "${mode}" >&2
        status=1
    }
    mkdir -p "${LOG_ROOT}"
    for file in "${temp_dir}"/*.log; do
        destination="${LOG_ROOT}/$(basename "${file}")"
        [[ ! -e "${destination}" ]] || destination="${destination%.log}_$RANDOM.log"
        mv "${file}" "${destination}"
    done
    printf '[INFO] collected %s node logs\n' "${log_count}"
    shopt -u nullglob; rm -rf -- "${temp_dir}"
    timing_event "ici.${mode}" end "runtime_scope=slice" "hosts=${#HOSTS[@]}"
    (( status == 0 )) || die "ICI ${mode} failed in the Slice process group"
}

if [[ -n "${SUBTREE_NODES_CSV}" ]]; then
    run_subtree
    exit $?
fi

[[ -n "${HOSTFILE}" && -f "${HOSTFILE}" ]] \
    || die "--hostfile must reference an existing file"
mapfile -t ALL_HOSTS < <(awk '!/^[[:space:]]*(#|$)/ {print $1}' "${HOSTFILE}")
(( ${#ALL_HOSTS[@]} > 1 )) || die "multi-node ICI requires at least two hosts"
HOSTS=("${ALL_HOSTS[@]}")
for host in "${HOSTS[@]}"; do validate_host_token "${host}"; done
[[ "$(printf '%s\n' "${HOSTS[@]}" | sort -u | wc -l | tr -d ' ')" == "${#HOSTS[@]}" ]] \
    || die "ICI hosts must be unique"

SUBTREE_COORDINATOR_ADDRESS="${HOSTS[0]}:${COORDINATOR_PORT}"
SUBTREE_PROCESS_COUNT="${#HOSTS[@]}"
printf 'ICI hosts: %s\n' "${HOSTS[*]}"
printf 'ICI runtime_scope=slice host_count=%s block_range=%s runtime_topology=auto launch_tree=binary\n' \
    "${SUBTREE_PROCESS_COUNT}" "${BLOCK_RANGE:-full_slice}"
printf 'ICI coordinator: %s process_count=%s\n' \
    "${SUBTREE_COORDINATOR_ADDRESS}" "${SUBTREE_PROCESS_COUNT}"
for mode in "${MODES[@]}"; do run_slice_mode "${mode}"; done
