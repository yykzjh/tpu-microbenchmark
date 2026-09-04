#!/usr/bin/env bash
# Measure pairwise TCP bandwidth on matching network devices with iperf.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PATH="${REPO_ROOT}/.venv/bin:${PATH}"


# CommPilot second-precision timing markers.
timing_event_at() {
    local timestamp="$1"
    local phase="$2"
    local event="$3"
    shift 3
    printf '[TIMING] timestamp=%s phase=%s event=%s script=%s' \
        "${timestamp}" "${phase}" "${event}" "${BASH_SOURCE[0]##*/}" >&2
    if (( $# > 0 )); then
        printf ' %s' "$@" >&2
    fi
    printf '\n' >&2
}

timing_event() {
    local phase="$1"
    local event="$2"
    shift 2
    timing_event_at "$(date '+%Y-%m-%dT%H:%M:%S%z')" "${phase}" "${event}" "$@"
}

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

HOSTFILE=""
START_INDEX=0
END_INDEX=2
HOST_RANGES=""
SSH_PORT=22
SSH_USER="${USER:-$(id -un)}"
NET_DEVICES_CSV="eth0,eth1"
PARALLEL_STREAMS=32
DURATION=30
INTERVAL=1
BUFFER_LENGTH="1M"
TCP_WINDOW="16M"
TRANSFER_SIZE=""
BASE_PORT=5201
TCP_NO_DELAY=0
BATCH_TEST=0

CURRENT_SERVER_HOST=""
CURRENT_SERVER_PIDFILE=""
CURRENT_SERVER_LOG=""

die() { printf '[ERROR] %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<'EOF'
Usage: test_multinode_iperf.sh [options]
  --hostfile <path>          Hostfile containing the test nodes.
  --start-index <N>          Inclusive host index. Default: 0.
  --end-index <N>            Exclusive host index. Default: 2.
  --host-ranges <ranges>     End-exclusive host range; exactly two hosts are required.
  --ssh-user <name>           SSH login (default: current user).
  --ssh-port <N>             SSH port. Default: 22.
  --net-devices <csv>        Matching device names to test serially. Default: eth0,eth1.
  --parallel-streams <N>     iperf parallel TCP streams (-P). Default: 32.
  --duration <seconds>       Duration per direction (-t). Default: 30.
  --interval <seconds>       Report interval (-i). Default: 1.
  --buffer-length <size>     Read/write buffer length (-l). Default: 1M.
  --tcp-window <size>        TCP window size (-w). Default: 16M.
  --transfer-size <size>     Send this many bytes (-n) instead of using --duration.
  --base-port <N>            First iperf server port. Default: 5201.
  --tcp-no-delay             Disable Nagle buffering (-N).
  --batch-test               Write only to stdout for the Web-owned log capture.
  --help                     Show this help.

Each selected device is tested in both directions. The script discovers the
device-local NUMA node from sysfs and binds both iperf endpoints with numactl.
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --hostfile) [[ $# -ge 2 ]] || die "--hostfile requires a value"; HOSTFILE="$2"; shift 2 ;;
        --start-index) [[ $# -ge 2 ]] || die "--start-index requires a value"; START_INDEX="$2"; shift 2 ;;
        --end-index) [[ $# -ge 2 ]] || die "--end-index requires a value"; END_INDEX="$2"; shift 2 ;;
        --host-ranges) [[ $# -ge 2 ]] || die "--host-ranges requires a value"; HOST_RANGES="$2"; shift 2 ;;
        --ssh-user) [[ $# -ge 2 ]] || die "--ssh-user requires a value"; SSH_USER="$2"; shift 2 ;;
        --ssh-port) [[ $# -ge 2 ]] || die "--ssh-port requires a value"; SSH_PORT="$2"; shift 2 ;;
        --net-devices) [[ $# -ge 2 ]] || die "--net-devices requires a value"; NET_DEVICES_CSV="$2"; shift 2 ;;
        --parallel-streams) [[ $# -ge 2 ]] || die "--parallel-streams requires a value"; PARALLEL_STREAMS="$2"; shift 2 ;;
        --duration) [[ $# -ge 2 ]] || die "--duration requires a value"; DURATION="$2"; shift 2 ;;
        --interval) [[ $# -ge 2 ]] || die "--interval requires a value"; INTERVAL="$2"; shift 2 ;;
        --buffer-length) [[ $# -ge 2 ]] || die "--buffer-length requires a value"; BUFFER_LENGTH="$2"; shift 2 ;;
        --tcp-window) [[ $# -ge 2 ]] || die "--tcp-window requires a value"; TCP_WINDOW="$2"; shift 2 ;;
        --transfer-size) [[ $# -ge 2 ]] || die "--transfer-size requires a value"; TRANSFER_SIZE="$2"; shift 2 ;;
        --base-port) [[ $# -ge 2 ]] || die "--base-port requires a value"; BASE_PORT="$2"; shift 2 ;;
        --tcp-no-delay) TCP_NO_DELAY=1; shift ;;
        --batch-test) BATCH_TEST=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done
[[ "${SSH_USER}" =~ ^[A-Za-z_][A-Za-z0-9_.-]*$ ]] || die "invalid --ssh-user"

for numeric in START_INDEX END_INDEX SSH_PORT PARALLEL_STREAMS DURATION INTERVAL BASE_PORT; do
    [[ "${!numeric}" =~ ^[0-9]+$ ]] || die "${numeric} must be a non-negative integer"
done
(( END_INDEX > START_INDEX )) || die "--end-index must be greater than --start-index"
(( SSH_PORT >= 1 && SSH_PORT <= 65535 )) || die "--ssh-port must be between 1 and 65535"
(( PARALLEL_STREAMS >= 1 )) || die "--parallel-streams must be at least 1"
(( DURATION >= 1 )) || die "--duration must be at least 1"
(( INTERVAL >= 1 )) || die "--interval must be at least 1"
(( BASE_PORT >= 1 && BASE_PORT <= 65535 )) || die "--base-port must be between 1 and 65535"
for size_value in BUFFER_LENGTH TCP_WINDOW; do
    [[ "${!size_value}" =~ ^[1-9][0-9]*[KMGkmg]?$ ]] \
        || die "${size_value} must be an iperf size such as 1M or 16M"
done
if [[ -n "${TRANSFER_SIZE}" ]]; then
    [[ "${TRANSFER_SIZE}" =~ ^[1-9][0-9]*[KMGkmg]?$ ]] \
        || die "TRANSFER_SIZE must be an iperf size such as 4G"
fi
[[ -n "${HOSTFILE}" && -r "${HOSTFILE}" ]] || die "--hostfile must name a readable file"

mapfile -t ALL_HOSTS < <(awk 'NF && $1 !~ /^#/ {print $1}' "${HOSTFILE}")
(( ${#ALL_HOSTS[@]} >= 2 )) || die "hostfile must contain at least two nodes"

SELECTED_INDEXES=()
if [[ -n "${HOST_RANGES}" ]]; then
    [[ "${HOST_RANGES}" =~ ^[0-9]+-[0-9]+$ ]] \
        || die "--host-ranges must contain one end-exclusive range, for example 0-2"
    range_start="${HOST_RANGES%-*}"
    range_end="${HOST_RANGES#*-}"
    (( range_end > range_start )) || die "invalid --host-ranges value: ${HOST_RANGES}"
    for ((index = range_start; index < range_end; index++)); do
        SELECTED_INDEXES+=("${index}")
    done
else
    for ((index = START_INDEX; index < END_INDEX; index++)); do
        SELECTED_INDEXES+=("${index}")
    done
fi
(( ${#SELECTED_INDEXES[@]} == 2 )) || die "iperf requires exactly two selected hosts"
for index in "${SELECTED_INDEXES[@]}"; do
    (( index >= 0 && index < ${#ALL_HOSTS[@]} )) || die "host index is out of range: ${index}"
done
HOSTS=("${ALL_HOSTS[${SELECTED_INDEXES[0]}]}" "${ALL_HOSTS[${SELECTED_INDEXES[1]}]}")

IFS=',' read -r -a NET_DEVICES <<< "${NET_DEVICES_CSV}"
(( ${#NET_DEVICES[@]} > 0 )) || die "--net-devices must select at least one device"
declare -A SEEN_DEVICES=()
for device in "${NET_DEVICES[@]}"; do
    [[ "${device}" =~ ^[A-Za-z0-9_.:-]+$ ]] || die "unsupported network device name: ${device}"
    [[ -z "${SEEN_DEVICES[${device}]:-}" ]] || die "duplicate network device: ${device}"
    SEEN_DEVICES["${device}"]=1
done

SSH=(ssh -p "${SSH_PORT}" -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
     -o LogLevel=ERROR -o ConnectTimeout=10)

remote_shell() {
    local host="$1"
    shift
    "${SSH[@]}" "${SSH_USER}@${host}" "$*"
}

cleanup_server() {
    if [[ -n "${CURRENT_SERVER_HOST}" ]]; then
        remote_shell "${CURRENT_SERVER_HOST}" \
            "if [[ -s '${CURRENT_SERVER_PIDFILE}' ]]; then kill \$(cat '${CURRENT_SERVER_PIDFILE}') 2>/dev/null || true; fi; rm -f -- '${CURRENT_SERVER_PIDFILE}' '${CURRENT_SERVER_LOG}'" \
            >/dev/null 2>&1 || true
        CURRENT_SERVER_HOST=""
        CURRENT_SERVER_PIDFILE=""
        CURRENT_SERVER_LOG=""
    fi
}
trap 'status=$?; cleanup_server; timing_finish "${status}"' EXIT

device_ipv4() {
    local host="$1"
    local device="$2"
    remote_shell "${host}" \
        "ip -4 -o addr show dev '${device}' scope global | awk 'NR == 1 {sub(/\\/.*/, \"\", \$4); print \$4}'"
}

device_numa_node() {
    local host="$1"
    local device="$2"
    remote_shell "${host}" \
        "test -r '/sys/class/net/${device}/device/numa_node' && cat '/sys/class/net/${device}/device/numa_node' || true"
}

validate_endpoint() {
    local host="$1"
    local device="$2"
    remote_shell "${host}" \
        "command -v iperf >/dev/null; iperf --version 2>&1 | grep -Eiq 'iperf version 2|iperf 2'; command -v numactl >/dev/null; ip link show dev '${device}' >/dev/null"
}

wait_for_server() {
    local client_host="$1"
    local server_ip="$2"
    local port="$3"
    local attempt
    for attempt in $(seq 1 20); do
        if remote_shell "${client_host}" "nc -z -w 1 '${server_ip}' '${port}'" >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.25
    done
    return 1
}

emit_metric() {
    local value="$1"
    local device="$2"
    local direction="$3"
    local client_host="$4"
    local server_host="$5"
    local client_numa="$6"
    local server_numa="$7"
    jq -cn \
        --argjson value "${value}" \
        --arg device "${device}_${direction}" \
        --arg net_device "${device}" \
        --arg direction "${direction}" \
        --arg client "${client_host}" \
        --arg server "${server_host}" \
        --argjson client_numa "${client_numa}" \
        --argjson server_numa "${server_numa}" \
        --argjson parallel_streams "${PARALLEL_STREAMS}" \
        '{testcase:"iperf",metric:"bandwidth",value:$value,unit:"Gb/s",device:$device,dimensions:{net_device:$net_device,direction:$direction,client:$client,server:$server,client_numa:$client_numa,server_numa:$server_numa,parallel_streams:$parallel_streams}}' \
        | sed 's/^/[COMMPILOT_METRIC] /'
}

run_direction() {
    local device="$1"
    local direction="$2"
    local client_host="$3"
    local server_host="$4"
    local client_ip="$5"
    local server_ip="$6"
    local client_numa="$7"
    local server_numa="$8"
    local port="$9"
    local token="commpilot-iperf-$$_${device}_${port}"
    local client_output server_output bandwidth_bps bandwidth_gbps

    CURRENT_SERVER_HOST="${server_host}"
    CURRENT_SERVER_PIDFILE="${REPO_ROOT}/logs/.tmp/${token}.pid"
    CURRENT_SERVER_LOG="${REPO_ROOT}/logs/.tmp/${token}.server.log"
    remote_shell "${server_host}" \
        "mkdir -p '${REPO_ROOT}/logs/.tmp'; nohup numactl --cpunodebind='${server_numa}' --membind='${server_numa}' iperf -s -B '${server_ip}' -p '${port}' -y C >'${CURRENT_SERVER_LOG}' 2>&1 & echo \$! >'${CURRENT_SERVER_PIDFILE}'"
    wait_for_server "${client_host}" "${server_ip}" "${port}" \
        || die "iperf server did not become ready: ${server_host}/${device}:${port}"

    client_args="-c '${server_ip}' -B '${client_ip}' -p '${port}' -P '${PARALLEL_STREAMS}' -i '${INTERVAL}' -l '${BUFFER_LENGTH}' -w '${TCP_WINDOW}' -y C"
    if [[ -n "${TRANSFER_SIZE}" ]]; then
        client_args+=" -n '${TRANSFER_SIZE}'"
    else
        client_args+=" -t '${DURATION}'"
    fi
    (( TCP_NO_DELAY == 0 )) || client_args+=" -N"
    timing_event iperf start "device=${device}" "direction=${direction}" \
        "client=${client_host}" "server=${server_host}" "port=${port}"
    client_output="$(remote_shell "${client_host}" \
        "numactl --cpunodebind='${client_numa}' --membind='${client_numa}' iperf ${client_args}")"
    printf '%s\n' "${client_output}"
    server_output="$(remote_shell "${server_host}" "cat '${CURRENT_SERVER_LOG}' 2>/dev/null || true")"
    [[ -z "${server_output}" ]] || printf '%s\n' "${server_output}"
    cleanup_server
    bandwidth_bps="$(printf '%s\n' "${client_output}" | awk -F, '
        function interval_duration(interval, bounds) {
            split(interval, bounds, "-")
            return (bounds[2] + 0) - (bounds[1] + 0)
        }
        $6 ~ /^[0-9]+$/ && $7 ~ /^[0-9.]+-[0-9.]+$/ && \
                $9 ~ /^[0-9]+([.][0-9]+)?$/ {
            stream_id = $6
            duration = interval_duration($7)
            if (!(stream_id in longest_duration) || duration > longest_duration[stream_id]) {
                longest_duration[stream_id] = duration
                average_bps[stream_id] = $9 + 0
            }
        }
        END {
            for (stream_id in average_bps) total_bps += average_bps[stream_id]
            if (total_bps > 0) printf "%.0f", total_bps
        }
    ')"
    [[ -n "${bandwidth_bps}" ]] || die "unable to parse iperf CSV bandwidth for ${device}/${direction}"
    bandwidth_gbps="$(awk -v bits="${bandwidth_bps}" 'BEGIN {printf "%.3f", bits / 1000000000}')"
    emit_metric "${bandwidth_gbps}" "${device}" "${direction}" \
        "${client_host}" "${server_host}" "${client_numa}" "${server_numa}"
    timing_event iperf end "device=${device}" "direction=${direction}" \
        "bandwidth_Gbps=${bandwidth_gbps}"
}

run_suite() {
    local device_index device port
    local host0_ip host1_ip host0_numa host1_numa
    printf '[INFO] iperf pair: %s <-> %s\n' "${HOSTS[0]}" "${HOSTS[1]}"
    printf '[INFO] devices: %s; parallel streams: %s\n' "${NET_DEVICES_CSV}" "${PARALLEL_STREAMS}"
    for device_index in "${!NET_DEVICES[@]}"; do
        device="${NET_DEVICES[${device_index}]}"
        port=$((BASE_PORT + device_index))
        (( port <= 65535 )) || die "derived iperf port exceeds 65535: ${port}"
        validate_endpoint "${HOSTS[0]}" "${device}"
        validate_endpoint "${HOSTS[1]}" "${device}"
        host0_ip="$(device_ipv4 "${HOSTS[0]}" "${device}")"
        host1_ip="$(device_ipv4 "${HOSTS[1]}" "${device}")"
        host0_numa="$(device_numa_node "${HOSTS[0]}" "${device}")"
        host1_numa="$(device_numa_node "${HOSTS[1]}" "${device}")"
        [[ "${host0_ip}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] \
            || die "no IPv4 address found for ${HOSTS[0]}/${device}"
        [[ "${host1_ip}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] \
            || die "no IPv4 address found for ${HOSTS[1]}/${device}"
        [[ "${host0_numa}" =~ ^[0-9]+$ ]] \
            || die "no usable NUMA node found for ${HOSTS[0]}/${device}"
        [[ "${host1_numa}" =~ ^[0-9]+$ ]] \
            || die "no usable NUMA node found for ${HOSTS[1]}/${device}"
        printf '[INFO] %s: %s/%s NUMA=%s; %s/%s NUMA=%s\n' \
            "${device}" "${HOSTS[0]}" "${host0_ip}" "${host0_numa}" \
            "${HOSTS[1]}" "${host1_ip}" "${host1_numa}"

        run_direction "${device}" "forward" \
            "${HOSTS[0]}" "${HOSTS[1]}" "${host0_ip}" "${host1_ip}" \
            "${host0_numa}" "${host1_numa}" "${port}"
        run_direction "${device}" "reverse" \
            "${HOSTS[1]}" "${HOSTS[0]}" "${host1_ip}" "${host0_ip}" \
            "${host1_numa}" "${host0_numa}" "${port}"
    done
}

if (( BATCH_TEST == 1 )); then
    run_suite
else
    LOG_DIR="${REPO_ROOT}/logs/iperf"
    mkdir -p "${LOG_DIR}"
    NODE_IDENTITY="${NODE_SN_NAME:-$(hostname)}"
    NODE_IDENTITY="$(printf '%s' "${NODE_IDENTITY}" | tr -c 'A-Za-z0-9._-' '_')"
    LOG_FILE="${LOG_DIR}/$(date '+%Y%m%d_%H%M%S')_multinode_iperf_${NODE_IDENTITY}.log"
    run_suite 2>&1 | tee "${LOG_FILE}"
fi
