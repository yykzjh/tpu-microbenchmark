#!/usr/bin/env bash
# Upload selected benchmark logs to per-testcase OSS prefixes.
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

PLATFORM="TPU"

ARG_OSS_PATH=""
ARG_OSS_AK=""
ARG_OSS_SK=""
ARG_OSS_ENDPOINT=""
ARG_OSS_REGION=""
LOGS_DIR="${REPO_ROOT}/logs"
TESTCASES_CSV=""
FILENAME_PREFIX=""
CLEANUP_AFTER_UPLOAD=0
SKIP_EMPTY_TESTCASES=0

die() { printf '[ERROR] %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<'EOF'
Usage: upload-logs-to-oss.sh [options]
  --testcase <csv>       Testcases to upload: gemm,tpubandwidth,iperf,ici.
                         Default: all supported testcase directories that exist.
  --oss-path <uri>       Overrides OSS_PATH.
  --oss-ak <value>       Overrides OSS_AK.
  --oss-sk <value>       Overrides OSS_SK.
  --oss-endpoint <value> Overrides OSS_ENDPOINT.
  --oss-region <value>   Overrides OSS_REGION.
  --logs-dir <path>      Log root. Default: <repository>/logs.
  --filename-prefix <s>  Upload only .log files whose basename starts with s.
  --cleanup-after-upload Remove each selected testcase directory only after all
                         of its log files have uploaded successfully.
  --skip-empty-testcases Skip selected testcase directories without matching
                         logs. At least one log must still be uploaded.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --testcase) [[ $# -ge 2 ]] || die "--testcase requires a value"; TESTCASES_CSV="$2"; shift 2 ;;
        --oss-path) [[ $# -ge 2 ]] || die "--oss-path requires a value"; ARG_OSS_PATH="$2"; shift 2 ;;
        --oss-ak) [[ $# -ge 2 ]] || die "--oss-ak requires a value"; ARG_OSS_AK="$2"; shift 2 ;;
        --oss-sk) [[ $# -ge 2 ]] || die "--oss-sk requires a value"; ARG_OSS_SK="$2"; shift 2 ;;
        --oss-endpoint) [[ $# -ge 2 ]] || die "--oss-endpoint requires a value"; ARG_OSS_ENDPOINT="$2"; shift 2 ;;
        --oss-region) [[ $# -ge 2 ]] || die "--oss-region requires a value"; ARG_OSS_REGION="$2"; shift 2 ;;
        --logs-dir) [[ $# -ge 2 ]] || die "--logs-dir requires a value"; LOGS_DIR="$2"; shift 2 ;;
        --filename-prefix) [[ $# -ge 2 ]] || die "--filename-prefix requires a value"; FILENAME_PREFIX="$2"; shift 2 ;;
        --cleanup-after-upload) CLEANUP_AFTER_UPLOAD=1; shift ;;
        --skip-empty-testcases) SKIP_EMPTY_TESTCASES=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

OSS_PATH="${ARG_OSS_PATH:-${OSS_PATH:-}}"
OSS_AK="${ARG_OSS_AK:-${OSS_AK:-}}"
OSS_SK="${ARG_OSS_SK:-${OSS_SK:-}}"
OSS_ENDPOINT="${ARG_OSS_ENDPOINT:-${OSS_ENDPOINT:-}}"
OSS_REGION="${ARG_OSS_REGION:-${OSS_REGION:-}}"
for variable in OSS_PATH OSS_AK OSS_SK OSS_ENDPOINT OSS_REGION; do
    [[ -n "${!variable}" ]] || die "${variable} is required"
done
command -v ossutil >/dev/null 2>&1 || die "ossutil is not installed"
[[ -d "${LOGS_DIR}" ]] || die "log directory not found"
[[ -z "${FILENAME_PREFIX}" || "${FILENAME_PREFIX}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
    || die "--filename-prefix contains unsupported characters"
OSS_PATH="${OSS_PATH%/}"
[[ "${OSS_PATH}" =~ ^oss://[^/]+/.+ ]] || die "OSS_PATH must be an oss:// URI with a prefix"

case "${PLATFORM}" in
    NVIDIA) SUPPORTED=(gemm nvbandwidth nccl accl acclep acclepv2) ;;
    PPU) SUPPORTED=(gemm sailbandwidth pccl) ;;
    TPU) SUPPORTED=(gemm tpubandwidth iperf ici) ;;
    AMD) SUPPORTED=(gemm rocbandwidth rccl) ;;
    *) die "unsupported platform: ${PLATFORM}" ;;
esac

is_supported() {
    local candidate="$1" supported
    for supported in "${SUPPORTED[@]}"; do
        [[ "${candidate}" == "${supported}" ]] && return 0
    done
    return 1
}

is_selected() {
    local candidate="$1" selected
    for selected in "${SELECTED_TESTCASES[@]:-}"; do
        [[ "${candidate}" == "${selected}" ]] && return 0
    done
    return 1
}

if [[ -n "${TESTCASES_CSV}" ]]; then
    IFS=',' read -r -a TESTCASES <<< "${TESTCASES_CSV}"
else
    TESTCASES=()
    for testcase in "${SUPPORTED[@]}"; do
        [[ -d "${LOGS_DIR}/${testcase}" ]] && TESTCASES+=("${testcase}")
    done
fi
[[ ${#TESTCASES[@]} -gt 0 ]] || die "no testcase log directory was selected"

upload_count=0
SELECTED_TESTCASES=()
UPLOADED_LOGS=()
for testcase in "${TESTCASES[@]}"; do
    testcase="${testcase//[[:space:]]/}"
    [[ -n "${testcase}" ]] || die "testcase list contains an empty value"
    is_supported "${testcase}" || die "unsupported ${PLATFORM} testcase: ${testcase}"
    is_selected "${testcase}" && continue
    source_dir="${LOGS_DIR}/${testcase}"
    if [[ ! -d "${source_dir}" ]]; then
        (( SKIP_EMPTY_TESTCASES == 1 )) || die "testcase log directory not found"
        printf '[INFO] skipped missing %s log directory\n' "${testcase}"
        continue
    fi
    SELECTED_TESTCASES+=("${testcase}")
    destination="${OSS_PATH}/${testcase}/"
    testcase_started_epoch="$(date +%s)"
    timing_event "oss.${testcase}" start
    ossutil mkdir "${destination}" \
        -i "${OSS_AK}" -k "${OSS_SK}" -e "${OSS_ENDPOINT}" \
        --region "${OSS_REGION}" >/dev/null 2>&1 || true

    testcase_count=0
    while IFS= read -r -d '' log_file; do
        ossutil cp "${log_file}" "${destination}$(basename "${log_file}")" \
            -i "${OSS_AK}" -k "${OSS_SK}" -e "${OSS_ENDPOINT}" \
            --region "${OSS_REGION}" --force >/dev/null
        ((testcase_count += 1))
        ((upload_count += 1))
        UPLOADED_LOGS+=("${log_file}")
    done < <(find "${source_dir}" -type f -name "${FILENAME_PREFIX}*.log" -print0 | sort -z)
    if (( testcase_count == 0 )); then
        (( SKIP_EMPTY_TESTCASES == 1 )) || die "no matching .log files found"
        printf '[INFO] skipped %s without matching log files\n' "${testcase}"
        timing_event "oss.${testcase}" end "status=skipped" "files=0" \
            "duration_seconds=$(($(date +%s) - testcase_started_epoch))"
        continue
    fi
    printf '[INFO] uploaded %d %s log(s)\n' "${testcase_count}" "${testcase}"
    timing_event "oss.${testcase}" end "status=succeeded" "files=${testcase_count}" \
        "duration_seconds=$(($(date +%s) - testcase_started_epoch))"
done

(( upload_count > 0 )) || die "no matching .log files were uploaded"

if [[ ${CLEANUP_AFTER_UPLOAD} -eq 1 ]]; then
    for log_file in "${UPLOADED_LOGS[@]}"; do
        rm -f -- "${log_file}"
    done
    for testcase in "${SELECTED_TESTCASES[@]}"; do
        find "${LOGS_DIR}/${testcase}" -depth -type d -empty -delete
    done
    printf '[INFO] removed %d uploaded local log file(s)\n' "${#UPLOADED_LOGS[@]}"
fi

printf '[INFO] uploaded %d log file(s)\n' "${upload_count}"
