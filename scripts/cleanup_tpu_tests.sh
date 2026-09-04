#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
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

DRY_RUN=0
TERM_GRACE_SECONDS=3
LIBTPU_LOCKFILE="/tmp/libtpu_lockfile"

usage() {
  cat <<'EOF'
Usage: cleanup_tpu_tests.sh [options]

Stop CommPilot TPU benchmark processes on the current host and remove the
libtpu lock file.

Options:
  --dry-run             Print actions without changing the host.
  --grace-seconds <N>   Wait before SIGKILL. Default: 3.
  --lockfile <path>     libtpu lock file. Default: /tmp/libtpu_lockfile.
  -h, --help            Show this help.
EOF
}

while (($# > 0)); do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      ;;
    --grace-seconds)
      [[ $# -ge 2 ]] || { echo "--grace-seconds requires a value" >&2; exit 2; }
      TERM_GRACE_SECONDS="$2"
      shift
      ;;
    --lockfile)
      [[ $# -ge 2 ]] || { echo "--lockfile requires a value" >&2; exit 2; }
      LIBTPU_LOCKFILE="$2"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

[[ "${TERM_GRACE_SECONDS}" =~ ^[0-9]+$ ]] \
  || { echo "--grace-seconds must be a non-negative integer" >&2; exit 2; }

TPU_TEST_PATTERN='(src/(ici/test_ici|gemm/test_gemm|memory/test_(hbm|vmem)|pcie/test_pcie)\.py|scripts/(gemm/test_gemm|tpubandwidth/test_tpubandwidth|ici/test_multinode_ici)\.sh|python[^ ]* -m src\.(ici\.test_ici|gemm\.test_gemm|memory\.test_(hbm|vmem)|pcie\.test_pcie))'

list_matching_pids() {
  pgrep -f "$TPU_TEST_PATTERN" | awk -v self="$$" '$0 != self' || true
}

print_matching_processes() {
  if (($# == 0)); then
    echo "No matching TPU benchmark processes found."
    return
  fi
  echo "Matching TPU benchmark processes:"
  ps -o pid=,ppid=,stat=,command= -p "$(IFS=,; echo "$*")" || true
}

remove_libtpu_lockfile() {
  if [[ ! -e "$LIBTPU_LOCKFILE" ]]; then
    echo "No libtpu lock file found: $LIBTPU_LOCKFILE"
    return
  fi

  if ((DRY_RUN)); then
    echo "[dry-run] Would remove $LIBTPU_LOCKFILE"
    return
  fi

  if rm -f "$LIBTPU_LOCKFILE" 2>/dev/null; then
    echo "Removed $LIBTPU_LOCKFILE"
    return
  fi

  if command -v sudo >/dev/null 2>&1 && sudo -n rm -f "$LIBTPU_LOCKFILE" 2>/dev/null; then
    echo "Removed $LIBTPU_LOCKFILE with sudo"
    return
  fi

  echo "Failed to remove $LIBTPU_LOCKFILE; remove it manually or rerun with sufficient permission." >&2
  return 1
}

pids_output="$(list_matching_pids)"
if [[ -n "$pids_output" ]]; then
  pids=()
  while IFS= read -r pid; do
    [[ -n "$pid" ]] && pids+=("$pid")
  done <<< "$pids_output"
  print_matching_processes "${pids[@]}"

  if ((DRY_RUN)); then
    echo "[dry-run] Would send SIGTERM to: ${pids[*]}"
    echo "[dry-run] Would send SIGKILL to survivors after ${TERM_GRACE_SECONDS}s"
  else
    echo "Sending SIGTERM to TPU benchmark processes: ${pids[*]}"
    kill -TERM "${pids[@]}" 2>/dev/null || true
    sleep "$TERM_GRACE_SECONDS"

    survivors_output="$(list_matching_pids)"
    if [[ -n "$survivors_output" ]]; then
      survivors=()
      while IFS= read -r pid; do
        [[ -n "$pid" ]] && survivors+=("$pid")
      done <<< "$survivors_output"
      echo "Sending SIGKILL to remaining TPU benchmark processes: ${survivors[*]}"
      kill -KILL "${survivors[@]}" 2>/dev/null || true
    else
      echo "All matching TPU benchmark processes exited after SIGTERM."
    fi
  fi
else
  print_matching_processes
fi

remove_libtpu_lockfile
