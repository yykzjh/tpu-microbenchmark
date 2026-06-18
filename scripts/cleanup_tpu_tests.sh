#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=0
TERM_GRACE_SECONDS="${TERM_GRACE_SECONDS:-3}"
LIBTPU_LOCKFILE="${LIBTPU_LOCKFILE:-/tmp/libtpu_lockfile}"

usage() {
  cat <<'EOF'
Usage: cleanup_tpu_tests.sh [--dry-run]

Stop CommPilot TPU benchmark processes on the current host and remove the
libtpu lock file.

Options:
  --dry-run   Print matching processes and cleanup actions without killing
              processes or removing files.
  -h, --help  Show this help.

Environment:
  TERM_GRACE_SECONDS  Seconds to wait after SIGTERM before SIGKILL. Default: 3.
  LIBTPU_LOCKFILE     Lock file to remove. Default: /tmp/libtpu_lockfile.
EOF
}

while (($# > 0)); do
  case "$1" in
    --dry-run)
      DRY_RUN=1
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

TPU_TEST_PATTERN='acceptance/platform/TPU/common/src/(ici/test_ici|gemm/test_gemm|memory/test_(hbm|vmem)|pcie/test_pcie)\.py'

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
