#!/usr/bin/env bash
set -euo pipefail

# Defaults
RUNNER="run_seg_bop.py"
CONFIG="configs/bop_seg/config.py"

usage() {
  cat <<EOF
Usage:
  ./run.sh [--runner NAME] [--config PATH] [-- <extra python args...>]

Defaults:
  --runner  run_seg_bop.py
  --config  configs/bop_seg/config.py

Examples:
  ./run.sh
  ./run.sh --runner run_seg_bop.py --config configs/bop_seg/config.py
  ./run.sh --runner run_seg_bop.py --config configs/bop_seg/config.py -- --foo 1 --bar 2
EOF
}

# Parse flags
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -r|--runner) RUNNER="$2"; shift 2;;
    -c|--config) CONFIG="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    --) shift; EXTRA+=("$@"); break;;
    *) EXTRA+=("$1"); shift;;  # allow passing extra args without --
  esac
done

export OPENCV_IO_ENABLE_OPENEXR=1

eval "$(scripts/bproc env)"

python "runers/${RUNNER}" --config_path "${CONFIG}" "${EXTRA[@]}"
