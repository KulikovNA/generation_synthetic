#!/usr/bin/env bash
set -euo pipefail
die() { echo "ERROR: $*" >&2; exit 1; }

# где лежит bop_custom_patch.py
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SITEPKG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sitepkgs|--site-packages) SITEPKG="$2"; shift 2;;
    *) die "Unknown arg: $1";;
  esac
done
[[ -n "$SITEPKG" ]] || die "Missing --sitepkgs PATH"
mkdir -p "$SITEPKG"

AUTO_PTH="$SITEPKG/zzz_bop_autopatch.pth"

# ВАЖНО:
#  - первая строка (не import) = добавится в sys.path как путь
#  - следующая строка ДОЛЖНА начинаться с "import" и быть ОДНОЙ строкой
{
  echo "$PATCH_DIR"
  echo "import os,importlib,traceback; exec(\"try:\\n importlib.import_module('bop_custom_patch').patch_bop_dataset_params()\\nexcept Exception:\\n  (traceback.print_exc() if os.environ.get('BOP_AUTOPATCH_DEBUG')=='1' else None)\\n\")"
} > "$AUTO_PTH"

echo "Installed BOP autopatch loader:"
echo "  - $AUTO_PTH"
echo "Patch dir added to sys.path:"
echo "  - $PATCH_DIR"
echo "Debug: set BOP_AUTOPATCH_DEBUG=1 to print traceback if patch fails."
