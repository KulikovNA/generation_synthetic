#!/usr/bin/env bash
set -euo pipefail

# === Параметры (можно переопределить через переменные окружения/argv) ===
DATASET_ROOT="${DATASET_ROOT:-/home/nikita/data_generator/dataset_07_10/differBig/2024-10-07/differBig/train_pbr}"
OUT_DIR="${OUT_DIR:-/home/nikita/data_generator/dataset_07_10/differBig/2024-10-07/_report}"
THRESH="${THRESH:-/home/nikita/data_generator/dataset_07_10/thresholds.yaml}"

# Кол-во воркеров: по умолчанию = числу логических ядер, но не менее 1
if command -v nproc >/dev/null 2>&1; then
  DEFAULT_WORKERS="$(nproc)"
else
  DEFAULT_WORKERS="4"
fi
WORKERS="${WORKERS:-$DEFAULT_WORKERS}"

# Включить прогресс-бар (tqdm). Установи PROGRESS=0, чтобы отключить.
PROGRESS="${PROGRESS:-1}"

# Путь к валидатору (обновлённый файл)
VALIDATOR="${VALIDATOR:-validate_bop.py}"

# === Парсинг позиционных аргументов (необязательно) ===
# usage: validate_bop.sh [DATASET_ROOT] [OUT_DIR] [THRESH]
if [[ "${1-}" != "" ]]; then DATASET_ROOT="$1"; fi
if [[ "${2-}" != "" ]]; then OUT_DIR="$2"; fi
if [[ "${3-}" != "" ]]; then THRESH="$3"; fi

# === Подготовка ===
mkdir -p "$OUT_DIR"

if [[ ! -f "$VALIDATOR" ]]; then
  echo "❌ Не найден валидатор: $VALIDATOR"
  echo "   Ожидается validate_bop.py в текущем каталоге или укажи VALIDATOR=/путь/к/скрипту"
  exit 1
fi

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "❌ Не найдена папка с сценами: $DATASET_ROOT"
  exit 1
fi

THRESH_ARG=()
if [[ -f "$THRESH" ]]; then
  THRESH_ARG=(--thresholds "$THRESH")
else
  echo "⚠️  Файл порогов не найден: $THRESH — будут использованы значения по умолчанию."
fi

# Если хотим прогресс — можно установить tqdm (мягко)
if [[ "$PROGRESS" == "1" ]]; then
  PROGRESS_FLAG=(--progress)
else
  PROGRESS_FLAG=()
fi

# === Запуск ===
echo "▶︎ Валидация BOP"
echo "   root:     $DATASET_ROOT"
echo "   out:      $OUT_DIR"
echo "   thresh:   ${THRESH_ARG[*]:-<defaults>}"
echo "   workers:  $WORKERS"
echo "   progress: $PROGRESS"
START_TS="$(date +%s)"

set +e
python3 "$VALIDATOR" \
  --dataset_root "$DATASET_ROOT" \
  --out_dir      "$OUT_DIR" \
  "${THRESH_ARG[@]}" \
  --workers "$WORKERS" \
  "${PROGRESS_FLAG[@]}"
RC=$?
set -e

END_TS="$(date +%s)"
DUR=$((END_TS - START_TS))

echo "⏱  Готово за ${DUR}s. Сводка:"
echo "   $OUT_DIR/run.json"
echo "   $OUT_DIR/issues.csv"
if [[ -f "$OUT_DIR/metrics/perf.csv" ]]; then
  echo "   $OUT_DIR/metrics/perf.csv"
fi

# Не валить пайплайн, но показать код (0 — ок, 2 — есть Critical)
if [[ "$RC" -ne 0 ]]; then
  echo "⚠️  Валидатор завершился с кодом: $RC (см. issues.csv и run.json)"
else
  echo "✅ PASS: критических ошибок нет."
fi

exit 0
