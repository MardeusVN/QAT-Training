#!/bin/bash
set -uo pipefail

CKPT_DIR="/d/QAT_Transfer/data_preprocessed/qat_finetune_run3_ext/lightning_logs/version_0/checkpoints"
TARGET_EPOCH=150
PID=24880
KNOWN_BEST_EPOCH=62
QAT_TRANSFER_DIR="/d/QAT_Transfer/qat_transfer"
PYTHON="/d/QAT_Transfer/.venv/Scripts/python.exe"

while true; do
  if ! wmic process where "ProcessId=$PID" get ProcessId 2>/dev/null | grep -q "$PID"; then
    echo "PROCESS_DIED_EARLY pid=$PID"
    exit 2
  fi

  latest=$(find "$CKPT_DIR" -iname "qat-last-epoch=*.ckpt" -printf "%f\n" 2>/dev/null \
    | sed -E 's/qat-last-epoch=([0-9]+)\.ckpt/\1/' | sort -n | tail -1)

  if [ -n "$latest" ] && [ "$latest" -ge "$TARGET_EPOCH" ]; then
    echo "REACHED_EPOCH latest=$latest"
    break
  fi

  sleep 300
done

# find current best checkpoint
best_file=$(find "$CKPT_DIR" -iname "qat-best-epoch=*.ckpt" 2>/dev/null | head -1)
best_epoch=$(basename "$best_file" | sed -E 's/qat-best-epoch=([0-9]+)-.*/\1/')

echo "CURRENT_BEST_FILE=$best_file"
echo "CURRENT_BEST_EPOCH=$best_epoch"

if [ -z "$best_epoch" ] || [ "$best_epoch" -le "$KNOWN_BEST_EPOCH" ]; then
  echo "NO_IMPROVEMENT_SINCE_EPOCH_${KNOWN_BEST_EPOCH} -- stopping training"

  taskkill //PID "$PID" //F
  sleep 5

  echo "TRAINING_STOPPED"
  echo "RUNNING_INT8_EXPORT from $best_file"

  cd "$QAT_TRANSFER_DIR" || exit 3
  "$PYTHON" -m piper_train.export_int8 \
    "$best_file" \
    "voice.int8.qat_final.onnx" \
    --calibration-dataset "../data_preprocessed/dataset.jsonl" \
    --calibration-sentences 60

  export_status=$?
  if [ $export_status -eq 0 ]; then
    echo "EXPORT_DONE voice.int8.qat_final.onnx"
  else
    echo "EXPORT_FAILED status=$export_status"
  fi
  exit 0
else
  echo "IMPROVEMENT_FOUND new_best_epoch=$best_epoch -- leaving training running"
  exit 0
fi
