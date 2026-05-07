#!/usr/bin/env bash
set -euo pipefail

# Universal ChunkFT launcher.
# Configure by environment variables, then append extra arguments after `--`.
# Examples:
#   TASK_MODE=glue TASK_NAME=sst2 MODEL_NAME_OR_PATH=/path/to/model bash scripts/run_chunkft.sh
#   TASK_MODE=qa DATASET_NAME=squad MODEL_NAME_OR_PATH=/path/to/model bash scripts/run_chunkft.sh -- --max_seq_length 384
#   TASK_MODE=tasks TASK_NAME=boolq MODEL_NAME_OR_PATH=/path/to/model bash scripts/run_chunkft.sh 4 1

TASK_MODE="${TASK_MODE:-glue}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/path/to/model}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${TASK_MODE}_chunkft}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
PORT="${PORT:-$(shuf -i25000-30000 -n1)}"

TASK_TYPE="${TASK_TYPE:-}"
TASK_NAME="${TASK_NAME:-sst2}"
DATASET_NAME="${DATASET_NAME:-e2e_nlg}"
DATASET_DIR="${DATASET_DIR:-data}"
MODEL_TYPE="${MODEL_TYPE:-auto}"
PEFT_TYPE="${PEFT_TYPE:-}"
LORA_RANK="${LORA_RANK:-8}"

DO_TRAIN="${DO_TRAIN:-1}"
DO_EVAL="${DO_EVAL:-1}"
DO_PREDICT="${DO_PREDICT:-0}"
USE_DEEPSPEED="${USE_DEEPSPEED:-0}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-dsconfig/zero0_config.json}"
FP16="${FP16:-1}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-0}"

OPTIM="${OPTIM:-adamw_hf}"
LEARNING_RATE="${LEARNING_RATE:-3e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
MAX_STEPS="${MAX_STEPS:-}"
WARMUP_RATIO="${WARMUP_RATIO:-0.0}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-linear}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-8}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-512}"
MAX_SOURCE_LENGTH="${MAX_SOURCE_LENGTH:-1024}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-64}"
BLOCK_SIZE="${BLOCK_SIZE:-512}"
SEED="${SEED:-42}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
EVALUATION_STRATEGY="${EVALUATION_STRATEGY:-epoch}"
SAVE_STRATEGY="${SAVE_STRATEGY:-epoch}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"

ENABLE_CHUNKFT="${ENABLE_CHUNKFT:-1}"
CHUNK_NUM="${CHUNK_NUM:-1}"
CHUNK_STRATEGY="${CHUNK_STRATEGY:-row}"
CHUNK_UPDATE_INTERVAL="${CHUNK_UPDATE_INTERVAL:-1}"
ENABLE_CHUNK_PREFETCH="${ENABLE_CHUNK_PREFETCH:-true}"

if [[ "${1:-}" != "" && "${1:-}" != "--" ]]; then CHUNK_NUM="$1"; shift; fi
if [[ "${1:-}" != "" && "${1:-}" != "--" ]]; then CHUNK_UPDATE_INTERVAL="$1"; shift; fi
if [[ "${1:-}" == "--" ]]; then shift; fi
EXTRA_ARGS=("$@")

case "$TASK_MODE" in
  glue)
    ENTRYPOINT="examples/run_glue.py"
    MODE_ARGS=(--task_name "$TASK_NAME" --max_seq_length "$MAX_SEQ_LENGTH")
    TASK_TYPE="${TASK_TYPE:-SEQ_CLS}"
    ;;
  generation)
    ENTRYPOINT="examples/run_generation.py"
    MODE_ARGS=(--model_type "$MODEL_TYPE" --dataset_name "$DATASET_NAME" --model_max_length "$MODEL_MAX_LENGTH" --predict_with_generate)
    TASK_TYPE="${TASK_TYPE:-CAUSAL_LM}"
    ;;
  qa)
    ENTRYPOINT="examples/run_qa.py"
    MODE_ARGS=(--dataset_name "$DATASET_NAME" --max_seq_length "$MAX_SEQ_LENGTH")
    TASK_TYPE="${TASK_TYPE:-QUESTION_ANS}"
    ;;
  ner)
    ENTRYPOINT="examples/run_ner.py"
    MODE_ARGS=(--dataset_name "$DATASET_NAME" --max_seq_length "$MAX_SEQ_LENGTH")
    TASK_TYPE="${TASK_TYPE:-TOKEN_CLS}"
    ;;
  pretrain)
    ENTRYPOINT="examples/pretrain_tuning.py"
    MODE_ARGS=(--model_type "$MODEL_TYPE" --dataset_dir "$DATASET_DIR" --block_size "$BLOCK_SIZE")
    TASK_TYPE="${TASK_TYPE:-CAUSAL_LM}"
    ;;
  instruct)
    ENTRYPOINT="examples/instruct_tuning.py"
    MODE_ARGS=(--model_type "$MODEL_TYPE" --dataset_dir "$DATASET_DIR" --max_seq_length "$MAX_SEQ_LENGTH")
    TASK_TYPE="${TASK_TYPE:-CAUSAL_LM}"
    ;;
  tasks)
    ENTRYPOINT="examples/run_tasks.py"
    MODE_ARGS=(--task_name "$TASK_NAME" --max_source_length "$MAX_SOURCE_LENGTH" --max_target_length "$MAX_TARGET_LENGTH" --model_max_length "$MODEL_MAX_LENGTH" --predict_with_generate)
    TASK_TYPE="${TASK_TYPE:-CAUSAL_LM}"
    ;;
  *)
    echo "Unsupported TASK_MODE: $TASK_MODE" >&2
    echo "Supported: glue, generation, qa, ner, pretrain, instruct, tasks" >&2
    exit 2
    ;;
esac

ARGS=(
  --model_name_or_path "$MODEL_NAME_OR_PATH"
  --output_dir "$OUTPUT_DIR/model"
  --overwrite_output_dir
  --logging_steps "$LOGGING_STEPS"
  --logging_dir "$OUTPUT_DIR/log"
  --seed "$SEED"
)

ARGS+=(
  --TaskType "$TASK_TYPE"
  --optim "$OPTIM"
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
  --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE"
  --learning_rate "$LEARNING_RATE"
  --weight_decay "$WEIGHT_DECAY"
  --lr_scheduler_type "$LR_SCHEDULER_TYPE"
  --warmup_ratio "$WARMUP_RATIO"
  --evaluation_strategy "$EVALUATION_STRATEGY"
  --save_strategy "$SAVE_STRATEGY"
  --save_total_limit "$SAVE_TOTAL_LIMIT"
)

if [[ -n "$MAX_STEPS" ]]; then
  ARGS+=(--max_steps "$MAX_STEPS")
else
  ARGS+=(--num_train_epochs "$NUM_TRAIN_EPOCHS")
fi

[[ "$DO_TRAIN" == "1" ]] && ARGS+=(--do_train)
[[ "$DO_EVAL" == "1" ]] && ARGS+=(--do_eval)
[[ "$DO_PREDICT" == "1" ]] && ARGS+=(--do_predict)
[[ "$FP16" == "1" ]] && ARGS+=(--fp16)
[[ "$GRADIENT_CHECKPOINTING" == "1" ]] && ARGS+=(--gradient_checkpointing)
[[ "$SAVE_STRATEGY" == "steps" ]] && ARGS+=(--save_steps "$SAVE_STEPS")
[[ "$USE_DEEPSPEED" == "1" ]] && ARGS+=(--deepspeed "$DEEPSPEED_CONFIG")

if [[ -n "$PEFT_TYPE" ]]; then
  ARGS+=(--peft_type "$PEFT_TYPE" --lora_rank "$LORA_RANK")
fi

if [[ "$ENABLE_CHUNKFT" == "1" ]]; then
  ARGS+=(
    --chunk_tuning
    --chunk_num "$CHUNK_NUM"
    --chunk_strategy "$CHUNK_STRATEGY"
    --chunk_update_interval "$CHUNK_UPDATE_INTERVAL"
  )
  if [[ "$TASK_MODE" != "instruct" ]]; then
    ARGS+=(--enable_chunk_prefetch "$ENABLE_CHUNK_PREFETCH")
  fi
fi

set -x
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" torchrun \
  --master_port "$PORT" \
  --nproc_per_node "$NUM_GPUS" \
  "$ENTRYPOINT" \
  "${MODE_ARGS[@]}" \
  "${ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
