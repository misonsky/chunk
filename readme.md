# ChunkFT

ChunkFT is a memory-efficient fine-tuning codebase for large language models. It extends Hugging Face Trainer workflows with chunk-wise parameter updates, optional PEFT modules, optimizer-state offloading/prefetching, and DeepSpeed integration.

## Highlights

- Chunk-wise training: update a subset of parameter chunks at a time.
- PEFT integration through `peft_function`, including LoRA-style methods.
- Trainer variants for classification, sequence-to-sequence, generation, QA, NER, pretraining, instruction tuning, and general task-collection experiments.
- Optional DeepSpeed mixed-precision optimizer monkey patches.
- Chunk optimizer-state offloading and prefetching controlled by `CHUNKFT_*` environment variables.
- One universal launch script: `scripts/run_chunkft.sh`.

## Repository Layout

```text
chnk/                         # Core ChunkFT package
chnk/trainer.py               # ChunkTrainer and chunk state management
chnk/seqtrainer.py            # ChunkSeq2SeqTrainer
chnk/qatrainer.py             # ChunkQuestionAnsweringTrainer
chnk/registerCallBack.py      # Model-family callback and parameter selection rules
chnk/optimizers/              # Optimizer wrappers and DeepSpeed patching
chnk/utils/                   # PEFT helpers, chunk utilities, checkpoint helpers
examples/                     # Python training entry points
scripts/run_chunkft.sh        # Universal launch script
dsconfig/                     # DeepSpeed configs
glue/                         # GLUE data/resources
models/                       # Local model customizations
metrics/                      # Local metric implementations
```

## Installation

Create an environment with PyTorch, Transformers, Datasets, PEFT, Accelerate, and DeepSpeed:

```bash
pip install torch transformers datasets peft accelerate deepspeed evaluate
```

Install task-specific metric dependencies from `metrics/*/requirements.txt` only when needed.

## Universal Script

All old task-specific scripts under `scripts/` have been removed. Use the single launcher:

```bash
bash scripts/run_chunkft.sh
```

The script is configured with environment variables and supports these task modes:

- `TASK_MODE=glue`
- `TASK_MODE=generation`
- `TASK_MODE=qa`
- `TASK_MODE=ner`
- `TASK_MODE=pretrain`
- `TASK_MODE=instruct`
- `TASK_MODE=tasks`

The first two positional arguments are optional shortcuts. The chunk strategy defaults to `row`:

```bash
bash scripts/run_chunkft.sh <chunk_num> <chunk_update_interval>
```

Extra arguments after `--` are passed directly to the selected Python entry point.

## Quick Start

Run GLUE SST-2 with ChunkFT:

```bash
TASK_MODE=glue \
TASK_NAME=sst2 \
MODEL_NAME_OR_PATH=/path/to/model \
OUTPUT_DIR=outputs/sst2_chunkft \
bash scripts/run_chunkft.sh 4 1
```

## Common Script Variables

Model and task:

- `TASK_MODE`: one of `glue`, `generation`, `qa`, `ner`, `pretrain`, `instruct`, `tasks`.
- `MODEL_NAME_OR_PATH`: base model path or Hugging Face model name.
- `TASK_TYPE`: PEFT task type, for example `SEQ_CLS`, `CAUSAL_LM`, `SEQ_2_SEQ_LM`, `TOKEN_CLS`, or `QUESTION_ANS`.
- `TASK_NAME`: task name for GLUE or task-collection runs.
- `DATASET_NAME`: dataset name for generation, QA, or NER entry points.
- `DATASET_DIR`: local dataset directory for pretraining or instruction tuning.
- `MODEL_TYPE`: model family hint for some entry points, such as `llama`, `opt`, or `gpt2`.

Training:

- `NUM_GPUS`: number of processes for `torchrun`.
- `CUDA_VISIBLE_DEVICES`: visible GPU ids.
- `OUTPUT_DIR`: output directory prefix.
- `LEARNING_RATE`: learning rate.
- `NUM_TRAIN_EPOCHS`: number of epochs when `MAX_STEPS` is unset.
- `MAX_STEPS`: optional step-based training limit.
- `PER_DEVICE_TRAIN_BATCH_SIZE`: train batch size per device.
- `PER_DEVICE_EVAL_BATCH_SIZE`: eval batch size per device.
- `FP16`: set `1` to add `--fp16`.

ChunkFT:

- `ENABLE_CHUNKFT`: set `1` to add `--chunk_tuning`; set `0` for normal training.
- `CHUNK_NUM`: number of parameter chunks.
- `CHUNK_UPDATE_INTERVAL`: number of optimizer updates before switching chunks.
- `ENABLE_CHUNK_PREFETCH`: controls `--enable_chunk_prefetch` for entry points that support it.

## Direct Python Usage

You can still call entry points directly:

```bash
python examples/run_glue.py \
  --model_name_or_path /path/to/model \
  --task_name sst2 \
  --do_train \
  --do_eval \
  --output_dir outputs/sst2_chunkft/model \
  --TaskType SEQ_CLS \
  --chunk_tuning \
  --chunk_num 4 \
  --chunk_update_interval 4
```

## Runtime Flags

- `CHUNKFT_ENABLE_MONKEY_PATCHES=0`: disable DeepSpeed monkey patches.
- `CHUNKFT_ENABLE_PREFETCH=0`: disable chunk optimizer-state prefetching.
- `CHUNKFT_ASYNC_OFFLOAD=0`: disable async offload behavior.
- `CHUNKFT_PIN_MEMORY=0`: disable pinned-memory transfers.
- `CHUNKFT_CUDA_SYNC=1`: force CUDA synchronization for debugging.
- `CHUNKFT_EMPTY_CACHE=1`: empty CUDA cache at selected runtime points.
- `CHUNKFT_DEBUG_GPU_USAGE=1`: enable GPU usage debug logging.