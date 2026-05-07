#!/usr/bin/env python
# coding=utf-8
"""Run ChunkFT causal LM training on the task collection task collection."""

import logging
import math
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
import warnings
import json
import shutil
from statistics import mean
from dataclasses import dataclass, field
from typing import List, Optional

import datasets
import torch
import transformers
from peft import PeftModel
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, get_last_checkpoint
from task_dataset_utils import (
    TaskCollectionClassificationCollator,
    build_task_collection_classification_dataset,
    build_task_collection_icl_records,
    build_task_collection_generation_dataset,
    build_task_collection_supervised_dataset,
    decode_generation_predictions,
    evaluate_task_collection_predictions,
    forward_wrap_with_option_len,
    load_task_collection_records,
    sample_task_collection_subset,
    sample_task_collection_train_sets,
    score_task_collection_record_candidates
)
sys.path.append(os.path.abspath("."))

from chnk import ( 
    ChunkSeq2SeqTrainer,
    GetCallBack,
    Seq2SeqTrainer,
    apply_gradient_checkpointing_strategy,
    normalize_chunk_args,
    peft_function,
    rebuild_layer,
)

from wandb_utils import configure_wandb 


logger = logging.getLogger(__name__)


@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum combined prompt+target length after tokenization."},
    )
    optim: str = field(
        default="adamw_torch",
        metadata={"help": "The optimizer to use."},
    )
    pretraining_tp: int = field(
        default=1,
        metadata={"help": "Tensor parallel degree used by some decoder-only checkpoints."},
    )
    force_do_sample: bool = field(
        default=False,
        metadata={
            "help": "Force generation_config.do_sample=True after model loading. Useful when you want "
            "temperature/top_p/top_k sampling to stay active during eval/predict."
        },
    )
    train_as_classification: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Use task collection's classification-style option grouping during supervised training. "
            "If unset, the script follows the original task collection default for each task."
        },
    )


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name_or_path."},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name_or_path."},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store downloaded model files."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use a fast tokenizer."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "Model version to use."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={"help": "Use the token from `huggingface-cli login` when loading private models."},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Allow execution of custom Hub modeling code."},
    )
    TaskType: str = field(
        default="CAUSAL_LM",
        metadata={"help": "ChunkFT task type, should stay aligned with PEFT task type."},
    )
    peft_type: Optional[str] = field(
        default=None,
        metadata={"help": "Optional PEFT mode: lora, adalora, ia3, p_tuning, prefix_tuning, prompt_tuning."},
    )
    init_text: Optional[str] = field(
        default=None,
        metadata={"help": "Initialization text used by prompt tuning."},
    )
    lora_rank: int = field(
        default=8,
        metadata={"help": "Rank for lora or adalora."},
    )
    peft_path: Optional[str] = field(
        default=None,
        metadata={"help": "Load an existing PEFT checkpoint before training."},
    )
    virtual_tokens: int = field(
        default=20,
        metadata={"help": "Number of virtual tokens for p_tuning, prefix_tuning or prompt_tuning."},
    )
    chunk_num: int = field(
        default=1,
        metadata={"help": "Number of parameter chunks for ChunkFT."},
    )
    chunk_update_interval: int = field(
        default=1,
        metadata={"help": "How many optimizer updates to keep the same chunk before switching."},
    )
    enable_chunk_prefetch: bool = field(
        default=True,
        metadata={"help": "Enable async prefetch of the next chunk optimizer state/fp32 slice."},
    )
    chunk_strategy: str = field(
        default="row",
        metadata={"help": "Chunk selection strategy: row or column."},
    )
    chunk_tuning: bool = field(
        default=False,
        metadata={"help": "Enable ChunkFT optimization."},
    )
    gradient_checkpointing_layers: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated transformer block indices or ranges for selective checkpointing."},
    )
    gradient_checkpointing_ratio: float = field(
        default=1.0,
        metadata={"help": "If 0<ratio<1 and no explicit layers are set, checkpoint this fraction of blocks."},
    )
    gradient_checkpointing_mode: str = field(
        default="tail",
        metadata={"help": "How to select checkpointed layers when using a ratio: tail, head, or uniform."},
    )
    chunk_tuning: bool = field(
        default=False,
        metadata={"help": "Deprecated alias for --chunk_tuning."},
    )
    group_element: Optional[int] = field(
        default=None,
        metadata={"help": "Deprecated alias for --chunk_num."},
    )
    optimizer_strategy: Optional[str] = field(
        default=None,
        metadata={"help": "Deprecated alias for --chunk_strategy."},
    )
    freeze_layers: List[str] = field(
        default_factory=list,
        metadata={"help": "Optional transformer layer indices to freeze."},
    )


@dataclass
class DataTrainingArguments:
    task_name: str = field(
        metadata={
            "help": "Task name. Supported: SST2, RTE, CB, BoolQ, WSC, WIC, MultiRC, Copa, ReCoRD, SQuAD, DROP."
        }
    )
    dataset_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Optional cache dir passed to `datasets.load_dataset`."},
    )
    max_source_length: int = field(
        default=1536,
        metadata={"help": "Maximum prompt length after tokenization."},
    )
    max_target_length: int = field(
        default=128,
        metadata={"help": "Maximum target length after tokenization and also used as generation max_new_tokens."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Randomly subsample this many training examples."},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Randomly subsample this many validation examples."},
    )
    max_dev_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Randomly subsample this many dev examples from the sampled training pool."},
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Randomly subsample this many test examples."},
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Randomly subsample this many prediction examples. Defaults to max_eval_samples."},
    )
    num_train: Optional[int] = field(
        default=None,
        metadata={"help": "task-collection-compatible alias of --max_train_samples."},
    )
    num_eval: Optional[int] = field(
        default=None,
        metadata={"help": "task-collection-compatible alias of --max_eval_samples."},
    )
    train_set_seed: Optional[int] = field(
        default=42,
        metadata={"help": "task-collection-compatible alias of --data_seed."},
    )
    num_train_sets: Optional[int] = field(
        default=None,
        metadata={"help": "task-collection-compatible number of sampled train sets. Use with --train_set_id to select one."},
    )
    train_set_id: int = field(
        default=0,
        metadata={"help": "When --num_train_sets is set and --train_set_seed is not, select which sampled set index to use."},
    )
    num_dev: Optional[int] = field(
        default=None,
        metadata={"help": "task-collection-compatible alias of --max_dev_samples."},
    )
    num_test: Optional[int] = field(
        default=None,
        metadata={"help": "task-collection-compatible alias of --max_test_samples."},
    )
    predict_split: str = field(
        default="test",
        metadata={"help": "Split used for generation-based prediction. Supports train, dev, validation, or test."},
    )
    tag: str = field(
        default="",
        metadata={"help": "Optional experiment tag used in task-collection result filenames."},
    )
    result_file: Optional[str] = field(
        default=None,
        metadata={"help": "Optional explicit path for the final merged result JSON."},
    )

    def __post_init__(self) -> None:
        self.predict_split = self.predict_split.lower()
        if self.predict_split not in {"train", "dev", "validation", "test"}:
            raise ValueError("--predict_split must be one of: train, dev, validation, test.")
        if self.num_train is not None and self.max_train_samples is None:
            self.max_train_samples = self.num_train
        if self.num_eval is not None and self.max_eval_samples is None:
            self.max_eval_samples = self.num_eval
        if self.num_dev is not None and self.max_dev_samples is None:
            self.max_dev_samples = self.num_dev
        if self.num_test is not None and self.max_test_samples is None:
            self.max_test_samples = self.num_test
        if self.train_set_seed is not None:
            self.data_seed = self.train_set_seed
        elif self.num_train_sets is not None:
            if self.train_set_id < 0 or self.train_set_id >= self.num_train_sets:
                raise ValueError("--train_set_id must be in [0, num_train_sets).")
            self.data_seed = self.train_set_id


def resolve_task_collection_sampling_seed(data_args: DataTrainingArguments) -> int:
    if data_args.train_set_seed is not None:
        return data_args.train_set_seed
    if data_args.num_train_sets is not None:
        return data_args.train_set_id


def save_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as writer:
        json.dump(payload, writer, ensure_ascii=False, indent=2)


def build_task_collection_result_tag(model_name_or_path: str, data_args: DataTrainingArguments) -> str:
    model_name = os.path.basename(model_name_or_path.rstrip("/"))
    parts = [data_args.task_name, model_name]
    if data_args.max_eval_samples is not None:
        parts.append(f"sampleeval{data_args.max_eval_samples}")
    if data_args.max_train_samples is not None:
        parts.append(f"ntrain{data_args.max_train_samples}")
    if data_args.max_dev_samples not in (None, 0):
        parts.append(f"ndev{data_args.max_dev_samples}")
    if data_args.tag:
        parts.append(data_args.tag)
    return "-".join(parts)


def prune_to_best_checkpoint(output_dir: str, best_checkpoint: Optional[str]) -> None:
    if not best_checkpoint or not os.path.isdir(output_dir):
        return
    best_checkpoint = os.path.abspath(best_checkpoint)
    for entry in os.listdir(output_dir):
        if not entry.startswith("checkpoint-"):
            continue
        candidate = os.path.abspath(os.path.join(output_dir, entry))
        if candidate != best_checkpoint and os.path.isdir(candidate):
            shutil.rmtree(candidate)


def build_selection_compute_metrics(records, prompt_lengths, tokenizer, task_name: str):
    def compute_metrics(eval_prediction):
        predictions = eval_prediction.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        decoded_predictions = decode_generation_predictions(
            predictions,
            tokenizer=tokenizer,
            prompt_lengths=prompt_lengths,
        )
        return evaluate_task_collection_predictions(task_name, decoded_predictions, records)

    return compute_metrics

def build_classification_compute_metrics(records, task_name: str):
    candidate_sizes = [len(record.get("candidates", [])) for record in records]

    def compute_metrics(eval_prediction):
        predictions = eval_prediction.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        if hasattr(predictions, "tolist"):
            predictions = predictions.tolist()

        flat_scores = []
        for prediction in predictions:
            if isinstance(prediction, (list, tuple)):
                flat_scores.append(float(prediction[0]))
            else:
                flat_scores.append(float(prediction))

        predicted_texts = []
        offset = 0
        for record, candidate_size in zip(records, candidate_sizes):
            if candidate_size <= 0 or offset + candidate_size > len(flat_scores):
                break
            group_scores = flat_scores[offset : offset + candidate_size]
            predicted_id = max(range(candidate_size), key=group_scores.__getitem__)
            predicted_texts.append(str(record["candidates"][predicted_id]).strip())
            offset += candidate_size
        return evaluate_task_collection_predictions(task_name, predicted_texts, records[: len(predicted_texts)])

    return compute_metrics

def preprocess_classification_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.contiguous()


def is_task_collection_generation_task(task) -> bool:
    return getattr(task, "generation", False)

def align_generation_sampling_config(model, logger, force_do_sample: bool = False) -> None:
    if not hasattr(model, "generation_config") or model.generation_config is None:
        return
    generation_config = model.generation_config
    if getattr(generation_config, "_from_model_config", False):
        generation_config._from_model_config = False

    if force_do_sample:
        temperature = getattr(generation_config, "temperature", None)
        top_p = getattr(generation_config, "top_p", None)
        top_k = getattr(generation_config, "top_k", None)
        if temperature is not None and temperature <= 0:
            logger.warning("generation_config.temperature=%s is invalid for sampling; resetting to 1.0.", temperature)
            generation_config.temperature = 1.0
        if top_p is not None and not (0 < top_p <= 1.0):
            logger.warning("generation_config.top_p=%s is invalid for sampling; resetting to 1.0.", top_p)
            generation_config.top_p = 1.0
        if top_k is not None and top_k < 0:
            logger.warning("generation_config.top_k=%s is invalid for sampling; resetting to 50.", top_k)
            generation_config.top_k = 50
        generation_config.do_sample = True
        logger.info(
            "Forced generation_config.do_sample=True "
            "(temperature=%s, top_p=%s, top_k=%s).",
            getattr(generation_config, "temperature", None),
            getattr(generation_config, "top_p", None),
            getattr(generation_config, "top_k", None),
        )
        return

    if getattr(generation_config, "do_sample", False):
        logger.info(
            "Set generation_config.do_sample=False for deterministic task collection evaluation "
            "(temperature=%s, top_p=%s, top_k=%s).",
            getattr(generation_config, "temperature", None),
            getattr(generation_config, "top_p", None),
            getattr(generation_config, "top_k", None),
        )
    generation_config.do_sample = False
    if getattr(generation_config, "temperature", None) not in (None, 1.0):
        generation_config.temperature = 1.0
    if getattr(generation_config, "top_p", None) not in (None, 1.0):
        generation_config.top_p = 1.0
    if getattr(generation_config, "top_k", None) is not None and generation_config.top_k <= 0:
        generation_config.top_k = 50

def build_icl_records_for_split(
    *,
    data_args: DataTrainingArguments,
    split_name: str,
    train_pool: List[dict],
    target_records: List[dict],
    target_pool: List[dict],
    target_limit: Optional[int],
    target_seed: int,
    sampled_train_records: List[dict],
) -> tuple[List[dict], dict]:
    if not target_records and not target_pool:
        return [], {"split_name": split_name, "mode": "empty", "demo_seeds": [], "demo_ids": []}

    if data_args.train_set_seed is not None or data_args.num_train_sets is not None:
        icl_records = build_task_collection_icl_records(target_records, [sampled_train_records])
        return icl_records, {
            "split_name": split_name,
            "mode": "shared_train_set",
            "demo_seeds": [resolve_task_collection_sampling_seed(data_args)],
            "demo_ids": [record["id"] for record in sampled_train_records],
        }

    if data_args.max_dev_samples not in (None, 0):
        raise ValueError("task-collection one-train-set-per-eval-sample mode does not support num_dev/max_dev_samples.")

    sampled_targets = sample_task_collection_subset(target_pool, target_limit, target_seed)
    sampled_train_sets, demo_seeds = sample_task_collection_train_sets(
        train_records=train_pool,
        num_train=data_args.max_train_samples or 0,
        num_target_samples=len(sampled_targets),
        num_train_sets=None,
        train_set_seed=None,
    )
    icl_records = build_task_collection_icl_records(sampled_targets, sampled_train_sets)
    return icl_records, {
        "split_name": split_name,
        "mode": "one_train_set_per_target",
        "demo_seeds": demo_seeds,
        "demo_ids": [[record["id"] for record in demo_set] for demo_set in sampled_train_sets],
    }


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        align_generation_sampling_config(kwargs["model"], logger, force_do_sample=getattr(args, "force_do_sample", False))
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "peft_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "peft_model")
        kwargs["model"].save_pretrained(peft_model_path)
        kwargs["tokenizer"].save_pretrained(peft_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        align_generation_sampling_config(kwargs["model"], logger, force_do_sample=getattr(args, "force_do_sample", False))
        peft_model_path = os.path.join(args.output_dir, "peft_model")
        kwargs["model"].save_pretrained(peft_model_path)
        kwargs["tokenizer"].save_pretrained(peft_model_path)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    normalize_chunk_args(model_args, logger)
    configure_wandb(training_args, logger)

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        training_args.parallel_mode.value == "distributed",
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    if training_args.do_train and training_args.load_best_model_at_end:
        if training_args.save_total_limit != 1:
            logger.info("Setting save_total_limit=1 so only the best checkpoint is kept.")
            training_args.save_total_limit = 1

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to train from scratch."
            )
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                "Checkpoint detected, resuming training at %s. To avoid this behavior, change the "
                "--output_dir or add --overwrite_output_dir.",
                last_checkpoint,
            )

    set_seed(training_args.seed)

    sampling_seed = resolve_task_collection_sampling_seed(data_args)

    split_records, task = load_task_collection_records(
        task_name=data_args.task_name,
        cache_dir=data_args.dataset_cache_dir or model_args.cache_dir,
        max_train_samples=data_args.max_train_samples,
        max_dev_samples=data_args.max_dev_samples,
        max_eval_samples=data_args.max_eval_samples,
        max_test_samples=data_args.max_test_samples,
        data_seed=sampling_seed,
    )
    train_records = split_records["train"]
    dev_records = split_records["dev"]
    validation_records = split_records["validation"]
    test_records = split_records["test"]
    full_split_records, _ = load_task_collection_records(
        task_name=data_args.task_name,
        cache_dir=data_args.dataset_cache_dir or model_args.cache_dir,
        data_seed=sampling_seed,
    )
    full_train_pool = full_split_records["train"]
    full_validation_pool = full_split_records["validation"]
    full_test_pool = full_split_records["test"]
    if training_args.train_as_classification is None:
        training_args.train_as_classification = task.train_as_classification_default
    logger.info(
        "Task-collection train_as_classification=%s for task=%s",
        training_args.train_as_classification,
        task.name,
    )
    icl_mode = not training_args.do_train and (data_args.max_train_samples or 0) > 0
    multi_train_set_icl_mode = icl_mode and data_args.num_train_sets is not None and data_args.train_set_seed is None
    if multi_train_set_icl_mode and data_args.max_dev_samples not in (None, 0):
        raise ValueError("task-collection multi train-set ICL mode does not support num_dev/max_dev_samples.")

    sampling_manifest = {
        "task_name": data_args.task_name,
        "icl_mode": icl_mode,
        "sampling_seed": sampling_seed,
        "train_set_seed": data_args.train_set_seed,
        "num_train_sets": data_args.num_train_sets,
        "train_set_id": data_args.train_set_id,
        "num_train": data_args.max_train_samples,
        "num_dev": data_args.max_dev_samples,
        "num_eval": data_args.max_eval_samples,
        "num_test": data_args.max_test_samples,
        "split_sizes": {
            "train": len(train_records),
            "dev": len(dev_records),
            "validation": len(validation_records),
            "test": len(test_records),
        },
        "split_ids": {
            "train": [record["id"] for record in train_records],
            "dev": [record["id"] for record in dev_records],
            "validation": [record["id"] for record in validation_records],
            "test": [record["id"] for record in test_records],
        },
    }
    os.makedirs(training_args.output_dir, exist_ok=True)
    sampling_manifest_path = os.path.join(training_args.output_dir, "sampling_manifest.json")
    save_json(sampling_manifest_path, sampling_manifest)
    logger.info("Saved sampling manifest to %s", sampling_manifest_path)
    result_tag = build_task_collection_result_tag(model_args.model_name_or_path, data_args)
    final_results = {
        "result_tag": result_tag,
        "task_name": data_args.task_name,
        "model_name_or_path": model_args.model_name_or_path,
        "icl_mode": icl_mode,
        "sampling_manifest_path": sampling_manifest_path,
    }

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        trust_remote_code=model_args.trust_remote_code,
    )
    if hasattr(config, "pretraining_tp"):
        config.pretraining_tp = training_args.pretraining_tp
    if hasattr(config, "use_cache") and training_args.gradient_checkpointing:
        config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
        revision=model_args.model_revision,
        model_max_length=training_args.model_max_length,
        use_auth_token=True if model_args.use_auth_token else None,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if training_args.train_as_classification or not is_task_collection_generation_task(task):
        tokenizer.padding_side = "left"

    if model_args.chunk_tuning:
        rebuild_layer()

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch.float16,
    )
    if model_args.peft_type:
        if model_args.peft_path is not None:
            logger.info("Loading PEFT weights from %s", model_args.peft_path)
            model = PeftModel.from_pretrained(model, model_args.peft_path)
        else:
            model = peft_function(
                model,
                config=config,
                peft_type=model_args.peft_type,
                task_type=model_args.TaskType,
                rank=model_args.lora_rank,
                virtual_tokens=model_args.virtual_tokens,
                tokenizer_name_or_path=model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
                init_text=model_args.init_text if model_args.peft_type == "prompt_tuning" else None,
                peft_config=None,
            )
    apply_gradient_checkpointing_strategy(model, config, model_args, training_args, logger)

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model.config, "decoder_start_token_id") and model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = model.config.bos_token_id
    if hasattr(model, "generation_config"):
        model.generation_config.max_new_tokens = data_args.max_target_length
    align_generation_sampling_config(model, logger, force_do_sample=training_args.force_do_sample)
    
    if training_args.train_as_classification or not is_task_collection_generation_task(task):
        if getattr(model, "original_forward", None) is None:
            model.original_forward = model.forward
        model.forward = forward_wrap_with_option_len.__get__(model, type(model))
    
    train_dataset = None

    if training_args.do_train:
        if training_args.train_as_classification:
            train_dataset = build_task_collection_classification_dataset(
                records=train_records,
                tokenizer=tokenizer,
                model_max_length=training_args.model_max_length,
            )
        else:
            train_dataset = build_task_collection_supervised_dataset(
                records=train_records,
                tokenizer=tokenizer,
                max_source_length=data_args.max_source_length,
                max_target_length=data_args.max_target_length,
                model_max_length=training_args.model_max_length,
        )

    dev_dataset = None
    if dev_records:
        if training_args.train_as_classification:
            dev_dataset = build_task_collection_classification_dataset(
                records=dev_records,
                tokenizer=tokenizer,
                model_max_length=training_args.model_max_length,
            )
        else:
            dev_dataset = build_task_collection_supervised_dataset(
                records=dev_records,
                tokenizer=tokenizer,
                max_source_length=data_args.max_source_length,
                max_target_length=data_args.max_target_length,
                model_max_length=training_args.model_max_length,
            )
    validation_dataset = None
    if validation_records:
        if training_args.train_as_classification:
            validation_dataset = build_task_collection_classification_dataset(
                records=validation_records,
                tokenizer=tokenizer,
                model_max_length=training_args.model_max_length,
            )
        else:
            validation_dataset = build_task_collection_supervised_dataset(
                records=validation_records,
                tokenizer=tokenizer,
                max_source_length=data_args.max_source_length,
                max_target_length=data_args.max_target_length,
                model_max_length=training_args.model_max_length,
            )
    eval_dataset = dev_dataset if dev_dataset is not None else validation_dataset
    selection_records = dev_records if dev_records else validation_records
    selection_eval_dataset = None
    selection_compute_metrics = None
    if training_args.do_train and training_args.do_eval and selection_records and is_task_collection_generation_task(task):
        selection_eval_dataset, selection_prompt_lengths = build_task_collection_generation_dataset(
            records=selection_records,
            tokenizer=tokenizer,
            max_source_length=data_args.max_source_length,
            model_max_length=training_args.model_max_length,
            include_labels=True,
        )
        selection_compute_metrics = build_selection_compute_metrics(
            records=selection_records,
            prompt_lengths=selection_prompt_lengths,
            tokenizer=tokenizer,
            task_name=data_args.task_name,
        )
        training_args.predict_with_generate = True
        if task.metric_name in {"accuracy", "f1"}:
            training_args.metric_for_best_model = task.metric_name
            training_args.greater_is_better = True
            logger.info(
                "Best checkpoint will be selected by eval_%s using generation-based validation.",
                task.metric_name,
            )
        else:
            logger.info("Falling back to eval_loss for best checkpoint selection.")
    elif training_args.do_train and training_args.do_eval and selection_records and not is_task_collection_generation_task(task):
        training_args.predict_with_generate = False
        selection_eval_dataset = build_task_collection_classification_dataset(
            records=selection_records,
            tokenizer=tokenizer,
            model_max_length=training_args.model_max_length,
        )
        selection_compute_metrics = build_classification_compute_metrics(selection_records, data_args.task_name)
        training_args.metric_for_best_model = "accuracy"
        training_args.greater_is_better = True
        logger.info(
            "Best checkpoint will be selected by eval_accuracy using candidate scoring validation "
            "for classification task %s.",
            task.name,
        )
    elif training_args.do_train and training_args.do_eval:
            training_args.predict_with_generate = False
            logger.info("No labeled eval records available for validation selection; best checkpoint will use eval_loss.")
        
    if training_args.train_as_classification or not is_task_collection_generation_task(task):
        data_collator = TaskCollectionClassificationCollator(
            tokenizer=tokenizer,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    trainer_cls = ChunkSeq2SeqTrainer if model_args.chunk_tuning else Seq2SeqTrainer
    trainer_kwargs = {
        "args": training_args,
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": selection_eval_dataset if selection_eval_dataset is not None else eval_dataset,
        "tokenizer": tokenizer,
        "data_collator": data_collator,
        "compute_metrics": selection_compute_metrics,
        "preprocess_logits_for_metrics": preprocess_classification_logits_for_metrics if (
            training_args.do_train and training_args.do_eval and selection_records and not is_task_collection_generation_task(task)
        ) else None,
    }
    if model_args.chunk_tuning:
        trainer = trainer_cls(
            FThandler=GetCallBack(config),
            TaskType=model_args.TaskType,
            chunk_num=model_args.chunk_num,
            chunk_update_interval=model_args.chunk_update_interval,
            strategy=model_args.chunk_strategy,
            peft_type=model_args.peft_type,
            freeze_layers=model_args.freeze_layers,
            **trainer_kwargs,
        )
    else:
        trainer = trainer_cls(
            peft_type=model_args.peft_type,
            **trainer_kwargs,
        )

    if model_args.peft_type:
        trainer.add_callback(SavePeftModelCallback)
    align_generation_sampling_config(trainer.model, logger, force_do_sample=training_args.force_do_sample)
    if training_args.do_train:
        checkpoint = training_args.resume_from_checkpoint or last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        prune_to_best_checkpoint(training_args.output_dir, trainer.state.best_model_checkpoint)

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        final_results["train"] = metrics

    def build_generation_records(split_key: str, sampled_records: List[dict], full_pool: List[dict], target_limit: Optional[int]):
        if not icl_mode:
            return sampled_records, {"split_name": split_key, "mode": "no_icl"}
        target_seed = 0 if split_key == "validation" else sampling_seed
        return build_icl_records_for_split(
            data_args=data_args,
            split_name=split_key,
            train_pool=full_train_pool,
            target_records=sampled_records,
            target_pool=full_pool,
            target_limit=target_limit,
            target_seed=target_seed,
            sampled_train_records=train_records,
        )

    def run_scored_split(
        split_name: str,
        metric_key_prefix: str,
        records_for_split: List[dict],
    ) -> dict:
        if is_task_collection_generation_task(task):
            generation_dataset, prompt_lengths = build_task_collection_generation_dataset(
                records=records_for_split,
                tokenizer=tokenizer,
                max_source_length=data_args.max_source_length,
                model_max_length=training_args.model_max_length,
            )
            predict_with_generate = trainer.args.predict_with_generate
            trainer.args.predict_with_generate = True
            generation_output = trainer.predict(
                generation_dataset,
                metric_key_prefix=metric_key_prefix,
                max_new_tokens=data_args.max_target_length,
            )
            trainer.args.predict_with_generate = predict_with_generate
            decoded_predictions = decode_generation_predictions(
                generation_output.predictions,
                tokenizer=tokenizer,
                prompt_lengths=prompt_lengths,
            )
        else:
            decoded_predictions = []
            for record in records_for_split:
                candidate_scores = score_task_collection_record_candidates(
                    model=trainer.model,
                    tokenizer=tokenizer,
                    record=record,
                    model_max_length=training_args.model_max_length,
                )
                best_candidate_id = max(range(len(candidate_scores)), key=candidate_scores.__getitem__)
                decoded_predictions.append(str(record["candidates"][best_candidate_id]).strip())

        metrics = evaluate_task_collection_predictions(data_args.task_name, decoded_predictions, records_for_split)
        metrics["samples"] = len(records_for_split)
        return {
            "metrics": metrics,
            "predictions": decoded_predictions,
            "records": records_for_split,
        }

    def evaluate_supervised_split(split_name: str, split_key: str, records, dataset_obj, full_pool, target_limit):
        if not records and not full_pool:
            return
        if not icl_mode:
            logger.info("*** Evaluate loss on %s split ***", split_name)
            original_compute_metrics = trainer.compute_metrics
            original_predict_with_generate = trainer.args.predict_with_generate
            trainer.compute_metrics = None
            trainer.args.predict_with_generate = False
            metrics = trainer.evaluate(eval_dataset=dataset_obj, metric_key_prefix=split_name)
            trainer.compute_metrics = original_compute_metrics
            trainer.args.predict_with_generate = original_predict_with_generate
            metrics[f"{split_name}_samples"] = len(dataset_obj)
            loss_key = f"{split_name}_loss"
            if loss_key in metrics:
                try:
                    metrics[f"{split_name}_perplexity"] = math.exp(metrics[loss_key])
                except OverflowError:
                    metrics[f"{split_name}_perplexity"] = float("inf")
            trainer.log_metrics(split_name, metrics)
            trainer.save_metrics(split_name, metrics)
            final_results[split_name] = metrics

        generation_records, icl_manifest = build_generation_records(split_key, records, full_pool, target_limit)
        logger.info("*** Evaluate generation on %s split ***", split_name)
        generation_result = run_scored_split(split_name, f"{split_name}_generation", generation_records)
        generation_metrics = generation_result["metrics"]
        trainer.log_metrics(f"{split_name}_generation", generation_metrics)
        trainer.save_metrics(f"{split_name}_generation", generation_metrics)
        icl_manifest_path = os.path.join(training_args.output_dir, f"{split_name}_icl_manifest.json")
        save_json(icl_manifest_path, icl_manifest)
        final_results[f"{split_name}_generation"] = generation_metrics

    if multi_train_set_icl_mode:
        aggregate_results = {"eval": [], "predict": []}
        for trainset_id in range(data_args.num_train_sets or 0):
            logger.info("*** Running task-collection ICL train set %s/%s ***", trainset_id, data_args.num_train_sets)
            shared_train_records = sample_task_collection_subset(full_train_pool, data_args.max_train_samples, trainset_id)
            trainset_manifest = {
                "trainset_id": trainset_id,
                "sampling_seed": trainset_id,
                "demo_ids": [record["id"] for record in shared_train_records],
            }
            save_json(
                os.path.join(training_args.output_dir, f"trainset{trainset_id}_sampling_manifest.json"),
                trainset_manifest,
            )

            if training_args.do_eval and full_validation_pool:
                eval_records_for_set = sample_task_collection_subset(full_validation_pool, data_args.max_eval_samples, trainset_id)
                eval_generation_records = build_task_collection_icl_records(eval_records_for_set, [shared_train_records])
                eval_result = run_scored_split("eval", f"trainset{trainset_id}_eval_generation", eval_generation_records)
                aggregate_results["eval"].append(eval_result["metrics"])
                save_json(
                    os.path.join(training_args.output_dir, f"trainset{trainset_id}_eval_generation_metrics.json"),
                    eval_result["metrics"],
                )

            if training_args.do_predict:
                prediction_source = full_test_pool if full_test_pool else full_validation_pool
                prediction_split = "test" if full_test_pool else "validation"
                prediction_limit = data_args.max_test_samples if prediction_split == "test" else data_args.max_eval_samples
                prediction_records_for_set = sample_task_collection_subset(prediction_source, prediction_limit, trainset_id)
                prediction_generation_records = build_task_collection_icl_records(prediction_records_for_set, [shared_train_records])
                prediction_result = run_scored_split(
                    "predict",
                    f"trainset{trainset_id}_predict_generation",
                    prediction_generation_records,
                )
                aggregate_results["predict"].append(prediction_result["metrics"])
                save_json(
                    os.path.join(training_args.output_dir, f"trainset{trainset_id}_predict_metrics.json"),
                    prediction_result["metrics"],
                )
                predictions_path = os.path.join(
                    training_args.output_dir,
                    f"trainset{trainset_id}_predictions_{prediction_split}.txt",
                )
                with open(predictions_path, "w", encoding="utf-8") as writer:
                    for record, prediction in zip(prediction_result["records"], prediction_result["predictions"]):
                        writer.write(f"{record['id']}\t{prediction}\n")

        summary = {}
        for split_name, metrics_list in aggregate_results.items():
            if metrics_list:
                metric_names = sorted({name for metrics in metrics_list for name in metrics.keys()})
                summary[split_name] = {
                    metric_name: mean(
                        [metrics[metric_name] for metrics in metrics_list if metric_name in metrics]
                    )
                    for metric_name in metric_names
                }
        if summary:
            save_json(os.path.join(training_args.output_dir, "multi_train_set_summary.json"), summary)
            final_results["multi_train_set_summary"] = summary
        logger.info("Finished multi train-set ICL evaluation. Summary saved to %s", training_args.output_dir)
        result_path = data_args.result_file or os.path.join(training_args.output_dir, f"{result_tag}.json")
        save_json(result_path, final_results)
        return

    if training_args.do_eval:
        if dev_dataset is not None:
            evaluate_supervised_split(
                "dev",
                "dev",
                dev_records,
                dev_dataset,
                dev_records,
                len(dev_records),
            )
        if validation_dataset is not None:
            evaluate_supervised_split(
                "eval",
                "validation",
                validation_records,
                validation_dataset,
                full_validation_pool,
                data_args.max_eval_samples,
            )

    if training_args.do_predict:
        logger.info("*** Predict with generation on %s split ***", data_args.predict_split)
        available_prediction_splits = {
            "train": train_records,
            "dev": dev_records,
            "validation": validation_records,
            "test": test_records,
        }
        available_prediction_pools = {
            "train": full_train_pool,
            "dev": dev_records,
            "validation": full_validation_pool,
            "test": full_test_pool,
        }
        available_prediction_limits = {
            "train": data_args.max_train_samples,
            "dev": data_args.max_dev_samples,
            "validation": data_args.max_eval_samples,
            "test": data_args.max_test_samples,
        }
        actual_predict_split = data_args.predict_split
        prediction_records = available_prediction_splits.get(actual_predict_split, [])
        if not prediction_records and actual_predict_split == "test" and validation_records:
            logger.warning("Task %s has no accessible test split with current dataset source; falling back to validation.", task.name)
            actual_predict_split = "validation"
            prediction_records = validation_records
        prediction_pool = available_prediction_pools.get(actual_predict_split, prediction_records)
        prediction_limit = available_prediction_limits.get(actual_predict_split, len(prediction_records))
        if data_args.max_predict_samples is not None and data_args.max_predict_samples < len(prediction_records):
            prediction_records = prediction_records[: data_args.max_predict_samples]
            prediction_limit = data_args.max_predict_samples
        if not prediction_records:
            raise ValueError(f"No records available for predict split '{data_args.predict_split}'.")
        generation_records, icl_manifest = build_generation_records(
            actual_predict_split,
            prediction_records,
            prediction_pool,
            prediction_limit,
        )
        prediction_result = run_scored_split("predict", "predict", generation_records)
        decoded_predictions = prediction_result["predictions"]
        prediction_metrics = prediction_result["metrics"]
        trainer.log_metrics("predict", prediction_metrics)
        trainer.save_metrics("predict", prediction_metrics)
        final_results["predict"] = prediction_metrics

        output_predictions_path = os.path.join(training_args.output_dir, f"predictions_{actual_predict_split}.txt")
        os.makedirs(training_args.output_dir, exist_ok=True)
        with open(output_predictions_path, "w", encoding="utf-8") as writer:
            for record, prediction in zip(prediction_result["records"], decoded_predictions):
                writer.write(f"{record['id']}\t{prediction}\n")
        logger.info("Saved decoded predictions to %s", output_predictions_path)
        predict_icl_manifest_path = os.path.join(training_args.output_dir, f"predict_{actual_predict_split}_icl_manifest.json")
        save_json(predict_icl_manifest_path, icl_manifest)

    result_path = data_args.result_file or os.path.join(training_args.output_dir, f"{result_tag}.json")
    save_json(result_path, final_results)
    logger.info("Saved final merged results to %s", result_path)


if __name__ == "__main__":
    warnings.filterwarnings("default", category=FutureWarning)
    main()
