# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A subclass of `Trainer` specific to Question-Answering tasks
"""
import numpy as np
import importlib.util
import math
import time
import random
import re
import sys
import shutil
import json
from tqdm import tqdm
import os
from packaging import version
import torch.distributed as dist
from typing import TYPE_CHECKING, Any, Dict, Tuple, Union,Optional,List
import torch
from torch import nn
from torch.utils.data import RandomSampler,Dataset
from transformers import Trainer, is_torch_tpu_available
from distutils.util import strtobool
from transformers.trainer_utils import (
    TrainOutput,
    speed_metrics,
    PREFIX_CHECKPOINT_DIR,
    has_length,
    HPSearchBackend)
from transformers.trainer_pt_utils import (
    get_parameter_names,
    get_dataloader_sampler,
    get_model_param_count)
from transformers.utils import is_sagemaker_mp_enabled,logging,is_accelerate_available
try:
    from transformers.trainer_callback import TrainerState, ExportableState
except ImportError:
    from transformers.trainer_callback import TrainerState
    ExportableState = None

###
from transformers.dependency_versions_check import dep_version_check
from transformers.training_args import ParallelMode
from transformers.integrations.integration_utils import hp_params
from transformers.integrations.deepspeed import (
    deepspeed_load_checkpoint,
    deepspeed_init,
    is_deepspeed_available,
    is_deepspeed_zero3_enabled
)
from transformers.pytorch_utils import (
    ALL_LAYERNORM_LAYERS)
from transformers.training_args import (
    TrainingArguments)
from .optimizers.optimization import Adafactor, get_scheduler
from .optimizers import ExtendOptimizerNames as OptimizerNames
from .optimizers.clip_grad import clip_grad_norm_ as chunk_clip_grad_norm_
from transformers.utils.import_utils import (
    is_bitsandbytes_available,
)
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
###
from torch.nn.parallel import DistributedDataParallel
if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False


if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import (
        DistributedDataParallelKwargs,
        GradientAccumulationPlugin,
        load_fsdp_model,
        load_fsdp_optimizer,
        save_fsdp_model,
        save_fsdp_optimizer,
    )

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]

    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper
is_torch_less_than_1_11 = version.parse(torch.__version__.split("+")[0]) < version.parse("1.11")
# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
CHUNK_STATE_NAME = "chunkft_state.json"

logger = logging.get_logger(__name__)
from .optimizers import replace_backward
from .utils import set_chunk_checkpoint_layers


def _bytes_to_gib(num_bytes):
    return num_bytes / 1024**3


def get_torch_cuda_memory_stats(prefix="train"):
    try:
        import torch
    except ImportError:
        return {}

    if not torch.cuda.is_available():
        return {}

    device = torch.cuda.current_device()
    stats = {
        f"{prefix}_gpu_allocated_gb": _bytes_to_gib(torch.cuda.memory_allocated(device)),
        f"{prefix}_gpu_reserved_gb": _bytes_to_gib(torch.cuda.memory_reserved(device)),
        f"{prefix}_gpu_max_allocated_gb": _bytes_to_gib(torch.cuda.max_memory_allocated(device)),
        f"{prefix}_gpu_max_reserved_gb": _bytes_to_gib(torch.cuda.max_memory_reserved(device)),
    }
    return stats

def _debug_gpu_usage_enabled():
    return os.environ.get("CHUNKFT_DEBUG_GPU_USAGE", "0").lower() in {"1", "true", "yes", "on"}


def _bytes_to_gib(value):
    return value / (1024**3)


def _safe_cuda_utilization(device):
    utilization_fn = getattr(torch.cuda, "utilization", None)
    if not callable(utilization_fn):
        return None
    try:
        return float(utilization_fn(device))
    except Exception:
        return None


def _safe_cuda_memory_stats(prefix="train"):
    if not torch.cuda.is_available():
        return {}

    device = torch.cuda.current_device()
    stats = {
        f"{prefix}_gpu_allocated_gb": _bytes_to_gib(torch.cuda.memory_allocated(device)),
        f"{prefix}_gpu_reserved_gb": _bytes_to_gib(torch.cuda.memory_reserved(device)),
        f"{prefix}_gpu_max_allocated_gb": _bytes_to_gib(torch.cuda.max_memory_allocated(device)),
        f"{prefix}_gpu_max_reserved_gb": _bytes_to_gib(torch.cuda.max_memory_reserved(device)),
    }

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        stats.update(
            {
                f"{prefix}_gpu_free_gb": _bytes_to_gib(free_bytes),
                f"{prefix}_gpu_total_gb": _bytes_to_gib(total_bytes),
                f"{prefix}_gpu_used_gb": _bytes_to_gib(total_bytes - free_bytes),
                f"{prefix}_gpu_memory_usage_pct": (total_bytes - free_bytes) * 100.0 / total_bytes,
            }
        )
    except Exception:
        pass

    utilization = _safe_cuda_utilization(device)
    if utilization is not None:
        stats[f"{prefix}_gpu_utilization_pct"] = utilization
    return stats


def _reset_cuda_peak_memory_stats():
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
    except Exception:
        pass


class CudaMemoryStatsMixin:
    def _cuda_memory_stats_with_fluctuation(self, prefix="train"):
        stats = _safe_cuda_memory_stats(prefix)
        if not stats:
            return stats

        tracked_keys = [
            f"{prefix}_gpu_allocated_gb",
            f"{prefix}_gpu_reserved_gb",
            f"{prefix}_gpu_used_gb",
            f"{prefix}_gpu_memory_usage_pct",
        ]
        current = {key: stats[key] for key in tracked_keys if key in stats}
        previous = getattr(self, "_cuda_memory_previous_stats", None)
        running_min = getattr(self, "_cuda_memory_min_stats", {}).copy()
        running_max = getattr(self, "_cuda_memory_max_stats", {}).copy()

        for key, value in current.items():
            running_min[key] = min(running_min.get(key, value), value)
            running_max[key] = max(running_max.get(key, value), value)

            suffix = key[len(prefix) + 1 :]
            if previous is not None and key in previous:
                stats[f"{prefix}_{suffix}_delta_since_last_log"] = value - previous[key]
            stats[f"{prefix}_{suffix}_min"] = running_min[key]
            stats[f"{prefix}_{suffix}_max"] = running_max[key]
            stats[f"{prefix}_{suffix}_range"] = running_max[key] - running_min[key]

        self._cuda_memory_previous_stats = current
        self._cuda_memory_min_stats = running_min
        self._cuda_memory_max_stats = running_max
        return stats

    def _reset_cuda_memory_fluctuation_stats(self):
        self._cuda_memory_previous_stats = None
        self._cuda_memory_min_stats = {}
        self._cuda_memory_max_stats = {}

    def log(self, logs, *args, **kwargs):
        if isinstance(logs, dict):
            logs.update(self._cuda_memory_stats_with_fluctuation())
        return super().log(logs, *args, **kwargs)

def _iter_checkpoint_cache_owners(root):
    seen = set()
    stack = [root]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        for attr in ("optimizer", "deepspeed", "model_wrapped", "module"):
            child = getattr(current, attr, None)
            if child is not None:
                stack.append(child)


def _clear_checkpoint_runtime_caches(root):
    for owner in _iter_checkpoint_cache_owners(root):
        clear_fn = getattr(owner, "clear_checkpoint_transients", None)
        if callable(clear_fn):
            clear_fn()

def _initialize_trainer_state_compat(trainer, trial=None):
    state_fields = getattr(TrainerState, "__dataclass_fields__", {})
    state_kwargs = {}

    if "is_local_process_zero" in state_fields:
        state_kwargs["is_local_process_zero"] = trainer.is_local_process_zero()
    if "is_world_process_zero" in state_fields:
        state_kwargs["is_world_process_zero"] = trainer.is_world_process_zero()
    if "train_batch_size" in state_fields:
        state_kwargs["train_batch_size"] = getattr(trainer, "_train_batch_size", None)
    if "stateful_callbacks" in state_fields:
        if ExportableState is not None:
            state_kwargs["stateful_callbacks"] = [
                cb for cb in trainer.callback_handler.callbacks + [trainer.control] if isinstance(cb, ExportableState)
            ]
        else:
            state_kwargs["stateful_callbacks"] = []

    state = TrainerState(**state_kwargs)
    state.is_hyper_param_search = trial is not None
    if hasattr(state, "train_batch_size") and getattr(state, "train_batch_size", None) is None:
        state.train_batch_size = getattr(trainer, "_train_batch_size", None)
    return state

class PEFTrainer(CudaMemoryStatsMixin,Trainer):
    def __init__(self,peft_type=None,*args,**kwargs):
        super().__init__(*args, **kwargs)
        self.peft_type = peft_type
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        if _debug_gpu_usage_enabled():
            self.check_GPU_usage()
        return super().training_step(model, inputs)
    def check_GPU_usage(self):
        if not torch.cuda.is_available():
            return
        trainable_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)#334094338 # 44381186 #12598274
        trainable_parameters = trainable_parameters /1e6 #M
        peak_meory = torch.cuda.max_memory_allocated() /1024/1024/1024 #G
        utilization = torch.cuda.utilization(0)
        print("Trainable parameters {} peak_meory {} utilization {}".format(trainable_parameters,peak_meory,utilization))
    def _save_checkpoint(self, model, trial, metrics=None):
        _clear_checkpoint_runtime_caches(self)
        return super()._save_checkpoint(model, trial, metrics)
    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        _reset_cuda_peak_memory_stats()
        self._reset_cuda_memory_fluctuation_stats()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # self.state = TrainerState()
        # self.state.is_hyper_param_search = trial is not None
        # self.state.train_batch_size = self._train_batch_size
        self.state = _initialize_trainer_state_compat(self, trial=trial)

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            if getattr(self.model, "_chunkft_selective_gradient_checkpointing", None) is None:
                if args.gradient_checkpointing_kwargs is None:
                    gradient_checkpointing_kwargs = {}
                else:
                    gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

                self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(self.model_wrapped, resume_from_checkpoint)
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        _clear_checkpoint_runtime_caches(self)
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )
        if resume_from_checkpoint is not None:
            self._load_chunk_state(resume_from_checkpoint)

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                sampler = get_dataloader_sampler(train_dataloader)
                sampler_kinds = [RandomSampler]
                if version.parse(accelerate_version) > version.parse("0.23.0"):
                    sampler_kinds.append(SeedableRandomSampler)
                is_random_sampler = isinstance(sampler, tuple(sampler_kinds))
                if is_torch_less_than_1_11 or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    sampler = sampler if sampler is not None else []
                    _ = list(sampler)

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1

                if self.args.include_num_input_tokens_seen:
                    main_input_name = getattr(self.model, "main_input_name", "input_ids")
                    if main_input_name not in inputs:
                        logger.warning(
                            "Tried to track the number of tokens seen, however the current model is "
                            "not configured properly to know what item is the input. To fix this, add "
                            "a `main_input_name` attribute to the model class you are using."
                        )
                    else:
                        self.state.num_input_tokens_seen += self.accelerator.gather(inputs[main_input_name]).numel()
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                with self.accelerator.accumulate(model):
                    tr_loss_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                )

                if (
                    total_batched_samples % args.gradient_accumulation_steps == 0
                    or
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    is_last_step_and_steps_less_than_grad_acc
                ):
                    # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                    # in accelerate. So, explicitly enable sync gradients to True in that case.
                    if is_last_step_and_steps_less_than_grad_acc:
                        self.accelerator.gradient_state._set_sync_gradients(True)

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        # deepspeed does its own clipping

                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif self.use_apex:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm,
                            )
                        else:
                            self.accelerator.clip_grad_norm_(
                                self.trainable_parameters,
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    self.optimizer.step()
                    optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            
                            self.lr_scheduler.step()

                    if self.peft_type == "adalora":
                        if isinstance(model,DistributedDataParallel):
                            if not model.module.peft_config["default"].total_step:
                                model.module.peft_config["default"].total_step = max_steps
                            model.module.base_model.update_and_allocate(self.state.global_step)
                        else:
                            if not model.peft_config["default"].total_step:
                                model.peft_config["default"].total_step = max_steps
                            model.base_model.update_and_allocate(self.state.global_step)
                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, None,model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, None,model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss
        metrics.update(self._cuda_memory_stats_with_fluctuation("train"))

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)

class ChunkTrainer(CudaMemoryStatsMixin,Trainer):
    def __init__(self, 
                FThandler,
                TaskType,
                chunk_num,
                chunk_update_interval,
                strategy,
                peft_type,
                freeze_layers,
                enable_chunk_prefetch=True,
                *args,
                **kwargs):
        super().__init__(*args, **kwargs)
        if self.is_deepspeed_enabled:
            replace_backward()
        self.fthl = FThandler(freeze_layers,strategy,chunk_num,TaskType,peft_type)
        self.counter = 0
        self._should_advance_chunk = True
        self.chunk_num = chunk_num
        self.chunk_update_interval = int(chunk_update_interval)
        if self.chunk_update_interval < 1:
            raise ValueError("`chunk_update_interval` must be a positive integer.")
        self.enable_chunk_prefetch = bool(enable_chunk_prefetch)
        os.environ["CHUNKFT_ENABLE_PREFETCH"] = "1" if self.enable_chunk_prefetch else "0"
        self.strategy = strategy
        self.peft_type = peft_type
        self.active_chunk_update_count = 0
        self.base_requires_grad = {name: p.requires_grad for name, p in self.model.named_parameters()}
        self.trainable_param_names = self._build_trainable_param_names()
        self.parameter_budget_bytes = self._build_parameter_budget_bytes()
        self.fthl.set_parameter(
            self.model,
            active_param_names=self.trainable_param_names,
            parameter_budget_bytes=self.parameter_budget_bytes,
        )
        self.chunk_dict = self.fthl.get_chunk_dict
        self.decay_parameter_names = {
            name for name in self.get_decay_parameter_names(self.model) if name in self.trainable_param_names
        }
        self.trainable_parameter_infos = []
        self.chunk_parameter_infos_by_counter = [[] for _ in range(self.chunk_num)]
        self.active_chunk_parameters_by_counter = [[] for _ in range(self.chunk_num)]
        self.active_chunk_parameters = []
        for name, p in self.model.named_parameters():
            p.requires_grad = name in self.trainable_param_names
            if p.requires_grad:
                p.chunk_num = self.chunk_num
                chunk_ranges = getattr(p, "chunk_ranges", None)
                if chunk_ranges is None:
                    chunk_dim = 1 if p.strategy else 0
                    chunk_size = p.shape[chunk_dim]
                    chunk_ranges = self.chunk_dict[chunk_size]
                if len(chunk_ranges) != self.chunk_num:
                    raise RuntimeError(
                        f"Parameter `{name}` expected {self.chunk_num} chunk ranges, got {len(chunk_ranges)}."
                    )
                p.chunk_ranges = chunk_ranges
                self.trainable_parameter_infos.append((name, p, chunk_ranges))
                for counter, upd_ran in enumerate(chunk_ranges):
                    self.chunk_parameter_infos_by_counter[counter].append((p, upd_ran))
                    if upd_ran[0] < upd_ran[1]:
                        self.active_chunk_parameters_by_counter[counter].append(p)
        self.trainable_parameters = [p for _, p, _ in self.trainable_parameter_infos]
        self.active_chunk_index = 0
        set_chunk_checkpoint_layers(self.model, self.counter)
        # self.init_state = {name:p.requires_grad for name,p in self.model.named_parameters()}
        # self.group_parameters = self.fthl.group_model(self.model)
    def _build_optimizer_budget_config(self):
        optim_name = str(getattr(self.args, "optim", "")).lower()
        state_tensors = 0
        if any(name in optim_name for name in ["adam", "adamw", "lamb"]):
            state_tensors = 2
        elif any(name in optim_name for name in ["lion", "rmsprop", "sgd", "adagrad", "lars"]):
            state_tensors = 1
        elif "adafactor" in optim_name:
            state_tensors = 1

        optim_bits = 32
        if "8bit" in optim_name:
            optim_bits = 8
        block_wise = True
        min_8bit_size = 4096
        return {
            "optim_name": optim_name,
            "state_tensors": state_tensors,
            "optim_bits": optim_bits,
            "block_wise": block_wise,
            "min_8bit_size": min_8bit_size,
        }
    def _estimate_optimizer_state_bytes(self, name, p, optimizer_budget_config):
        state_tensors = optimizer_budget_config["state_tensors"]
        if state_tensors <= 0:
            return 0

        param_numel = p.numel()
        optim_bits = optimizer_budget_config["optim_bits"]
        min_8bit_size = optimizer_budget_config["min_8bit_size"]
        block_wise = optimizer_budget_config["block_wise"]
        use_8bit_state = optim_bits == 8 and param_numel >= min_8bit_size
        if (
            use_8bit_state
            and "adam" in optimizer_budget_config["optim_name"]
            and self.fthl.check_selection(self.fthl.emb_pattern, [name])
        ):
            use_8bit_state = False
        state_element_size = 1 if use_8bit_state else 4
        state_bytes = param_numel * state_tensors * state_element_size

        if use_8bit_state:
            if block_wise:
                blocks = math.ceil(param_numel / 2048)
                state_bytes += state_tensors * blocks * 4
            else:
                state_bytes += state_tensors * 8
        return state_bytes
    def _build_parameter_budget_bytes(self):
        optimizer_budget_config = self._build_optimizer_budget_config()
        parameter_budget_bytes = {}
        for name, p in self.model.named_parameters():
            param_bytes = p.numel() * p.element_size()
            optimizer_state_bytes = self._estimate_optimizer_state_bytes(name, p, optimizer_budget_config)
            parameter_budget_bytes[name] = param_bytes + optimizer_state_bytes
        return parameter_budget_bytes
    def _get_chunk_state_path(self, output_dir):
        return os.path.join(output_dir, CHUNK_STATE_NAME)
    def _save_chunk_state(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        chunk_state = {
            "next_chunk_idx": int(self.counter),
            "should_advance_chunk": bool(self._should_advance_chunk),
            "chunk_num": int(self.chunk_num),
            "chunk_update_interval": int(self.chunk_update_interval),
            "active_chunk_idx": int(self.active_chunk_index),
            "active_chunk_update_count": int(self.active_chunk_update_count),
        }
        with open(self._get_chunk_state_path(output_dir), "w", encoding="utf-8") as f:
            json.dump(chunk_state, f)
    def _load_chunk_state(self, checkpoint):
        completed_updates = int(self.state.global_step)
        default_counter = (completed_updates // self.chunk_update_interval) % self.chunk_num if self.chunk_num else 0
        default_active_chunk_update_count = completed_updates % self.chunk_update_interval if self.chunk_num else 0
        default_should_advance_chunk = default_active_chunk_update_count == 0
        default_active_chunk_index = 0
        if self.chunk_num:
            if default_should_advance_chunk:
                default_active_chunk_index = (default_counter - 1) % self.chunk_num
            else:
                default_active_chunk_index = default_counter
        chunk_state = {
            "next_chunk_idx": default_counter,
            "should_advance_chunk": default_should_advance_chunk,
            "chunk_num": self.chunk_num,
            "chunk_update_interval": self.chunk_update_interval,
            "active_chunk_idx": default_active_chunk_index,
            "active_chunk_update_count": default_active_chunk_update_count,
        }
        chunk_state_path = self._get_chunk_state_path(checkpoint)
        if os.path.isfile(chunk_state_path):
            with open(chunk_state_path, "r", encoding="utf-8") as f:
                loaded_state = json.load(f)
            if loaded_state.get("chunk_num") not in (None, self.chunk_num):
                logger.warning(
                    "Chunk state chunk_num=%s does not match current chunk_num=%s. Falling back to step-derived chunk index.",
                    loaded_state.get("chunk_num"),
                    self.chunk_num,
                )
            elif loaded_state.get("chunk_update_interval") not in (None, self.chunk_update_interval):
                logger.warning(
                    "Chunk state chunk_update_interval=%s does not match current chunk_update_interval=%s. Falling back to step-derived chunk index.",
                    loaded_state.get("chunk_update_interval"),
                    self.chunk_update_interval,
                )
            else:
                chunk_state.update(loaded_state)
        self.counter = int(chunk_state["next_chunk_idx"]) % self.chunk_num
        self._should_advance_chunk = bool(chunk_state.get("should_advance_chunk", True))
        self.active_chunk_index = int(chunk_state.get("active_chunk_idx", self.counter)) % self.chunk_num
        self.active_chunk_update_count = int(chunk_state.get("active_chunk_update_count", 0))
        if self._should_advance_chunk:
            self.active_chunk_update_count = 0
            self.active_chunk_parameters = []
        else:
            self.select_parameter(self.active_chunk_index)
        set_chunk_checkpoint_layers(self.model, self.active_chunk_index)
    def _build_trainable_param_names(self):
        active_groups = set(self.fthl.group_model(self.model))
        trainable_param_names = set()
        for name, p in self.model.named_parameters():
            if not self.base_requires_grad.get(name, p.requires_grad):
                continue
            group_name = self.fthl.match_group(name)
            if group_name is None or group_name in active_groups:
                trainable_param_names.add(name)
        return trainable_param_names
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        if self.optimizer is None:
            trainable_named_parameters = [
                (n, p) for n, p in opt_model.named_parameters() if n in self.trainable_param_names
            ]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in trainable_named_parameters if n in self.decay_parameter_names
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in trainable_named_parameters if n not in self.decay_parameter_names
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = ChunkTrainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")
        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)
        return self.optimizer
    def compute_parameter_memory(self):
        print(f"Optimizer state size: {sum(p.element_size() * p.nelement() for p in self.optimizer.state.values()) / (1024 ** 2):.2f} MB")
    def select_parameter(self, chunk_index=None):
        current_counter = self.counter if chunk_index is None else int(chunk_index) % self.chunk_num
        for p, upd_ran in self.chunk_parameter_infos_by_counter[current_counter]:
            p.counter = current_counter
            p.upd_ran = upd_ran
        self.active_chunk_parameters = self.active_chunk_parameters_by_counter[current_counter]
        self.active_chunk_index = current_counter
        set_chunk_checkpoint_layers(self.model, current_counter)
        if self.enable_chunk_prefetch and self.optimizer is not None:
            prefetch_fn = getattr(self.optimizer, "prefetch_chunk_states", None)
            if callable(prefetch_fn) and self.active_chunk_parameters:
                prefetch_fn(self.active_chunk_parameters, target_counter=current_counter)
                if self.chunk_num > 1:
                    next_counter = (current_counter + 1) % self.chunk_num
                    next_chunk_parameters = self.active_chunk_parameters_by_counter[next_counter]
                    if next_chunk_parameters:
                        prefetch_fn(next_chunk_parameters, target_counter=next_counter)
    def clip_active_chunk_grads(self, max_grad_norm):
        active_parameters = [p for p in self.active_chunk_parameters if getattr(p, "new_grad", None) is not None]
        return chunk_clip_grad_norm_(active_parameters, max_grad_norm)
    def get_optimizer_state(self):
        states_groups={}
        optimizer_grouped_parameters = self.optimizer.param_groups
        for i, param_group in enumerate(optimizer_grouped_parameters):
            param_group.pop("params")
            states_groups[i] = param_group
        return states_groups
    def update_parameter_state(self):
        self.select_parameter(self.counter)
        self.active_chunk_update_count = 0
    def complete_chunk_update(self):
        self.active_chunk_update_count += 1
        if self.active_chunk_update_count >= self.chunk_update_interval:
            self.counter = (self.active_chunk_index + 1) % self.chunk_num
            self.active_chunk_update_count = 0
            self._should_advance_chunk = True
        else:
            self.counter = self.active_chunk_index
            self._should_advance_chunk = False
            if self.enable_chunk_prefetch and self.optimizer is not None:
                prefetch_fn = getattr(self.optimizer, "prefetch_chunk_states", None)
                if callable(prefetch_fn) and self.active_chunk_parameters:
                    prefetch_fn(self.active_chunk_parameters, target_counter=self.active_chunk_index)
    def check_optimizer_groups_size(self):
        total_size = 0
        optimizer_grouped_parameters = self.optimizer.param_groups
        for i, param_group in enumerate(optimizer_grouped_parameters):
            params = param_group['params']
            for _index,param in enumerate(params):
                param_size = param.numel() * param.element_size()
                total_size += param_size
        total_size_mb = total_size / (1024 ** 2)
        print(f'Total size of the parameters in this group: {total_size_mb:.2f} MB')
    def check_optimizer_state_dict(self):
        optimizer_state = self.optimizer.state_dict()
        states = optimizer_state["state"]
        # optimizer_state = {k: v.cpu() for k, v in states.items()}
        # self.optimizer.load_state_dict(optimizer_state)
        # param_group = optimizer_state["param_groups"]
        total_size = 0
        total_gpu_size =0
        print(states.keys())
        for step in states:
            print("------------------{}----------------".format(step))
            step_info = states[step]
            p_step = step_info["step"]
            exp_avg = step_info["exp_avg"]
            exp_avg_sq = step_info["exp_avg_sq"]
            print("exp_avg devices",exp_avg.device)
            print("exp_avg_sq devices",exp_avg_sq.device)
            # print(p_step)
            param_size1 = exp_avg.numel() * exp_avg.element_size()
            param_size2 = exp_avg_sq.numel() * exp_avg_sq.element_size()
            total_size += param_size1 + param_size2
        # print(k)
        total_size_mb = total_size / (1024 ** 2)
        print(f'Total size of the states in optimizer: {total_size_mb:.2f} MB')
        print("**"*50)
    def check_trainable(self):
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        for n, p in opt_model.named_parameters():
            if p.requires_grad:
                print(n)
        print("***********************************************")
    def check_GPU_usage(self):
        trainable_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)#334094338 # 44381186 #12598274
        trainable_parameters = trainable_parameters /1e6 #M
        peak_meory = torch.cuda.max_memory_allocated() /1024/1024/1024 #G
        utilization = torch.cuda.utilization(0)
        print("Trainable parameters {} peak_meory {} utilization {}".format(trainable_parameters,peak_meory,utilization))
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        if self._should_advance_chunk:
            self.update_parameter_state()
            self._should_advance_chunk = False
        # self.check_optimizer_state_dict()
        # self.check_optimizer_groups_size()
        # trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)#334094338 # 44381186 #12598274
        # print(f"Trainable parameters: {trainable_parameters}")
        # self.compute_parameter_memory()
        # self.check_GPU_usage()
        # self.check_trainable()
        return super().training_step(model, inputs)
    def _save_checkpoint(self, model, trial, metrics=None):
        _clear_checkpoint_runtime_caches(self)
        super()._save_checkpoint(model, trial, metrics)
        output_dir = os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")
        self._save_chunk_state(output_dir)
    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        _reset_cuda_peak_memory_stats()
        self._reset_cuda_memory_fluctuation_stats()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # self.state = TrainerState()
        # self.state.is_hyper_param_search = trial is not None
        # self.state.train_batch_size = self._train_batch_size
        self.state = _initialize_trainer_state_compat(self, trial=trial)

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            if getattr(self.model, "_chunkft_selective_gradient_checkpointing", None) is None:
                if args.gradient_checkpointing_kwargs is None:
                    gradient_checkpointing_kwargs = {}
                else:
                    gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

                self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(self.model_wrapped, resume_from_checkpoint)
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        _clear_checkpoint_runtime_caches(self)
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )
        if resume_from_checkpoint is not None:
            self._load_chunk_state(resume_from_checkpoint)

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                sampler = get_dataloader_sampler(train_dataloader)
                sampler_kinds = [RandomSampler]
                if version.parse(accelerate_version) > version.parse("0.23.0"):
                    sampler_kinds.append(SeedableRandomSampler)
                is_random_sampler = isinstance(sampler, tuple(sampler_kinds))
                if is_torch_less_than_1_11 or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    sampler = sampler if sampler is not None else []
                    _ = list(sampler)

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1

                if self.args.include_num_input_tokens_seen:
                    main_input_name = getattr(self.model, "main_input_name", "input_ids")
                    if main_input_name not in inputs:
                        logger.warning(
                            "Tried to track the number of tokens seen, however the current model is "
                            "not configured properly to know what item is the input. To fix this, add "
                            "a `main_input_name` attribute to the model class you are using."
                        )
                    else:
                        self.state.num_input_tokens_seen += self.accelerator.gather(inputs[main_input_name]).numel()
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                with self.accelerator.accumulate(model):
                    tr_loss_step = self.training_step(model, inputs)
                
                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step
                # print("="*40)
                # for name,p in model.named_parameters():
                #     if not hasattr(p,"counter"):
                #         print(name)
                # print("="*40)
                # print(k)
                self.current_flos += float(self.floating_point_ops(inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                )

                if (
                    total_batched_samples % args.gradient_accumulation_steps == 0
                    or
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    is_last_step_and_steps_less_than_grad_acc
                ):
                    # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                    # in accelerate. So, explicitly enable sync gradients to True in that case.
                    if is_last_step_and_steps_less_than_grad_acc:
                        self.accelerator.gradient_state._set_sync_gradients(True)

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        # deepspeed does its own clipping

                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif self.use_apex:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm,
                            )
                        else:
                            self.clip_active_chunk_grads(args.max_grad_norm)

                    # Optimizer step
                    self.optimizer.step()
                    optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                    if optimizer_was_run:
                        self.complete_chunk_update()
                    else:
                        self.counter = self.active_chunk_index
                        self._should_advance_chunk = False
                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            if self.counter == 0:
                                self.lr_scheduler.step()
                    else:
                        if self.counter == 0:
                            self.lr_scheduler.step()
                    if self.peft_type == "adalora":
                        if isinstance(model,DistributedDataParallel):
                            if not model.module.peft_config["default"].total_step:
                                model.module.peft_config["default"].total_step = max_steps
                            model.module.base_model.update_and_allocate(self.state.global_step)
                        else:
                            if not model.peft_config["default"].total_step:
                                model.peft_config["default"].total_step = max_steps
                            model.base_model.update_and_allocate(self.state.global_step)
                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, None,model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, None,model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss
        metrics.update(self._cuda_memory_stats_with_fluctuation("train"))

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)
    @staticmethod
    def get_optimizer_cls_and_kwargs(args: TrainingArguments) -> Tuple[Any, Any]:
        """
        Returns the optimizer class and optimizer parameters based on the training arguments.

        Args:
            args (`transformers.training_args.TrainingArguments`):
                The training arguments for the training session.

        """

        for optimizer in OptimizerNames:
            print(optimizer.name, optimizer.value)
        # parse args.optim_args
        optim_args = {}
        if args.optim_args:
            for mapping in args.optim_args.replace(" ", "").split(","):
                key, value = mapping.split("=")
                optim_args[key] = value

        optimizer_kwargs = {"lr": args.learning_rate}

        adam_kwargs = {
            "betas": (args.adam_beta1, args.adam_beta2),
            "eps": args.adam_epsilon,
        }
        if args.optim == OptimizerNames.ADAFACTOR:
            optimizer_cls = Adafactor
            optimizer_kwargs.update({"scale_parameter": False, "relative_step": False})
        elif args.optim == OptimizerNames.ADAMW_HF:
            from .optimizers.optimization import AdamW

            optimizer_cls = AdamW
            optimizer_kwargs.update(adam_kwargs)
        elif args.optim in [OptimizerNames.ADAMW_TORCH, OptimizerNames.ADAMW_TORCH_FUSED]:
            from .optimizers.torchAdamw import AdamW

            optimizer_cls = AdamW
            optimizer_kwargs.update(adam_kwargs)
            if args.optim == OptimizerNames.ADAMW_TORCH_FUSED:
                optimizer_kwargs.update({"fused": True})
        elif args.optim == OptimizerNames.ADAMW_TORCH_XLA:
            try:
                from torch_xla.amp.syncfree import AdamW

                optimizer_cls = AdamW
                optimizer_kwargs.update(adam_kwargs)
            except ImportError:
                raise ValueError("Trainer failed to import syncfree AdamW from torch_xla.")
        elif args.optim == OptimizerNames.ADAMW_TORCH_NPU_FUSED:
            try:
                from torch_npu.optim import NpuFusedAdamW

                optimizer_cls = NpuFusedAdamW
                optimizer_kwargs.update(adam_kwargs)
            except ImportError:
                raise ValueError("Trainer failed to import FusedAdamW from torch_npu.")
        elif args.optim == OptimizerNames.ADAMW_APEX_FUSED:
            try:
                from apex.optimizers import FusedAdam

                optimizer_cls = FusedAdam
                optimizer_kwargs.update(adam_kwargs)
            except ImportError:
                raise ValueError("Trainer tried to instantiate apex FusedAdam but apex is not installed!")
        
        elif args.optim in [
            OptimizerNames.ADAMW_BNB,
            OptimizerNames.ADAMW_8BIT,
            OptimizerNames.PAGED_ADAMW,
            OptimizerNames.PAGED_ADAMW_8BIT,
            OptimizerNames.LION,
            OptimizerNames.LION_8BIT,
            OptimizerNames.PAGED_LION,
            OptimizerNames.PAGED_LION_8BIT,
        ]:
            from .optimizers import BitAdamW, BitLion
            is_paged = False
            optim_bits = 32
            optimizer_cls = None
            additional_optim_kwargs = adam_kwargs
            if "paged" in args.optim:
                is_paged = True
            if "8bit" in args.optim:
                optim_bits = 8
            if "adam" in args.optim:
                optimizer_cls = BitAdamW
            elif "lion" in args.optim:
                optimizer_cls = BitLion
                additional_optim_kwargs = {"betas": (args.adam_beta1, args.adam_beta2)}
            bnb_kwargs = {"is_paged": is_paged, "optim_bits": optim_bits}
            optimizer_kwargs.update(additional_optim_kwargs)
            optimizer_kwargs.update(bnb_kwargs)
        elif args.optim in [
            OptimizerNames.ADAM,
            OptimizerNames.ADAM_32BIT,
            OptimizerNames.ADAM_8BIT,
            OptimizerNames.PAGED_ADAM,
            OptimizerNames.PAGED_ADAM_32BIT,
            OptimizerNames.PAGED_ADAM_8BIT
        ]:
            from .optimizers import Adam32bit,Adam8bit,PagedAdam8bit,PagedAdam32bit
            
            additional_optim_kwargs = adam_kwargs
            if "paged" in args.optim:
                if "8bit" in args.optim:
                    optimizer_cls = PagedAdam8bit
                else:
                    optimizer_cls = PagedAdam32bit
            else:
                if "8bit" in args.optim:
                    optimizer_cls = Adam8bit
                else:
                    optimizer_cls = Adam32bit
            optimizer_kwargs.update(additional_optim_kwargs)
        elif args.optim in [
            OptimizerNames.LAMB,
            OptimizerNames.LAMB_32BIT,
            OptimizerNames.LAMB_8BIT
        ]:
            from .optimizers import BitLAMB,LAMB32bit,LAMB8bit
            if "8bit" in args.optim:
                optimizer_cls = LAMB8bit
            else:
                ## bitsandbytes only support 8-bit lamb
                optimizer_cls = LAMB8bit
            additional_optim_kwargs = {"betas": (args.adam_beta1, args.adam_beta2)}
            optimizer_kwargs.update(additional_optim_kwargs)
        elif args.optim in [
            OptimizerNames.LARS,
            OptimizerNames.LARS_32BIT,
            OptimizerNames.LARS_8BIT
        ]:
            from .optimizers import BitLARS,LARS8bit,LARS32bit
            if "8bit" in args.optim:
                optimizer_cls = LARS8bit
            else:
                ## bitsandbytes only support 8-bit lars
                optimizer_cls = LARS8bit
            additional_optim_kwargs = {"momentum": args.adam_beta1}
            optimizer_kwargs.update(additional_optim_kwargs)
        elif args.optim in [
            OptimizerNames.BSGD,
            OptimizerNames.BSGD_32BIT,
            OptimizerNames.BSGD_8BIT
        ]:
            from .optimizers import BitSGD,SGD8bit,SGD32bit
            if "8bit" in args.optim:
                optimizer_cls = SGD8bit
            else:
                ## bitsandbytes only support 8-bit lars
                optimizer_cls = SGD32bit
        elif args.optim in [
            OptimizerNames.BRMSPROP,
            OptimizerNames.BRMSPROP_32BIT,
            OptimizerNames.BRMSPROP_8BIT
        ]:
            from .optimizers import BitRMSprop,RMSprop8bit,RMSprop32bit
            if "8bit" in args.optim:
                optimizer_cls = RMSprop8bit
            else:
                ## bitsandbytes only support 8-bit lars
                optimizer_cls = RMSprop32bit
        elif args.optim == OptimizerNames.ADAMW_ANYPRECISION:
            try:
                from torchdistx.optimizers import AnyPrecisionAdamW

                optimizer_cls = AnyPrecisionAdamW
                optimizer_kwargs.update(adam_kwargs)

                # TODO Change dtypes back to M=FP32, Var = BF16, Kahan = False once they can be cast together in torchdistx.
                optimizer_kwargs.update(
                    {
                        "use_kahan_summation": strtobool(optim_args.get("use_kahan_summation", "False")),
                        "momentum_dtype": getattr(torch, optim_args.get("momentum_dtype", "float32")),
                        "variance_dtype": getattr(torch, optim_args.get("variance_dtype", "float32")),
                        "compensation_buffer_dtype": getattr(
                            torch, optim_args.get("compensation_buffer_dtype", "bfloat16")
                        ),
                    }
                )
            except ImportError:
                raise ValueError("Please install https://github.com/pytorch/torchdistx")
        elif args.optim == OptimizerNames.SGD:
            from .optimizers.sgd import SGD
            optimizer_cls = SGD
        
        elif args.optim == OptimizerNames.ADAGRAD:
            from .optimizers.adagrad import Adagrad
            optimizer_cls = Adagrad
        elif args.optim == OptimizerNames.RMSPROP:
            from .optimizers.rmsprop import RMSprop
            optimizer_cls = RMSprop
        else:
            raise ValueError(f"Trainer cannot instantiate unsupported optimizer: {args.optim}")
        logger.info(f"the optimizer you are using: {optimizer_cls}")
        return optimizer_cls, optimizer_kwargs
