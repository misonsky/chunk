# coding=utf-8
"""Deprecated compatibility shim for the old QA trainer module.

The active implementation lives in ``chnk.qatrainer``. This module is kept only
to preserve legacy import paths while avoiding a second copy of trainer logic.
"""

import warnings

from .qatrainer import (
    ChunkQuestionAnsweringTrainer as _ChunkQuestionAnsweringTrainer,
    QuestionAnsweringTrainer as _QuestionAnsweringTrainer,
)


_LEGACY_QA_KWARGS = {
    "num_group",
    "optimizer_strategy",
    "keep_position",
    "keeping_layers",
    "layer_names",
    "freeze_emb",
    "freeze_output",
    "random_tuning",
}


def _warn_deprecated_module():
    warnings.warn(
        "`chnk.trainer_qa` is deprecated and now forwards to `chnk.qatrainer`. "
        "Please update imports to `chnk.qatrainer` or `chnk`.",
        DeprecationWarning,
        stacklevel=3,
    )


def _extract_legacy_kwargs(kwargs):
    legacy_kwargs = {key: kwargs.pop(key) for key in list(kwargs.keys()) if key in _LEGACY_QA_KWARGS}
    active_legacy_kwargs = {}
    for key, value in legacy_kwargs.items():
        if value is None:
            continue
        if key in {"freeze_emb", "freeze_output", "random_tuning"} and value is False:
            continue
        if key == "num_group" and value == 1:
            continue
        active_legacy_kwargs[key] = value
    return active_legacy_kwargs


class QuestionAnsweringTrainer(_QuestionAnsweringTrainer):
    def __init__(self, *args, **kwargs):
        _warn_deprecated_module()
        active_legacy_kwargs = _extract_legacy_kwargs(kwargs)
        if active_legacy_kwargs:
            legacy_keys = ", ".join(sorted(active_legacy_kwargs))
            raise ValueError(
                "Legacy QA trainer arguments are no longer supported in `chnk.trainer_qa`: "
                f"{legacy_keys}. Use the active implementation in `chnk.qatrainer`/`chnk.trainer`."
            )
        super().__init__(*args, **kwargs)


class ChunkQuestionAnsweringTrainer(_ChunkQuestionAnsweringTrainer):
    def __init__(self, *args, **kwargs):
        _warn_deprecated_module()
        active_legacy_kwargs = _extract_legacy_kwargs(kwargs)
        if active_legacy_kwargs:
            legacy_keys = ", ".join(sorted(active_legacy_kwargs))
            raise ValueError(
                "Legacy QA trainer arguments are no longer supported in `chnk.trainer_qa`: "
                f"{legacy_keys}. Use the active implementation in `chnk.qatrainer`/`chnk.trainer`."
            )
        super().__init__(*args, **kwargs)

