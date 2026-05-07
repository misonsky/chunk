import random
import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from collections.abc import Mapping
import datasets
import numpy as np
from datasets import Dataset, load_dataset
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset as TorchDataset
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import PaddingStrategy

IGNORE_INDEX = -100


def _normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def _qa_f1(prediction: str, answers: Sequence[str]) -> float:
    normalized_prediction = _normalize_answer(prediction)
    prediction_tokens = normalized_prediction.split()
    if not answers:
        return 0.0
    if answers[0] in {"CANNOTANSWER", "no answer"}:
        return float(normalized_prediction == _normalize_answer(answers[0]))

    best_f1 = 0.0
    for answer in answers:
        answer_tokens = _normalize_answer(answer).split()
        common = Counter(prediction_tokens) & Counter(answer_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = num_same / max(len(prediction_tokens), 1)
        recall = num_same / max(len(answer_tokens), 1)
        best_f1 = max(best_f1, (2 * precision * recall) / (precision + recall))
    return best_f1


def _canonical_task_name(task_name: str) -> str:
    normalized = task_name.strip().lower()
    aliases = {
        "sst2": "sst2",
        "sst-2": "sst2",
        "rte": "rte",
        "cb": "cb",
        "boolq": "boolq",
        "wsc": "wsc",
        "wsc.fixed": "wsc",
        "wic": "wic",
        "multirc": "multirc",
        "multi_rc": "multirc",
        "copa": "copa",
        "record": "record",
        "recorddataset": "record",
        "squad": "squad",
        "drop": "drop",
    }
    if normalized not in aliases:
        supported = ", ".join(sorted(set(aliases.values())))
        raise ValueError(f"Unsupported task collection task '{task_name}'. Supported tasks: {supported}")
    return aliases[normalized]


def _sample_records(records: List[Dict], max_samples: Optional[int], seed: int) -> List[Dict]:
    if max_samples is None or max_samples >= len(records):
        return records
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    return [records[idx] for idx in indices[:max_samples]]


def _split_train_dev_records(
    records: List[Dict],
    max_train_samples: Optional[int],
    max_dev_samples: Optional[int],
    seed: int,
) -> Tuple[List[Dict], List[Dict]]:
    if max_train_samples is None and max_dev_samples is None:
        return records, []

    train_size = max_train_samples or 0
    dev_size = max_dev_samples or 0
    total = train_size + dev_size
    sampled_records = _sample_records(records, total if total > 0 else None, seed)
    if dev_size == 0:
        return sampled_records, []
    return sampled_records[:train_size], sampled_records[train_size : train_size + dev_size]


@dataclass
class TaskCollectionTask:
    name: str
    metric_name: str = "accuracy"
    train_as_classification_default: bool = False
    generation: bool = False
    def load_raw_splits(self, cache_dir: Optional[str] = None) -> Dict[str, Sequence[Dict]]:
        raise NotImplementedError

    def convert_example(self, example: Dict, split: str, idx: int) -> Dict:
        raise NotImplementedError

    def load_records_by_split(
        self,
        cache_dir: Optional[str] = None,
        max_train_samples: Optional[int] = None,
        max_dev_samples: Optional[int] = None,
        max_eval_samples: Optional[int] = None,
        max_test_samples: Optional[int] = None,
        data_seed: int = 42,
    ) -> Dict[str, List[Dict]]:
        raw_splits = self.load_raw_splits(cache_dir=cache_dir)
        records_by_split = {
            split_name: [self.convert_example(example, split_name, idx) for idx, example in enumerate(split_records)]
            for split_name, split_records in raw_splits.items()
        }
        train_records, dev_records = _split_train_dev_records(
            records=records_by_split.get("train", []),
            max_train_samples=max_train_samples,
            max_dev_samples=max_dev_samples,
            seed=data_seed,
        )
        validation_records = _sample_records(records_by_split.get("validation", []), max_eval_samples, data_seed)
        test_records = _sample_records(records_by_split.get("test", []), max_test_samples, data_seed)
        return {
            "train": train_records,
            "dev": dev_records,
            "validation": validation_records,
            "test": test_records,
        }


def _classification_target(verbalizer: Dict[int, str], example: Dict) -> Tuple[str, List[str]]:
    label = example.get("label")
    if label is None or int(label) < 0:
        return "", []
    answer = verbalizer[int(label)]
    return answer, [answer]

def _resolve_candidate_label(record: Dict) -> Optional[int]:
    label = record.get("label")
    if label is not None and int(label) >= 0:
        return int(label)

    candidates = record.get("candidates") or []
    answers = record.get("answers") or []
    normalized_answers = {_normalize_answer(str(answer)) for answer in answers}
    for candidate_id, candidate in enumerate(candidates):
        if _normalize_answer(str(candidate)) in normalized_answers:
            return candidate_id
    return None

class SST2Task(TaskCollectionTask):
    verbalizer = {0: " terrible", 1: " great"}

    def __init__(self) -> None:
        super().__init__(name="sst2", metric_name="accuracy", train_as_classification_default=True)

    def load_raw_splits(self, cache_dir: Optional[str] = None) -> Dict[str, Sequence[Dict]]:
        dataset = load_dataset("glue", "sst2", cache_dir=cache_dir)
        splits = {"train": dataset["train"], "validation": dataset["validation"]}
        if "test" in dataset:
            splits["test"] = dataset["test"]
        return splits

    def convert_example(self, example: Dict, split: str, idx: int) -> Dict:
        target, answers = _classification_target(self.verbalizer, example)
        label = example.get("label")
        return {
            "id": str(example.get("idx", f"{split}-{idx}")),
            "task_name": self.name,
            "metric_name": self.metric_name,
            "prompt": f"{example['sentence'].strip()} It was",
            "target": target,
            "answers": [answer.strip() for answer in answers],
            "candidates": [self.verbalizer[0], self.verbalizer[1]],
            "label": int(label) if label is not None and int(label) >= 0 else None,
        }


class RTETask(TaskCollectionTask):
    verbalizer = {0: "Yes", 1: "No"}

    def __init__(self) -> None:
        super().__init__(name="rte", metric_name="accuracy", train_as_classification_default=True)

    def load_raw_splits(self, cache_dir: Optional[str] = None) -> Dict[str, Sequence[Dict]]:
        dataset = load_dataset("super_glue", "rte", cache_dir=cache_dir)
        splits = {"train": dataset["train"], "validation": dataset["validation"]}
        if "test" in dataset:
            splits["test"] = dataset["test"]
        return splits

    def convert_example(self, example: Dict, split: str, idx: int) -> Dict:
        prompt = (
            f"{example['premise']}\n"
            f"Does this mean that \"{example['hypothesis']}\" is true? Yes or No?\n"
        )
        target, answers = _classification_target(self.verbalizer, example)
        label = example.get("label")
        return {
            "id": str(example.get("idx", f"{split}-{idx}")),
            "task_name": self.name,
            "metric_name": self.metric_name,
            "prompt": prompt,
            "target": target,
            "answers": answers,
            "candidates": [self.verbalizer[0], self.verbalizer[1]],
            "label": int(label) if label is not None and int(label) >= 0 else None,
        }


class CBTask(TaskCollectionTask):
    verbalizer = {0: "Yes", 1: "No", 2: "Maybe"}

    def __init__(self) -> None:
        super().__init__(name="cb", metric_name="accuracy", train_as_classification_default=True)

    def load_raw_splits(self, cache_dir: Optional[str] = None) -> Dict[str, Sequence[Dict]]:
        dataset = load_dataset("super_glue", "cb", cache_dir=cache_dir)
        splits = {"train": dataset["train"], "validation": dataset["validation"]}
        if "test" in dataset:
            splits["test"] = dataset["test"]
        return splits

    def convert_example(self, example: Dict, split: str, idx: int) -> Dict:
        prompt = (
            f"Suppose {example['premise']} "
            f"Can we infer that \"{example['hypothesis']}\"? Yes, No, or Maybe?\n"
        )
        target, answers = _classification_target(self.verbalizer, example)
        label = example.get("label")
        return {
            "id": str(example.get("idx", f"{split}-{idx}")),
            "task_name": self.name,
            "metric_name": self.metric_name,
            "prompt": prompt,
            "target": target,
            "answers": answers,
            "candidates": [self.verbalizer[0], self.verbalizer[1], self.verbalizer[2]],
            "label": int(label) if label is not None and int(label) >= 0 else None,
        }


class BoolQTask(TaskCollectionTask):
    def __init__(self) -> None:
        super().__init__(name="boolq", metric_name="accuracy", train_as_classification_default=True)

    def load_raw_splits(self, cache_dir: Optional[str] = None) -> Dict[str, Sequence[Dict]]:
        dataset = load_dataset("boolq", cache_dir=cache_dir)
        splits = {"train": dataset["train"], "validation": dataset["validation"]}
        if "test" in dataset:
            splits["test"] = dataset["test"]
        return splits

    def convert_example(self, example: Dict, split: str, idx: int) -> Dict:
        question = example["question"]
        if not question.endswith("?"):
            question = question + "?"
        question = question[0].upper() + question[1:]
        answer_value = example.get("answer")
        answer = ""
        answers: List[str] = []
        if answer_value is not None:
            answer = "Yes" if answer_value else "No"
            answers = [answer]
        label = None
        if answer_value is not None:
            label = 0 if answer_value else 1
        return {
            "id": f"{split}-{idx}",
            "task_name": self.name,
            "metric_name": self.metric_name,
            "prompt": f"{example['passage']} {question}\n",
            "target": answer,
            "answers": answers,
            "candidates": ["Yes", "No"],
            "label": label,
        }


class WSCTask(TaskCollectionTask):
    verbalizer = {0: "No", 1: "Yes"}

    def __init__(self) -> None:
        super().__init__(name="wsc", metric_name="accuracy", train_as_classification_default=True)

    def load_raw_splits(self, cache_dir: Optional[str] = None) -> Dict[str, Sequence[Dict]]:
        dataset = load_dataset("super_glue", "wsc.fixed", cache_dir=cache_dir)
        splits = {"train": dataset["train"], "validation": dataset["validation"]}
        if "test" in dataset:
            splits["test"] = dataset["test"]
        return splits

    def convert_example(self, example: Dict, split: str, idx: int) -> Dict:
        prompt = (
            f"{example['text']}\n"
            f"In the previous sentence, does the pronoun \"{example['span2_text'].lower()}\" "
            f"refer to {example['span1_text']}? Yes or No?\n"
        )
        target, answers = _classification_target(self.verbalizer, example)
        label = example.get("label")
        return {
            "id": str(example.get("idx", f"{split}-{idx}")),
            "task_name": self.name,
            "metric_name": self.metric_name,
            "prompt": prompt,
            "target": target,
            "answers": answers,
            "candidates": [self.verbalizer[0], self.verbalizer[1]],
            "label": int(label) if label is not None and int(label) >= 0 else None,
        }


class WICTask(TaskCollectionTask):
    verbalizer = {0: "No", 1: "Yes"}

    def __init__(self) -> None:
        super().__init__(name="wic", metric_name="accuracy", train_as_classification_default=True)

    def load_raw_splits(self, cache_dir: Optional[str] = None) -> Dict[str, Sequence[Dict]]:
        dataset = load_dataset("super_glue", "wic", cache_dir=cache_dir)
        splits = {"train": dataset["train"], "validation": dataset["validation"]}
        if "test" in dataset:
            splits["test"] = dataset["test"]
        return splits

    def convert_example(self, example: Dict, split: str, idx: int) -> Dict:
        prompt = (
            f"Does the word \"{example['word']}\" have the same meaning in these two sentences? Yes, No?\n"
            f"{example['sentence1']}\n"
            f"{example['sentence2']}\n"
        )
        target, answers = _classification_target(self.verbalizer, example)
        label = example.get("label")
        return {
            "id": str(example.get("idx", f"{split}-{idx}")),
            "task_name": self.name,
            "metric_name": self.metric_name,
            "prompt": prompt,
            "target": target,
            "answers": answers,
            "candidates": [self.verbalizer[0], self.verbalizer[1]],
            "label": int(label) if label is not None and int(label) >= 0 else None,
        }


class MultiRCTask(TaskCollectionTask):
    verbalizer = {0: "No", 1: "Yes"}

    def __init__(self) -> None:
        super().__init__(name="multirc", metric_name="accuracy", train_as_classification_default=True)

    def load_raw_splits(self, cache_dir: Optional[str] = None) -> Dict[str, Sequence[Dict]]:
        dataset = load_dataset("super_glue", "multirc", cache_dir=cache_dir)
        splits = {"train": dataset["train"], "validation": dataset["validation"]}
        if "test" in dataset:
            splits["test"] = dataset["test"]
        return splits

    def convert_example(self, example: Dict, split: str, idx: int) -> Dict:
        prompt = (
            f"{example['paragraph']}\n"
            f"Question: {example['question']}\n"
            f"I found this answer \"{example['answer']}\". Is that correct? Yes or No?\n"
        )
        target, answers = _classification_target(self.verbalizer, example)
        example_id = example.get("idx")
        label = example.get("label")
        return {
            "id": str(example_id) if example_id is not None else f"{split}-{idx}",
            "task_name": self.name,
            "metric_name": self.metric_name,
            "prompt": prompt,
            "target": target,
            "answers": answers,
            "candidates": [self.verbalizer[0], self.verbalizer[1]],
            "label": int(label) if label is not None and int(label) >= 0 else None,
        }


class CopaTask(TaskCollectionTask):
    def __init__(self) -> None:
        super().__init__(name="copa", metric_name="accuracy")

    def load_raw_splits(self, cache_dir: Optional[str] = None) -> Dict[str, Sequence[Dict]]:
        dataset = load_dataset("super_glue", "copa", cache_dir=cache_dir)
        splits = {"train": dataset["train"], "validation": dataset["validation"]}
        if "test" in dataset:
            splits["test"] = dataset["test"]
        return splits

    def convert_example(self, example: Dict, split: str, idx: int) -> Dict:
        premise = example["premise"].rstrip()
        if premise.endswith("."):
            premise = premise[:-1]
        conjunction = " so " if example["question"] == "effect" else " because "
        prompt = premise + conjunction
        answer = ""
        answers: List[str] = []
        label = example.get("label")
        candidates = []
        for choice_id in (1, 2):
            candidate = example[f"choice{choice_id}"]
            if candidate and candidate.split(" ")[0] != "I":
                candidate = candidate[0].lower() + candidate[1:]
            candidates.append(candidate)
        if label is not None and int(label) >= 0:
            answer = candidates[int(label)]
            answers = [answer]
        return {
            "id": str(example.get("idx", f"{split}-{idx}")),
            "task_name": self.name,
            "metric_name": self.metric_name,
            "prompt": prompt,
            "target": answer,
            "answers": answers,
            "candidates": candidates,
            "label": int(label) if label is not None and int(label) >= 0 else None,
        }

class ReCoRDTask(TaskCollectionTask):
    def __init__(self) -> None:
        super().__init__(name="record", metric_name="accuracy")

    def load_raw_splits(self, cache_dir: Optional[str] = None) -> Dict[str, Sequence[Dict]]:
        dataset = load_dataset("super_glue", "record", cache_dir=cache_dir)
        splits = {"train": dataset["train"], "validation": dataset["validation"]}
        if "test" in dataset:
            splits["test"] = dataset["test"]
        return splits

    def convert_example(self, example: Dict, split: str, idx: int) -> Dict:
        passage = example["passage"].replace("@highlight\n", "- ")
        raw_answers = example.get("answers") or []
        answers = list(dict.fromkeys(raw_answers))
        first_answer = answers[0] if answers else None
        query = example["query"]
        target = " " + query.replace("@placeholder", first_answer) if first_answer is not None else ""
        candidates = [" " + query.replace("@placeholder", entity) for entity in example.get("entities", [])]
        example_id = example.get("idx")
        return {
            "id": str(example_id) if example_id is not None else f"{split}-{idx}",
            "task_name": self.name,
            "metric_name": self.metric_name,
            "prompt": f"{passage}\n-",
            "target": target,
            "answers": [query.replace("@placeholder", answer) for answer in answers],
            "candidates": candidates,
            "label": None,
        }


class SQuADTask(TaskCollectionTask):
    def __init__(self) -> None:
        super().__init__(name="squad", metric_name="f1", generation=True)

    def load_raw_splits(self, cache_dir: Optional[str] = None) -> Dict[str, Sequence[Dict]]:
        dataset = load_dataset("squad", cache_dir=cache_dir)
        splits = {"train": dataset["train"], "validation": dataset["validation"]}
        if "test" in dataset:
            splits["test"] = dataset["test"]
        return splits

    def convert_example(self, example: Dict, split: str, idx: int) -> Dict:
        answer_dict = example.get("answers", {})
        answers = [answer for answer in answer_dict.get("text", []) if answer]
        prompt = (
            f"Title: {example['title']}\n"
            f"Context: {example['context']}\n"
            f"Question: {example['question'].strip()}\n"
            "Answer:"
        )
        return {
            "id": example["id"],
            "task_name": self.name,
            "metric_name": self.metric_name,
            "prompt": prompt,
            "target": f" {answers[0]}\n" if answers else "",
            "answers": answers,
        }


class DROPTask(TaskCollectionTask):
    def __init__(self) -> None:
        super().__init__(name="drop", metric_name="f1", generation=True)

    def load_raw_splits(self, cache_dir: Optional[str] = None) -> Dict[str, Sequence[Dict]]:
        dataset = load_dataset("drop", cache_dir=cache_dir)
        splits = {"train": dataset["train"], "validation": dataset["validation"]}
        if "test" in dataset:
            splits["test"] = dataset["test"]
        return splits

    def convert_example(self, example: Dict, split: str, idx: int) -> Dict:
        answer_spans = example.get("answers_spans", {})
        answers = [answer for answer in answer_spans.get("spans", []) if answer]
        prompt = (
            f"Passage: {example['passage']}\n"
            f"Question: {example['question'].strip()}\n"
            "Answer:"
        )
        example_id = example.get("query_id", f"{split}-{idx}")
        return {
            "id": str(example_id),
            "task_name": self.name,
            "metric_name": self.metric_name,
            "prompt": prompt,
            "target": f" {answers[0]}\n" if answers else "",
            "answers": answers,
        }


TASK_REGISTRY: Dict[str, Callable[[], TaskCollectionTask]] = {
    "sst2": SST2Task,
    "rte": RTETask,
    "cb": CBTask,
    "boolq": BoolQTask,
    "wsc": WSCTask,
    "wic": WICTask,
    "multirc": MultiRCTask,
    "copa": CopaTask,
    "record": ReCoRDTask,
    "squad": SQuADTask,
    "drop": DROPTask,
}


def get_task_collection_task(task_name: str) -> TaskCollectionTask:
    canonical = _canonical_task_name(task_name)
    return TASK_REGISTRY[canonical]()


def load_task_collection_records(
    task_name: str,
    cache_dir: Optional[str] = None,
    max_train_samples: Optional[int] = None,
    max_dev_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    data_seed: int = 42,
) -> Tuple[Dict[str, List[Dict]], TaskCollectionTask]:
    task = get_task_collection_task(task_name)
    split_records = task.load_records_by_split(
        cache_dir=cache_dir,
        max_train_samples=max_train_samples,
        max_dev_samples=max_dev_samples,
        max_eval_samples=max_eval_samples,
        max_test_samples=max_test_samples,
        data_seed=data_seed,
    )
    return split_records, task


def sample_task_collection_subset(records: Sequence[Dict], num_samples: Optional[int], seed: int) -> List[Dict]:
    return _sample_records(list(records), num_samples, seed)


def sample_task_collection_train_sets(
    train_records: Sequence[Dict],
    num_train: int,
    num_target_samples: int,
    num_train_sets: Optional[int] = None,
    train_set_seed: Optional[int] = None,
) -> Tuple[List[List[Dict]], List[int]]:
    if train_set_seed is not None:
        seeds = [int(train_set_seed)]
    elif num_train_sets is not None:
        seeds = list(range(int(num_train_sets)))
    else:
        seeds = np.random.RandomState(0).randint(0, 10000, size=num_target_samples).tolist()
    sampled_sets = [_sample_records(list(train_records), num_train, int(seed)) for seed in seeds]
    return sampled_sets, [int(seed) for seed in seeds]


def build_task_collection_icl_records(
    target_records: Sequence[Dict],
    demo_sets: Sequence[Sequence[Dict]],
    train_sep: str = "\n\n",
) -> List[Dict]:
    if not demo_sets:
        return [dict(record) for record in target_records]
    if len(demo_sets) not in {1, len(target_records)}:
        raise ValueError("demo_sets must have length 1 or the same length as target_records.")

    icl_records: List[Dict] = []
    one_set_per_target = len(demo_sets) == len(target_records)
    for idx, record in enumerate(target_records):
        demos = demo_sets[idx] if one_set_per_target else demo_sets[0]
        demo_text = train_sep.join(
            f"{demo['prompt']}{demo['target']}".strip() for demo in demos if demo.get("target", "") != ""
        ).strip()
        merged_prompt = f"{demo_text}{train_sep}{record['prompt']}".strip() if demo_text else record["prompt"]
        merged_record = dict(record)
        merged_record["base_prompt"] = record["prompt"]
        merged_record["prompt"] = merged_prompt
        merged_record["demo_ids"] = [demo["id"] for demo in demos]
        icl_records.append(merged_record)
    return icl_records

class TaskCollectionNestedDataset(TorchDataset):
    def __init__(self, records: Sequence[Any]) -> None:
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Any:
        return self.records[idx]


def _left_truncate_like_task_collection(tokenizer, input_ids: List[int], max_length: int) -> List[int]:
    if len(input_ids) <= max_length:
        return input_ids
    if getattr(tokenizer, "add_bos_token", False):
        return input_ids[0:1] + input_ids[1:][-(max_length - 1) :]
    return input_ids[-max_length:]


def _encode_classification_record(
    record: Dict,
    tokenizer,
    model_max_length: int,
) -> List[Dict]:
    candidates = record.get("candidates") or []
    label = _resolve_candidate_label(record)
    if label is None or int(label) < 0:
        raise ValueError(f"Record {record.get('id')} is missing a valid classification label.")
    if not candidates:
        raise ValueError(f"Record {record.get('id')} does not define classification candidates.")

    prompt_ids = tokenizer(record["prompt"], add_special_tokens=True)["input_ids"]
    prompt_length = len(prompt_ids)
    encoded_candidates: List[Dict] = []
    for candidate in candidates:
        full_ids = tokenizer(f"{record['prompt']}{candidate}", add_special_tokens=True)["input_ids"]
        option_len = max(len(full_ids) - prompt_length, 1)
        full_ids = _left_truncate_like_task_collection(tokenizer, full_ids, model_max_length)
        encoded_candidates.append(
            {
                "input_ids": full_ids,
                "labels": int(label),
                "option_len": option_len,
                "num_options": len(candidates),
            }
        )
    return encoded_candidates
def build_task_collection_candidate_encodings(
    record: Dict,
    tokenizer,
    model_max_length: int,
) -> Tuple[List[List[int]], List[int]]:
    candidates = record.get("candidates") or []
    if not candidates:
        raise ValueError(f"Record {record.get('id')} does not define classification candidates.")

    prompt_ids = tokenizer(record["prompt"], add_special_tokens=True)["input_ids"]
    prompt_length = len(prompt_ids)
    encoded_candidates: List[List[int]] = []
    option_lens: List[int] = []
    for candidate in candidates:
        full_ids = tokenizer(f"{record['prompt']}{candidate}", add_special_tokens=True)["input_ids"]
        option_lens.append(max(len(full_ids) - prompt_length, 1))
        encoded_candidates.append(_left_truncate_like_task_collection(tokenizer, full_ids, model_max_length))
    return encoded_candidates, option_lens

def score_task_collection_record_candidates(
    model,
    tokenizer,
    record: Dict,
    model_max_length: int,
) -> List[float]:
    encoded_candidates, option_lens = build_task_collection_candidate_encodings(
        record=record,
        tokenizer=tokenizer,
        model_max_length=model_max_length,
    )

    try:
        model_device = model.device
    except AttributeError:
        model_device = next(model.parameters()).device

    scores: List[float] = []
    model.eval()
    with torch.inference_mode():
        for encoded_candidate, option_len in zip(encoded_candidates, option_lens):
            input_ids = torch.tensor([encoded_candidate], device=model_device)
            logits = model(input_ids=input_ids).logits
            labels = input_ids[0, 1:]
            shifted_logits = logits[0, :-1]
            log_probs = F.log_softmax(shifted_logits, dim=-1)
            selected_log_probs = log_probs[torch.arange(labels.size(0), device=labels.device), labels]
            scores.append(selected_log_probs[-option_len:].mean().item())
    return scores

def _encode_supervised_record(
    record: Dict,
    tokenizer,
    max_source_length: int,
    max_target_length: int,
    model_max_length: int,
) -> Dict:
    target_ids = tokenizer(
        record["target"],
        add_special_tokens=False,
        truncation=True,
        max_length=max_target_length,
    )["input_ids"]
    if tokenizer.eos_token_id is not None and (not target_ids or target_ids[-1] != tokenizer.eos_token_id):
        target_ids = target_ids + [tokenizer.eos_token_id]

    source_budget = max(1, min(max_source_length, model_max_length - len(target_ids)))
    source_ids = tokenizer(
        record["prompt"],
        add_special_tokens=True,
        truncation=True,
        max_length=source_budget,
    )["input_ids"]

    input_ids = (source_ids + target_ids)[:model_max_length]
    labels = ([IGNORE_INDEX] * len(source_ids) + target_ids)[:model_max_length]
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def build_task_collection_supervised_dataset(
    records: Sequence[Dict],
    tokenizer,
    max_source_length: int,
    max_target_length: int,
    model_max_length: int,
) -> Dataset:
    encoded_records = [
        _encode_supervised_record(
            record=record,
            tokenizer=tokenizer,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            model_max_length=model_max_length,
        )
        for record in records
    ]
    return Dataset.from_list(encoded_records)


def build_task_collection_classification_dataset(
    records: Sequence[Dict],
    tokenizer,
    model_max_length: int,
) -> TorchDataset:
    encoded_records = [
        _encode_classification_record(
            record=record,
            tokenizer=tokenizer,
            model_max_length=model_max_length,
        )
        for record in records
    ]
    return TaskCollectionNestedDataset(encoded_records)


def build_task_collection_generation_dataset(
    records: Sequence[Dict],
    tokenizer,
    max_source_length: int,
    model_max_length: int,
    include_labels: bool = False,
) -> Tuple[Dataset, List[int]]:
    encoded_records: List[Dict] = []
    prompt_lengths: List[int] = []
    prompt_budget = min(max_source_length, model_max_length)
    for record in records:
        encoded_prompt = tokenizer(
            record["prompt"],
            add_special_tokens=True,
            truncation=True,
            max_length=prompt_budget,
        )["input_ids"]
        encoded_records.append(
            {
                "input_ids": encoded_prompt,
                "attention_mask": [1] * len(encoded_prompt),
                **({"labels": list(encoded_prompt)} if include_labels else {}),
            }
        )
        prompt_lengths.append(len(encoded_prompt))
    return Dataset.from_list(encoded_records), prompt_lengths

def forward_wrap_with_option_len(
    self,
    input_ids=None,
    labels=None,
    option_len=None,
    num_options=None,
    return_dict=None,
    **kwargs,
):
    if labels is None or option_len is None:
        return self.original_forward(input_ids=input_ids, labels=labels, return_dict=return_dict, **kwargs)

    outputs = self.original_forward(input_ids=input_ids, return_dict=return_dict, **kwargs)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
    logits_for_return = logits

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = torch.clone(input_ids)[..., 1:].contiguous()
    shift_labels[shift_labels == self.config.pad_token_id] = -100

    for idx, current_option_len in enumerate(option_len):
        shift_labels[idx, :-int(current_option_len)] = -100

    loss_fct = CrossEntropyLoss(ignore_index=-100)
    acc = None
    if num_options is not None:
        log_probs = F.log_softmax(shift_logits, dim=-1)
        mask = shift_labels != -100
        shift_labels[~mask] = 0
        selected_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
        selected_log_probs = (selected_log_probs * mask).sum(-1) / mask.sum(-1)
        logits_for_return = selected_log_probs.contiguous()

        if any(current_num_options != num_options[0] for current_num_options in num_options):
            loss = 0
            start_id = 0
            count = 0
            while start_id < len(num_options):
                end_id = start_id + num_options[start_id]
                option_logits = selected_log_probs[start_id:end_id].unsqueeze(0)
                option_labels = labels[start_id:end_id][0].unsqueeze(0)
                loss = loss_fct(option_logits, option_labels) + loss
                count += 1
                start_id = end_id
            loss = loss / max(count, 1)
        else:
            option_count = int(num_options[0])
            selected_log_probs = selected_log_probs.view(-1, option_count)
            labels = labels.view(-1, option_count)[:, 0]
            loss = loss_fct(selected_log_probs, labels)
            acc = torch.tensor(
                np.mean(
                    [
                        np.argmax(selected_log_prob) == label
                        for selected_log_prob, label in zip(
                            selected_log_probs.detach().cpu(),
                            labels.detach().cpu(),
                        )
                    ]
                )
            )
    else:
        loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

    if not return_dict:
        output = (logits_for_return,) + outputs[1:]
        if acc is not None:
            output = output + (acc,)
        return (loss,) + output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits_for_return,
        past_key_values=getattr(outputs, "past_key_values", None),
        hidden_states=getattr(outputs, "hidden_states", None),
        attentions=getattr(outputs, "attentions", None),
    )

@dataclass
class TaskCollectionClassificationCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Classification training stores one sample as a list of candidate dicts,
        # while generation-based eval still yields plain dict features.
        if features and isinstance(features[0], Mapping):
            label_name = "label" if "label" in features[0] else "labels" if "labels" in features[0] else None
            labels = [feature[label_name] for feature in features] if label_name is not None else None
            features_to_pad = [{k: v for k, v in feature.items() if k != label_name} for feature in features]
        else:
            flattened_features = [nested_feature for feature in features for nested_feature in feature]
            label_name = None
            labels = None
            features_to_pad = flattened_features
        batch = self.tokenizer.pad(
            features_to_pad,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        if labels is not None:
            sequence_length = batch["input_ids"].shape[1]
            padding_side = self.tokenizer.padding_side

            def to_list(value):
                if isinstance(value, torch.Tensor):
                    return value.tolist()
                return list(value)

            if padding_side == "right":
                batch[label_name] = [
                    to_list(label) + [-100] * (sequence_length - len(label)) for label in labels
                ]
            else:
                batch[label_name] = [
                    [-100] * (sequence_length - len(label)) + to_list(label) for label in labels
                ]
            batch[label_name] = torch.tensor(batch[label_name], dtype=torch.int64)
        if "label" in batch:
            batch["labels"] = batch.pop("label")
        if "label_ids" in batch:
            batch["labels"] = batch.pop("label_ids")
        return batch


def decode_generation_predictions(predictions, tokenizer, prompt_lengths: Sequence[int]) -> List[str]:
    decoded_predictions: List[str] = []
    for generated_ids, prompt_length in zip(predictions, prompt_lengths):
        # Trainer gather/padding can append -100 sentinels to generated sequences.
        # Fast tokenizers choke on negative ids during decode, so strip them first.
        continuation_ids = []
        for token_id in generated_ids[prompt_length:]:
            try:
                token_id = int(token_id)
            except (TypeError, ValueError, OverflowError):
                continue
            if token_id < 0:
                continue
            continuation_ids.append(token_id)
        decoded_predictions.append(tokenizer.decode(continuation_ids, skip_special_tokens=True).strip())
    return decoded_predictions


def evaluate_task_collection_predictions(task_name: str, predictions: Sequence[str], records: Sequence[Dict]) -> Dict[str, float]:
    task = get_task_collection_task(task_name)
    answers = [record["answers"] for record in records]
    if not records or not any(answers):
        return {}
    if task.metric_name == "accuracy":
        score = 0.0
        for prediction, reference_answers in zip(predictions, answers):
            normalized_prediction = _normalize_answer(prediction.splitlines()[0].strip() if prediction else "")
            normalized_answers = {_normalize_answer(answer) for answer in reference_answers}
            score += float(normalized_prediction in normalized_answers)
        return {"accuracy": score / max(len(predictions), 1)}
    if task.metric_name == "f1":
        f1_total = 0.0
        em_total = 0.0
        for prediction, reference_answers in zip(predictions, answers):
            cleaned_prediction = prediction.strip()
            normalized_prediction = _normalize_answer(cleaned_prediction)
            normalized_answers = {_normalize_answer(answer) for answer in reference_answers}
            em_total += float(normalized_prediction in normalized_answers)
            f1_total += _qa_f1(cleaned_prediction, reference_answers)
        denom = max(len(predictions), 1)
        return {"f1": f1_total / denom, "exact_match": em_total / denom}
    raise ValueError(f"Unsupported metric '{task.metric_name}' for task '{task.name}'.")
