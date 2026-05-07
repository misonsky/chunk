import os
from pathlib import Path


_DISABLED_VALUES = {"1", "true", "yes", "on", "disabled"}


def _clear_disabled_env_var(name):
    value = os.environ.get(name)
    if value is None:
        return False
    if str(value).strip().lower() not in _DISABLED_VALUES:
        return False
    os.environ.pop(name, None)
    return True


def _normalize_report_to(report_to):
    if report_to is None:
        return ["wandb"]
    if isinstance(report_to, str):
        normalized = report_to.strip().lower()
        if normalized == "none":
            return []
        if normalized == "all":
            return "all"
        return [report_to]
    if not report_to:
        return ["wandb"]
    return list(report_to)


def configure_wandb(training_args, logger=None, default_project="ChunkFT"):
    cleared = []
    for env_name in ("WANDB_MODE", "WANDB_DISABLED"):
        if _clear_disabled_env_var(env_name):
            cleared.append(env_name)

    os.environ.setdefault("WANDB_PROJECT", default_project)

    report_to = _normalize_report_to(getattr(training_args, "report_to", None))
    if report_to not in ("all", []):
        normalized_targets = {str(target).lower() for target in report_to}
        if "wandb" not in normalized_targets:
            report_to.append("wandb")
    training_args.report_to = report_to

    run_name = getattr(training_args, "run_name", None)
    if not run_name:
        output_dir = getattr(training_args, "output_dir", None)
        training_args.run_name = Path(output_dir).name if output_dir else "chunkft"

    if logger is not None:
        if training_args.report_to == []:
            logger.info("W&B logging remains disabled because `report_to` is set to none.")
        else:
            cleared_text = f", cleared {', '.join(cleared)}" if cleared else ""
            logger.info(
                "W&B logging enabled: report_to=%s, run_name=%s, project=%s%s",
                training_args.report_to,
                training_args.run_name,
                os.environ.get("WANDB_PROJECT"),
                cleared_text,
            )
