import gc
import shutil
from pathlib import Path

import torch
from transformers import EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer

from .lwf_online import LwFSFTTrainerOnline
from ...utils.configs import TrainConfig
from ...utils.utility import MemoryCleanupCallback


class DataCollatorWithIsOld:
    def __init__(self, base_collator, debug: bool = False):
        self.base_collator = base_collator
        self.debug = debug
        self._printed = False

    def __call__(self, features):
        batch = self.base_collator(features)

        if self.debug and (not self._printed) and len(features) > 0:
            print(f"[LwF][collator] feature keys: {list(features[0].keys())}")
            self._printed = True

        if len(features) > 0 and "is_old" in features[0]:
            batch["is_old"] = torch.tensor(
                [bool(f["is_old"]) for f in features],
                dtype=torch.bool,
            )
            if self.debug:
                print(
                    f"[LwF][collator] added is_old to batch: "
                    f"{batch['is_old'][:8].tolist()}"
                )

        return batch

def _has_teacher_adapter(llm, teacher_adapter_name: str = "teacher") -> bool:
    peft_cfg = getattr(llm, "peft_config", None)
    if peft_cfg is None:
        return False
    return teacher_adapter_name in peft_cfg


def _normalize_checkpoint_adapter_layout(adapter_dir: Path) -> None:
    """
    For LwF dual-adapter checkpoints, keep only student adapter at checkpoint root.
    If files are under checkpoint-*/student, move them up to checkpoint-*.
    """
    checkpoints = sorted(
        [p for p in adapter_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[-1]),
    )
    for checkpoint_dir in checkpoints:
        student_dir = checkpoint_dir / "student"
        if student_dir.is_dir():
            for item in student_dir.iterdir():
                target = checkpoint_dir / item.name
                if target.exists():
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                shutil.move(str(item), str(target))
            shutil.rmtree(student_dir, ignore_errors=True)
            print(f"[LwF][save] moved student adapter files to {checkpoint_dir}")

        teacher_dir = checkpoint_dir / "teacher"
        if teacher_dir.is_dir():
            shutil.rmtree(teacher_dir, ignore_errors=True)
            print(f"[LwF][save] removed teacher adapter folder from {checkpoint_dir}")


def train_lwf(
    trainConfig: TrainConfig,
    model,
    dataset_train,
    dataset_val,
    experience_name,
    alpha: float = 1.0,
    temperature: float = 2.0,
    kd_on_new: bool = False,
    teacher_adapter_name: str = "teacher",
    student_adapter_name: str = "student",
):
    wandb_logger = None
    print("WandB logging is disabled for LwF training.")

    adapter_dir = (
        Path(trainConfig.output_path)
        / f"{experience_name}-{trainConfig.variant_name}{f'-{trainConfig.extra_info}' if trainConfig.extra_info != '' else ''}"
    )
    if trainConfig.retriever is not None:
        adapter_dir = adapter_dir.with_name(adapter_dir.name + "-" + trainConfig.retriever)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[LwF][train] exp={experience_name} output_dir={adapter_dir} "
        f"epochs={trainConfig.epochs} batch_size={trainConfig.batch_size} grad_accum={trainConfig.grad_accum}"
    )

    tok = model.tokenizer
    llm = model.model

    tok.padding_side = "right"
    tok.add_eos_token = False

    cfg_kwargs = dict(
        **(
            {"seed": trainConfig.seed, "data_seed": trainConfig.seed}
            if trainConfig.seed is not None
            else {}
        ),
        gradient_checkpointing=trainConfig.activation_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        gradient_accumulation_steps=trainConfig.grad_accum,
        per_device_train_batch_size=trainConfig.batch_size,
        max_length=trainConfig.max_length,
        group_by_length=trainConfig.group_by_length,
        packing=trainConfig.packing,
        weight_decay=trainConfig.weight_decay,
        learning_rate=trainConfig.lr,
        lr_scheduler_type=trainConfig.lr_scheduler_type,
        optim=trainConfig.optim,
        warmup_steps=trainConfig.warmup_steps,
        max_grad_norm=trainConfig.max_grad_norm,
        label_smoothing_factor=trainConfig.label_smoothing,
        num_train_epochs=trainConfig.epochs,
        logging_steps=trainConfig.logging_steps,
        logging_dir=str(adapter_dir / "logs"),
        output_dir=str(adapter_dir),
        report_to="none",
        disable_tqdm=False,
        completion_only_loss=trainConfig.completion_only_loss,
        save_strategy=trainConfig.save_strategy,
        save_total_limit=trainConfig.save_total_limit,
        eval_strategy="no" if trainConfig.no_validation else "epoch",
        load_best_model_at_end=trainConfig.hyperparameters_search,
        metric_for_best_model=trainConfig.metric_for_best_model
        if not trainConfig.no_validation
        else None,
        greater_is_better=trainConfig.greater_is_better,
        dataloader_pin_memory=False,
        eval_accumulation_steps=4,
        save_safetensors=True,
        remove_unused_columns=False,
    )

    if trainConfig.low_memory_mode:
        cfg_kwargs["gradient_checkpointing_kwargs"] = {
            "use_reentrant": False,
            "offload_to_cpu": True,
        }
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            cfg_kwargs["bf16"] = True
        else:
            cfg_kwargs["fp16"] = True

    sft_cfg = SFTConfig(**cfg_kwargs)

    callbacks = [MemoryCleanupCallback()]

    if not trainConfig.no_validation:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=trainConfig.early_stopping_patience,
                early_stopping_threshold=trainConfig.early_stopping_threshold,
            )
        )

    trainer_kwargs = {
        "model": llm,
        "processing_class": tok,
        "args": sft_cfg,
        "train_dataset": dataset_train,
        "callbacks": callbacks,
    }

    if not trainConfig.no_validation and dataset_val is not None:
        trainer_kwargs["eval_dataset"] = dataset_val

    use_lwf = _has_teacher_adapter(llm, teacher_adapter_name=teacher_adapter_name)
    print(
        f"[LwF][train] use_lwf={use_lwf} teacher_adapter={teacher_adapter_name} "
        f"student_adapter={student_adapter_name}"
    )

    if use_lwf:
        print(
            f"Teacher adapter '{teacher_adapter_name}' found. "
            "Using LwFSFTTrainer with single-backbone dual-adapter setup."
        )
        trainer = LwFSFTTrainerOnline(
            alpha=alpha,
            temperature=temperature,
            kd_on_new=kd_on_new,
            teacher_adapter_name=teacher_adapter_name,
            student_adapter_name=student_adapter_name,
            **trainer_kwargs,
        )
    else:
        print(
            f"Teacher adapter '{teacher_adapter_name}' not found. "
            "Using standard SFTTrainer."
        )
        trainer = SFTTrainer(**trainer_kwargs)
    trainer.data_collator = DataCollatorWithIsOld(
        trainer.data_collator,
        debug=False,
    )
    torch.cuda.empty_cache()
    gc.collect()

    if trainConfig.resume_from:
        base_path = "./core/experiments/"
        resume_path = base_path + f"{trainConfig.resume_from}/"
        print(f"[LwF][train] resume_from_checkpoint={resume_path}")
        trainer.train(resume_from_checkpoint=resume_path)
    else:
        print("[LwF][train] starting fresh training")
        trainer.train()

    _normalize_checkpoint_adapter_layout(adapter_dir)

    print(f"[LwF][train] training_completed exp={experience_name}")

    try:
        torch.save(trainer.args, str(adapter_dir / "training_args.bin"))
    except Exception:
        pass

    if trainConfig.hyperparameters_search:
        if trainConfig.no_validation:
            raise ValueError(
                "Cannot perform hyperparameter search with no_validation=True. Evaluation is required for hyperparameter search."
            )
        eval_results = trainer.evaluate()
        return eval_results["eval_loss"]

    return None