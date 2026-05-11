from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    bias: str = "none"


@dataclass
class TrainingHyperparams:
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2.0e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 0.3
    optim: str = "adamw_8bit"
    logging_steps: int = 10
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    group_by_length: bool = False


@dataclass
class TrainingConfig:
    mode: str
    base_model: str
    load_in_4bit: bool
    max_seq_length: int
    train_file: str
    val_split: float
    seed: int
    lora: LoRAConfig
    training: TrainingHyperparams
    output_dir: str
    adapter_save_dir: str
    mlflow_enabled: bool = False
    mlflow_experiment: str = ""
    mlflow_tracking_uri: str = ""

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

        lora_data = data.get("lora", {})
        training_data = data.get("training", {})
        mlflow_data = data.get("mlflow", {})

        return cls(
            mode=data["mode"],
            base_model=data["base_model"],
            load_in_4bit=data.get("load_in_4bit", True),
            max_seq_length=data.get("max_seq_length", 4096),
            train_file=data["train_file"],
            val_split=data.get("val_split", 0.1),
            seed=data.get("seed", 42),
            lora=LoRAConfig(**lora_data),
            training=TrainingHyperparams(**training_data),
            output_dir=data["output_dir"],
            adapter_save_dir=data["adapter_save_dir"],
            mlflow_enabled=mlflow_data.get("enabled", False),
            mlflow_experiment=mlflow_data.get("experiment_name", ""),
            mlflow_tracking_uri=mlflow_data.get("tracking_uri", ""),
        )


def load_jsonl(path: str) -> list[dict[str, Any]]:
    examples = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def split_train_val(
    examples: list[dict],
    val_split: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    import random as rnd

    r = rnd.Random(seed)
    r.shuffle(examples)
    n_val = max(1, int(len(examples) * val_split))
    return examples[n_val:], examples[:n_val]


def train(config_path: str) -> None:
    """Run LoRA fine-tuning."""
    config = TrainingConfig.from_yaml(config_path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    logger.info("=" * 70)
    logger.info("LoRA Fine-tuning: %s", config.mode)
    logger.info("=" * 70)
    logger.info("Base model:      %s (4bit=%s)", config.base_model, config.load_in_4bit)
    logger.info("Train data:      %s", config.train_file)
    logger.info("Output:          %s", config.output_dir)
    logger.info("Adapter dir:     %s", config.adapter_save_dir)
    logger.info(
        "LoRA r=%d, alpha=%d, dropout=%.2f", config.lora.r, config.lora.alpha, config.lora.dropout
    )
    logger.info("Epochs:          %d", config.training.num_train_epochs)
    logger.info(
        "Effective batch: %d (per_device=%d × grad_accum=%d)",
        config.training.per_device_train_batch_size * config.training.gradient_accumulation_steps,
        config.training.per_device_train_batch_size,
        config.training.gradient_accumulation_steps,
    )
    logger.info(
        "LR:              %.2e (%s, warmup_ratio=%.2f)",
        config.training.learning_rate,
        config.training.lr_scheduler_type,
        config.training.warmup_ratio,
    )
    logger.info("=" * 70)

    try:
        from unsloth import FastLanguageModel, is_bfloat16_supported
    except ImportError as e:
        raise RuntimeError(
            "Unsloth not installed. Install via:\n  pip install -r training/requirements_server.txt"
        ) from e

    from datasets import Dataset
    from transformers import TrainingArguments
    from trl import SFTConfig, SFTTrainer

    logger.info("Loading training data from %s", config.train_file)
    examples = load_jsonl(config.train_file)
    logger.info("Total examples: %d", len(examples))

    train_examples, val_examples = split_train_val(
        examples,
        config.val_split,
        config.seed,
    )
    logger.info("Train/Val split: %d / %d", len(train_examples), len(val_examples))

    logger.info("Loading base model %s...", config.base_model)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.base_model,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
        dtype=None,  # auto-detect
    )

    logger.info("Applying LoRA adapter (r=%d)", config.lora.r)
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora.r,
        target_modules=config.lora.target_modules,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        bias=config.lora.bias,
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
        max_seq_length=config.max_seq_length,
    )

    def format_messages(example: dict) -> dict:
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    train_ds = Dataset.from_list(train_examples).map(format_messages)
    val_ds = Dataset.from_list(val_examples).map(format_messages)
    logger.info("Datasets prepared: train=%d, val=%d", len(train_ds), len(val_ds))

    if config.mlflow_enabled:
        try:
            import mlflow

            if config.mlflow_tracking_uri:
                mlflow.set_tracking_uri(config.mlflow_tracking_uri)
            mlflow.set_experiment(config.mlflow_experiment)
            os.environ["MLFLOW_LOG_MODEL"] = "false"
            logger.info(
                "MLflow tracking: %s / %s",
                config.mlflow_tracking_uri or "(default)",
                config.mlflow_experiment,
            )
        except Exception as e:
            logger.warning("MLflow init failed: %s — continuing without tracking", e)

    use_bf16 = config.training.bf16 and is_bfloat16_supported()
    use_fp16 = not use_bf16

    if not use_bf16:
        logger.warning("BF16 not supported, falling back to FP16")

    sft_config = SFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        lr_scheduler_type=config.training.lr_scheduler_type,
        warmup_ratio=config.training.warmup_ratio,
        weight_decay=config.training.weight_decay,
        max_grad_norm=config.training.max_grad_norm,
        optim=config.training.optim,
        logging_steps=config.training.logging_steps,
        save_strategy=config.training.save_strategy,
        eval_strategy=config.training.eval_strategy,
        save_total_limit=config.training.save_total_limit,
        load_best_model_at_end=config.training.load_best_model_at_end,
        metric_for_best_model=config.training.metric_for_best_model,
        greater_is_better=config.training.greater_is_better,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=config.training.gradient_checkpointing,
        group_by_length=config.training.group_by_length,
        seed=config.seed,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        packing=False,
        report_to=["mlflow"] if config.mlflow_enabled else [],
        run_name=f"lora_{config.mode}",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=sft_config,
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete")

    adapter_path = Path(config.adapter_save_dir)
    adapter_path.mkdir(parents=True, exist_ok=True)

    logger.info("Saving LoRA adapter to %s", adapter_path)
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))

    metadata = {
        "mode": config.mode,
        "base_model": config.base_model,
        "lora_r": config.lora.r,
        "lora_alpha": config.lora.alpha,
        "training_examples": len(train_examples),
        "validation_examples": len(val_examples),
        "epochs": config.training.num_train_epochs,
        "effective_batch_size": (
            config.training.per_device_train_batch_size
            * config.training.gradient_accumulation_steps
        ),
        "learning_rate": config.training.learning_rate,
    }
    (adapter_path / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    logger.info("Adapter saved: %s", adapter_path)


def main() -> None:
    import fire

    fire.Fire(train)


if __name__ == "__main__":
    main()
