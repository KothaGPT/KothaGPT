from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 50000
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 0
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True
    norm_type: str = "rmsnorm"
    architecture: str = "decoder_transformer"

    def to_dict(self) -> dict:
        return asdict(self)

    def effective_intermediate_size(self) -> int:
        if self.intermediate_size > 0:
            return self.intermediate_size
        return int(8 / 3 * self.hidden_size)

    def validate(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size {self.hidden_size} must be divisible by num_heads {self.num_heads}"
            )
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings must be positive")
        if self.norm_type not in ("rmsnorm", "layernorm"):
            raise ValueError(f"norm_type {self.norm_type} not supported")
        if self.architecture != "decoder_transformer":
            raise ValueError(f"architecture {self.architecture} not supported")


@dataclass
class TrainingConfig:
    batch_size: int = 2
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3.0e-4
    max_steps: int = 1000
    warmup_steps: int = 100
    min_lr: float = 3.0e-5
    lr_schedule: str = "cosine"
    mixed_precision: str = "bf16"
    seed: int = 0
    eval_interval: int = 50
    save_interval: int = 100
    gradient_checkpointing: bool = False
    trend_guard_patience: int = 0
    trend_guard_action: str = "abort"
    eval_batches: int = 20
    keep_last: int = 5
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    shuffle: bool = True

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.mixed_precision not in ("none", "bf16", "fp16"):
            raise ValueError(f"mixed_precision {self.mixed_precision} not supported")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DataConfig:
    train: str = "data/processed/train"
    validation: str = "data/processed/validation"
    tokenizer_path: str = "ml/tokenizer/artifacts/best/tokenizer.json"

    def validate(self) -> None:
        if not self.train:
            raise ValueError("train data path is required")
        if not self.tokenizer_path:
            raise ValueError("tokenizer_path is required")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BaseModelConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def validate(self) -> None:
        self.model.validate()
        self.training.validate()
        self.data.validate()

    def to_dict(self) -> dict:
        return asdict(self)


def config_digest(cfg: BaseModelConfig) -> str:
    d = asdict(cfg)
    json_str = json.dumps(d, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()


def model_digest(cfg: BaseModelConfig | ModelConfig | dict) -> str:
    d = asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else dict(cfg)
    json_str = json.dumps(d, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()


def load_config(path: str | Path) -> BaseModelConfig:
    from ml.tokenizer import load_tokenizer
    from ml.tokenizer.base import BaseTokenizer

    path = Path(path)
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    model_data = raw.get("model", {})
    training_data = raw.get("training", {})
    data_data = raw.get("data", {})

    if "vocab_size" not in model_data:
        tokenizer_path = data_data.get("tokenizer_path", "ml/tokenizer/artifacts/best/tokenizer.json")
        if not Path(tokenizer_path).is_file():
            raise FileNotFoundError(f"tokenizer not found: {tokenizer_path}")
        try:
            tokenizer = load_tokenizer(tokenizer_path)
            model_data["vocab_size"] = len(tokenizer.vocab)
        except Exception:
            try:
                tok_data = json.loads(Path(tokenizer_path).read_text(encoding="utf-8"))
                if "vocab" in tok_data and isinstance(tok_data["vocab"], dict):
                    model_data["vocab_size"] = len(tok_data["vocab"])
            except Exception:
                pass

    cfg = BaseModelConfig(
        model=ModelConfig(**model_data),
        training=TrainingConfig(**training_data),
        data=DataConfig(**data_data),
    )
    cfg.validate()
    return cfg
