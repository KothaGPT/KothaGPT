from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

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
    rope_scaling: str | None = None
    rope_scaling_factor: float = 1.0
    tie_word_embeddings: bool = True
    norm_type: str = "rmsnorm"
    architecture: str = "decoder_transformer"
    mm_hidden: int = 0
    vision_encoder: bool = False
    audio_encoder: bool = False
    document_encoder: bool = False
    # WS-3 reasoning head: optional auxiliary supervision during training.
    reasoning_head: bool = False
    # WS-3 coding token extensions: additive coding special tokens.
    coding_extensions: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    def effective_intermediate_size(self) -> int:
        if self.intermediate_size > 0:
            return self.intermediate_size
        return int(8 / 3 * self.hidden_size)

    def validate(self) -> None:
        # Coerce numeric strings (e.g. from YAML round-trips where ``1e-5`` is
        # parsed as the string ``"1e-5"``) so downstream code never sees a
        # ``str`` where a ``float`` is expected.
        for field_name in ("rms_norm_eps", "rope_theta", "rope_scaling_factor"):
            value = getattr(self, field_name)
            if isinstance(value, str):
                try:
                    setattr(self, field_name, float(value))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{field_name} must be a number, got {value!r}") from exc
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
        if self.mm_hidden < 0:
            raise ValueError("mm_hidden must be non-negative")
        if self.max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings must be positive")
        if self.norm_type not in ("rmsnorm", "layernorm"):
            raise ValueError(f"norm_type {self.norm_type} not supported")
        if self.architecture != "decoder_transformer":
            raise ValueError(f"architecture {self.architecture} not supported")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
        if self.rope_scaling is not None and self.rope_scaling not in ("linear", "ntk"):
            raise ValueError(f"rope_scaling {self.rope_scaling} not supported")
        if self.rope_scaling is not None and self.rope_scaling_factor <= 1.0:
            raise ValueError("rope_scaling_factor must be > 1.0 when rope_scaling is set")


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
        if self.min_lr is not None:
            # Coerce numeric strings (e.g. from YAML round-trips) so validation
            # never crashes on a type mismatch between a str and an int.
            if isinstance(self.min_lr, str):
                try:
                    self.min_lr = float(self.min_lr)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"min_lr must be a number, got {self.min_lr!r}") from exc
            if isinstance(self.min_lr, bool) or not isinstance(self.min_lr, (int, float)):
                raise ValueError(f"min_lr must be a number, got {type(self.min_lr).__name__}")
            if self.min_lr < 0:
                raise ValueError("min_lr must be non-negative")
            if self.min_lr >= self.learning_rate:
                raise ValueError("min_lr must be below learning_rate")
        if self.lr_schedule not in ("cosine", "linear"):
            raise ValueError(f"lr_schedule {self.lr_schedule} not supported")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.warmup_steps > self.max_steps:
            raise ValueError("warmup_steps must not exceed max_steps")

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
        except Exception:  # noqa: BLE001
            try:
                tok_data = json.loads(Path(tokenizer_path).read_text(encoding="utf-8"))
                if "vocab" in tok_data and isinstance(tok_data["vocab"], dict):
                    model_data["vocab_size"] = len(tok_data["vocab"])
            except Exception:  # noqa: BLE001, S110
                pass

    # Filter unknown keys so config files can carry extra metadata (e.g. the
    # ``tier`` block in ml/configs/*.yaml, ``task_type`` / ``base_model`` in
    # SFT configs) without breaking the dataclass contract.
    model_fields = {f for f in ModelConfig.__dataclass_fields__}
    training_fields = {f for f in TrainingConfig.__dataclass_fields__}
    data_fields = {f for f in DataConfig.__dataclass_fields__}
    cfg = BaseModelConfig(
        model=ModelConfig(**{k: v for k, v in model_data.items() if k in model_fields}),
        training=TrainingConfig(**{k: v for k, v in training_data.items() if k in training_fields}),
        data=DataConfig(**{k: v for k, v in data_data.items() if k in data_fields}),
    )
    cfg.validate()
    return cfg
