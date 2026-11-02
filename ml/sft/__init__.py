"""SFT package (WS-2)."""

from .templates import (
    apply_chat_template,
    check_tokenizer_coverage,
    get_template,
    parse_chat_template,
)

try:
    from .dataset import SFTDataset, SFTMix, load_mix
    from .trainer import run_sft_trainer
except ImportError as _e:  # torch not installed in API env
    SFTDataset = SFTMix = load_mix = run_sft_trainer = None  # type: ignore

__all__ = [
    "SFTDataset",
    "SFTMix",
    "apply_chat_template",
    "check_tokenizer_coverage",
    "get_template",
    "load_mix",
    "parse_chat_template",
    "run_sft_trainer",
]
