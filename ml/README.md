# ML

## Planned components
- `models/` model architecture/configuration
- `tokenizer/` tokenizer training and processing
- `pretrain/` causal language-model pretraining
- `sft/` supervised fine-tuning
- `preference/` DPO/reward-model pipelines
- `inference/` local inference adapters
- `configs/` training configurations

Never commit model weights or private datasets.

## GPU training verification

GPU training uses the dependencies in `ml/requirements.txt`, kept separate from the API environment. On a clean machine, run:

```bash
make gpu-env
make gpu-verify
```

The verifier reports the PyTorch/CUDA versions, visible NVIDIA devices, bf16 support, and distributed backend availability, then runs a one-step trainer smoke test. It exits with an actionable error when CUDA or NCCL is unavailable; this is expected on CPU-only development and CI machines. Use `make train-smoke` for the CPU fallback.

Prerequisites for GPU verification are a supported NVIDIA driver, a CUDA-capable PyTorch installation, and (for distributed readiness) NCCL. The optional Compose service is disabled by default and can be started with `docker compose --profile gpu run --rm trainer`; it requires the NVIDIA Container Toolkit. Training data is mounted from `data/tokenized` and outputs from `ml/pretrain/artifacts`.
