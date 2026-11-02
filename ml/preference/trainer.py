"""DPO stub wrapping SFT loss — preference alignment (WS-3)."""

from __future__ import annotations

import torch

from ml.models import KothaGPT
from ml.preference.dataset import PreferenceRecord


def _encode_pair(tok, prompt: str, completion: str, max_len: int):
    p_ids = tok.encode(prompt)
    c_ids = tok.encode(completion + "\n<eos>")
    ids = (p_ids + c_ids)[:max_len]
    prompt_len = min(len(p_ids), len(ids))
    labels = [-100]*prompt_len + ids[prompt_len:]
    return ids, labels


def _logprob_for_batch(
    model, tokenizer, texts: list[str], completions: list[str], max_len: int, device: str, *, detach: bool = False
) -> torch.Tensor:
    """Per-sequence logprob (sum of target-token logprobs, not batch scalar)."""
    pad_id = getattr(tokenizer, "vocab", {}).get("<pad>", 3)

    def pad(seqs, pad_val):
        w = max(len(s) for s in seqs)
        return [s + [pad_val] * (w - len(s)) for s in seqs]

    ids_list, labs_list = [], []
    for prompt, comp in zip(texts, completions):
        ids, labs = _encode_pair(tokenizer, prompt, comp, max_len)
        ids_list.append(ids)
        labs_list.append(labs)
    # Try batch per-token if available
    inp = torch.tensor(pad(ids_list, pad_id), dtype=torch.long, device=device)
    lab = torch.tensor(pad(labs_list, -100), dtype=torch.long, device=device)
    out = model(input_ids=inp, labels=lab)
    if isinstance(out, dict) and "logprobs" in out:
        lp = out["logprobs"] * (lab != -100).float()
        res = lp.sum(dim=1)
        return res.detach() if detach else res
    # Fallback: per-sequence CE
    per_seq = []
    for i in range(len(ids_list)):
        single_inp = torch.tensor([ids_list[i]], dtype=torch.long, device=device)
        single_lab = torch.tensor([labs_list[i]], dtype=torch.long, device=device)
        single_out = model(input_ids=single_inp, labels=single_lab)
        val = -single_out["loss"] * (single_lab != -100).sum().float()
        per_seq.append(val.detach() if detach else val)
    return torch.stack(per_seq).to(device)


def run_dpo(
    model: KothaGPT,
    records: list[PreferenceRecord],
    tokenizer,
    *,
    device: str = "cpu",
    max_steps: int = 5,
    batch_size: int = 2,
    learning_rate: float = 1e-5,
    max_length: int = 256,
    beta: float = 0.1,
):
    """DPO with per-pair logprobs and frozen reference model."""
    import copy

    model.to(device).train()
    # Frozen reference model (copy of initial policy)
    ref_model = copy.deepcopy(model)
    ref_model.to(device).eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    steps, total = 0, 0.0
    idx = 0
    while steps < max_steps:
        batch = records[idx : idx + batch_size]
        if not batch:
            idx = 0
            batch = records[idx : idx + batch_size]
        idx += batch_size
        prompts = [r.prompt for r in batch]
        chosens = [r.chosen for r in batch]
        rejecteds = [r.rejected for r in batch]
        logp_c = _logprob_for_batch(model, tokenizer, prompts, chosens, max_length, device, detach=False)
        logp_r = _logprob_for_batch(model, tokenizer, prompts, rejecteds, max_length, device, detach=False)
        with torch.no_grad():
            ref_c = _logprob_for_batch(ref_model, tokenizer, prompts, chosens, max_length, device, detach=True)
            ref_r = _logprob_for_batch(ref_model, tokenizer, prompts, rejecteds, max_length, device, detach=True)
        # Per-pair DPO objective: logsigmoid(beta * ((logp_c - logp_r) - (ref_c - ref_r)))
        per_pair = torch.nn.functional.logsigmoid(beta * ((logp_c - logp_r) - (ref_c - ref_r)))
        dpo_loss = -per_pair.mean()
        opt.zero_grad(set_to_none=True)
        dpo_loss.backward()
        opt.step()
        total += float(dpo_loss.detach())
        steps += 1
    return {"steps": steps, "dpo_loss": total / max(steps, 1)}
