from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from stage2.model import PasswordEncoder

from . import config
from .data import ContextVectorizer, Stage3Tokenizer, _load_json_array
from .model import Stage3Model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3 sampling")
    parser.add_argument("--tokenizer", required=True, help="Path to stage2 tokenizer.json")
    parser.add_argument("--encoder", required=True, help="Path to stage2 encoder_final.pt")
    parser.add_argument("--model", required=True, help="Path to stage3_final.pt")
    parser.add_argument("--cpm", required=True, help="Path to context/merged_context_data.json")
    parser.add_argument("--sample-id", type=int, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-len", type=int, default=config.MAX_SEQ_LEN)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--context-off", action="store_true")
    return parser.parse_args()


def load_context(cpm_path: str | Path, sample_id: int) -> dict:
    for obj in _load_json_array(Path(cpm_path)):
        if obj.get("sample_id") == sample_id:
            return obj
    return {}


def sample_next(logits: torch.Tensor, temperature: float, top_k: int) -> int:
    logits = logits / max(temperature, 1e-6)
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_val = values[-1]
        logits = torch.where(logits < min_val, torch.full_like(logits, -1e9), logits)
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1))


def main() -> None:
    args = parse_args()
    tokenizer = Stage3Tokenizer(args.tokenizer)

    stage2_encoder = PasswordEncoder(
        vocab_size=len(tokenizer.token_to_id),
        d_model=config.ENC_DIM,
        ffn_hidden=512,
        num_heads=4,
        num_layers=4,
        dropout=0.1,
        max_len=tokenizer.max_seq_len,
        pad_id=tokenizer.pad_id,
    )
    stage2_encoder.load_state_dict(torch.load(args.encoder, map_location="cpu"))
    stage2_encoder.to(args.device)

    model = Stage3Model(
        stage2_encoder=stage2_encoder,
        vocab_size=len(tokenizer.token_to_id),
        pad_id=tokenizer.pad_id,
        max_len=tokenizer.max_seq_len,
    ).to(args.device)
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

    context_vec = torch.zeros(config.CONTEXT_DIM)
    if not args.context_off:
        context_entry = load_context(args.cpm, args.sample_id)
        context_vectorizer = ContextVectorizer(device=args.device)
        context_vec = context_vectorizer.encode(
            context_entry.get("raw_context", ""),
            context_entry.get("structured_context", {}),
        )
    context_vec = context_vec.to(args.device)

    with torch.no_grad():
        z = torch.randn(1, config.LATENT_DIM, device=args.device)
        z_cond = model.context_mod(z, context_vec.unsqueeze(0))

        generated = [tokenizer.bos_id]
        for _ in range(args.max_len - 1):
            ids = generated + [tokenizer.pad_id] * (args.max_len - len(generated))
            mask = [1] * len(generated) + [0] * (args.max_len - len(generated))
            input_ids = torch.tensor([ids], device=args.device)
            attention_mask = torch.tensor([mask], device=args.device)
            logits = model.decoder(input_ids, attention_mask, z_cond)
            next_logits = logits[0, len(generated) - 1]
            next_id = sample_next(next_logits, args.temperature, args.top_k)
            generated.append(next_id)
            if next_id == tokenizer.eos_id:
                break

    tokens = [tokenizer.id_to_token.get(tid, "<UNK>") for tid in generated]
    print(json.dumps({"sample_id": args.sample_id, "tokens": tokens}))


if __name__ == "__main__":
    main()
