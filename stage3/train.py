from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from stage2.model import PasswordEncoder
from stage2.losses import label_smoothed_nll_loss

from . import config
from .data import ContextVectorizer, Stage3Dataset, Stage3Tokenizer, collate_batch
from .model import Stage3Model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3 VAE training")
    parser.add_argument("--tokenizer", required=True, help="Path to stage2 tokenizer.json")
    parser.add_argument("--encoder", required=True, help="Path to stage2 encoder_final.pt")
    parser.add_argument("--cpm", required=True, help="Path to context/merged_context_data.json")
    parser.add_argument(
        "--targets", required=True, help="Path to context/password_targets.json"
    )
    parser.add_argument("--fusion", default=None, help="Optional fusion model weights")
    parser.add_argument("--output-dir", default="stage3_output")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--context-off", action="store_true")
    parser.add_argument("--ctx-loss", action="store_true")
    return parser.parse_args()


def kl_anneal_beta(step: int) -> float:
    if step <= 0:
        return 0.0
    warmup = config.KL_WARMUP_STEPS
    return min(config.KL_MAX_BETA, step / max(1, warmup)) * config.KL_MAX_BETA


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.mean(torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=1))


def context_alignment_loss(
    input_ids: torch.Tensor,
    context_struct: list[dict],
    tokenizer: Stage3Tokenizer,
) -> torch.Tensor:
    batch_losses = []
    for ids, ctx in zip(input_ids, context_struct):
        tokens = []
        for key in ("names", "username", "location", "numbers"):
            for item in ctx.get(key, []):
                item = str(item).lower()
                if item in tokenizer.token_to_id:
                    tokens.append(tokenizer.token_to_id[item])
        if not tokens:
            batch_losses.append(torch.tensor(0.0, device=ids.device))
            continue
        token_set = set(tokens)
        present = sum(1 for tid in ids.tolist() if tid in token_set)
        batch_losses.append(1.0 - (present / max(1, len(ids))))
    return torch.mean(torch.stack(batch_losses))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Stage3Tokenizer(args.tokenizer)
    context_vectorizer = None if args.context_off else ContextVectorizer(args.fusion, args.device)
    dataset = Stage3Dataset(args.cpm, args.targets, tokenizer, context_vectorizer)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        drop_last=True,
    )

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

    encoder_params = list(model.stage2_encoder.parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith("stage2_encoder.")]

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": config.LR * config.ENCODER_LR_MULT},
            {"params": other_params, "lr": config.LR},
        ],
        betas=config.BETAS,
        weight_decay=config.WEIGHT_DECAY,
    )

    step = 0
    total_steps = args.epochs * max(1, len(loader))
    start_time = time.time()
    last_log = start_time
    model.train()
    for epoch in range(args.epochs):
        for input_ids, attention_mask, context_vec, structured_ctx in loader:
            step += 1
            input_ids = input_ids.to(args.device)
            attention_mask = attention_mask.to(args.device)
            context_vec = context_vec.to(args.device)

            outputs = model(input_ids, attention_mask, context_vec)
            logits = outputs.logits

            next_logits = logits[:, :-1]
            next_targets = input_ids[:, 1:]
            recon_loss = label_smoothed_nll_loss(
                next_logits.reshape(-1, next_logits.size(-1)),
                next_targets.reshape(-1),
                epsilon=config.LABEL_SMOOTHING,
                ignore_index=tokenizer.pad_id,
            )

            kl_loss = kl_divergence(outputs.mu, outputs.logvar)
            beta = kl_anneal_beta(step)

            if args.ctx_loss:
                ctx_loss = context_alignment_loss(input_ids, structured_ctx, tokenizer)
            else:
                ctx_loss = torch.tensor(0.0, device=input_ids.device)

            loss = recon_loss + beta * kl_loss + config.CTX_LOSS_WEIGHT * ctx_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()

            now = time.time()
            if now - last_log >= 10:
                elapsed = now - start_time
                steps_done = max(1, step)
                steps_left = max(0, total_steps - step)
                step_time = elapsed / steps_done
                eta_sec = steps_left * step_time
                eta_min = eta_sec / 60
                print(
                    f"step {step}/{total_steps} | loss {loss.item():.4f} | ETA {eta_min:.1f} min"
                )
                last_log = now

        torch.save(model.state_dict(), output_dir / f"stage3_epoch_{epoch + 1}.pt")

    torch.save(model.state_dict(), output_dir / "stage3_final.pt")


if __name__ == "__main__":
    main()
