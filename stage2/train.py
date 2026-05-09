from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from . import config
from .losses import WarmupLinearDecay, contrastive_loss, label_smoothed_nll_loss
from .model import PasswordEncoder
from .tokenizer import PasswordTokenizer, SegmentationDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2 training")
    parser.add_argument(
        "--segmentations",
        default="output/segmentations.tsv",
        help="Path to Stage 1 segmentations.tsv",
    )
    parser.add_argument(
        "--vocab",
        default="output/vocab_raw.tsv",
        help="Path to Stage 1 vocab_raw.tsv",
    )
    parser.add_argument(
        "--output-dir",
        default="stage2_output",
        help="Directory for checkpoints and tokenizer vocab",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=config.MAX_SEQ_LEN)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--contrastive-weight", type=float, default=1.0)
    parser.add_argument("--recon-weight", type=float, default=1.0)
    parser.add_argument("--next-weight", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def collate_batch(samples):
    input_ids = torch.tensor([s.input_ids for s in samples], dtype=torch.long)
    attention_mask = torch.tensor([s.attention_mask for s in samples], dtype=torch.long)
    return input_ids, attention_mask


def augment_tokens(
    input_ids: torch.Tensor,
    pad_id: int,
    unk_id: int,
    drop_prob: float,
    bos_id: int,
    eos_id: int,
) -> torch.Tensor:
    noise = torch.rand_like(input_ids.float())
    protected = (input_ids == pad_id) | (input_ids == bos_id) | (input_ids == eos_id)
    keep = (noise > drop_prob) | protected
    augmented = torch.where(keep, input_ids, torch.full_like(input_ids, unk_id))
    return augmented


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = PasswordTokenizer(args.vocab, max_seq_len=args.max_seq_len)
    tokenizer.save_vocab(output_dir / "tokenizer.json")

    dataset = SegmentationDataset(args.segmentations, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        drop_last=True,
    )

    model = PasswordEncoder(
        vocab_size=len(tokenizer.token_to_id),
        d_model=config.D_MODEL,
        ffn_hidden=config.FFN_HIDDEN,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        max_len=args.max_seq_len,
        pad_id=tokenizer.pad_id,
    ).to(args.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LR,
        betas=config.BETAS,
        weight_decay=config.WEIGHT_DECAY,
    )

    lr_schedule = WarmupLinearDecay(config.D_MODEL, config.WARMUP_STEPS)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: lr_schedule(step + 1)
    )

    step = 0
    model.train()
    for epoch in range(args.epochs):
        for input_ids, attention_mask in loader:
            step += 1
            input_ids = input_ids.to(args.device)
            attention_mask = attention_mask.to(args.device)

            outputs = model(input_ids, attention_mask)

            next_logits = outputs.logits_next[:, :-1]
            next_targets = input_ids[:, 1:]
            next_loss = label_smoothed_nll_loss(
                next_logits.reshape(-1, next_logits.size(-1)),
                next_targets.reshape(-1),
                epsilon=config.LABEL_SMOOTHING,
                ignore_index=tokenizer.pad_id,
            )

            recon_loss = label_smoothed_nll_loss(
                outputs.logits_recon.reshape(-1, outputs.logits_recon.size(-1)),
                input_ids.reshape(-1),
                epsilon=config.LABEL_SMOOTHING,
                ignore_index=tokenizer.pad_id,
            )

            aug_ids = augment_tokens(
                input_ids,
                tokenizer.pad_id,
                tokenizer.unk_id,
                config.CONTRASTIVE_DROPOUT,
                tokenizer.bos_id,
                tokenizer.eos_id,
            )
            aug_mask = (aug_ids != tokenizer.pad_id).long()
            z1 = outputs.pooled
            z2 = model.encode(aug_ids, aug_mask)
            cont_loss = contrastive_loss(z1, z2, temperature=config.CONTRASTIVE_TEMPERATURE)

            loss = (
                args.next_weight * next_loss
                + args.recon_weight * recon_loss
                + args.contrastive_weight * cont_loss
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()
            scheduler.step()

        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch + 1,
            "step": step,
        }
        torch.save(checkpoint, output_dir / f"encoder_epoch_{epoch + 1}.pt")

    torch.save(model.state_dict(), output_dir / "encoder_final.pt")


if __name__ == "__main__":
    main()

