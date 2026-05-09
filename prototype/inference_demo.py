"""
Inference flow file.

Stage 1 -> Stage 2 -> Stage 2.5 -> Stage 3 -> Stage 4
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from stage1 import stage1 as stage1_entry
from stage2 import train as stage2_entry
from stage25 import cli as stage25_entry

TORCH_OK = True
try:
    import torch  # type: ignore
except Exception:
    TORCH_OK = False

from stage25.extract import extract_context
from stage4 import config as gen_config
from stage4.constraints import repair, validate

from prototype.makewordlists import generate_passwords as rules_generate_passwords


@dataclass
class InferenceOutputs:
    context_text_path: str
    structured_context_path: str
    context_vector_path: str
    generated_passwords_path: str
    generated_details_path: str


@dataclass
class GeneratedItem:
    idx: int
    password: str
    repaired: bool
    valid: bool
    tokens: list[str]


if TORCH_OK:
    from stage2.model import PasswordEncoder
    from stage25.processor import ContextProcessor
    from stage3.config import CONTEXT_DIM, LATENT_DIM
    from stage3.data import Stage3Tokenizer
    from stage3.model import Stage3Model
    from stage4.decoder import decode_tokens
    from stage4.sampling import sample_from_logits


def _load_stage3_model(
    *,
    project_dir: Path,
    device: str,
) -> tuple[Stage3Tokenizer, Stage3Model]:
    tokenizer_path = project_dir / "stage2_output" / "tokenizer.json"
    encoder_path = project_dir / "stage2_output" / "encoder_final.pt"
    stage3_path = project_dir / "stage3_output" / "stage3_final.pt"

    missing = [p for p in (tokenizer_path, encoder_path, stage3_path) if not p.exists()]
    if missing:
        raise SystemExit(
            "Missing model artifact(s):\n"
            + "\n".join(f"- {p}" for p in missing)
            + "\n\nTrain / place artifacts first under stage2_output/ and stage3_output/."
        )

    tokenizer = Stage3Tokenizer(tokenizer_path)

    stage2_encoder = PasswordEncoder(
        vocab_size=len(tokenizer.token_to_id),
        d_model=256,  # stage3.config.ENC_DIM
        ffn_hidden=512,
        num_heads=4,
        num_layers=4,
        dropout=0.1,
        max_len=tokenizer.max_seq_len,
        pad_id=tokenizer.pad_id,
    )
    stage2_encoder.load_state_dict(torch.load(encoder_path, map_location="cpu"))
    stage2_encoder.to(device)
    stage2_encoder.eval()

    model = Stage3Model(
        stage2_encoder=stage2_encoder,
        vocab_size=len(tokenizer.token_to_id),
        pad_id=tokenizer.pad_id,
        max_len=tokenizer.max_seq_len,
    )
    model.load_state_dict(torch.load(stage3_path, map_location="cpu"))
    model.to(device)
    model.eval()

    return tokenizer, model


def _generate_one(
    *,
    tokenizer: Stage3Tokenizer,
    model: Stage3Model,
    context_vec: torch.Tensor,
    structured_context: dict,
    device: str,
    max_seq_len: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> tuple[str, list[str]]:
    with torch.no_grad():
        z = torch.randn(1, LATENT_DIM, device=device)
        z_cond = model.context_mod(z, context_vec.unsqueeze(0))

        generated: list[int] = [tokenizer.bos_id]
        for _ in range(max_seq_len - 1):
            ids = generated + [tokenizer.pad_id] * (max_seq_len - len(generated))
            mask = [1] * len(generated) + [0] * (max_seq_len - len(generated))
            input_ids = torch.tensor([ids], device=device)
            attention_mask = torch.tensor([mask], device=device)
            logits = model.decoder(input_ids, attention_mask, z_cond)
            next_logits = logits[0, len(generated) - 1]
            next_id = sample_from_logits(
                next_logits, temperature=temperature, top_k=top_k, top_p=top_p
            )
            generated.append(next_id)
            if next_id == tokenizer.eos_id:
                break

    token_strings = [tokenizer.id_to_token.get(tid, "<UNK>") for tid in generated]
    password = decode_tokens(token_strings, context_struct=structured_context)
    return password, token_strings


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference flow file (Stage 1 -> Stage 4).")
    p.add_argument("--context", default=None, help="Context text (if omitted, --context-file is required).")
    p.add_argument("--context-file", default=None, help="Path to context text file.")
    p.add_argument("--num", type=int, default=20, help="Number of passwords to generate.")
    p.add_argument("--output-dir", default="inference_output", help="Output folder (relative to repo root).")
    p.add_argument("--device", default=None, help="Torch device override (cpu, cuda).")
    p.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducible generation.")
    p.add_argument("--rockyou", default=None, help="Stage 1 input file (optional).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent.parent
    out_dir = project_dir / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cpu"
    if TORCH_OK:
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if args.seed is not None:
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)

    # Stage 1 -> Stage 1 outputs
    if args.rockyou:
        sys.argv = [
            "stage1",
            "--input",
            args.rockyou,
            "--output-dir",
            str(project_dir / "output"),
        ]
        stage1_entry.main()

    # Stage 2 -> Stage 2 outputs
    sys.argv = [
        "stage2",
        "--segmentations",
        str(project_dir / "output" / "segmentations.tsv"),
        "--vocab",
        str(project_dir / "output" / "vocab_raw.tsv"),
        "--output-dir",
        str(project_dir / "stage2_output"),
    ]
    stage2_entry.main()

    # Stage 2.5 -> Stage 2.5 outputs
    if args.context_file:
        context_text = Path(args.context_file).read_text(encoding="utf-8", errors="ignore").strip()
    else:
        context_text = (args.context or "").strip()
    if not context_text:
        raise SystemExit("Provide --context or --context-file.")

    context_text_path = out_dir / "context.txt"
    context_text_path.write_text(context_text, encoding="utf-8")

    structured_context = extract_context(context_text)
    structured_path = out_dir / "stage25_structured_context.json"
    structured_path.write_text(json.dumps(structured_context, indent=2), encoding="utf-8")

    context_vec_path = out_dir / "stage25_context_vec.json"
    if TORCH_OK:
        stage25_in_path = out_dir / "context.txt"
        stage25_out_path = out_dir / "stage25_context_vec.json"
        sys.argv = [
            "stage25",
            "--input",
            str(stage25_in_path),
            "--output",
            str(stage25_out_path),
            "--device",
            device,
        ]
        stage25_entry.main()
        context_vec = torch.tensor(
            json.loads(stage25_out_path.read_text(encoding="utf-8")),
            dtype=torch.float32,
            device=device,
        )
    else:
        # Stage 2.5 -> output placeholder
        context_vec_path.write_text("null\n", encoding="utf-8")

    # Stage 3 -> Stage 4 -> generated passwords
    items: list[GeneratedItem] = []
    passwords: list[str] = []

    tokenizer = None
    model = None
    if TORCH_OK:
        tokenizer, model = _load_stage3_model(project_dir=project_dir, device=device)

    for i in range(int(args.num)):
        raw_pwd = ""
        raw_tokens: list[str] = []
        final_pwd = ""
        was_repaired = False
        is_valid = False

        if TORCH_OK:
            for _ in range(gen_config.RESAMPLE_TRIES):
                raw_pwd, raw_tokens = _generate_one(
                    tokenizer=tokenizer,
                    model=model,
                    context_vec=context_vec,
                    structured_context=structured_context,
                    device=device,
                    max_seq_len=tokenizer.max_seq_len,
                    temperature=gen_config.TEMPERATURE,
                    top_k=gen_config.TOP_K,
                    top_p=gen_config.TOP_P,
                )
                if not raw_pwd:
                    continue

                if validate(
                    raw_pwd,
                    min_len=gen_config.MIN_LEN,
                    max_len=gen_config.MAX_LEN,
                    require_digit=gen_config.REQUIRE_DIGIT,
                    require_symbol=gen_config.REQUIRE_SYMBOL,
                    require_upper=gen_config.REQUIRE_UPPER,
                    require_lower=gen_config.REQUIRE_LOWER,
                ):
                    final_pwd = raw_pwd
                    is_valid = True
                    break

                repaired = repair(
                    raw_pwd,
                    min_len=gen_config.MIN_LEN,
                    max_len=gen_config.MAX_LEN,
                    require_digit=gen_config.REQUIRE_DIGIT,
                    require_symbol=gen_config.REQUIRE_SYMBOL,
                    require_upper=gen_config.REQUIRE_UPPER,
                    require_lower=gen_config.REQUIRE_LOWER,
                )
                if validate(
                    repaired,
                    min_len=gen_config.MIN_LEN,
                    max_len=gen_config.MAX_LEN,
                    require_digit=gen_config.REQUIRE_DIGIT,
                    require_symbol=gen_config.REQUIRE_SYMBOL,
                    require_upper=gen_config.REQUIRE_UPPER,
                    require_lower=gen_config.REQUIRE_LOWER,
                ):
                    final_pwd = repaired
                    was_repaired = repaired != raw_pwd
                    is_valid = True
                    break
        else:
            # Stage 3 (rules) -> Stage 4 constraints
            seed = args.seed + i if args.seed is not None else None
            raw_pwd = rules_generate_passwords(
                context=context_text,
                num_passwords=1,
                seed=seed,
                min_length=gen_config.MIN_LEN,
                max_length=gen_config.MAX_LEN,
                require_upper=gen_config.REQUIRE_UPPER,
                require_lower=gen_config.REQUIRE_LOWER,
                require_digit=gen_config.REQUIRE_DIGIT,
                require_symbol=gen_config.REQUIRE_SYMBOL,
                symbols="!@#$%&_-",
            )[0]
            raw_tokens = []
            if validate(
                raw_pwd,
                min_len=gen_config.MIN_LEN,
                max_len=gen_config.MAX_LEN,
                require_digit=gen_config.REQUIRE_DIGIT,
                require_symbol=gen_config.REQUIRE_SYMBOL,
                require_upper=gen_config.REQUIRE_UPPER,
                require_lower=gen_config.REQUIRE_LOWER,
            ):
                final_pwd = raw_pwd
                is_valid = True
            else:
                repaired = repair(
                    raw_pwd,
                    min_len=gen_config.MIN_LEN,
                    max_len=gen_config.MAX_LEN,
                    require_digit=gen_config.REQUIRE_DIGIT,
                    require_symbol=gen_config.REQUIRE_SYMBOL,
                    require_upper=gen_config.REQUIRE_UPPER,
                    require_lower=gen_config.REQUIRE_LOWER,
                )
                final_pwd = repaired
                was_repaired = repaired != raw_pwd
                is_valid = validate(
                    final_pwd,
                    min_len=gen_config.MIN_LEN,
                    max_len=gen_config.MAX_LEN,
                    require_digit=gen_config.REQUIRE_DIGIT,
                    require_symbol=gen_config.REQUIRE_SYMBOL,
                    require_upper=gen_config.REQUIRE_UPPER,
                    require_lower=gen_config.REQUIRE_LOWER,
                )

        if not final_pwd:
            final_pwd = raw_pwd or "Password2026!"
            was_repaired = final_pwd != raw_pwd
            is_valid = validate(
                final_pwd,
                min_len=gen_config.MIN_LEN,
                max_len=gen_config.MAX_LEN,
                require_digit=gen_config.REQUIRE_DIGIT,
                require_symbol=gen_config.REQUIRE_SYMBOL,
                require_upper=gen_config.REQUIRE_UPPER,
                require_lower=gen_config.REQUIRE_LOWER,
            )

        passwords.append(final_pwd)
        items.append(
            GeneratedItem(
                idx=i,
                password=final_pwd,
                repaired=was_repaired,
                valid=is_valid,
                tokens=raw_tokens,
            )
        )

    passwords_path = out_dir / "passwords.txt"
    passwords_path.write_text("\n".join(passwords) + "\n", encoding="utf-8")

    details_path = out_dir / "passwords_details.json"
    details_path.write_text(
        json.dumps([asdict(x) for x in items], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    manifest = InferenceOutputs(
        context_text_path=str(context_text_path),
        structured_context_path=str(structured_path),
        context_vector_path=str(context_vec_path),
        generated_passwords_path=str(passwords_path),
        generated_details_path=str(details_path),
    )
    (out_dir / "manifest.json").write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")

    print("Wrote inference output to:", out_dir)
    print("Passwords:", passwords_path)


if __name__ == "__main__":
    main()

