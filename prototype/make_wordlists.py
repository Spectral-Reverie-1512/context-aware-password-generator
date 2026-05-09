"""
Build targeted wordlists by calling the password generator API (same as the web UI).
Use these wordlists with crack_pdf_demo.py against your demo PDFs.

Start the light API first (matches frontend /api/generate-passwords behavior):
  python -m uvicorn prototype.api_server_light:app --reload --host 127.0.0.1 --port 8765

Then run:
  python make_wordlists.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: requests is required. Run: pip install requests")
    sys.exit(1)

DEFAULT_API_URL = "http://127.0.0.1:8765/api/generate-passwords"
WORDLISTS_DIR = Path(__file__).resolve().parent / "wordlists"

# Defaults aligned with frontend (index.html / app.js) and api_server_light.GenerateRequest
DEFAULT_MIN_LENGTH = 5
DEFAULT_MAX_LENGTH = 32
DEFAULT_REQUIRE_SYMBOL = True
DEFAULT_REQUIRE_DIGIT = True

# Default scenarios (used only if you don't provide --context or --scenarios-file)
SCENARIOS = [
    {
        "context": "College student, uses Instagram and Netflix, loves anime and K-pop, pet cat named Momo",
        "num_passwords": 2000,
        "outfile": "momo.txt",
    }
]


# api_server_light allows up to 5000 passwords per request (see GenerateRequest.num_passwords le=)
MAX_PER_REQUEST = 5000


def _post_json(api_url: str, json_body: dict, timeout: int = 120) -> requests.Response:
    """POST JSON; exit with a short message if the API is not running."""
    try:
        return requests.post(api_url, json=json_body, timeout=timeout)
    except requests.exceptions.ConnectionError as e:
        print(
            "Cannot reach the password API (connection refused).\n"
            "Start it in another terminal, then run this script again:\n"
            "  python -m uvicorn api_server_light:app --reload --host 127.0.0.1 --port 8765\n"
            f"If you use a different URL, pass: --api-url <your-endpoint>\n"
            f"Tried: {api_url}",
            file=sys.stderr,
        )
        raise SystemExit(1) from e
    except requests.exceptions.Timeout as e:
        print(f"API request timed out ({timeout}s): {api_url}", file=sys.stderr)
        raise SystemExit(1) from e


def _clean_password(pwd: object) -> str | None:
    """
    Normalize API outputs into usable wordlist entries.
    Filters blanks and any password containing whitespace.
    """
    if not isinstance(pwd, str):
        return None
    s = pwd.strip()
    if not s:
        return None
    if any(ch.isspace() for ch in s):
        return None
    return s


def make_wordlist(
    api_url: str,
    context: str,
    num: int,
    outfile: str,
    *,
    seed: int | None = None,
    min_length: int = DEFAULT_MIN_LENGTH,
    max_length: int = DEFAULT_MAX_LENGTH,
    require_symbol: bool = DEFAULT_REQUIRE_SYMBOL,
    require_digit: bool = DEFAULT_REQUIRE_DIGIT,
) -> None:
    out_path = WORDLISTS_DIR / outfile
    WORDLISTS_DIR.mkdir(parents=True, exist_ok=True)

    # Same JSON body shape as frontend/app.js fetch to /api/generate-passwords
    def build_payload(batch_size: int) -> dict:
        body: dict = {
            "context": context,
            "num_passwords": batch_size,
            "mode": "auto",
            "min_length": min_length,
            "max_length": max_length,
            "require_symbol": require_symbol,
            "require_digit": require_digit,
        }
        if seed is not None:
            body["seed"] = seed
        return body

    # Use a set to dedupe and avoid writing repeated junk.
    passwords_set: set[str] = set()
    remaining = num
    empty_rounds = 0
    max_empty_rounds = 20
    while remaining > 0:
        batch_size = min(MAX_PER_REQUEST, remaining)
        resp = _post_json(api_url, build_payload(batch_size), timeout=120)
        resp.raise_for_status()
        batch = resp.json().get("passwords", [])
        if not isinstance(batch, list):
            batch = []

        added_this_round = 0
        for raw in batch:
            cleaned = _clean_password(raw)
            if cleaned is None:
                continue
            if cleaned in passwords_set:
                continue
            passwords_set.add(cleaned)
            added_this_round += 1

        # Only decrement by the number of *valid new* passwords we accepted.
        remaining -= added_this_round

        # Safety: if API returns nothing usable, retry a few times then stop.
        if added_this_round == 0:
            empty_rounds += 1
            if empty_rounds >= max_empty_rounds:
                break
        else:
            empty_rounds = 0

    with out_path.open("w", encoding="utf-8") as f:
        for pwd in sorted(passwords_set):
            f.write(pwd + "\n")

    print(f"Wrote {len(passwords_set)} passwords to {out_path}")


def _scenario_options(s: dict) -> dict:
    """Optional keys from a scenario dict (for --scenarios-file or SCENARIOS)."""
    out: dict = {}
    if "seed" in s and s["seed"] is not None:
        out["seed"] = int(s["seed"])
    if "min_length" in s:
        out["min_length"] = int(s["min_length"])
    if "max_length" in s:
        out["max_length"] = int(s["max_length"])
    if "require_symbol" in s:
        out["require_symbol"] = bool(s["require_symbol"])
    if "require_digit" in s:
        out["require_digit"] = bool(s["require_digit"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate wordlists by calling the password generator API.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="API endpoint (default: %(default)s)")
    parser.add_argument("--context", default=None, help="Context text (if omitted, will prompt).")
    parser.add_argument("--num", type=int, default=2000, help="Number of passwords to generate (default: %(default)s)")
    parser.add_argument("--outfile", default="wordlist.txt", help="Output filename under wordlists/ (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed (same as web UI).")
    parser.add_argument("--min-length", type=int, default=DEFAULT_MIN_LENGTH, help="min_length (default: %(default)s)")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="max_length (default: %(default)s)")
    parser.add_argument("--no-require-symbol", action="store_true", help="Set require_symbol=false.")
    parser.add_argument("--no-require-digit", action="store_true", help="Set require_digit=false.")
    parser.add_argument(
        "--scenarios-file",
        default=None,
        help="Optional JSON list of scenarios. Each object may include context, num_passwords, outfile, "
        "and optionally seed, min_length, max_length, require_symbol, require_digit.",
    )
    args = parser.parse_args()

    api_url = args.api_url
    gen_kw: dict = {
        "seed": args.seed,
        "min_length": args.min_length,
        "max_length": args.max_length,
        "require_symbol": not args.no_require_symbol,
        "require_digit": not args.no_require_digit,
    }

    print("Calling API at", api_url)

    if args.scenarios_file:
        scenarios_path = Path(args.scenarios_file)
        data = scenarios_path.read_text(encoding="utf-8")
        scenarios = __import__("json").loads(data)
        if not isinstance(scenarios, list):
            raise SystemExit("--scenarios-file must contain a JSON list of scenarios.")
        for s in scenarios:
            if not isinstance(s, dict):
                continue
            merged = {**gen_kw, **_scenario_options(s)}
            make_wordlist(
                api_url,
                str(s.get("context", "")),
                int(s.get("num_passwords", 2000)),
                str(s.get("outfile", "wordlist.txt")),
                **merged,
            )
        print("Done.")
        return

    context = args.context
    if context is None:
        print("Enter context (press Enter to finish):")
        context = input("> ").strip()
        if not context:
            # fall back to defaults
            for s in SCENARIOS:
                merged = {**gen_kw, **_scenario_options(s)}
                make_wordlist(api_url, s["context"], s["num_passwords"], s["outfile"], **merged)
            print("Done.")
            return

    make_wordlist(api_url, context, args.num, args.outfile, **gen_kw)
    print("Done.")


if __name__ == "__main__":
    main()
