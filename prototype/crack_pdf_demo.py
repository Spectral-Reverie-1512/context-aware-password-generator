"""
Demo: try to crack a password-protected PDF using a context-generated wordlist.

Usage:
  python crack_pdf_demo.py <path-to-encrypted.pdf> <path-to-wordlist.txt>

Example:
  python crack_pdf_demo.py prototype/demos/secret_momo.pdf prototype/wordlists/momo.txt

PowerShell (one pair of quotes around the PDF path — not ""two pairs""):
  python crack_pdf_demo.py "C:\\Users\\You\\Downloads\\demo_protected.pdf" wordlists\\my_demo.txt

Requires: pip install pypdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def try_crack_pdf(pdf_path: str | Path, wordlist_path: str | Path) -> tuple[str | None, int]:
    """
    Try each line in the wordlist as the PDF user password.
    Returns (password, attempt_count): password if found else None, and attempts made.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        print("Error: pypdf is required. Run: pip install pypdf")
        sys.exit(1)

    pdf_path = Path(pdf_path)
    wordlist_path = Path(wordlist_path)

    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}")
        sys.exit(1)
    if not wordlist_path.exists():
        print(f"Error: Wordlist not found: {wordlist_path}")
        sys.exit(1)

    with wordlist_path.open("r", encoding="utf-8", errors="replace") as f:
        passwords = [line.strip() for line in f if line.strip()]

    for i, pwd in enumerate(passwords, start=1):
        try:
            reader = PdfReader(str(pdf_path))
            if not reader.is_encrypted:
                print(f"[INFO] PDF is not encrypted: {pdf_path}")
                return ("", 0)

            # Try to decrypt; some backends may always return 0 even
            # when the password is correct, so we verify by accessing
            # a page after decrypt instead of relying on the return code.
            try:
                reader.decrypt(pwd)
            except Exception:
                # If decrypt itself errors, treat as failure for this candidate.
                continue

            try:
                _ = reader.pages[0]
            except Exception:
                # Still encrypted / wrong password.
                continue

            # If we got here without an exception, the password worked.
            return (pwd, i)
        except Exception:
            continue

    return (None, len(passwords))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Try each line in a wordlist as the PDF user password.",
        epilog=(
            'Tip: In PowerShell use one pair of quotes around the PDF path, e.g. '
            '"C:\\path\\to\\file.pdf" wordlists\\list.txt'
        ),
    )
    parser.add_argument("pdf", type=Path, help="Path to the encrypted PDF")
    parser.add_argument("wordlist", type=Path, help="Path to the wordlist (.txt, one password per line)")
    args = parser.parse_args()
    pdf_path = args.pdf
    wordlist_path = args.wordlist

    print(f"Trying wordlist ({wordlist_path}) against PDF: {pdf_path}")
    password, attempts = try_crack_pdf(pdf_path, wordlist_path)

    if password is not None:
        print(f"[SUCCESS] Cracked with password: {password!r} after {attempts} attempt(s).")
    else:
        print(f"[FAIL] No matching password in wordlist (tried {attempts} candidates).")


if __name__ == "__main__":
    main()
