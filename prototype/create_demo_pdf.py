"""
Create a password-protected demo PDF for use with crack_pdf_demo.py.

Usage:
  python create_demo_pdf.py <output.pdf> <password> [optional text content]

Example:
  python create_demo_pdf.py prototype/demos/secret_momo.pdf "Momo2026!aniKpop#"
  python create_demo_pdf.py prototype/demos/secret_momo.pdf "Momo2026!aniKpop#" "Secret note for Momo"

Requires: pip install pypdf
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        print("Usage: python create_demo_pdf.py <output.pdf> <password> [content text]")
        sys.exit(1)

    out_path = Path(sys.argv[1])
    password = sys.argv[2]
    content = sys.argv[3] if len(sys.argv) > 3 else "This is a demo protected PDF. Password was set by create_demo_pdf.py."

    try:
        from pypdf import PdfWriter
    except ImportError:
        print("Error: pypdf is required. Run: pip install pypdf")
        sys.exit(1)

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_metadata({"/Title": "Demo protected PDF", "/Subject": content[:200]})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # RC4-128 works with pypdf only (AES-256 needs: pip install cryptography)
    writer.encrypt(password, algorithm="RC4-128")
    with out_path.open("wb") as f:
        writer.write(f)

    print(f"Created password-protected PDF: {out_path}")
    print(f"Password: {password!r}")
    print("Crack it with: python crack_pdf_demo.py", str(out_path), "<wordlist.txt>")


if __name__ == "__main__":
    main()
