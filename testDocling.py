#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import sys

from docling.document_converter import DocumentConverter


def convert_pdf_to_docling_json(input_pdf: Path, output_json: Path) -> None:
    if not input_pdf.exists():
        raise FileNotFoundError(f"PDF nicht gefunden: {input_pdf}")

    if input_pdf.suffix.lower() != ".pdf":
        raise ValueError(f"Eingabedatei ist keine PDF: {input_pdf}")

    converter = DocumentConverter()
    result = converter.convert(str(input_pdf))
    doc = result.document

    output_json.parent.mkdir(parents=True, exist_ok=True)

    # Variante 1: Offizielle Docling-JSON-Ausgabe
    doc.save_as_json(output_json, indent=2)

    # Alternative, falls du das Dict selbst weiterverarbeiten willst:
    # data = doc.export_to_dict()
    # output_json.write_text(
    #     json.dumps(data, ensure_ascii=False, indent=2),
    #     encoding="utf-8",
    # )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PDF mit Docling einlesen und als Docling JSON ausgeben."
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Pfad zur Ausgabe-JSON. Default: <pdf-name>.docling.json",
    )

    args = parser.parse_args()

    input_pdf = Path("data/Baloch_2022.pdf")
    output_json = (
        Path(args.output).expanduser().resolve()
        if args.output
        else input_pdf.with_suffix(".docling.json")
    )

    try:
        convert_pdf_to_docling_json(input_pdf, output_json)
    except Exception as exc:
        print(f"Fehler: {exc}", file=sys.stderr)
        return 1

    print(f"Docling JSON geschrieben: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())