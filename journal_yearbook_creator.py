#!/usr/bin/env python3
"""Create print-ready journals and photo yearbooks without image-generation APIs."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from xml.sax.saxutils import escape

from PIL import Image, ImageOps, UnidentifiedImageError
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader, simpleSplit
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

PAGE_WIDTH, PAGE_HEIGHT = letter
MARGIN = 48
TEAL = colors.HexColor("#174B58")
GOLD = colors.HexColor("#D4A64A")
INK = colors.HexColor("#263746")
RULE = colors.HexColor("#AAB9C2")
PALE = colors.HexColor("#EEF4F5")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "book"


def _make_output_dir(output_dir: Path, book_type: str, title: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    book_dir = output_dir / f"{book_type}_{_slugify(title)}_{timestamp}"
    book_dir.mkdir()
    return book_dir


def _write_manifest(book_dir: Path, metadata: Dict[str, Any]) -> None:
    manifest_path = book_dir / "metadata.json"
    manifest_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def _draw_cover(
    pdf: canvas.Canvas,
    title: str,
    subtitle: str,
    photo: Optional[Path] = None,
    show_photo_placeholder: bool = True,
) -> None:
    pdf.setFillColor(PALE)
    pdf.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, stroke=0, fill=1)
    pdf.setFillColor(TEAL)
    pdf.rect(0, PAGE_HEIGHT - 82, PAGE_WIDTH, 82, stroke=0, fill=1)
    pdf.rect(0, 0, PAGE_WIDTH, 24, stroke=0, fill=1)
    pdf.setFillColor(GOLD)
    pdf.rect(MARGIN, PAGE_HEIGHT - 91, PAGE_WIDTH - 2 * MARGIN, 4, stroke=0, fill=1)

    title_size = 34
    title_lines = simpleSplit(title, "Helvetica-Bold", title_size, PAGE_WIDTH - 2 * MARGIN)
    while len(title_lines) > 3 and title_size > 20:
        title_size -= 2
        title_lines = simpleSplit(title, "Helvetica-Bold", title_size, PAGE_WIDTH - 2 * MARGIN)

    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", title_size)
    y = PAGE_HEIGHT - 205
    for line in title_lines:
        pdf.drawCentredString(PAGE_WIDTH / 2, y, line)
        y -= title_size + 8

    pdf.setFillColor(TEAL)
    pdf.setFont("Helvetica", 14)
    pdf.drawCentredString(PAGE_WIDTH / 2, y - 10, subtitle)

    photo_box = (PAGE_WIDTH / 2 - 150, 210, 300, 225)
    if photo is not None or show_photo_placeholder:
        _draw_photo_box(pdf, photo, photo_box, "ADD A COVER PHOTO")
    else:
        center_x, center_y = PAGE_WIDTH / 2, 322
        pdf.setStrokeColor(GOLD)
        pdf.setLineWidth(2)
        for points in (
            [(center_x, center_y - 45), (center_x - 105, center_y - 30),
             (center_x - 105, center_y + 44), (center_x, center_y + 28)],
            [(center_x, center_y - 45), (center_x + 105, center_y - 30),
             (center_x + 105, center_y + 44), (center_x, center_y + 28)],
        ):
            pdf.lines(
                [(points[index][0], points[index][1],
                  points[index + 1][0], points[index + 1][1])
                 for index in range(len(points) - 1)]
            )
        pdf.setLineWidth(1)
        for offset in (0, 14, 28):
            pdf.line(center_x - 90, center_y + 25 - offset,
                     center_x - 18, center_y + 14 - offset)
            pdf.line(center_x + 18, center_y + 14 - offset,
                     center_x + 90, center_y + 25 - offset)
    pdf.setFillColor(TEAL)
    pdf.setFont("Helvetica", 10)
    pdf.drawCentredString(PAGE_WIDTH / 2, 145, "Made to keep, share, and revisit")
    pdf.showPage()


def _draw_fitted_image(
    pdf: canvas.Canvas,
    image_path: Path,
    box: tuple,
) -> None:
    x, y, width, height = box
    with Image.open(image_path) as source:
        oriented = ImageOps.exif_transpose(source)
        if "A" in oriented.getbands():
            rgba = oriented.convert("RGBA")
            background = Image.new("RGBA", rgba.size, "white")
            background.alpha_composite(rgba)
            image = background.convert("RGB")
        else:
            image = oriented.convert("RGB")
    image_width, image_height = image.size
    scale = min(width / image_width, height / image_height)
    draw_width = image_width * scale
    draw_height = image_height * scale
    draw_x = x + (width - draw_width) / 2
    draw_y = y + (height - draw_height) / 2
    pdf.drawImage(
        ImageReader(image),
        draw_x,
        draw_y,
        width=draw_width,
        height=draw_height,
        preserveAspectRatio=True,
        mask="auto",
    )


def _draw_photo_box(
    pdf: canvas.Canvas,
    photo: Optional[Path],
    box: tuple,
    placeholder: str,
) -> None:
    x, y, width, height = box
    pdf.setFillColor(colors.white)
    pdf.setStrokeColor(RULE)
    pdf.setLineWidth(0.8)
    pdf.rect(x, y, width, height, stroke=1, fill=1)
    inset = 7
    if photo is None:
        pdf.setFillColor(colors.HexColor("#71838D"))
        pdf.setFont("Helvetica", 9)
        pdf.drawCentredString(x + width / 2, y + height / 2, placeholder)
    else:
        _draw_fitted_image(
            pdf,
            photo,
            (x + inset, y + inset, width - 2 * inset, height - 2 * inset),
        )


def _draw_page_number(pdf: canvas.Canvas, number: int) -> None:
    pdf.setFillColor(colors.HexColor("#70818A"))
    pdf.setFont("Helvetica", 8)
    pdf.drawRightString(PAGE_WIDTH - MARGIN, 26, str(number))


def _draw_journal_prompt(
    pdf: canvas.Canvas,
    title: str,
    x: float,
    top: float,
    width: float,
    lines: int,
) -> None:
    pdf.setFillColor(TEAL)
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(x, top, title)
    pdf.setStrokeColor(RULE)
    pdf.setLineWidth(0.55)
    for index in range(lines):
        y = top - 24 - index * 22
        pdf.line(x, y, x + width, y)


def create_journal(
    title: str,
    pages: int = 30,
    output_dir: Path = Path("output"),
    author: str = "",
) -> Path:
    """Create a journal PDF with a cover, ownership page, and guided writing pages."""
    if not title.strip():
        raise ValueError("Journal title cannot be empty.")
    if pages < 1:
        raise ValueError("Journal pages must be at least one.")

    book_dir = _make_output_dir(output_dir, "Journal", title)
    pdf_path = book_dir / f"{_slugify(title)}.pdf"
    pdf = canvas.Canvas(str(pdf_path), pagesize=letter, pageCompression=1)
    pdf.setTitle(title)
    if author:
        pdf.setAuthor(author)

    _draw_cover(
        pdf,
        title,
        author or "A guided journal",
        show_photo_placeholder=False,
    )

    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 21)
    pdf.drawString(MARGIN, PAGE_HEIGHT - 115, "This journal belongs to")
    pdf.setStrokeColor(RULE)
    pdf.setLineWidth(0.8)
    pdf.line(MARGIN, PAGE_HEIGHT - 165, PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 165)
    pdf.setFont("Helvetica", 11)
    for label, y in (("Start date", PAGE_HEIGHT - 235), ("A few things I want to remember", PAGE_HEIGHT - 320)):
        pdf.setFillColor(TEAL)
        pdf.drawString(MARGIN, y, label)
        pdf.setStrokeColor(RULE)
        pdf.line(MARGIN, y - 28, PAGE_WIDTH - MARGIN, y - 28)
        if label.startswith("A few"):
            pdf.line(MARGIN, y - 54, PAGE_WIDTH - MARGIN, y - 54)
            pdf.line(MARGIN, y - 80, PAGE_WIDTH - MARGIN, y - 80)
    _draw_page_number(pdf, 1)
    pdf.showPage()

    for page_number in range(1, pages + 1):
        pdf.setFillColor(INK)
        pdf.setFont("Helvetica-Bold", 18)
        pdf.drawString(MARGIN, PAGE_HEIGHT - 62, "Daily reflection")
        pdf.setFillColor(TEAL)
        pdf.setFont("Helvetica", 10)
        pdf.drawString(MARGIN, PAGE_HEIGHT - 88, "Date:")
        pdf.setStrokeColor(RULE)
        pdf.line(MARGIN + 34, PAGE_HEIGHT - 90, MARGIN + 190, PAGE_HEIGHT - 90)

        content_width = PAGE_WIDTH - 2 * MARGIN
        _draw_journal_prompt(pdf, "TODAY'S FOCUS", MARGIN, PAGE_HEIGHT - 140, content_width, 2)
        _draw_journal_prompt(pdf, "ONE THING I'M GRATEFUL FOR", MARGIN, PAGE_HEIGHT - 250, content_width, 2)
        _draw_journal_prompt(pdf, "A MOMENT I WANT TO REMEMBER", MARGIN, PAGE_HEIGHT - 360, content_width, 2)
        _draw_journal_prompt(pdf, "NOTES AND THOUGHTS", MARGIN, PAGE_HEIGHT - 470, content_width, 8)
        _draw_page_number(pdf, page_number + 1)
        pdf.showPage()

    pdf.save()
    _write_manifest(
        book_dir,
        {
            "type": "journal",
            "title": title,
            "author": author,
            "writing_pages": pages,
            "total_pages": pages + 2,
            "pdf": pdf_path.name,
        },
    )
    return pdf_path


def _resolve_photo(value: Any, base_dir: Path, context: str) -> Optional[Path]:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError(f"{context} photo must be a path string or null.")
    photo_path = (base_dir / value).expanduser().resolve()
    if not photo_path.is_file():
        raise FileNotFoundError(f"{context} photo does not exist: {photo_path}")
    try:
        with Image.open(photo_path) as image:
            image.verify()
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"{context} photo is not a readable image: {photo_path}") from error
    return photo_path


def _load_yearbook_data(data_path: Path, title_override: Optional[str]) -> Dict[str, Any]:
    source = data_path.expanduser().resolve()
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Yearbook data file does not exist: {source}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"Yearbook data is not valid JSON: {error}") from error

    if not isinstance(data, dict):
        raise ValueError("Yearbook JSON must contain a top-level object.")

    title = title_override or data.get("title", "Our Yearbook")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("Yearbook title must be a non-empty string.")
    school = data.get("school", "")
    year = data.get("year", "")
    if not isinstance(school, str) or not isinstance(year, str):
        raise ValueError("Yearbook school and year fields must be strings.")

    sections_data = data.get("sections")
    if not isinstance(sections_data, list) or not sections_data:
        raise ValueError("Yearbook data must contain at least one section.")

    sections: List[Dict[str, Any]] = []
    for section_number, section_data in enumerate(sections_data, start=1):
        if not isinstance(section_data, dict):
            raise ValueError(f"Section {section_number} must be an object.")
        section_title = section_data.get("title")
        entries_data = section_data.get("entries")
        if not isinstance(section_title, str) or not section_title.strip():
            raise ValueError(f"Section {section_number} needs a non-empty title.")
        if not isinstance(entries_data, list) or not entries_data:
            raise ValueError(f"Section '{section_title}' needs at least one entry.")

        entries: List[Dict[str, Any]] = []
        for entry_number, entry_data in enumerate(entries_data, start=1):
            context = f"Section '{section_title}' entry {entry_number}"
            if not isinstance(entry_data, dict):
                raise ValueError(f"{context} must be an object.")
            name = entry_data.get("name")
            caption = entry_data.get("caption", "")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"{context} needs a non-empty name.")
            if not isinstance(caption, str):
                raise ValueError(f"{context} caption must be a string.")
            _validate_card_text(name.strip(), caption.strip())
            entries.append(
                {
                    "name": name.strip(),
                    "caption": caption.strip(),
                    "photo": _resolve_photo(entry_data.get("photo"), source.parent, context),
                }
            )
        sections.append({"title": section_title.strip(), "entries": entries})

    return {
        "title": title.strip(),
        "school": school.strip(),
        "year": year.strip(),
        "cover_photo": _resolve_photo(data.get("cover_photo"), source.parent, "Cover"),
        "sections": sections,
    }


def _draw_yearbook_card(
    pdf: canvas.Canvas,
    x: float,
    y: float,
    width: float,
    height: float,
    entry: Dict[str, Any],
) -> None:
    pdf.setFillColor(colors.white)
    pdf.setStrokeColor(RULE)
    pdf.setLineWidth(0.8)
    pdf.roundRect(x, y, width, height, 8, stroke=1, fill=1)

    _draw_photo_box(
        pdf,
        entry["photo"],
        (x + 12, y + 100, width - 24, height - 24 - 100),
        "ADD PHOTO",
    )

    name_style = ParagraphStyle(
        "YearbookName",
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=13,
        textColor=INK,
        alignment=1,
    )
    caption_style = ParagraphStyle(
        "YearbookCaption",
        fontName="Helvetica",
        fontSize=8.5,
        leading=10,
        textColor=colors.HexColor("#4F626D"),
        alignment=1,
    )
    name = Paragraph(escape(entry["name"]), name_style)
    _, name_height = name.wrap(width - 24, 28)
    if name_height > 28:
        raise ValueError(f"Yearbook name is too long to fit on a card: {entry['name']}")
    name.drawOn(pdf, x + 12, y + 70 + (28 - name_height) / 2)

    if entry["caption"]:
        caption = Paragraph(escape(entry["caption"]), caption_style)
        _, caption_height = caption.wrap(width - 24, 52)
        if caption_height > 52:
            raise ValueError(f"Yearbook caption is too long to fit on a card: {entry['name']}")
        caption.drawOn(pdf, x + 12, y + 12 + (52 - caption_height) / 2)


def _validate_card_text(name: str, caption: str) -> None:
    text_width = (PAGE_WIDTH - 2 * 44 - 20) / 2 - 24
    if len(simpleSplit(name, "Helvetica-Bold", 11, text_width)) * 13 > 28:
        raise ValueError(f"Yearbook name is too long to fit on a card: {name}")
    if caption and len(simpleSplit(caption, "Helvetica", 8.5, text_width)) * 10 > 52:
        raise ValueError(f"Yearbook caption is too long to fit on a card: {name}")


def create_yearbook(
    data_path: Path,
    output_dir: Path = Path("output"),
    title: Optional[str] = None,
) -> Path:
    """Create a photo yearbook PDF from a JSON file and user-supplied photos."""
    data = _load_yearbook_data(data_path, title)
    book_dir = _make_output_dir(output_dir, "Yearbook", data["title"])
    pdf_path = book_dir / f"{_slugify(data['title'])}.pdf"
    pdf = canvas.Canvas(str(pdf_path), pagesize=letter, pageCompression=1)
    pdf.setTitle(data["title"])

    subtitle_parts = [value for value in (data["school"], data["year"]) if value]
    _draw_cover(pdf, data["title"], " | ".join(subtitle_parts) or "Yearbook", data["cover_photo"])

    margin = 44
    gutter = 20
    card_width = (PAGE_WIDTH - 2 * margin - gutter) / 2
    card_height = 258
    card_gap = 18
    entries_per_page = 4
    total_pages = 1

    for section in data["sections"]:
        entries = section["entries"]
        for offset in range(0, len(entries), entries_per_page):
            page_entries = entries[offset : offset + entries_per_page]
            continuation = " (continued)" if offset else ""
            pdf.setFillColor(TEAL)
            pdf.setFont("Helvetica-Bold", 19)
            pdf.drawString(margin, PAGE_HEIGHT - 52, section["title"] + continuation)
            pdf.setStrokeColor(GOLD)
            pdf.setLineWidth(2)
            pdf.line(margin, PAGE_HEIGHT - 65, PAGE_WIDTH - margin, PAGE_HEIGHT - 65)

            top = PAGE_HEIGHT - 92
            for index, entry in enumerate(page_entries):
                row = index // 2
                column = index % 2
                x = margin + column * (card_width + gutter)
                y = top - card_height - row * (card_height + card_gap)
                _draw_yearbook_card(pdf, x, y, card_width, card_height, entry)

            total_pages += 1
            _draw_page_number(pdf, total_pages - 1)
            pdf.showPage()

    pdf.save()
    entry_count = sum(len(section["entries"]) for section in data["sections"])
    _write_manifest(
        book_dir,
        {
            "type": "yearbook",
            "title": data["title"],
            "school": data["school"],
            "year": data["year"],
            "sections": len(data["sections"]),
            "entries": entry_count,
            "total_pages": total_pages,
            "pdf": pdf_path.name,
        },
    )
    return pdf_path


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive whole number") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create print-ready journals and photo yearbooks."
    )
    parser.add_argument("--type", choices=("journal", "yearbook"), required=True)
    parser.add_argument("--title", help="Book title (overrides the yearbook JSON title)")
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--pages", type=_positive_int, help="Journal writing pages (default: 30)")
    parser.add_argument("--author", default="", help="Journal author")
    parser.add_argument("--data", type=Path, help="Yearbook JSON data file")
    args = parser.parse_args(argv)

    if args.type == "journal":
        if args.data:
            parser.error("--data is only valid with --type yearbook")
        if not args.title:
            parser.error("--title is required with --type journal")
        try:
            pdf_path = create_journal(args.title, args.pages or 30, args.output, args.author)
        except (OSError, ValueError) as error:
            parser.error(str(error))
    else:
        if not args.data:
            parser.error("--data is required with --type yearbook")
        if args.author:
            parser.error("--author is only valid with --type journal")
        if args.pages:
            parser.error("--pages is only valid with --type journal")
        try:
            pdf_path = create_yearbook(args.data, args.output, args.title)
        except (OSError, ValueError) as error:
            parser.error(str(error))

    print(f"Created {args.type} PDF: {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
