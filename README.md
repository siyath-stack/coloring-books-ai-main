# Journal and Yearbook Creator

Create print-ready guided journals and photo yearbooks as PDFs. These workflows
run locally and do not call an image-generation service.

## What it creates

- **Journal**: a cover, an ownership page, and repeatable daily reflection pages
  with room to write.
- **Yearbook**: a cover and photo-card pages with names and captions. Supply
  your own photos, or leave photo fields empty to get printable placeholders.

Both formats use US Letter pages. PDFs use vector text and lines; yearbook photos
are placed without distortion.

## Setup

```bash
python -m pip install Pillow reportlab
```

The journal/yearbook creator only needs Pillow and ReportLab. Install
`requirements.txt` if you also plan to run the legacy coloring-book tools.

## Create a journal

```bash
python journal_yearbook_creator.py \
  --type journal \
  --title "A Year of Ideas" \
  --author "Your Name" \
  --pages 60
```

The page count is for writing pages; the PDF also includes a cover and an
ownership page.

## Create a yearbook

Start with the included editable example:

```bash
python journal_yearbook_creator.py \
  --type yearbook \
  --data examples/yearbook.json
```

Edit the JSON to add people, captions, and photos. Photo paths are relative to
the JSON file. For example:

```json
{
  "title": "Our Yearbook",
  "school": "Your School",
  "year": "2025-2026",
  "cover_photo": "photos/cover.jpg",
  "sections": [
    {
      "title": "Class Memories",
      "entries": [
        {
          "name": "Student Name",
          "caption": "A favorite moment from the year.",
          "photo": "photos/student.jpg"
        }
      ]
    }
  ]
}
```

Each section can have any number of entries. The creator lays them out four per
page and reports missing or unreadable photo files instead of silently omitting
them. Leave a `photo` value as `null` to create a blank photo box.

```bash
python journal_yearbook_creator.py \
  --type yearbook \
  --data path/to/yearbook.json \
  --title "Class of 2026"
```

## Output

Each run creates a timestamped folder under `output/` containing the PDF and a
`metadata.json` summary. Pass `--output PATH` to use a different output folder.
Use `--help` for all command options.

## Tests

```bash
python -m unittest -v test_journal_yearbook_creator
```

## Legacy coloring-book tools

The original coloring-book scripts remain available, but are no longer the
project's default workflow:

- `coloring_book_generator.py` — image-backed coloring-book generation
- `complete_book_workflow.py` — legacy end-to-end coloring-book workflow
- `generate_book.py` — storybook generation
- `batch_generator.py` — batch coloring-book generation

Those tools may require external image providers and their own credentials or
quotas. Journal and yearbook creation do not use them.
