import json
import re
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from journal_yearbook_creator import create_journal, create_yearbook, main


def pdf_page_count(pdf_path):
    counts = re.findall(rb"/Count\s+(\d+)\s*/Kids", pdf_path.read_bytes())
    if not counts:
        raise AssertionError("PDF page tree does not contain a page count.")
    return int(counts[0])


class JournalYearbookCreatorTests(unittest.TestCase):
    def test_journal_creates_cover_owner_page_and_requested_writing_pages(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = create_journal(
                title="A Small Journal",
                pages=3,
                output_dir=Path(temp_dir),
                author="Alex",
            )

            self.assertTrue(pdf_path.is_file())
            self.assertEqual(pdf_page_count(pdf_path), 5)
            metadata = json.loads((pdf_path.parent / "metadata.json").read_text())
            self.assertEqual(metadata["type"], "journal")
            self.assertEqual(metadata["writing_pages"], 3)
            self.assertEqual(metadata["total_pages"], 5)

    def test_cli_selects_journal_mode(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = main(
                [
                    "--type",
                    "journal",
                    "--title",
                    "CLI Journal",
                    "--pages",
                    "1",
                    "--output",
                    temp_dir,
                ]
            )

            self.assertEqual(result, 0)
            self.assertEqual(len(list(Path(temp_dir).glob("Journal_*/*.pdf"))), 1)

    def test_yearbook_uses_relative_photos_and_paginates_entries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            photos = root / "photos"
            photos.mkdir()
            Image.new("RGB", (320, 200), "white").save(photos / "student.png")

            entries = [
                {
                    "name": f"Student {index}",
                    "caption": "A favorite school memory.",
                    "photo": "photos/student.png" if index == 1 else None,
                }
                for index in range(1, 6)
            ]
            data_path = root / "yearbook.json"
            data_path.write_text(
                json.dumps(
                    {
                        "title": "Our Class",
                        "school": "Example School",
                        "year": "2025-2026",
                        "sections": [{"title": "Class Memories", "entries": entries}],
                    }
                ),
                encoding="utf-8",
            )

            pdf_path = create_yearbook(data_path, root / "output")

            self.assertTrue(pdf_path.is_file())
            self.assertEqual(pdf_page_count(pdf_path), 3)
            metadata = json.loads((pdf_path.parent / "metadata.json").read_text())
            self.assertEqual(metadata["entries"], 5)
            self.assertEqual(metadata["total_pages"], 3)

    def test_yearbook_rejects_missing_photo_instead_of_silently_omitting_it(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_path = root / "yearbook.json"
            data_path.write_text(
                json.dumps(
                    {
                        "sections": [
                            {
                                "title": "Class",
                                "entries": [
                                    {"name": "Student", "photo": "missing.jpg"}
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(FileNotFoundError, "photo does not exist"):
                create_yearbook(data_path, root / "output")
            self.assertFalse((root / "output").exists())

    def test_yearbook_requires_at_least_one_section(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_path = root / "yearbook.json"
            data_path.write_text(json.dumps({"sections": []}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "at least one section"):
                create_yearbook(data_path, root / "output")


if __name__ == "__main__":
    unittest.main()
