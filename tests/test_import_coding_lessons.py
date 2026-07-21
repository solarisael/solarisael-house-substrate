from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import import_coding_lessons as importer


class FakeCursor:
    def __init__(self, existing: dict[tuple[str, str | None, str], int] | None = None) -> None:
        self.existing = dict(existing or {})
        self.pending = None
        self.inserted: list[tuple[str, str | None, str]] = []
        self.updated: list[int] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql: str, args: tuple) -> None:
        operation = sql.strip().split(None, 1)[0].upper()
        if operation == "SELECT":
            key = (args[0], args[1], args[2])
            lesson_id = self.existing.get(key)
            self.pending = (lesson_id,) if lesson_id is not None else None
        elif operation == "INSERT":
            key = (args[0], args[1], args[2])
            self.existing[key] = max(self.existing.values(), default=0) + 1
            self.inserted.append(key)
        elif operation == "UPDATE":
            self.updated.append(args[-1])
        else:
            raise AssertionError(f"unexpected operation: {operation}")

    def fetchone(self):
        return self.pending


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self.cursor_value = cursor

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self) -> FakeCursor:
        return self.cursor_value


class CodingLessonPackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pack = importer.load_pack(importer.DEFAULT_PACK)

    def test_bundled_pack_is_valid_and_unique(self) -> None:
        self.assertEqual(self.pack["id"], "solarisael-house-craft-starter")
        self.assertEqual(self.pack["version"], 1)
        self.assertEqual(len(self.pack["lessons"]), 14)
        keys = {
            (lesson["scope"], lesson["project"], lesson["title"])
            for lesson in self.pack["lessons"]
        }
        self.assertEqual(len(keys), 14)
        self.assertEqual(sum(lesson["always_on"] for lesson in self.pack["lessons"]), 1)
        self.assertTrue(all(lesson["meta"]["starter_pack_version"] == 1 for lesson in self.pack["lessons"]))

    def test_duplicate_lesson_keys_are_rejected(self) -> None:
        document = json.loads(importer.DEFAULT_PACK.read_text(encoding="utf-8"))
        document["lessons"].append(dict(document["lessons"][0]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate lesson key"):
                importer.load_pack(path)

    def test_default_import_preserves_existing_lessons(self) -> None:
        lessons = self.pack["lessons"][:2]
        key = (lessons[0]["scope"], lessons[0]["project"], lessons[0]["title"])
        cursor = FakeCursor({key: 41})
        counts = importer.import_pack(
            FakeConnection(cursor),
            {**self.pack, "lessons": lessons},
            update_existing=False,
        )
        self.assertEqual(counts, {"inserted": 1, "updated": 0, "skipped": 1})
        self.assertEqual(cursor.updated, [])

    def test_update_existing_requires_explicit_flag(self) -> None:
        lesson = self.pack["lessons"][0]
        key = (lesson["scope"], lesson["project"], lesson["title"])
        cursor = FakeCursor({key: 77})
        counts = importer.import_pack(
            FakeConnection(cursor),
            {**self.pack, "lessons": [lesson]},
            update_existing=True,
        )
        self.assertEqual(counts, {"inserted": 0, "updated": 1, "skipped": 0})
        self.assertEqual(cursor.updated, [77])


if __name__ == "__main__":
    unittest.main()
