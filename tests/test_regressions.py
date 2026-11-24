import contextlib
import io
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import audit_dataset
import pipeline
from run_utils import create_run_dir, latest_checkpoint
from scripts.split_dataset import main as split_dataset


class RegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def make_raw(self):
        raw = self.root / "raw"
        for class_index, name in enumerate(("apple", "banana")):
            for index in range(6):
                # Deliberately repeat basenames in different source directories.
                path = raw / name / str(index) / "sample.png"
                path.parent.mkdir(parents=True)
                Image.new("RGB", (32, 32), (class_index * 100, index * 30, 40)).save(path)
        return raw

    def split_args(self, raw, output):
        return Namespace(input=str(raw), output=str(output), train=0.5, val=0.25, test=0.25, seed=42)

    def audit(self, data):
        output = self.root / "audit.json"
        with contextlib.redirect_stdout(io.StringIO()):
            audit_dataset.main(Namespace(data=str(data), output=str(output)))
        return json.loads(output.read_text(encoding="utf-8"))

    def make_split(self):
        output = self.root / "split"
        with contextlib.redirect_stdout(io.StringIO()):
            split_dataset(self.split_args(self.make_raw(), output))
        return output

    def test_checkpoint_skips_incomplete_runs_and_sorts_numerically(self):
        for name in ("test_99", "test_100", "test_101", "test_other"):
            (self.root / name).mkdir()
        for name in ("test_99", "test_100", "test_other"):
            (self.root / name / "best.pt").touch()
        self.assertEqual(latest_checkpoint(self.root), self.root / "test_100" / "best.pt")
        self.assertEqual(create_run_dir(self.root).name, "test_102")

    def test_checkpoint_missing(self):
        self.assertIsNone(latest_checkpoint(self.root / "missing"))

    def test_split_preserves_all_nested_files_and_passes_audit(self):
        output = self.make_split()
        report = self.audit(output)
        self.assertTrue(report["passed"])
        self.assertEqual(sum(row["total"] for row in report["splits"].values()), 12)
        self.assertEqual(report["cross_split_duplicate_file_count"], 0)

    def test_split_is_reproducible(self):
        first = self.make_split()
        second = self.root / "second"
        with contextlib.redirect_stdout(io.StringIO()):
            split_dataset(self.split_args(self.root / "raw", second))
        self.assertEqual(
            sorted(str(p.relative_to(first)) for p in first.rglob("*.png")),
            sorted(str(p.relative_to(second)) for p in second.rglob("*.png")),
        )

    def test_split_refuses_to_mix_with_existing_data(self):
        output = self.make_split()
        before = {p: p.read_bytes() for p in output.rglob("*.png")}
        args = self.split_args(self.root / "raw", output)
        args.seed = 7
        with self.assertRaises(FileExistsError):
            split_dataset(args)
        self.assertEqual(before, {p: p.read_bytes() for p in output.rglob("*.png")})

    def test_split_rejects_overlapping_directories(self):
        raw = self.make_raw()
        with self.assertRaises(ValueError):
            split_dataset(self.split_args(raw, raw / "split"))
        self.assertFalse((raw / "split").exists())

    def test_split_rejects_invalid_ratios(self):
        args = self.split_args(self.make_raw(), self.root / "split")
        args.train = float("nan")
        with self.assertRaises(ValueError):
            split_dataset(args)

    def test_split_validates_all_classes_before_copying(self):
        raw = self.make_raw()
        (raw / "empty").mkdir()
        with self.assertRaises(ValueError):
            split_dataset(self.split_args(raw, self.root / "split"))
        self.assertFalse((self.root / "split").exists())

    def test_audit_fails_on_missing_dataset(self):
        with self.assertRaises(SystemExit):
            self.audit(self.root / "missing")
        report = json.loads((self.root / "audit.json").read_text(encoding="utf-8"))
        self.assertFalse(report["passed"])

    def test_audit_fails_on_cross_split_duplicates(self):
        output = self.make_split()
        train = next((output / "train").rglob("*.png"))
        target = output / "test" / "apple" / "duplicate.png"
        target.write_bytes(train.read_bytes())
        with self.assertRaises(SystemExit):
            self.audit(output)

    def test_audit_fails_on_corrupt_image(self):
        output = self.make_split()
        next((output / "val").rglob("*.png")).write_bytes(b"invalid image")
        with self.assertRaises(SystemExit):
            self.audit(output)

    def test_audit_fails_on_class_mismatch(self):
        output = self.make_split()
        (output / "val" / "banana").rename(output / "val" / "orange")
        with self.assertRaises(SystemExit):
            self.audit(output)

    def test_pipeline_evaluates_the_new_run_after_number_99(self):
        output = self.root / "outputs"
        (output / "test_99").mkdir(parents=True)
        (output / "test_99" / "best.pt").touch()
        commands = []

        def fake_run(command):
            commands.append(command)
            if Path(command[1]).name == "train.py":
                (output / "test_100").mkdir()
                (output / "test_100" / "best.pt").touch()

        args = Namespace(data=str(self.root), output=str(output), model="resnet18", epochs=1,
                         batch_size=2, lr=0.001, min_lr=0.000001, device="cpu", top_k=2,
                         no_pretrained=True, freeze_backbone=False)
        with patch.object(pipeline, "run", side_effect=fake_run), contextlib.redirect_stdout(io.StringIO()):
            pipeline.main(args)
        for command in commands[2:]:
            self.assertEqual(command[command.index("--checkpoint") + 1], str(output / "test_100" / "best.pt"))
        self.assertTrue((output / "test_100" / "pipeline_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
