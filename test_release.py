"""Checks for release boundaries and the clean-source requirement (stdlib only)."""
from pathlib import Path
import subprocess
import tempfile
import unittest
import zipfile

from prepare_release import VERSION, clean_commit, inspect_source, sha256


class ReleaseChecks(unittest.TestCase):
    def make_archive(self, path, extra=(), missing=()):
        with zipfile.ZipFile(path, "w") as archive:
            for name in ("main.py", "LICENSE", "RELEASE.md", "requirements-release.txt", *extra):
                if name not in missing:
                    archive.writestr(f"praj-{VERSION}/" + name, "example")

    def test_valid_archive_and_checksum(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.zip"
            self.make_archive(path)
            self.assertEqual(inspect_source(path), 4)
            self.assertEqual(len(sha256(path)), 64)

    def test_rejects_unverified_assets_and_binaries(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.zip"
            for name in ("logo.png", "stockfish_logo.png", "engine.exe", "weights.pt", "../escape"):
                with self.subTest(name=name):
                    self.make_archive(path, extra=(name,))
                    with self.assertRaises(RuntimeError):
                        inspect_source(path)

    def test_requires_source_and_license(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.zip"
            self.make_archive(path, missing=("LICENSE",))
            with self.assertRaises(RuntimeError):
                inspect_source(path)

    def test_git_exports_omit_artwork_and_reject_dirty_checkout(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            def run(*args):
                return subprocess.check_output(["git", "-C", folder, *args], stderr=subprocess.DEVNULL)
            run("init", "-q")
            (root / ".gitattributes").write_text("/logo.png export-ignore\n/stockfish_logo.png export-ignore\n")
            for name in ("logo.png", "stockfish_logo.png", "main.py"):
                (root / name).write_text("example")
            run("add", ".")
            run("-c", "user.name=Release Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture")
            self.assertEqual(len(clean_commit(root)), 40)
            # Place output outside the repository, so it does not make the checkout dirty.
            with tempfile.TemporaryDirectory() as output:
                archive = Path(output) / "source.zip"
                run("archive", "--format=zip", f"--output={archive}", "HEAD")
                with zipfile.ZipFile(archive) as bundle:
                    self.assertIn("main.py", bundle.namelist())
                    self.assertNotIn("logo.png", bundle.namelist())
                    self.assertNotIn("stockfish_logo.png", bundle.namelist())
            (root / "main.py").write_text("changed")
            with self.assertRaises(RuntimeError):
                clean_commit(root)


if __name__ == "__main__":
    unittest.main()
