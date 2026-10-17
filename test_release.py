"""Checks for release boundaries and the clean-source requirement (stdlib only)."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile

from prepare_release import ROOT, VERSION, clean_commit, inspect_source, sha256, write_manifest
from prepare_release import (write_runtime_archive, inspect_runtime,
                             write_developer_archive, verify_developer_archive)
from engine_info import EXECUTABLE, SOURCES_URL


class ReleaseChecks(unittest.TestCase):
    def make_candidate(self, output):
        (output / "bin").mkdir()
        (output / "bin" / EXECUTABLE).write_bytes(b"example binary")
        (output / "licenses").mkdir()
        (output / "licenses" / "example.txt").write_text("dependency notice")
        (output / "engines").mkdir()
        (output / "engines" / "stockfish.exe").write_bytes(b"official backend fixture")
        sf_notice = output / "licenses" / "Stockfish"
        sf_notice.mkdir()
        for name in ("Copying.txt", "AUTHORS", "ORIGIN.json"):
            (sf_notice / name).write_text("notice fixture")
        for name in ("LICENSE", "COPYING", "THIRD_PARTY_NOTICES.md"):
            (output / name).write_text("license fixture")

    def make_archive(self, path, extra=(), missing=()):
        with zipfile.ZipFile(path, "w") as archive:
            for name in ("main.py", "LICENSE", "COPYING", "RELEASE.md", "requirements-release.txt", *extra):
                if name not in missing:
                    archive.writestr(f"praj-{VERSION}/" + name, "example")

    def test_valid_archive_and_checksum(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.zip"
            self.make_archive(path)
            self.assertEqual(inspect_source(path), 5)
            self.assertEqual(len(sha256(path)), 64)

    def test_user_zip_runs_after_one_extraction(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            self.make_candidate(output)
            bundle = output / "windows.zip"
            write_runtime_archive(output, bundle)
            names = inspect_runtime(bundle)
            self.assertIn(EXECUTABLE, names)
            self.assertEqual({name.split("/")[0] for name in names},
                             {EXECUTABLE, "README.txt", "engines", "licenses"})
            self.assertIn("engines/stockfish.exe", names)
            self.assertFalse(any(name.endswith(".zip") for name in names))
            with zipfile.ZipFile(bundle) as archive:
                self.assertIn(SOURCES_URL, archive.read("README.txt").decode())

    def test_runtime_archive_requires_the_assisted_backend(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "incomplete.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(EXECUTABLE, b"example")
                archive.writestr("README.txt", SOURCES_URL)
                archive.writestr("licenses/COPYING.txt", "GPL")
            with self.assertRaises(RuntimeError):
                inspect_runtime(path)

    def test_developer_download_contains_expanded_project_source(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            self.make_candidate(output)
            self.make_archive(output / f"praj-{VERSION}-source.zip")
            (output / "sources").mkdir()
            (output / "sources" / "dependency.tar.gz").write_bytes(b"source fixture")
            manifest = {"commit": "fixture", "sha256": {"bin/" + EXECUTABLE:
                        sha256(output / "bin" / EXECUTABLE)},
                        "dependency_sources": [{"filename": "dependency.tar.gz"}]}
            bundle = output / "sources.zip"
            write_developer_archive(output, bundle, manifest)
            self.assertEqual(verify_developer_archive(bundle)["commit"], "fixture")
            with zipfile.ZipFile(bundle) as archive:
                self.assertIn("engine/main.py", archive.namelist())
                self.assertFalse(any(name.endswith(".zip") for name in archive.namelist()))

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

    def test_refresh_manifest_does_not_hash_itself(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / "payload").write_bytes(b"content")
            write_manifest(output, {"version": VERSION})
            write_manifest(output, {"version": VERSION})
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertNotIn("manifest.json", manifest["sha256"])
            self.assertEqual(manifest["sha256"]["payload"], sha256(output / "payload"))

    def test_alpha_beta_smoke_on_legacy_windows_output_encoding(self):
        env = dict(os.environ, PYTHONIOENCODING="cp1252")
        result = subprocess.run([sys.executable, str(ROOT / "alpha_beta.py")],
                                capture_output=True, env=env, timeout=30)
        # This regression checks console compatibility. The separate alpha_beta.py
        # CI step gates its chess assertions and machine-dependent timing limits.
        diagnostics = (result.stdout + result.stderr).decode("cp1252", errors="replace")
        self.assertIn(b"Results:", result.stdout, diagnostics)
        self.assertNotIn(b"UnicodeEncodeError", result.stderr, diagnostics)

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
