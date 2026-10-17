"""Prepare a source candidate and optionally a native binary; never publish it.

Use a clean committed checkout. Source archives honor .gitattributes export-ignore.
The binary is a candidate with the limitations documented in RELEASE.md.
"""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parent
VERSION = "2.1"
FORBIDDEN_SUFFIXES = {".exe", ".dll", ".pt", ".pth", ".onnx"}
FORBIDDEN_ASSETS = {"logo.png", "stockfish_logo.png", "desktop.ini"}


def git(*args, root=ROOT):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def clean_commit(root=ROOT):
    if git("status", "--porcelain", "--untracked-files=all", root=root):
        raise RuntimeError("Commit or preserve outstanding changes before preparing a release")
    return git("rev-parse", "HEAD", root=root)


def inspect_source(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        for name in names:
            item = Path(name)
            if (item.is_absolute() or ".." in item.parts
                    or item.suffix.lower() in FORBIDDEN_SUFFIXES
                    or item.name.lower() in FORBIDDEN_ASSETS
                    or ".git" in item.parts or "__pycache__" in item.parts):
                raise RuntimeError(f"Unexpected release archive entry: {name}")
        prefix = f"praj-{VERSION}/"
        for required in ("main.py", "LICENSE", "COPYING", "RELEASE.md", "requirements-release.txt"):
            if prefix + required not in names:
                raise RuntimeError(f"Missing source release file: {required}")
    return len(names)


def require_isolated_environment():
    config = Path(sys.prefix) / "pyvenv.cfg"
    if sys.prefix == sys.base_prefix or not config.is_file():
        raise RuntimeError("Binary builds require a fresh virtual environment")
    if "include-system-site-packages = false" not in config.read_text().lower():
        raise RuntimeError("Binary builds must not inherit system site-packages")
    if sys.version_info[:2] != (3, 13) or sys.platform != "win32":
        raise RuntimeError("The locked binary toolchain targets Windows CPython 3.13")
    for line in (ROOT / "requirements-release.txt").read_text().splitlines():
        requirement = line.split(";", 1)[0].strip()
        if not requirement or requirement.startswith("#"):
            continue
        name, version = requirement.split("==")
        if metadata.version(name) != version:
            raise RuntimeError(f"Build requires {requirement}")


def collect_notices(destination):
    components = []
    for dist in sorted(metadata.distributions(), key=lambda d: d.metadata["Name"].lower()):
        name = dist.metadata["Name"]
        # pip is an environment installer, not an input bundled by PyInstaller.
        if name.lower() == "pip":
            continue
        copied = []
        for entry in dist.files or []:
            lower = str(entry).lower()
            if any(word in lower for word in ("license", "copying", "notice", "authors")):
                relative = Path(str(entry))
                if relative.is_absolute() or ".." in relative.parts:
                    continue
                source = Path(dist.locate_file(entry))
                if source.is_file():
                    target = destination / name / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
                    copied.append(target.relative_to(destination).as_posix())
        components.append({"name": name, "version": dist.version, "notices": copied})
    python_license = Path(sys.base_prefix) / "LICENSE.txt"
    if not python_license.is_file():
        raise RuntimeError("Cannot locate the Python runtime license")
    target = destination / "Python" / "LICENSE.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(python_license, target)
    return components


def fetch_sources(components, destination):
    """Keep matching upstream source archives beside the distributed executable."""
    destination.mkdir(parents=True, exist_ok=True)
    sources = []
    for component in components:
        name, version = component["name"], component["version"]
        with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=60) as response:
            package = json.load(response)
        candidates = [item for item in package["urls"] if item["packagetype"] == "sdist"]
        if not candidates:
            raise RuntimeError(f"No corresponding source archive found for {name}=={version}")
        item = candidates[0]
        url = item["url"]
        if not url.startswith("https://files.pythonhosted.org/"):
            raise RuntimeError(f"Unexpected source host for {name}")
        filename = item["filename"]
        if Path(filename).name != filename:
            raise RuntimeError("Invalid source archive filename")
        target = destination / filename
        with urllib.request.urlopen(url, timeout=60) as response, target.open("wb") as output:
            shutil.copyfileobj(response, output)
        if sha256(target) != item["digests"]["sha256"]:
            raise RuntimeError(f"Source checksum mismatch for {name}")
        sources.append({"name": name, "version": version, "url": url,
                        "filename": filename, "sha256": item["digests"]["sha256"]})
    return sources


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_manifest(output, manifest):
    manifest["sha256"] = {
        path.relative_to(output).as_posix(): sha256(path)
        for path in sorted(output.rglob("*")) if path.is_file() and path.name != "manifest.json"
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def verify_manifest(output):
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    actual = {path.relative_to(output).as_posix() for path in output.rglob("*")
              if path.is_file() and path.name != "manifest.json"}
    if actual != set(manifest["sha256"]):
        raise RuntimeError("Release file inventory does not match manifest")
    for name, expected in manifest["sha256"].items():
        path = (output / name).resolve()
        if not path.is_relative_to(output.resolve()) or sha256(path) != expected:
            raise RuntimeError(f"Release checksum mismatch: {name}")
    return manifest


def protocol_check(binary, stockfish=None):
    env = dict(os.environ, PRAJ_ENGINE_EXE=str(binary), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env.pop("PRAJ_STOCKFISH_PATH", None)
    if stockfish:
        env["PRAJ_STOCKFISH_PATH"] = str(stockfish)
    result = subprocess.run([sys.executable, "-m", "unittest", "test_audit.AuditProtocol", "-v"],
                            cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    if result.returncode or "skipped" in result.stderr:
        raise RuntimeError("Packaged UCI checks failed:\n" + result.stdout + result.stderr)
    return result.stdout + result.stderr


def finalize_candidate(output, stockfish=None):
    output = output.resolve()
    if not output.is_relative_to((ROOT / "dist").resolve()):
        raise RuntimeError("Finalize a candidate directory inside this project's dist directory")
    manifest = verify_manifest(output)
    if manifest["commit"] != clean_commit():
        raise RuntimeError("Candidate must match the current clean source commit")
    binary = output / "bin" / "praj_mcts_2_1.exe"
    if not binary.is_file() or (binary.parent / "stockfish.exe").exists():
        raise RuntimeError("Candidate must contain the engine without a bundled Stockfish")
    if stockfish:
        stockfish = stockfish.resolve(strict=True)
    (output / "validation-fallback.txt").write_text(protocol_check(binary), encoding="utf-8")
    if stockfish:
        (output / "validation-stockfish.txt").write_text(protocol_check(binary, stockfish), encoding="utf-8")
    (output / "VALIDATION.md").write_text(
        "# Packaged-engine validation\n\n"
        "The fresh executable passed the complete UCI protocol suite without Stockfish.\n"
        + ("It also passed with an explicitly configured external Stockfish.\n" if stockfish else "")
        + "\nSee validation logs for individual checks, and manifest.json for build/source hashes.\n"
        "Stockfish, model weights, and unverified logos are excluded.\n", encoding="utf-8")
    write_manifest(output, manifest)
    verify_manifest(output)
    bundle = output.parent / "praj-2.1.0-rc.1-windows-x64.zip"
    with zipfile.ZipFile(bundle, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                archive.write(path, "praj-2.1/" + path.relative_to(output).as_posix())
    with tempfile.TemporaryDirectory(prefix="release-extract-", dir=ROOT / "dist") as folder:
        with zipfile.ZipFile(bundle) as archive:
            archive.extractall(folder)
        extracted = Path(folder) / "praj-2.1"
        verify_manifest(extracted)
        protocol_check(extracted / "bin" / "praj_mcts_2_1.exe")
    checksum = bundle.with_suffix(".zip.sha256")
    checksum.write_text(f"{sha256(bundle)}  {bundle.name}\n", encoding="ascii")
    print(bundle)
    print(checksum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", action="store_true", help="Also build a local Windows candidate")
    parser.add_argument("--finalize", type=Path, help="Validate and ZIP an existing candidate directory")
    parser.add_argument("--stockfish", type=Path, help="External Stockfish for optional assisted validation")
    args = parser.parse_args()
    if args.finalize:
        if args.binary:
            parser.error("--binary and --finalize are separate steps")
        finalize_candidate(args.finalize, args.stockfish)
        return
    commit = clean_commit()
    if args.binary:
        require_isolated_environment()
    (ROOT / "dist").mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=f"release-{VERSION}-", dir=ROOT / "dist"))
    source = output / f"praj-{VERSION}-source.zip"
    subprocess.run(["git", "-C", str(ROOT), "archive", "--format=zip",
                    f"--prefix=praj-{VERSION}/", f"--output={source}", commit], check=True)
    count = inspect_source(source)
    manifest = {"version": VERSION, "commit": commit, "source_entries": count,
                "python": sys.version, "platform": platform.platform(),
                "status": "release candidate; see RELEASE.md for limitations",
                "stockfish_bundled": False, "weights_bundled": False}
    if args.binary:
        work = Path(tempfile.mkdtemp(prefix="release-build-", dir=ROOT / "dist"))
        subprocess.run([sys.executable, "-m", "PyInstaller", "--clean", "--noconfirm",
                        "--onefile", "--noupx", "--name", "praj_mcts_2_1",
                        "--exclude-module", "torch", "--copy-metadata", "chess",
                        "--copy-metadata", "numpy", "--distpath", str(output / "bin"),
                        "--workpath", str(work / "work"), "--specpath", str(work),
                        str(ROOT / "main.py")], cwd=ROOT, check=True)
        manifest["build_components"] = collect_notices(output / "licenses")
        manifest["dependency_sources"] = fetch_sources(manifest["build_components"], output / "sources")
    for name in ("LICENSE", "COPYING", "THIRD_PARTY_NOTICES.md", "RELEASE.md"):
        shutil.copy2(ROOT / name, output / name)
    write_manifest(output, manifest)
    print(output)


if __name__ == "__main__":
    main()
