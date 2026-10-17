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
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

from engine_info import VERSION, RELEASE_VERSION, EXECUTABLE, RELEASE_URL, SOURCES_NAME, SOURCES_URL

ROOT = Path(__file__).resolve().parent
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


def bundle_stockfish(output):
    """Ship an unmodified, checksum-pinned upstream binary and its exact sources."""
    lock = json.loads((ROOT / "stockfish.lock.json").read_text())
    cache = ROOT / "dist" / "stockfish-cache"
    cache.mkdir(parents=True, exist_ok=True)
    package = cache / f"stockfish-{lock['version']}-windows-x86-64.zip"
    if not package.exists() or sha256(package) != lock["archive_sha256"]:
        with urllib.request.urlopen(lock["url"], timeout=60) as response, package.open("wb") as target:
            shutil.copyfileobj(response, target)
    if sha256(package) != lock["archive_sha256"]:
        raise RuntimeError("Official Stockfish archive checksum mismatch")
    binary = output / "engines" / "stockfish.exe"
    source = output / "stockfish-source"
    binary.parent.mkdir(parents=True, exist_ok=True)
    source.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(package) as archive:
        with archive.open(lock["binary_entry"]) as original, binary.open("wb") as target:
            shutil.copyfileobj(original, target)
        for entry in archive.infolist():
            if entry.is_dir() or entry.filename.lower().endswith(".exe"):
                continue
            if not entry.filename.startswith(lock["source_prefix"]):
                raise RuntimeError("Unexpected Stockfish source entry")
            relative = Path(entry.filename.removeprefix(lock["source_prefix"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError("Invalid Stockfish source path")
            target = source / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(entry) as original, target.open("wb") as destination:
                shutil.copyfileobj(original, destination)
    # Export the exact embedded NNUE files so the corresponding sources can
    # rebuild without having to locate/download the network weights later.
    export = subprocess.run([str(binary)], input="export_net\nquit\n", text=True,
                            capture_output=True, cwd=source / "src", timeout=90,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if export.returncode:
        raise RuntimeError("Stockfish neural-network export failed: " + export.stderr)
    networks = re.findall(r'"(nn-[0-9a-f]{12}\.nnue)"',
                          (source / "src" / "evaluate.h").read_text())
    if len(set(networks)) != 2:
        raise RuntimeError("Expected both Stockfish neural networks")
    network_hashes = {}
    for name in networks:
        path = source / "src" / name
        if not path.exists() or not sha256(path).startswith(name[3:15]):
            raise RuntimeError("Exported Stockfish network checksum mismatch: " + name)
        network_hashes[name] = sha256(path)
    notice = output / "licenses" / "Stockfish"
    notice.mkdir(parents=True, exist_ok=True)
    for name in ("Copying.txt", "AUTHORS", "README.md"):
        shutil.copy2(source / name, notice / name)
    origin = dict(lock, binary_sha256=sha256(binary), embedded_networks=network_hashes)
    (notice / "ORIGIN.json").write_text(json.dumps(origin, indent=2) + "\n", encoding="utf-8")
    return origin


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


def protocol_check(binary, stockfish=None, bundled=False):
    env = dict(os.environ, PRAJ_ENGINE_EXE=str(binary), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env.pop("PRAJ_STOCKFISH_PATH", None)
    if stockfish:
        env["PRAJ_STOCKFISH_PATH"] = str(stockfish)
    tests = ["test_audit.AuditProtocol"]
    if bundled:
        tests.append("test_bundle")
    result = subprocess.run([sys.executable, "-m", "unittest", *tests, "-v"],
                            cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    if result.returncode or "skipped" in result.stderr:
        raise RuntimeError("Packaged UCI checks failed:\n" + result.stdout + result.stderr)
    return result.stdout + result.stderr


def quick_start():
    return (
        f"Praj {VERSION} - Windows x64\nBy Prabhas\n\n"
        f"Extract this ZIP once. The engine is {EXECUTABLE} in this folder.\n"
        "You do not need to extract or download any source archive to run it.\n\n"
        "PLAY WITH A CHESS BOARD\n"
        f"Add {EXECUTABLE} as a UCI engine in your preferred chess GUI.\n\n"
        "USE THE TERMINAL\n"
        f"Double-click {EXECUTABLE}, or run it in a terminal. Like Stockfish,\n"
        "this is a plain-text UCI console; it does not display a graphical board.\n"
        "Type help to see commands. For a quick search, enter these one at a time:\n"
        "  uci\n  isready\n  position startpos\n  go movetime 1000\n"
        "Wait for bestmove, then type quit to exit. Type d to display the board.\n\n"
        "READY-TO-PLAY ASSISTANCE\n"
        "Official Stockfish 18 and both of its neural networks are included.\n"
        "Praj automatically starts engines\\stockfish.exe; no separate install,\n"
        "download, network connection, or StockfishPath command is required.\n"
        "Keep the entire extracted folder together, including engines and licenses.\n"
        "Praj is a Stockfish-assisted MCTS engine; it does not claim Stockfish's Elo.\n\n"
        "LICENSES AND SOURCE\n"
        "Keep the licenses folder when redistributing the engine. The combined\n"
        "application is GPL-3.0-or-later; original project code retains MIT.\n"
        "Complete matching project/dependency sources are a separate free download:\n"
        f"{SOURCES_URL}\n\nRelease notes and checksums:\n{RELEASE_URL}\n"
    )


def write_runtime_archive(output, bundle):
    """The user download contains just the engine, quick start, and notices."""
    with zipfile.ZipFile(bundle, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(output / "bin" / EXECUTABLE, EXECUTABLE)
        archive.write(output / "engines" / "stockfish.exe", "engines/stockfish.exe")
        archive.writestr("README.txt", quick_start())
        for path in sorted((output / "licenses").rglob("*")):
            if path.is_file():
                archive.write(path, "licenses/" + path.relative_to(output / "licenses").as_posix())
        for name, target in (("LICENSE", "PROJECT-MIT.txt"), ("COPYING", "COPYING.txt"),
                             ("THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.txt")):
            archive.write(output / name, "licenses/" + target)


def inspect_runtime(bundle):
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
        if {name.split("/")[0] for name in names} != {EXECUTABLE, "README.txt", "engines", "licenses"}:
            raise RuntimeError("User ZIP must contain the engine, bundled backend, README, and licenses")
        for name in names:
            if (".." in Path(name).parts or Path(name).is_absolute()
                    or name.lower().endswith((".zip", ".tar.gz", ".tgz"))):
                raise RuntimeError(f"Unexpected user ZIP entry: {name}")
        for name in (EXECUTABLE, "README.txt", "engines/stockfish.exe", "licenses/COPYING.txt",
                     "licenses/PROJECT-MIT.txt", "licenses/Stockfish/ORIGIN.json",
                     "licenses/Stockfish/Copying.txt", "licenses/Stockfish/AUTHORS"):
            if name not in names:
                raise RuntimeError(f"Missing runtime file: {name}")
        if SOURCES_URL not in archive.read("README.txt").decode("utf-8"):
            raise RuntimeError("User ZIP must point to the matching corresponding-source download")
    return names


def write_developer_archive(output, bundle, manifest):
    """Expand project source into the developer download; no nested project ZIP."""
    hashes = {}
    with zipfile.ZipFile(bundle, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        def add_file(path, name):
            archive.write(path, name)
            hashes[name] = sha256(path)

        with zipfile.ZipFile(output / f"praj-{VERSION}-source.zip") as source:
            for entry in source.infolist():
                if entry.is_dir():
                    continue
                name = "engine/" + entry.filename.removeprefix(f"praj-{VERSION}/")
                data = source.read(entry)
                archive.writestr(name, data)
                hashes[name] = hashlib.sha256(data).hexdigest()
        for folder, target in (("sources", "third-party"), ("stockfish-source", "third-party/stockfish"),
                               ("licenses", "licenses")):
            for path in sorted((output / folder).rglob("*")):
                if path.is_file():
                    add_file(path, target + "/" + path.relative_to(output / folder).as_posix())
        for name in ("validation-fallback.txt", "validation-stockfish.txt", "validation-bundled.txt", "VALIDATION.md"):
            if (output / name).exists():
                add_file(output / name, "build-info/" + name)
        source_manifest = dict(manifest, sha256=hashes,
                               binary_sha256=manifest["sha256"]["bin/" + EXECUTABLE])
        archive.writestr("build-info/manifest.json", json.dumps(source_manifest, indent=2) + "\n")


def verify_developer_archive(bundle):
    with zipfile.ZipFile(bundle) as archive:
        manifest = json.loads(archive.read("build-info/manifest.json"))
        actual = {name for name in archive.namelist() if name != "build-info/manifest.json"}
        if actual != set(manifest["sha256"]):
            raise RuntimeError("Corresponding-source inventory does not match")
        for name, expected in manifest["sha256"].items():
            with archive.open(name) as source:
                actual_hash = hashlib.file_digest(source, "sha256").hexdigest()
            if actual_hash != expected:
                raise RuntimeError(f"Corresponding-source checksum mismatch: {name}")
        for name in ("engine/main.py", "engine/COPYING"):
            if name not in actual:
                raise RuntimeError(f"Missing corresponding source file: {name}")
        for component in manifest.get("dependency_sources", []):
            if "third-party/" + component["filename"] not in actual:
                raise RuntimeError("Missing matching dependency source")
        if manifest.get("stockfish_bundled"):
            for name, expected in manifest["stockfish"]["embedded_networks"].items():
                target = "third-party/stockfish/src/" + name
                if target not in actual or manifest["sha256"][target] != expected:
                    raise RuntimeError("Missing matching Stockfish neural-network source")
    return manifest


def finalize_candidate(output, stockfish=None):
    output = output.resolve()
    if not output.is_relative_to((ROOT / "dist").resolve()):
        raise RuntimeError("Finalize a candidate directory inside this project's dist directory")
    manifest = verify_manifest(output)
    if manifest["commit"] != clean_commit():
        raise RuntimeError("Candidate must match the current clean source commit")
    binary = output / "bin" / EXECUTABLE
    if not binary.is_file() or not (output / "engines" / "stockfish.exe").is_file():
        raise RuntimeError("Candidate must include both Praj and its bundled Stockfish")
    stockfish = (stockfish or output / "engines" / "stockfish.exe").resolve(strict=True)
    (output / "validation-fallback.txt").write_text(protocol_check(binary), encoding="utf-8")
    if stockfish:
        (output / "validation-stockfish.txt").write_text(protocol_check(binary, stockfish), encoding="utf-8")
    (output / "VALIDATION.md").write_text(
        "# Packaged-engine validation\n\n"
        "The fresh executable passed the complete UCI protocol suite without Stockfish.\n"
        + ("It also passed with an explicitly configured external Stockfish.\n" if stockfish else "")
        + "\nSee validation logs for individual checks, and manifest.json for build/source hashes.\n"
        "Official Stockfish and its embedded neural networks are included; no separately trained Praj model is supplied.\n",
        encoding="utf-8")
    write_manifest(output, manifest)
    verify_manifest(output)
    bundle = output.parent / f"praj-{RELEASE_VERSION}-windows-x64.zip"
    sources = output.parent / SOURCES_NAME
    write_runtime_archive(output, bundle)
    inspect_runtime(bundle)
    with tempfile.TemporaryDirectory(prefix="release-extract-", dir=ROOT / "dist") as folder:
        with zipfile.ZipFile(bundle) as archive:
            archive.extractall(folder)
        extracted = Path(folder)
        if sha256(extracted / EXECUTABLE) != sha256(binary):
            raise RuntimeError("Extracted executable checksum does not match")
        if sha256(extracted / "engines" / "stockfish.exe") != manifest["stockfish"]["binary_sha256"]:
            raise RuntimeError("Extracted bundled evaluator checksum does not match")
        bundled_log = protocol_check(extracted / EXECUTABLE, bundled=True)
        (output / "validation-bundled.txt").write_text(bundled_log, encoding="utf-8")
    # Include the result of the real extracted-package test in the developer asset.
    # The runtime archive stays untouched after its executable was verified.
    write_manifest(output, manifest)
    write_developer_archive(output, sources, manifest)
    verify_developer_archive(sources)
    checksum = output.parent / f"praj-{RELEASE_VERSION}-SHA256SUMS.txt"
    checksum.write_text("".join(f"{sha256(path)}  {path.name}\n" for path in (bundle, sources)), encoding="ascii")
    print(bundle)
    print(sources)
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
                "stockfish_bundled": bool(args.binary), "praj_weights_bundled": False}
    if args.binary:
        work = Path(tempfile.mkdtemp(prefix="release-build-", dir=ROOT / "dist"))
        subprocess.run([sys.executable, "-m", "PyInstaller", "--clean", "--noconfirm",
                        "--onefile", "--noupx", "--name", "praj",
                        "--exclude-module", "torch", "--copy-metadata", "chess",
                        "--copy-metadata", "numpy", "--distpath", str(output / "bin"),
                        "--workpath", str(work / "work"), "--specpath", str(work),
                        str(ROOT / "main.py")], cwd=ROOT, check=True)
        manifest["build_components"] = collect_notices(output / "licenses")
        manifest["dependency_sources"] = fetch_sources(manifest["build_components"], output / "sources")
        manifest["stockfish"] = bundle_stockfish(output)
    for name in ("LICENSE", "COPYING", "THIRD_PARTY_NOTICES.md", "RELEASE.md"):
        shutil.copy2(ROOT / name, output / name)
    write_manifest(output, manifest)
    print(output)


if __name__ == "__main__":
    main()
