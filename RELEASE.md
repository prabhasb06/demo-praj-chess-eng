# Praj 2.1 release candidate

Praj is the engine name; Prabhas is the project author. Development used AI-generated code. The Windows runtime combines MCTS with bundled Stockfish 18 and its neural networks. It does not load an independently trained Praj checkpoint.

## Distribution terms

The project's original code retains the MIT grant in `LICENSE`. The complete application incorporating GPL-3.0-or-later python-chess is distributed under GPL-3.0-or-later; preserve both the MIT notice and `COPYING`. Dependencies retain their own notices. This is compatible distribution of the combined application, not revocation of the existing MIT grant. See <https://www.gnu.org/licenses/gpl-faq.html#AllCompatibility> and `THIRD_PARTY_NOTICES.md`.

Source archives omit the unverified logo files. The Windows package includes an unmodified official Stockfish 18 baseline x64 binary and its embedded NNUE networks, pinned in `stockfish.lock.json`. No independently trained Praj checkpoint is included.

## Download layout

The Windows ZIP extracts directly to `praj.exe`, `README.txt`, `engines/`, and `licenses/`. Open `praj.exe` or add it to a UCI chess GUI. It resolves `engines/stockfish.exe` relative to its own location even when the GUI uses a different working directory. Stockfish and its neural networks are ready automatically, without an installer, administrator privileges, a first-run download, or manual path configuration. Keep all extracted files together. The console provides `help` and `d`.

Complete matching source is offered as `praj-2.1.0-rc.3-sources.zip` on the same release page, linked from the Windows README. It includes project code in `engine/`, dependency sources in `third-party/`, the exact Stockfish source and both exported neural networks in `third-party/stockfish/`, notices in `licenses/`, and build/test records in `build-info/`. Stockfish's included Makefile can use those network files without downloading them again. These developer sources are unnecessary for playing; retain public source access and notices when redistributing.

## Build and validate on Windows x86-64

Use CPython 3.13 (the validated local runtime is 3.13.14), Git, and a clean committed checkout. Create an isolated environment; do not use `--system-site-packages`:

```powershell
python -m venv .venv-release
.\.venv-release\Scripts\python.exe -m pip install -r requirements-release.txt
.\.venv-release\Scripts\python.exe -m unittest tests test_audit test_release -v
.\.venv-release\Scripts\python.exe alpha_beta.py
.\.venv-release\Scripts\python.exe prepare_release.py --binary
```

The last command prints a unique output directory under `dist`. It creates the source archive from the exact Git commit, builds the console executable, copies Python/dependency notices, downloads matching dependency source archives from PyPI, verifies their published SHA-256 digests, and records all output hashes and build versions in `manifest.json`. Network access is required for source downloads. Existing artifacts are preserved. This is a repeatable recipe, not a claim of byte-identical executables across machines.

The builder retrieves the official archive recorded in `stockfish.lock.json`, checks its full SHA-256, and leaves the upstream binary unmodified. It copies Stockfish's GPL license, authors, and README; preserves exact source; exports both embedded NNUE files; and verifies each network's documented SHA-256 prefix. `licenses/Stockfish/ORIGIN.json` records full binary/network hashes and upstream provenance. The baseline x64 build does not require AVX2. Assisted-mode strength is not a calibrated Praj Elo rating. Static fallback remains available only when the backend is missing or fails.

Finalize the printed candidate directory before publishing it. This checks every hash, tests fallback and the pinned backend, creates runtime/source ZIPs and checksums, verifies the source/network inventory, and tests automatic assisted play from an extracted copy launched in an unrelated directory:

```powershell
.\.venv-release\Scripts\python.exe prepare_release.py --finalize dist\release-2.1-EXAMPLE --stockfish C:\path\to\stockfish.exe
```

For individual protocol checks, replace the example paths with the printed output directory and your Stockfish path:

```powershell
$env:PRAJ_ENGINE_EXE = 'C:\path\to\extracted\praj.exe'
.\.venv-release\Scripts\python.exe -m unittest test_audit.AuditProtocol -v
$env:PRAJ_STOCKFISH_PATH = 'C:\path\to\stockfish.exe'
.\.venv-release\Scripts\python.exe -m unittest test_audit.AuditProtocol -v
Remove-Item Env:PRAJ_ENGINE_EXE, Env:PRAJ_STOCKFISH_PATH
```

`python prepare_release.py` creates only the source candidate using the standard library on Python 3.11 or later. The Git archive export rules exclude unverified artwork. The script refuses dirty checkouts and rejects unexpected executables, checkpoints, and unverified logos in the source archive.

## Publication and limitations

Publish as **Praj 2.1 RC3 - ready-to-run Stockfish assistance**. Upload the Windows ZIP, complete sources ZIP, and shared checksum file. Keep the corresponding-source link beside the Windows download. Finalization checks raw fallback mode, assisted mode, and automatic bundled assistance from an extracted copy launched in an unrelated directory. The runtime remains simple; the optional developer download includes full dependency/Stockfish sources, exact NNUE data, license notices, provenance, and test records.

There is no calibrated absolute Elo, trained-weight release, long tournament run, or guarantee of error-free play. Current local release verification covers Windows x86-64; Linux source tests run separately in CI. The Windows binary is unsigned. The neural training tests require optional PyTorch and are skipped in the lightweight build environment; they are separate from binary-runtime validation. See `AUDIT_REPORT.md` for historical audit results and `RELEASE_REPORT.md` for this release's checks.
