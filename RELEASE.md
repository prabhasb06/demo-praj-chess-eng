# Praj 2.1 release candidate

Praj is the engine name; Prabhas is the project author. Development used AI-generated code. The runtime combines MCTS with an optional, separately installed Stockfish evaluator. The current UCI entry point does not load a trained neural checkpoint.

## Distribution terms

The project's original code retains the MIT grant in `LICENSE`. The complete application incorporating GPL-3.0-or-later python-chess is distributed under GPL-3.0-or-later; preserve both the MIT notice and `COPYING`. Dependencies retain their own notices. This is compatible distribution of the combined application, not revocation of the existing MIT grant. See <https://www.gnu.org/licenses/gpl-faq.html#AllCompatibility> and `THIRD_PARTY_NOTICES.md`.

Source archives omit `logo.png` and `stockfish_logo.png` because their provenance has not been established. Neither image is used by the engine. Stockfish and model weights are not included in the release assets.

## Download layout

The Windows ZIP extracts directly to three items: `praj.exe`, `README.txt`, and `licenses/`. There are no nested ZIP files or build/source folders in the runnable download. Open `praj.exe` for the plain UCI console, or add it to a chess GUI. Enter `help` for commands and `d` for an ASCII board.

Complete matching source is offered as a separate `praj-2.1.0-rc.2-sources.zip` on the same release page, with a direct link in the Windows README. It contains expanded project files in `engine/`, exact dependency source archives in `third-party/`, notices in `licenses/`, and manifests/test logs in `build-info/`. Downloading or extracting it is unnecessary for playing. Preserve the source download alongside the binary release, and retain the licenses when redistributing the executable.

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

The bundle contains no Stockfish executable. In a chess GUI, set the engine's `StockfishPath` to a separately obtained native Stockfish executable for assisted strength. Without it, the static evaluator provides legal play with substantially reduced, uncalibrated strength. Obtain Stockfish from <https://stockfishchess.org/download/> and retain its license/source information if redistributing it separately.

Finalize the printed candidate directory before publishing it. This verifies every hash, tests the packaged executable without Stockfish and optionally with a known external Stockfish, creates separate runtime/source ZIPs plus checksums, checks the developer-source inventory, and retests the extracted runtime executable:

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

Publish as **Praj 2.1 RC2 - simpler Windows download**. Upload the Windows ZIP, the separate complete sources ZIP, and the shared SHA-256 checksum file. Put a clear corresponding-source link beside the Windows download and preserve that access. Keep the public tag pointing to the exact source commit in the developer manifest. The full original MIT notice, GPL text, and dependency notices remain in the Windows ZIP's `licenses/` folder.

There is no calibrated absolute Elo, trained-weight release, long tournament run, or guarantee of error-free play. Current local release verification covers Windows x86-64; Linux source tests run separately in CI. The Windows binary is unsigned. The neural training tests require optional PyTorch and are skipped in the lightweight build environment; they are separate from binary-runtime validation. See `AUDIT_REPORT.md` for historical audit results and `RELEASE_REPORT.md` for this release's checks.
