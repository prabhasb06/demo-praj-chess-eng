# Third-party notices and release boundaries

The original project code retains the MIT grant in `LICENSE`; the complete application incorporating GPL-licensed python-chess is distributed under GPL-3.0-or-later, with its full license in `COPYING`. Release candidates prepared with `prepare_release.py --binary` include an exact component inventory, copied package/Python notices, matching dependency source archives, and SHA-256 hashes. This inventory is a build manifest rather than a standardized SPDX/CycloneDX SBOM. See `RELEASE.md` for the build recipe and limitations.

## Runtime Python dependencies

- **python-chess / chess** — GPL-3.0-or-later: <https://github.com/niklasf/python-chess>. The release lock specifies the `python-chess` 1.999 metapackage and `chess` 1.11.2 implementation. Their exact upstream source archives accompany the binary. The application as a whole is distributed under GPL-3.0-or-later while preserving the MIT notice for original project code.
- **NumPy** — The runtime declares `numpy>=1.26,<3`. Consult the license and notices for the exact installed release: <https://github.com/numpy/numpy>.
- **PyTorch** — Optional training/test dependency declared in `requirements.txt`, not required by the lightweight runtime manifest. Consult the exact release's license and bundled notices if distributing a build that includes it: <https://github.com/pytorch/pytorch>.
- **Python / PyInstaller** — The Windows bundle includes the Python runtime and an unmodified PyInstaller bootloader. The build copies installed Python/package notices into `licenses/` and records the Python version and build-tool versions. PyInstaller's bootloader exception is documented at <https://pyinstaller.org/en/stable/license.html>. PyTorch is explicitly excluded from the packaged runtime.

## Separate programs and assets

- **Stockfish** — An optional external UCI engine, not tracked in this repository. The official project is GPLv3: <https://github.com/official-stockfish/Stockfish>. If a release bundles a Stockfish binary, include the matching license and comply with the source/corresponding-source requirements for that exact binary. A compatible system-installed Stockfish is a separate deployment choice.
- **Logo files** — `logo.png` and `stockfish_logo.png` have unverified provenance and remain in historical project files. They are unused by the engine and excluded from source release archives through `.gitattributes`; the binary recipe does not include them. They are not covered by this release's licensing claims.
- **Model weights** — No trained checkpoint or weight file is tracked or published. Therefore this repository is not currently an open-weight model release.

## Project license

Prabhas identified himself as the creator of Praj and described development as AI-generated. The original code retains its existing MIT grant, while distribution of the complete application uses GPL-3.0-or-later. This attribution records the creator's statement; it is not an independent determination of copyright protection for AI-generated material. Existing third-party rights and notices remain applicable.
