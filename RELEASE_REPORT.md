# Release preparation report — 16 October 2026

## Findings and fixes

- The historical audit environment inherited system site-packages. Added a pinned build-only dependency manifest and an isolated-venv check; the new runtime build explicitly excludes PyTorch.
- There was no checked-in binary packaging recipe. Added `prepare_release.py`, a clean-commit source archive, native Windows build, copied third-party/Python notices, verified matching dependency source downloads, and a SHA-256 inventory.
- All 14 UCI tests were skipped without Stockfish, hiding failures in the supported static-fallback configuration. Removed that skip and added an explicit backend-path option for packaged-engine tests.
- Logo provenance was unresolved. Both unused logo files are excluded from release archives and binary bundles while their originals remain preserved in project history/files.
- The complete distribution's terms were unclear. Original code retains MIT; the combined application uses GPL-3.0-or-later, with the full GPL text and corresponding dependency sources supplied beside the binary.
- CI covered only Linux source execution. Added Windows packaging and packaged-UCI validation, while retaining the Linux neural/source suite.
- Updated the README to explain the build process, author/engine attribution, actual evaluator behavior, and explicit limitations.

## Validation

Fresh Windows x86-64 CPython 3.13.14 environment using `requirements-release.txt`: **59 tests run, 48 passed, 11 optional PyTorch tests skipped**. All 14 source UCI tests ran; none were skipped for missing Stockfish. Standalone alpha-beta self-test: **24/24 passed**. Four new packaging tests cover archive boundaries, required files, Git export rules, and dirty-checkout rejection.

The release asset's `VALIDATION.md` records the fresh binary's protocol results, bundle verification, and backend configurations. `manifest.json` records the exact source commit, build packages, matching dependency-source hashes, and file hashes. Remote CI results must be checked separately; a local pass does not claim that GitHub Actions has completed.

## Remaining product limitations

This is a release candidate. Absolute Elo is uncalibrated; long tournament validation and trained model weights are outside this release. The main UCI engine uses Stockfish advice when configured and static fallback otherwise. The Windows executable is unsigned, and the local native-binary validation does not certify other operating systems or CPU architectures.
