# Third-party notices and release boundaries

This document is an initial release checklist, not a complete software bill of materials or legal opinion. Exact contents depend on the versions and build options used. Before distributing a binary, record its exact dependency versions, generate an SBOM, preserve each applicable license and notice, and review any corresponding-source obligations. This repository does not yet contain a reproducible binary-release recipe or a complete, version-specific SBOM.

## Runtime Python dependencies

- **python-chess** — The runtime declares `python-chess>=1.999,<2`. Upstream identifies python-chess as GPL-3.0-or-later: <https://github.com/niklasf/python-chess>. The range is not an exact lock, so determine the precise version included in each build. The PyInstaller executable bundles Python dependencies. This notice does not determine whether or how the complete engine may be distributed under the combination of licenses; obtain an appropriate licensing review or resolve the dependency/licensing design before release. Do not assume the root MIT file alone covers the combined application.
- **NumPy** — The runtime declares `numpy>=1.26,<3`. Consult the license and notices for the exact installed release: <https://github.com/numpy/numpy>.
- **PyTorch** — Optional training/test dependency declared in `requirements.txt`, not required by the lightweight runtime manifest. Consult the exact release's license and bundled notices if distributing a build that includes it: <https://github.com/pytorch/pytorch>.

## Separate programs and assets

- **Stockfish** — An optional external UCI engine, not tracked in this repository. The official project is GPLv3: <https://github.com/official-stockfish/Stockfish>. If a release bundles a Stockfish binary, include the matching license and comply with the source/corresponding-source requirements for that exact binary. A compatible system-installed Stockfish is a separate deployment choice.
- **Logo files** — `logo.png` and `stockfish_logo.png` are tracked in the repository, but their origin and licenses have not been verified. Permission to redistribute them has not been established here; verify provenance or replace/remove them before a public release.
- **Model weights** — No trained checkpoint or weight file is tracked or published. Therefore this repository is not currently an open-weight model release.

## Project license

The root `LICENSE` declares MIT terms for project-owned code. It does not relicense third-party components or assets, nor does this notice determine the license of a combined application. Confirm that the person/entity named in `LICENSE` owns or is authorized to license the project-owned code; separately verify logo rights and resolve the complete application's dependency-license obligations before making a public release claim.
