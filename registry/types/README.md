# The zyth-types registry

Type definitions (`.zyi`, ZYTH-SPEC.md 14.6) for Python and npm packages, one file per
package and minor version: `<python|npm>/<package>/<major>.<minor>.zyi`, where `<package>` is
the name `zyth add` takes (the distribution name for Python: `pyyaml`, whose module is
`yaml`). `zyth add py:<package>` and `zyth add npm:<package>` fetch the file for the
installed version (the same major.minor, else the newest earlier one with the same major
version) into the project's `zyth-types/`, after checking its SHA-256 against the index and
that it parses; `zyth types fetch` does it for packages already installed. A project's own
`zyth-types/` files are never replaced.

Contributing: start from `zyth types gen py:<package>` or `zyth types gen npm:<package>`, add
tensor shapes and tighter types by hand, check a program that uses them with `zyth check`,
and send the file. `scripts/types-registry.py` builds the site (`index.json` with each
file's checksum) and `scripts/types-registry.py --check` parses every file.

Served at https://types.zyth.io (override with `ZYTH_TYPES_URL`; `off` stops fetching).
