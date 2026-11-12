# Licensing

Zyth is free to download and use. The parts you build on are open source; the compiler is not.

## Open source: this repository

Everything in this repository is licensed under the Apache License 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
or the MIT License ([LICENSE-MIT](LICENSE-MIT)), at your option:

- the language specification (`ZYTH-SPEC.md`) and the reference for AI models (`docs/llms.txt`);
- the standard library written in Zyth (`stdlib/`);
- the type definitions for Python and npm packages (`registry/types/`);
- the package tools (`crates/zyth-pkg/`): the checksum log, its auditor and witness, the registry, and package content hashes;
- editor support (`editors/`);
- the examples, samples and showcases (`examples/`, `samples/`, `showcases/`, `packages/`).

Contributions to this repository are accepted under the same terms.

## Free to use, source closed: the compiler and runtimes

The `zyth` program (parser, checker, interpreter, compiler, tensor and GPU runtimes), the runtimes that
built programs carry (`zyth-rt`, `zyth.wasm`, `libzyth_native`) and the npm package
`@biodream-llc/zyth-runtime` are proprietary to BioDream LLC. They come with the Zyth Binary License,
included in every release: free for any use, including commercial; the programs you make with them are
yours to use, sell and share; the source code is confidential, and the binaries may not be modified or
reverse engineered.

The names "Zyth" and "BioDream" belong to BioDream LLC.
