# Zyth for VS Code

Syntax highlighting plus diagnostics, inline tensor shapes (`▸ tensor<f32, [64, 512]>`), hover, go to definition, find references, and quick fixes from `zyth lsp`.

```bash
cd editors/vscode
npm install            # installs vscode-languageclient
code --extensionDevelopmentPath="$PWD"   # or package with `npx @vscode/vsce package`
```

Set `zyth.path` if `zyth` is not on your PATH. Any LSP-capable editor works the same way: run `zyth lsp` over stdio for `.zyth` files.
