// Starts `zyth lsp` for .zyth files.
const vscode = require("vscode");
const { LanguageClient } = require("vscode-languageclient/node");

let client;

function activate(context) {
  const command = vscode.workspace.getConfiguration("zyth").get("path") || "zyth";
  client = new LanguageClient(
    "zyth",
    "Zyth",
    { command, args: ["lsp"] },
    { documentSelector: [{ scheme: "file", language: "zyth" }] }
  );
  context.subscriptions.push(client);
  client.start();
}

function deactivate() {
  return client ? client.stop() : undefined;
}

module.exports = { activate, deactivate };
