#!/usr/bin/env node
"use strict";

const child = require("child_process");
const path = require("path");

const root = path.resolve(__dirname, "../..");
const commit = "f8ac14003a5f27e1bdc8d8c59608a773c1cb96e5";
const showdown = path.join(root, "experimental/cache/cycle8_showdown/f8ac1400");
const actual = child.execFileSync("git", ["-C", showdown, "rev-parse", "HEAD"], {
  encoding: "utf8",
}).trim();
if (actual !== commit) {
  throw new Error("pinned Showdown pressure contract commit mismatch");
}
const {Moves} = require(path.join(showdown, "dist/data/moves.js"));
const rows = Object.entries(Moves).map(([id, move]) => ({
  id,
  target: move.target,
  mustpressure: Boolean(move.flags && move.flags.mustpressure),
})).sort((a, b) => a.id.localeCompare(b.id));
process.stdout.write(JSON.stringify({
  schema: "metagross-showdown-pressure-target-contract/v1",
  showdown_commit: commit,
  rows,
  row_count: rows.length,
}) + "\n");
