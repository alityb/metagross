#!/usr/bin/env node
"use strict";

const path = require("path");
const root = path.resolve(__dirname, "../..");
const {Dex} = require(path.join(root, "external/pokemon-showdown/dist/sim"));
const direct = new Map();

function add(source, target, authority) {
  const previous = direct.get(source);
  if (previous && previous.target !== target) {
    throw new Error(`conflicting public form mapping for ${source}`);
  }
  direct.set(source, {target, authority});
}

for (const species of Dex.species.all()) {
  if (species.battleOnly) {
    const target = Array.isArray(species.battleOnly)
      ? species.battleOnly[0]
      : species.battleOnly;
    add(species.id, Dex.toID(target), "battleOnly");
  }
  for (const cosmetic of species.cosmeticFormes || []) {
    add(Dex.toID(cosmetic), species.id, "cosmeticFormes");
  }
}

function terminal(source) {
  const seen = new Set();
  let current = source;
  while (direct.has(current)) {
    if (seen.has(current)) throw new Error(`public form mapping cycle at ${source}`);
    seen.add(current);
    current = direct.get(current).target;
  }
  return current;
}

const rows = [...direct].sort(([left], [right]) => left.localeCompare(right)).map(
  ([source, row]) => ({source, target: terminal(source), authority: row.authority})
);
process.stdout.write(JSON.stringify({
  schema: "metagross-showdown-public-form-contract/v1",
  mapping_count: rows.length,
  rows,
}) + "\n");

