#!/usr/bin/env node
"use strict";

const path = require("path");

const root = path.resolve(__dirname, "../../..");
const {Dex} = require(path.join(root, "external/pokemon-showdown/dist/sim"));
const mapping = new Map();

function install(source, target, authority) {
  const previous = mapping.get(source);
  if (previous && previous.target !== target) {
    throw new Error(`conflicting public form mapping for ${source}`);
  }
  mapping.set(source, {target, authority});
}

for (const species of Dex.species.all()) {
  if (species.battleOnly) {
    const target = Array.isArray(species.battleOnly)
      ? species.battleOnly[0]
      : species.battleOnly;
    install(species.id, Dex.toID(target), "battleOnly");
  }
  for (const cosmetic of species.cosmeticFormes || []) {
    install(Dex.toID(cosmetic), species.id, "cosmeticFormes");
  }
}

function terminal(source) {
  const seen = new Set();
  let current = source;
  while (mapping.has(current)) {
    if (seen.has(current)) throw new Error(`public form mapping cycle at ${source}`);
    seen.add(current);
    current = mapping.get(current).target;
  }
  return current;
}

const rows = [...mapping.keys()].sort().map(source => ({
  source,
  target: terminal(source),
  authority: mapping.get(source).authority,
}));
process.stdout.write(JSON.stringify({
  schema: "metagross-showdown-public-form-contract/v1",
  mapping_count: rows.length,
  rows,
}) + "\n");

