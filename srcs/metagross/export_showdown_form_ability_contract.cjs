#!/usr/bin/env node
"use strict";

// Source-pinned mechanical contract: an exact public target form implies its
// current ability only when Showdown declares exactly one ability for that form.
const path = require("path");
const root = path.resolve(__dirname, "../..");
const {Dex} = require(path.join(root, "external/pokemon-showdown/dist/sim"));

const rows = [];
for (const species of Dex.species.all()) {
  const exactSpecies = Dex.toID(species.name);
  const baseSpecies = Dex.toID(species.baseSpecies);
  const abilities = [...new Set(
    Object.values(species.abilities || {}).map(Dex.toID).filter(Boolean)
  )].sort();
  if (abilities.length !== 1) continue;
  rows.push({
    exact_species: exactSpecies,
    base_species: baseSpecies,
    current_ability: abilities[0],
    battle_only: Boolean(species.battleOnly),
    is_form: exactSpecies !== baseSpecies,
    authority: "showdown-exact-species-unique-ability",
  });
}
rows.sort((left, right) => left.exact_species.localeCompare(right.exact_species));
process.stdout.write(JSON.stringify({
  schema: "metagross-showdown-form-ability-contract/v1",
  row_count: rows.length,
  showdown_commit: require("child_process").execFileSync(
    "git", ["-C", path.join(root, "external/pokemon-showdown"), "rev-parse", "HEAD"],
    {encoding: "utf8"}
  ).trim(),
  rows,
}) + "\n");
