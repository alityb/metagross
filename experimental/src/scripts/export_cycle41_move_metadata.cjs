#!/usr/bin/env node
"use strict";

const path = require("path");
const fs = require("fs");
const showdown = path.resolve(__dirname, "../../../external/pokemon-showdown");
const {Dex} = require(path.join(showdown, "dist/sim/dex"));
const ids = JSON.parse(fs.readFileSync(0, "utf8"));
if (!Array.isArray(ids) || ids.some(x => typeof x !== "string")) {
  throw new Error("expected a JSON string array");
}
const output = {};
for (const raw of ids) {
  const id = raw.replace(/-tera$/, "");
  const move = Dex.moves.get(id);
  output[raw] = {
    id: move.id,
    exists: move.exists,
    category: move.category,
    basePower: move.basePower,
    target: move.target,
    boosts: move.boosts || null,
    selfBoosts: (move.self && move.self.boosts) || null,
    sideCondition: move.sideCondition || null,
    slotCondition: move.slotCondition || null,
    weather: move.weather || null,
    terrain: move.terrain || null,
    pseudoWeather: move.pseudoWeather || null,
    status: move.status || null,
    volatileStatus: move.volatileStatus || null,
    heal: move.heal || null,
  };
}
process.stdout.write(JSON.stringify(output));
