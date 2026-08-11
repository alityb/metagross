#!/usr/bin/env node
'use strict';

const crypto = require('crypto');
const path = require('path');

function usage() {
  throw new Error('usage: generate_mirrored_randbats_pair.cjs FORMAT TEAM1_SEED TEAM2_SEED');
}

function sha256(value) {
  return crypto.createHash('sha256').update(value, 'utf8').digest('hex');
}

function main() {
  if (process.argv.length !== 5) usage();
  const [, , formatID, team1Seed, team2Seed] = process.argv;
  const root = path.resolve(__dirname, '../../..');
  const {Teams} = require(path.join(root, 'external/pokemon-showdown/dist/sim'));
  const {Dex} = require(path.join(root, 'external/pokemon-showdown/dist/sim/dex'));
  const format = Dex.formats.get(formatID);
  if (!format.exists || !format.team) throw new Error(`unknown random format: ${formatID}`);

  const team1 = Teams.pack(Teams.getGenerator(format, team1Seed).getTeam());
  const team2 = Teams.pack(Teams.getGenerator(format, team2Seed).getTeam());
  process.stdout.write(JSON.stringify({
    format: format.id,
    team_1_seed: team1Seed,
    team_2_seed: team2Seed,
    team_1_packed: team1,
    team_2_packed: team2,
    team_1_sha256: sha256(team1),
    team_2_sha256: sha256(team2),
  }));
}

main();
