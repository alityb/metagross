'use strict';

const fs = require('fs');
const path = require('path');
const readline = require('readline');

function parseArgs(argv) {
  const args = {
    format: 'gen9randombattle',
    samples: 16,
    maxTeams: 20000,
    maxMillis: 250,
    server: false,
    showdownDir: path.resolve(__dirname, '..', 'external', 'pokemon-showdown'),
  };
  for (let i = 2; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--format') args.format = argv[++i];
    else if (arg === '--samples') args.samples = Number.parseInt(argv[++i], 10);
    else if (arg === '--max-teams') args.maxTeams = Number.parseInt(argv[++i], 10);
    else if (arg === '--max-ms') args.maxMillis = Number.parseInt(argv[++i], 10);
    else if (arg === '--server') args.server = true;
    else if (arg === '--showdown-dir') args.showdownDir = argv[++i];
    else throw new Error(`Unknown argument: ${arg}`);
  }
  if (!Number.isInteger(args.samples) || args.samples <= 0) throw new Error('--samples must be positive');
  if (!Number.isInteger(args.maxTeams) || args.maxTeams <= 0) throw new Error('--max-teams must be positive');
  if (!Number.isInteger(args.maxMillis) || args.maxMillis <= 0) throw new Error('--max-ms must be positive');
  return args;
}

function toID(value) {
  return String(value || '').toLowerCase().replace(/[^a-z0-9]+/g, '');
}

function slimSet(set) {
  return {
    species: set.species,
    speciesId: set.speciesId,
    level: set.level,
    moves: set.moves,
    ability: set.ability,
    item: set.item || '',
    teraType: set.teraType || '',
    evs: set.evs || {},
    role: set.role || '',
  };
}

function normalizeConstraint(raw) {
  return {
    speciesKeys: new Set((raw.speciesKeys || []).map(toID).filter(Boolean)),
    moves: (raw.moves || []).map(toID).filter(Boolean),
    item: raw.item ? toID(raw.item) : null,
    ability: raw.ability ? toID(raw.ability) : null,
    teraType: raw.teraType ? toID(raw.teraType) : null,
  };
}

function setMatchesConstraint(set, constraint) {
  const speciesId = toID(set.speciesId || set.species);
  if (!constraint.speciesKeys.has(speciesId)) return false;

  const moves = new Set((set.moves || []).map(toID));
  for (const move of constraint.moves) {
    if (!moves.has(move)) return false;
  }

  if (constraint.item && toID(set.item) !== constraint.item) return false;
  if (constraint.ability && toID(set.ability) !== constraint.ability) return false;
  if (constraint.teraType && toID(set.teraType) !== constraint.teraType) return false;
  return true;
}

function teamMatchesConstraints(team, constraints) {
  const used = new Set();
  for (const constraint of constraints) {
    let matched = false;
    for (let i = 0; i < team.length; i++) {
      if (used.has(i)) continue;
      if (!setMatchesConstraint(team[i], constraint)) continue;
      used.add(i);
      matched = true;
      break;
    }
    if (!matched) return false;
  }
  return true;
}

function makeSampler(args) {
  const simPath = path.join(args.showdownDir, 'dist', 'sim');
  const dexPath = path.join(args.showdownDir, 'dist', 'sim', 'dex');
  const {Teams} = require(simPath);
  const {Dex} = require(dexPath);
  const format = Dex.formats.get(args.format);

  // Warm up dynamic imports.
  Teams.getGenerator(format).getTeam();

  return function sample(input) {
    const constraints = (input.constraints || []).map(normalizeConstraint);
    const targetSamples = input.samples || args.samples;
    const maxTeams = input.maxTeams || args.maxTeams;
    const maxMillis = input.maxMillis || args.maxMillis;
    const deadline = Date.now() + maxMillis;

    const accepted = [];
    let generated = 0;
    while (accepted.length < targetSamples && generated < maxTeams && Date.now() < deadline) {
      generated += 1;
      const team = Teams.getGenerator(format).getTeam();
      if (teamMatchesConstraints(team, constraints)) accepted.push(team.map(slimSet));
    }

    return {
      format: args.format,
      targetSamples,
      maxTeams,
      maxMillis,
      generated,
      accepted: accepted.length,
      teams: accepted,
    };
  };
}

function main() {
  const args = parseArgs(process.argv);
  const sample = makeSampler(args);

  if (args.server) {
    const rl = readline.createInterface({input: process.stdin, crlfDelay: Infinity});
    rl.on('line', line => {
      try {
        const input = JSON.parse(line || '{}');
        process.stdout.write(`${JSON.stringify(sample(input))}\n`);
      } catch (error) {
        process.stdout.write(`${JSON.stringify({error: String(error && error.stack || error)})}\n`);
      }
    });
    return;
  }

  const input = JSON.parse(fs.readFileSync(0, 'utf8') || '{}');
  process.stdout.write(JSON.stringify(sample(input)));
}

main();
