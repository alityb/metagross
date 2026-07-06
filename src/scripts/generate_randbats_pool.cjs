'use strict';

const fs = require('fs');
const path = require('path');

function parseArgs(argv) {
  const args = {
    format: 'gen9randombattle',
    n: 10000,
    output: null,
    showdownDir: path.resolve(__dirname, '..', 'external', 'pokemon-showdown'),
  };
  for (let i = 2; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--format') args.format = argv[++i];
    else if (arg === '--n') args.n = Number.parseInt(argv[++i], 10);
    else if (arg === '--output') args.output = argv[++i];
    else if (arg === '--showdown-dir') args.showdownDir = argv[++i];
    else throw new Error(`Unknown argument: ${arg}`);
  }
  if (!args.output) throw new Error('--output is required');
  if (!Number.isInteger(args.n) || args.n <= 0) throw new Error('--n must be a positive integer');
  return args;
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

function main() {
  const args = parseArgs(process.argv);
  const simPath = path.join(args.showdownDir, 'dist', 'sim');
  const dexPath = path.join(args.showdownDir, 'dist', 'sim', 'dex');
  const {Teams} = require(simPath);
  const {Dex} = require(dexPath);
  const format = Dex.formats.get(args.format);
  const teams = [];

  // Warm up dynamic imports before measuring/generating the pool.
  Teams.getGenerator(format).getTeam();
  const startedAt = new Date().toISOString();
  const t0 = Date.now();
  for (let i = 0; i < args.n; i++) {
    const generator = Teams.getGenerator(format);
    teams.push(generator.getTeam().map(slimSet));
  }

  const payload = {
    format: args.format,
    n: args.n,
    generatedAt: startedAt,
    elapsedMs: Date.now() - t0,
    source: 'Pokemon Showdown Teams.getGenerator(format).getTeam()',
    teams,
  };
  fs.mkdirSync(path.dirname(args.output), {recursive: true});
  fs.writeFileSync(args.output, JSON.stringify(payload));
  console.error(`wrote ${args.n} ${args.format} teams to ${args.output}`);
}

main();
