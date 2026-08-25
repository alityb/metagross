#!/usr/bin/env node
'use strict';

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '../..');
const SHOWDOWN = path.join(ROOT, 'external/pokemon-showdown');
const {BattleStream, getPlayerStreams, Teams} = require(path.join(SHOWDOWN, 'dist/sim'));
const {Dex} = require(path.join(SHOWDOWN, 'dist/sim/dex'));
const {RandomPlayerAI} = require(path.join(SHOWDOWN, 'dist/sim/tools/random-player-ai'));

function seed(master, battle, channel) {
  const digest = crypto.createHash('sha256')
    .update(`${master}\0${battle}\0${channel}`, 'utf8').digest();
  const values = [];
  for (let index = 0; index < 4; index++) values.push(digest.readUInt16BE(index * 2));
  return values.join(',');
}

function sha256(value) {
  return crypto.createHash('sha256').update(value, 'utf8').digest('hex');
}

function canonicalAction(choice, request) {
  const command = choice.split(',', 1)[0].trim().split(/\s+/);
  if (command[0] === 'move') {
    const slot = Number(command[1]) - 1;
    const move = request.active?.[0]?.moves?.[slot]?.id;
    if (!move) throw new Error(`cannot resolve move choice: ${choice}`);
    return command.includes('terastallize') ? `${move}-tera` : move;
  }
  if (command[0] === 'switch') {
    const slot = Number(command[1]) - 1;
    const details = request.side?.pokemon?.[slot]?.details;
    if (!details) throw new Error(`cannot resolve switch choice: ${choice}`);
    const species = details.split(',', 1)[0].toLowerCase().replace(/[^a-z0-9]/g, '');
    return `switch ${species}`;
  }
  if (command[0] === 'pass') return 'pass';
  throw new Error(`unsupported choice: ${choice}`);
}

class CapturingRandomPlayerAI extends RandomPlayerAI {
  constructor(stream, options) {
    super(stream, options);
    this.chunks = [];
    this.decisions = [];
    this.pendingRequest = null;
  }

  receive(chunk) {
    this.chunks.push(chunk);
    super.receive(chunk);
  }

  receiveRequest(request) {
    this.pendingRequest = request.wait ? null : request;
    super.receiveRequest(request);
  }

  choose(choice) {
    if (this.pendingRequest) {
      this.decisions.push({
        chunk_count: this.chunks.length,
        action: canonicalAction(choice, this.pendingRequest),
      });
      this.pendingRequest = null;
    }
    super.choose(choice);
  }
}

async function generateBattle(master, index) {
  const formatID = 'gen9randombattle';
  const format = Dex.formats.get(formatID);
  const team1Seed = seed(master, index, 'team-p1');
  const team2Seed = seed(master, index, 'team-p2');
  const battleSeed = seed(master, index, 'battle');
  const ai1Seed = seed(master, index, 'ai-p1');
  const ai2Seed = seed(master, index, 'ai-p2');
  const team1 = Teams.pack(Teams.getGenerator(format, team1Seed).getTeam());
  const team2 = Teams.pack(Teams.getGenerator(format, team2Seed).getTeam());
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const p1 = new CapturingRandomPlayerAI(streams.p1, {seed: ai1Seed, move: 0.7, mega: 0.6});
  const p2 = new CapturingRandomPlayerAI(streams.p2, {seed: ai2Seed, move: 0.7, mega: 0.6});
  void p1.start();
  void p2.start();
  const drain = (async () => {
    for await (const _chunk of streams.omniscient) {}
  })();
  await streams.omniscient.write(
    `>start ${JSON.stringify({formatid: formatID, seed: battleSeed})}\n` +
    `>player p1 ${JSON.stringify({name: 'Truth P1', team: team1})}\n` +
    `>player p2 ${JSON.stringify({name: 'Truth P2', team: team2})}`
  );
  await drain;
  const truth1 = Teams.unpack(team1);
  const truth2 = Teams.unpack(team2);
  if (!truth1 || !truth2) throw new Error('generated packed team could not be unpacked');
  return {
    schema: 'metagross-known-team-battle/v1',
    battle_id: `known-team-${String(index).padStart(6, '0')}`,
    format: formatID,
    seeds: {master, battle: battleSeed, team_p1: team1Seed, team_p2: team2Seed, ai_p1: ai1Seed, ai_p2: ai2Seed},
    teams: {
      p1: {packed_sha256: sha256(team1), sets: truth1},
      p2: {packed_sha256: sha256(team2), sets: truth2},
    },
    views: {
      p1: {chunks: p1.chunks, decisions: p1.decisions},
      p2: {chunks: p2.chunks, decisions: p2.decisions},
    },
  };
}

async function main() {
  const count = Number(process.argv[2]);
  const output = process.argv[3];
  const master = process.argv[4] || 'metagross-known-team-v1';
  if (!Number.isInteger(count) || count < 1 || !output) {
    throw new Error('usage: generate_known_team_randbats.cjs COUNT OUTPUT [MASTER_SEED]');
  }
  const descriptor = fs.openSync(output, 'w', 0o600);
  try {
    for (let index = 0; index < count; index++) {
      const battle = await generateBattle(master, index);
      fs.writeSync(descriptor, `${JSON.stringify(battle)}\n`);
    }
    fs.fsyncSync(descriptor);
  } finally {
    fs.closeSync(descriptor);
  }
}

main().catch(error => {
  console.error(error.stack || error);
  process.exitCode = 1;
});
