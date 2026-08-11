#!/usr/bin/env node

import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import {execFileSync} from 'node:child_process';

function parseArgs() {
  const args = {};
  for (let index = 2; index < process.argv.length; index += 2) {
    const name = process.argv[index];
    const value = process.argv[index + 1];
    if (!name?.startsWith('--') || value === undefined) throw new Error('arguments must be --name value pairs');
    args[name.slice(2)] = value;
  }
  for (const required of ['url', 'pair-directory', 'team-generator', 'output']) {
    if (!args[required]) throw new Error(`missing --${required}`);
  }
  const url = new URL(args.url);
  if (url.protocol !== 'ws:' || url.hostname !== '127.0.0.1' || url.pathname !== '/showdown/websocket') {
    throw new Error('diagnostic requires the exact loopback Showdown websocket');
  }
  return args;
}

const sha256 = value => crypto.createHash('sha256').update(value, 'utf8').digest('hex');
const now = () => process.hrtime.bigint().toString();

function toID(value) {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, '');
}

function loginStatus(raw, username) {
  const expected = toID(username);
  for (const line of raw.split('\n')) {
    const parts = line.split('|');
    if (parts[1] === 'nametaken' && toID(parts[2] || '') === expected) return 'rejected';
    if (
      parts[1] === 'updateuser' &&
      toID(parts[2] || '') === expected &&
      parts[3] === '1'
    ) return 'confirmed';
  }
  return null;
}

function client(side, url, trace) {
  const socket = new WebSocket(url);
  const messages = [];
  const waiters = [];
  socket.addEventListener('message', event => {
    const raw = String(event.data);
    trace.push({time_ns: now(), side, direction: 'received', raw});
    messages.push(raw);
    for (const wake of waiters.splice(0)) wake();
  });
  socket.addEventListener('error', () => {
    for (const wake of waiters.splice(0)) wake();
  });
  return {
    socket,
    send(raw) {
      if (!/^\|\/(?:trn|challenge|accept)\b|^battle-[^|]+\|\/forfeit$/.test(raw)) {
        throw new Error(`outbound command is not diagnostic-safe: ${raw}`);
      }
      trace.push({time_ns: now(), side, direction: 'sent', raw});
      socket.send(raw);
    },
    async waitFor(predicate, timeoutMs = 10000) {
      const deadline = Date.now() + timeoutMs;
      while (true) {
        const found = messages.find(predicate);
        if (found !== undefined) return found;
        if (Date.now() >= deadline) throw new Error(`${side} timed out waiting for protocol evidence`);
        await new Promise((resolve, reject) => {
          const timer = setTimeout(() => {
            const index = waiters.indexOf(wake);
            if (index >= 0) waiters.splice(index, 1);
            reject(new Error(`${side} timed out waiting for protocol evidence`));
          }, Math.max(1, deadline - Date.now()));
          const wake = () => {
            clearTimeout(timer);
            resolve();
          };
          waiters.push(wake);
        });
      }
    },
  };
}

function writeRegistration(directory, userid, payload) {
  const destination = path.join(directory, `${userid}.json`);
  const temporary = `${destination}.tmp-${process.pid}`;
  fs.writeFileSync(temporary, `${JSON.stringify(payload)}\n`, {encoding: 'utf8', mode: 0o600, flag: 'wx'});
  fs.renameSync(temporary, destination);
}

function battleTag(raw) {
  return raw.split('\n').find(line => line.startsWith('>battle-'))?.slice(1) ?? null;
}

async function main() {
  const args = parseArgs();
  const output = path.resolve(args.output);
  if (fs.existsSync(output)) throw new Error(`output exists: ${output}`);
  const pairDirectory = path.resolve(args['pair-directory']);
  fs.mkdirSync(pairDirectory, {recursive: true, mode: 0o700});
  if (fs.readdirSync(pairDirectory).length) throw new Error('pair directory must be empty');
  const trace = [];
  const challengerName = 'mtdiagca';
  const acceptorName = 'mtdiagac';
  const challenger = client('challenger', args.url, trace);
  const acceptor = client('acceptor', args.url, trace);
  const result = {
    schema_version: 1,
    mode: 'local_pair_matchmaking_only',
    source_sha256: sha256(fs.readFileSync(new URL(import.meta.url), 'utf8')),
    url: args.url,
    pair_directory: pairDirectory,
    users: {challenger: challengerName, acceptor: acceptorName},
    trace,
    login: {},
    battle_tag: null,
    registrations_consumed: false,
    forbidden_decision_commands: 0,
    passed: false,
    public_ladder: false,
  };
  try {
    await Promise.all([
      challenger.waitFor(raw => raw.includes('|challstr|')),
      acceptor.waitFor(raw => raw.includes('|challstr|')),
    ]);
    challenger.send(`|/trn ${challengerName},0,`);
    acceptor.send(`|/trn ${acceptorName},0,`);
    for (const [name, connection] of [[challengerName, challenger], [acceptorName, acceptor]]) {
      const response = await connection.waitFor(raw => loginStatus(raw, name) !== null);
      result.login[name] = response;
      if (loginStatus(response, name) !== 'confirmed') {
        throw new Error(`Showdown rejected diagnostic username ${name}`);
      }
    }

    const generated = JSON.parse(execFileSync(
      process.execPath,
      [path.resolve(args['team-generator']), 'gen9randombattle', '1,2,3,4', '5,6,7,8'],
      {encoding: 'utf8'},
    ));
    const common = {
      schema_version: 1,
      pair_id: 'matchmaking-diagnostic-v1',
      leg: 1,
      format: 'gen9randombattle',
      battle_seed: '9,10,11,12',
      team_1_sha256: generated.team_1_sha256,
      team_2_sha256: generated.team_2_sha256,
    };
    writeRegistration(pairDirectory, challengerName, {
      ...common,
      assigned_team_sha256: generated.team_1_sha256,
      packed_team: generated.team_1_packed,
    });
    writeRegistration(pairDirectory, acceptorName, {
      ...common,
      assigned_team_sha256: generated.team_2_sha256,
      packed_team: generated.team_2_packed,
    });
    challenger.send(`|/challenge ${acceptorName},gen9randombattle`);
    await acceptor.waitFor(raw => raw.includes('/challenge') && raw.includes('gen9randombattle'));
    acceptor.send(`|/accept ${challengerName}`);
    const [challengerBattle, acceptorBattle] = await Promise.all([
      challenger.waitFor(raw => battleTag(raw) !== null),
      acceptor.waitFor(raw => battleTag(raw) !== null),
    ]);
    const tags = [battleTag(challengerBattle), battleTag(acceptorBattle)];
    if (!tags[0] || tags[0] !== tags[1]) throw new Error(`battle tags differ: ${tags}`);
    result.battle_tag = tags[0];
    challenger.send(`${tags[0]}|/forfeit`);
    await challenger.waitFor(raw => raw.includes('|win|'), 5000).catch(() => null);
    result.registrations_consumed = fs.readdirSync(pairDirectory).length === 0;
    result.forbidden_decision_commands = trace.filter(
      row => row.direction === 'sent' && /\|\/(?:choose|switch|team|search)\b/.test(row.raw)
    ).length;
    result.passed = result.registrations_consumed && result.forbidden_decision_commands === 0;
  } catch (error) {
    result.error = `${error.name}: ${error.message}`;
  } finally {
    challenger.socket.close();
    acceptor.socket.close();
    fs.writeFileSync(output, `${JSON.stringify(result, null, 2)}\n`, {encoding: 'utf8', flag: 'wx'});
  }
  if (!result.passed) process.exitCode = 1;
}

await main();
