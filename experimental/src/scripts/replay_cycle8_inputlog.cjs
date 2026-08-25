#!/usr/bin/env node
"use strict";

/**
 * Replay one raw Pokemon Showdown inputlog at its pinned build.
 *
 * This program is deliberately a capture boundary, not an information-state
 * exporter.  It serializes spectator/public output and each side's private
 * request stream into three separate files.  It never serializes Battle,
 * `end`'s omniscient team payload, or the other side's request into a POV file.
 */

const crypto = require("crypto");
const fs = require("fs");
const path = require("path");

function parseArgs(argv) {
  const args = {};
  for (let index = 2; index < argv.length; index += 2) {
    const key = argv[index];
    const value = argv[index + 1];
    if (!key?.startsWith("--") || value === undefined) {
      throw new Error(`invalid argument near ${key || "<end>"}`);
    }
    args[key.slice(2)] = value;
  }
  for (const required of ["showdown", "input", "out-dir"]) {
    if (!args[required]) throw new Error(`missing --${required}`);
  }
  return args;
}

function sha256(value) {
  return crypto.createHash("sha256").update(value).digest("hex");
}

function stable(value) {
  if (Array.isArray(value)) return value.map(stable);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.keys(value).sort().map(key => [key, stable(value[key])])
    );
  }
  return value;
}

function writeJson(target, value) {
  fs.writeFileSync(target, JSON.stringify(stable(value)) + "\n");
}

function main() {
  const args = parseArgs(process.argv);
  const showdown = path.resolve(args.showdown);
  const inputPath = path.resolve(args.input);
  const outDir = path.resolve(args["out-dir"]);
  const {BattleStream} = require(path.join(showdown, "dist/sim/battle-stream"));
  const {extractChannelMessages} = require(path.join(showdown, "dist/sim/battle"));
  const recordBytes = fs.readFileSync(inputPath);
  const record = JSON.parse(recordBytes.toString("utf8"));
  if (typeof record.inputlog !== "string" || !record.inputlog.trim()) {
    throw new Error("raw replay lacks inputlog");
  }

  const publicChunks = [];
  const pov = {
    p1: {schema: "metagross-cycle8-pov-capture/v1", role: "p1", sideupdate_chunks: [], requests: [], commands: [], errors: []},
    p2: {schema: "metagross-cycle8-pov-capture/v1", role: "p2", sideupdate_chunks: [], requests: [], commands: [], errors: []},
  };
  const latest = {p1: null, p2: null};
  let terminal = null;
  let inputIndex = -1;

  class CaptureStream extends BattleStream {
    constructor() {
      super({noCatch: true, keepAlive: true});
    }

    pushMessage(type, data) {
      if (type === "update") {
        const channels = extractChannelMessages(data, [0]);
        const spectator = channels[0].join("\n");
        if (spectator) publicChunks.push({input_index: inputIndex, data: spectator});
        return;
      }
      if (type === "sideupdate") {
        const newline = data.indexOf("\n");
        const role = data.slice(0, newline);
        const body = newline < 0 ? "" : data.slice(newline + 1);
        if (!(role in pov)) throw new Error(`unexpected sideupdate role ${role}`);
        pov[role].sideupdate_chunks.push({
          emitted_after_input_index: inputIndex,
          public_chunk_count: publicChunks.length,
          body,
          body_sha256: sha256(body),
        });
        for (const line of body.split("\n")) {
          if (line.startsWith("|request|")) {
            const payload = line.slice("|request|".length);
            if (payload === "null") {
              latest[role] = null;
              continue;
            }
            const request = JSON.parse(payload);
            const requestIndex = pov[role].requests.length;
            pov[role].requests.push({
              request_index: requestIndex,
              emitted_after_input_index: inputIndex,
              public_chunk_count: publicChunks.length,
              request,
              request_sha256: sha256(JSON.stringify(stable(request))),
            });
            latest[role] = requestIndex;
          } else if (line.startsWith("|error|")) {
            pov[role].errors.push({input_index: inputIndex, line});
          }
        }
        return;
      }
      if (type === "end") {
        const payload = JSON.parse(data);
        terminal = {
          winner: typeof payload.winner === "string" ? payload.winner : "",
          turns: Number.isInteger(payload.turns) ? payload.turns : null,
        };
      }
    }
  }

  const stream = new CaptureStream();
  const inputLines = record.inputlog.split("\n").filter(Boolean);
  for (inputIndex = 0; inputIndex < inputLines.length; inputIndex += 1) {
    const line = inputLines[inputIndex];
    const match = line.match(/^>(p[12])\s+(.+)$/);
    if (match) {
      const role = match[1];
      const requestIndex = latest[role];
      pov[role].commands.push({
        input_index: inputIndex,
        command: match[2],
        preceding_request_index: requestIndex,
      });
      // A second command from the same side cannot silently reuse a request.
      latest[role] = null;
    }
    stream._write(line);
  }

  fs.mkdirSync(outDir, {recursive: true});
  const publicCapture = {
    schema: "metagross-cycle8-public-capture/v1",
    battle_id: String(record.id || ""),
    raw_sha256: sha256(recordBytes),
    public_chunks: publicChunks,
    terminal,
    replay_ended: Boolean(stream.battle?.ended),
  };
  writeJson(path.join(outDir, "public.json"), publicCapture);
  writeJson(path.join(outDir, "p1.json"), pov.p1);
  writeJson(path.join(outDir, "p2.json"), pov.p2);
}

try {
  main();
} catch (error) {
  process.stderr.write(`${error?.stack || error}\n`);
  process.exitCode = 1;
}
