import { timingSafeEqual } from "node:crypto";
import { getStatus, setStatus } from "./_redis.js";

const MAX_BODY_BYTES = 128 * 1024;

function authorized(request) {
  const expected = process.env.DASHBOARD_INGEST_SECRET;
  const supplied = request.headers.authorization?.replace(/^Bearer\s+/i, "");
  if (!expected || !supplied) return false;
  const left = Buffer.from(expected);
  const right = Buffer.from(supplied);
  return left.length === right.length && timingSafeEqual(left, right);
}

function validSnapshot(value) {
  return value?.schema === 1
    && typeof value.updatedAt === "string"
    && typeof value.run?.username === "string"
    && ["healthy", "degraded", "offline"].includes(value.health?.overall);
}

async function notifyDiscord(title, description, color = 0xe47070) {
  const url = process.env.DISCORD_WEBHOOK_URL;
  if (!url) return;
  const result = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      username: "Oikawa",
      embeds: [{
        title,
        description,
        color,
        timestamp: new Date().toISOString(),
        footer: { text: "metagross ladder monitor" },
      }],
    }),
  });
  if (!result.ok) throw new Error(`Discord webhook failed: ${result.status}`);
}

function newTechnicalLosses(previous, current) {
  const seen = new Set((previous?.recentBattles || []).map((battle) => battle.id));
  return (current.recentBattles || []).filter(
    (battle) => !seen.has(battle.id) && battle.result === "loss" && battle.reason === "inactivity",
  );
}

export const config = { api: { bodyParser: { sizeLimit: "128kb" } } };

export default async function handler(request, response) {
  if (request.method !== "POST") {
    response.setHeader("Allow", "POST");
    return response.status(405).json({ error: "method_not_allowed" });
  }
  if (!authorized(request)) return response.status(401).json({ error: "unauthorized" });
  const rawSize = Buffer.byteLength(JSON.stringify(request.body || {}));
  if (rawSize > MAX_BODY_BYTES) return response.status(413).json({ error: "payload_too_large" });
  if (!validSnapshot(request.body)) return response.status(400).json({ error: "invalid_snapshot" });
  try {
    const previous = await getStatus();
    const stored = await setStatus(request.body);
    if (!stored) return response.status(503).json({ error: "redis_not_configured" });
    const issue = request.body.health?.issue;
    if (issue && issue.id !== previous?.health?.issue?.id) {
      await notifyDiscord(
        `Ladder failure · ${issue.category}`,
        `${issue.message}\nAccount: **${request.body.run.username}**`,
      ).catch((error) => console.error(error));
    }
    for (const battle of newTechnicalLosses(previous, request.body)) {
      await notifyDiscord(
        "Technical ladder loss",
        `**${request.body.run.username}** lost to **${battle.opponent || "unknown"}** by inactivity/auto-forfeit on turn ${battle.turn ?? "?"}.`,
      ).catch((error) => console.error(error));
    }
    return response.status(202).json({ ok: true, sequence: request.body.sequence });
  } catch (error) {
    return response.status(503).json({ error: "ingest_failed", detail: error.message });
  }
}
