import { createPublicKey, verify } from "node:crypto";
import { getStatus } from "./_redis.js";

const PING = 1;
const APPLICATION_COMMAND = 2;
const CHANNEL_MESSAGE = 4;
const ED25519_PREFIX = Buffer.from("302a300506032b6570032100", "hex");

export const config = { api: { bodyParser: false } };

async function rawBody(request) {
  const chunks = [];
  for await (const chunk of request) chunks.push(Buffer.from(chunk));
  return Buffer.concat(chunks);
}

function authentic(request, body) {
  const signature = request.headers["x-signature-ed25519"];
  const timestamp = request.headers["x-signature-timestamp"];
  const publicKey = process.env.DISCORD_PUBLIC_KEY;
  if (!signature || !timestamp || !publicKey) return false;
  try {
    const key = createPublicKey({
      key: Buffer.concat([ED25519_PREFIX, Buffer.from(publicKey, "hex")]),
      format: "der",
      type: "spki",
    });
    return verify(
      null,
      Buffer.concat([Buffer.from(timestamp), body]),
      key,
      Buffer.from(signature, "hex"),
    );
  } catch {
    return false;
  }
}

export async function ensureCommand() {
  const applicationId = process.env.DISCORD_APPLICATION_ID;
  const token = process.env.DISCORD_BOT_TOKEN;
  if (!applicationId || !token) throw new Error("Discord application credentials are missing");
  const response = await fetch(`https://discord.com/api/v10/applications/${applicationId}/commands`, {
    method: "PUT",
    headers: { Authorization: `Bot ${token}`, "Content-Type": "application/json" },
    body: JSON.stringify([{
      name: "ladder",
      description: "Show the live Metagross ladder record and historical comparisons",
      type: 1,
    }]),
  });
  if (!response.ok) throw new Error(`Discord command registration failed: ${response.status}`);
  return response.json();
}

export async function configureApplication() {
  const token = process.env.DISCORD_BOT_TOKEN;
  if (!token) throw new Error("Discord bot token is missing");
  const response = await fetch("https://discord.com/api/v10/applications/@me", {
    method: "PATCH",
    headers: { Authorization: `Bot ${token}`, "Content-Type": "application/json" },
    body: JSON.stringify({
      interactions_endpoint_url: "https://pokemon.amtayeb.dev/api/discord",
      install_params: { scopes: ["applications.commands", "bot"], permissions: "0" },
    }),
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`Discord application configuration failed: ${response.status} ${detail}`);
  }
  const application = await response.json();
  return {
    interactionsEndpointUrl: application.interactions_endpoint_url,
    installParams: application.install_params,
  };
}

function fmt(value, digits = 0) {
  return value == null ? "—" : Number(value).toFixed(digits);
}

function ladderMessage(status) {
  const rating = status?.rating || {};
  const wins = rating.wins || 0;
  const losses = rating.losses || 0;
  const games = wins + losses;
  const winRate = games ? 100 * wins / games : 0;
  const health = status?.health?.overall || "offline";
  return {
    type: CHANNEL_MESSAGE,
    data: {
      embeds: [{
        title: `${status?.run?.username || "Metagross"} · live ladder`,
        description: `**${wins}-${losses}** · ${winRate.toFixed(1)}% win rate · ${games}/${status?.run?.requestedGames || 600} games`,
        color: health === "healthy" ? 0x72d572 : 0xe47070,
        fields: [
          { name: "Elo", value: fmt(rating.elo), inline: true },
          { name: "GXE", value: `${fmt(rating.gxe, 1)}%`, inline: true },
          { name: "Glicko", value: `${fmt(rating.glicko)} ± ${fmt(rating.glickoDeviation, 1)}`, inline: true },
          {
            name: "Historical checkpoints",
            value: [
              "**Frozen r1** · 218-122 · Elo ~2362 · 92.4 GXE · 1973 ± 39",
              "**Frozen G3** · 209-131 · Elo 2141 · 89.4 GXE · 1901 ± 25",
              "**Frozen G4** · 132-86 · 85.0 GXE",
            ].join("\n"),
          },
        ],
        footer: { text: `r1 checkpoint 5 · 250/500ms/P16 · ${health}` },
        timestamp: status?.updatedAt || new Date().toISOString(),
      }],
    },
  };
}

export default async function handler(request, response) {
  if (request.method !== "POST") return response.status(405).json({ error: "method_not_allowed" });
  const body = await rawBody(request);
  if (!authentic(request, body)) return response.status(401).json({ error: "invalid_signature" });
  const interaction = JSON.parse(body.toString("utf8"));
  if (interaction.type === PING) {
    await ensureCommand().catch((error) => console.error(error));
    return response.status(200).json({ type: PING });
  }
  if (interaction.type === APPLICATION_COMMAND && interaction.data?.name === "ladder") {
    const status = await getStatus();
    if (!status) {
      return response.status(200).json({
        type: CHANNEL_MESSAGE,
        data: { content: "Ladder telemetry is not available.", flags: 64 },
      });
    }
    return response.status(200).json(ladderMessage(status));
  }
  return response.status(400).json({ error: "unsupported_interaction" });
}
