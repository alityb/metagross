import { configureApplication, ensureCommand } from "./discord.js";

function authorized(request) {
  const expected = process.env.DASHBOARD_INGEST_SECRET;
  const supplied = request.headers.authorization?.replace(/^Bearer\s+/i, "");
  return Boolean(expected && supplied && expected === supplied);
}

export default async function handler(request, response) {
  if (request.method !== "POST") return response.status(405).json({ error: "method_not_allowed" });
  if (!authorized(request)) return response.status(401).json({ error: "unauthorized" });
  try {
    const application = await configureApplication();
    const commands = await ensureCommand();
    return response.status(200).json({
      ok: true,
      application,
      commands: commands.map(({ id, name, description }) => ({ id, name, description })),
    });
  } catch (error) {
    return response.status(502).json({ error: "registration_failed", detail: error.message });
  }
}
