import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { getStatus } from "./_redis.js";

export default async function handler(_request, response) {
  if (_request.method !== "GET") {
    response.setHeader("Allow", "GET");
    return response.status(405).json({ error: "method_not_allowed" });
  }
  try {
    const live = await getStatus();
    if (live) return response.status(200).json(live);
    for (const name of ["status.json", "status.example.json"]) {
      try {
        const fallback = await readFile(join(process.cwd(), "public", name), "utf8");
        return response.status(200).json(JSON.parse(fallback));
      } catch (error) {
        if (error.code !== "ENOENT") throw error;
      }
    }
    throw new Error("no dashboard status is available");
  } catch (error) {
    return response.status(503).json({ error: "status_unavailable", detail: error.message });
  }
}
