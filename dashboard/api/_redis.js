const key = "metagross:ladder:public-status";

function config() {
  const url = process.env.KV_REST_API_URL || process.env.UPSTASH_REDIS_REST_URL;
  const token = process.env.KV_REST_API_TOKEN || process.env.UPSTASH_REDIS_REST_TOKEN;
  return url && token ? { url: url.replace(/\/$/, ""), token } : null;
}

async function command(args) {
  const redis = config();
  if (!redis) return null;
  const response = await fetch(redis.url, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${redis.token}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(args),
  });
  if (!response.ok) throw new Error(`Redis request failed: ${response.status}`);
  return response.json();
}

export async function getStatus() {
  const response = await command(["GET", key]);
  if (!response?.result) return null;
  return JSON.parse(response.result);
}

export async function setStatus(status) {
  const response = await command(["SET", key, JSON.stringify(status)]);
  return response !== null;
}
