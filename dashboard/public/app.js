const $ = (id) => document.getElementById(id);

function slug(species) {
  return String(species || "").toLowerCase().replace(/[.’']/g, "").replace(/[^a-z0-9-]+/g, "-").replace(/^-|-$/g, "");
}

function sprite(species, back = false) {
  return `https://play.pokemonshowdown.com/sprites/${back ? "gen5ani-back" : "gen5ani"}/${slug(species)}.gif`;
}

function text(id, value, fallback = "--") { $(id).textContent = value ?? fallback; }

function setSprite(id, mon, back) {
  const image = $(id);
  if (!mon?.species) return image.removeAttribute("src");
  image.src = sprite(mon.species, back);
  image.alt = mon.species;
  image.onerror = () => image.removeAttribute("src");
}

function setMon(prefix, mon, back) {
  text(`${prefix}-species`, mon?.species, "waiting");
  const hp = Math.max(0, Math.min(100, Number(mon?.hpPercent ?? 0)));
  text(`${prefix}-hp`, mon ? `${Math.round(hp)}%` : "--%");
  text(`${prefix}-status`, mon?.status || "", "");
  const bar = $(`${prefix}-hp-bar`);
  bar.style.width = `${hp}%`;
  bar.className = hp <= 25 ? "low" : hp <= 50 ? "medium" : "";
  setSprite(`${prefix}-sprite`, mon, back);
}

function setTeam(id, team, back) {
  const root = $(id);
  root.replaceChildren();
  for (const mon of team || []) {
    const image = document.createElement("img");
    image.className = `team-icon${mon.fainted ? " fainted" : ""}`;
    image.src = sprite(mon.species, back);
    image.alt = mon.species;
    image.title = `${mon.species} · ${Math.round(mon.hpPercent ?? 0)}%`;
    image.onerror = () => image.remove();
    root.append(image);
  }
}

function render(snapshot) {
  const age = Date.now() - Date.parse(snapshot.updatedAt);
  const stale = !Number.isFinite(age) || age > 30_000;
  const battle = snapshot.battle;
  text("username", snapshot.run?.username);
  text("record", `${snapshot.rating?.wins ?? 0}-${snapshot.rating?.losses ?? 0}`);
  text("elo", snapshot.rating?.elo == null ? null : Math.round(snapshot.rating.elo));
  text("gxe", snapshot.rating?.gxe == null ? null : `${Number(snapshot.rating.gxe).toFixed(1)}%`);
  text("presence-label", stale ? "offline" : battle ? "in battle" : "searching");
  $("live-dot").className = stale ? "stale" : "live";
  text("opponent-name", battle ? `vs. ${battle.opponent}` : "searching...");
  text("turn", battle?.turn);
  $("showdown-link").href = battle ? `https://play.pokemonshowdown.com/${battle.id}` : "https://play.pokemonshowdown.com/";

  setMon("our", battle?.us?.active, true);
  setMon("opponent", battle?.opponentSide?.active, false);
  setTeam("our-team", battle?.us?.revealed, true);
  setTeam("opponent-team", battle?.opponentSide?.revealed, false);

  const events = $("events");
  events.replaceChildren();
  for (const event of (battle?.events || []).filter((item) => item.kind !== "turn").slice(-3).reverse()) {
    const item = document.createElement("li");
    item.className = event.side || "";
    item.textContent = event.label;
    events.append(item);
  }

  const results = $("recent-results");
  results.replaceChildren();
  for (const result of (snapshot.recentBattles || []).slice(0, 8)) {
    const dot = document.createElement("i");
    dot.className = `result ${result.result}`;
    dot.title = `${result.result} vs. ${result.opponent}`;
    results.append(dot);
  }
  text("updated", stale ? "stale" : `${Math.max(0, Math.floor(age / 1000))}s ago`);
}

async function update() {
  try {
    let response = await fetch(`/api/status?ts=${Date.now()}`, { cache: "no-store" });
    if (!response.ok) response = await fetch(`/status.json?ts=${Date.now()}`, { cache: "no-store" });
    if (!response.ok) throw new Error("status unavailable");
    render(await response.json());
  } catch {
    $("live-dot").className = "stale";
    text("presence-label", "offline");
  }
}

await update();
setInterval(update, 3000);
