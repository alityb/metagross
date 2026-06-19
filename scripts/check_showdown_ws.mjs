const uri = process.argv[2] ?? "ws://localhost:8000/showdown/websocket";
const timeoutMs = Number(process.argv[3] ?? 10000);

const timeout = setTimeout(() => {
  console.error(`Timed out waiting for websocket: ${uri}`);
  process.exit(1);
}, timeoutMs);

const ws = new WebSocket(uri);

ws.addEventListener("open", () => {
  console.log(`websocket open: ${uri}`);
  ws.close();
  clearTimeout(timeout);
});

ws.addEventListener("error", (event) => {
  console.error(`websocket error: ${event.message ?? event.type}`);
  clearTimeout(timeout);
  process.exit(1);
});

ws.addEventListener("close", () => process.exit(0));
