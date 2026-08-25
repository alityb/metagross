#!/usr/bin/env bash
set -euo pipefail

# Week-scale, crash-resumable local execution of the admitted 5,000-game stage.
# Strata run sequentially so the 10-core / 24-GiB host is not oversubscribed.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCALE_DIR="${1:?usage: run_fresh_schema6_scale_5000_local.sh SCALE_DIR AUTHORIZATION_REPORT [fresh|resume]}"
AUTHORIZATION_REPORT="${2:?authorization screen report is required}"
RUN_MODE="${3:-fresh}"
AUTHORIZATION_SHA256="61115745a20193c56b633d04e0c8dd497771f3ad580cb470c8de05d577bc50e8"

if [[ "$RUN_MODE" != fresh && "$RUN_MODE" != resume ]]; then
  echo "RUN_MODE must be fresh or resume" >&2
  exit 2
fi
if [[ ! -f "$AUTHORIZATION_REPORT" ]]; then
  echo "authorization report does not exist: $AUTHORIZATION_REPORT" >&2
  exit 2
fi
actual_authorization_sha256="$(shasum -a 256 "$AUTHORIZATION_REPORT" | awk '{print $1}')"
if [[ "$actual_authorization_sha256" != "$AUTHORIZATION_SHA256" ]]; then
  echo "authorization report hash mismatch: $actual_authorization_sha256" >&2
  exit 2
fi
if ! jq -e '
  .schema == "metagross-schema6-20k-50k-screen/v1" and
  .admitted == true and
  .scale_gate_admitted == true and
  .eligible_roots == 113 and
  .withheld_roots_processed == 0 and
  .confirmation_rows_materialized == 0 and
  .calibration_rows_materialized == 0
' "$AUTHORIZATION_REPORT" >/dev/null; then
  echo "authorization report failed the frozen training-only scale gate" >&2
  exit 2
fi

if [[ "$RUN_MODE" == fresh && -e "$SCALE_DIR" ]] && find "$SCALE_DIR" -mindepth 1 -print -quit | grep -q .; then
  echo "fresh SCALE_DIR must be absent or empty: $SCALE_DIR" >&2
  exit 2
fi
mkdir -p "$SCALE_DIR"
SCALE_DIR="$(cd "$SCALE_DIR" && pwd)"
AUTHORIZATION_REPORT="$(cd "$(dirname "$AUTHORIZATION_REPORT")" && pwd)/$(basename "$AUTHORIZATION_REPORT")"

available_kib="$(df -Pk "$SCALE_DIR" | awk 'NR == 2 {print $4}')"
minimum_kib="$((70 * 1024 * 1024))"
if ((available_kib < minimum_kib)); then
  echo "local scale requires at least 70 GiB free; only ${available_kib} KiB available" >&2
  exit 2
fi

printf '%s\n' "$$" >"$SCALE_DIR/supervisor.pid"
printf '%s\n' "running" >"$SCALE_DIR/supervisor.status"
finish() {
  status="$?"
  trap - EXIT
  if ((status == 0)); then
    printf '%s\n' "complete" >"$SCALE_DIR/supervisor.status"
  else
    printf '%s\n' "failed:$status" >"$SCALE_DIR/supervisor.status"
  fi
  exit "$status"
}
trap finish EXIT

profile_complete() {
  local directory="$1"
  local games="$2"
  [[ -f "$directory/result.json" ]] || return 1
  [[ -f "$directory/schema6-capture-audit.json" ]] || return 1
  [[ -f "$directory/schema6-panel-bridge-audit.json" ]] || return 1
  jq -e --argjson games "$games" '.summary.completed_games == $games and .summary.void_games == 0' \
    "$directory/result.json" >/dev/null || return 1
  jq -e '.admitted == true' "$directory/schema6-capture-audit.json" >/dev/null || return 1
  jq -e '.admitted == true' "$directory/schema6-panel-bridge-audit.json" >/dev/null || return 1
}

run_profile() {
  local profile="$1"
  local games="$2"
  local base_port="$3"
  local mirror_seed="$4"
  local production_seed="$5"
  local username_prefix="$6"
  local directory="$SCALE_DIR/$profile"
  local attempt mode

  if profile_complete "$directory" "$games"; then
    echo "[$(date -u +%FT%TZ)] $profile already complete; reusing audited stratum"
    return 0
  fi
  for attempt in $(seq 1 100); do
    if [[ -f "$directory/result.json.progress.json" ]]; then
      mode=resume
    else
      mode=fresh
      if [[ -d "$directory" ]] && find "$directory" -mindepth 1 -print -quit | grep -q .; then
        echo "$profile has partial files but no atomic progress snapshot; refusing destructive restart" >&2
        return 1
      fi
    fi
    echo "[$(date -u +%FT%TZ)] starting $profile attempt=$attempt mode=$mode games=$games"
    if env \
      METAGROSS_CAPTURE_MIRROR_SEED="$mirror_seed" \
      METAGROSS_CAPTURE_PRODUCTION_SEED="$production_seed" \
      METAGROSS_CAPTURE_USERNAME_PREFIX="$username_prefix" \
      METAGROSS_WEBSOCKET_RECEIVE_TIMEOUT_SECONDS=600 \
      "$ROOT_DIR/experimental/src/scripts/run_fresh_schema6_capture.sh" \
        "$directory" "$games" "$base_port" "$profile" 1 "$mode"; then
      if profile_complete "$directory" "$games"; then
        echo "[$(date -u +%FT%TZ)] completed and audited $profile"
        return 0
      fi
      echo "$profile command returned success without a complete admitted stratum" >&2
      return 1
    fi
    if [[ ! -f "$directory/result.json.progress.json" ]]; then
      echo "$profile failed without a resumable atomic progress snapshot" >&2
      return 1
    fi
    echo "[$(date -u +%FT%TZ)] $profile interrupted; retrying from atomic progress in 10 seconds"
    sleep 10
  done
  echo "$profile exceeded 100 resumable attempts" >&2
  return 1
}

run_profile \
  peer 3000 8140 2027082900 \
  07e7a60daa08d673e24b3d495027ef6bce3258f361c4e0afde8773112abb8769 l5kpeer
run_profile \
  direct_r1 1000 8140 2028082900 \
  e7774edda69aad4dbd1b3bbbe81349d1db92dc78f3ba3e0c45f6c0dbcd40bc5b l5kdr1
run_profile \
  unguided 1000 8140 2029082900 \
  d75aac661ea163cbb796c3598d4c7108dde0f4e50b307d9ad47667ad1b75bf66 l5kung

"$ROOT_DIR/.venv-metamon/bin/python" \
  "$ROOT_DIR/experimental/src/scripts/summarize_schema6_capture_pilot.py" \
  --root "$SCALE_DIR" \
  --output "$SCALE_DIR/summary.json" \
  --stage-games 5000 \
  --authorization-report "$AUTHORIZATION_REPORT"

echo "[$(date -u +%FT%TZ)] local 5,000-game stage completed and admitted"
