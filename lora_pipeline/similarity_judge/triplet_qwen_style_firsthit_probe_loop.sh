#!/usr/bin/env bash
set -euo pipefail

SOURCE_SCRIPT="${1:-/data/benchmark_metrics/lora_pipeline/similarity_judge/triplet_qwen_style_firsthit_judge.sh}"
INTERVAL_SEC="${INTERVAL_SEC:-2}"
CONNECT_TIMEOUT_SEC="${CONNECT_TIMEOUT_SEC:-3}"
REQUEST_TIMEOUT_SEC="${REQUEST_TIMEOUT_SEC:-30}"
MAX_TOKENS="${MAX_TOKENS:-8}"
API_KEY="${API_KEY:-EMPTY}"
PROMPT="${PROMPT:-Reply with exactly: pong}"

timestamp() {
    date "+%F %T"
}

trim_body() {
    local body="$1"
    body="${body//$'\n'/ }"
    body="${body//$'\r'/ }"
    printf "%s" "${body:0:300}"
}

load_endpoints() {
    if [[ ! -f "$SOURCE_SCRIPT" ]]; then
        echo "source script not found: $SOURCE_SCRIPT" >&2
        return 1
    fi

    grep -E '^endpoint[0-9]+=' "$SOURCE_SCRIPT" | while IFS= read -r line; do
        line="${line#*=}"
        line="${line#\"}"
        line="${line%\"}"
        [[ -n "$line" ]] && printf '%s\n' "$line"
    done
}

build_payload() {
    local model="$1"
    python3 - "$model" "$PROMPT" "$MAX_TOKENS" <<'PY'
import json
import sys

model = sys.argv[1]
prompt = sys.argv[2]
max_tokens = int(sys.argv[3])

payload = {
    "model": model,
    "temperature": 0,
    "max_tokens": max_tokens,
    "messages": [
        {
            "role": "system",
            "content": "You are a healthcheck endpoint. Reply with exactly pong.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ],
}
print(json.dumps(payload, ensure_ascii=False))
PY
}

worker_loop() {
    local endpoint_spec="$1"
    local model="${endpoint_spec%@*}"
    local base_url="${endpoint_spec#*@}"
    local url="${base_url%/}/chat/completions"
    local payload
    local seq=0

    payload="$(build_payload "$model")"

    echo "[$(timestamp)] [START] model=${model} url=${url} interval=${INTERVAL_SEC}s" >&2

    while true; do
        seq=$((seq + 1))

        local raw=""
        if raw="$(
            curl -sS \
                --connect-timeout "$CONNECT_TIMEOUT_SEC" \
                --max-time "$REQUEST_TIMEOUT_SEC" \
                -H "Authorization: Bearer ${API_KEY}" \
                -H "Content-Type: application/json" \
                -d "$payload" \
                -w $'\n%{http_code} %{time_total}' \
                "$url" 2>&1
        )"; then
            local meta="${raw##*$'\n'}"
            local body="${raw%$'\n'*}"
            local http_code="${meta%% *}"
            local elapsed="${meta##* }"

            if [[ "$http_code" == "200" && "$body" == *'"choices"'* ]]; then
                printf '[%s] [OK] model=%s url=%s seq=%d code=%s time=%ss\n' \
                    "$(timestamp)" "$model" "$base_url" "$seq" "$http_code" "$elapsed"
            else
                printf '[%s] [FAIL] model=%s url=%s seq=%d code=%s time=%ss body=%s\n' \
                    "$(timestamp)" "$model" "$base_url" "$seq" "$http_code" "$elapsed" "$(trim_body "$body")"
            fi
        else
            local rc=$?
            printf '[%s] [FAIL] model=%s url=%s seq=%d curl_exit=%d err=%s\n' \
                "$(timestamp)" "$model" "$base_url" "$seq" "$rc" "$(trim_body "$raw")"
        fi

        sleep "$INTERVAL_SEC"
    done
}

cleanup() {
    trap - INT TERM EXIT
    if [[ "${#PIDS[@]}" -gt 0 ]]; then
        kill "${PIDS[@]}" 2>/dev/null || true
        wait "${PIDS[@]}" 2>/dev/null || true
    fi
}

mapfile -t ENDPOINTS < <(load_endpoints)
if [[ "${#ENDPOINTS[@]}" -eq 0 ]]; then
    echo "no endpoints found in $SOURCE_SCRIPT" >&2
    exit 1
fi

echo "[$(timestamp)] loaded ${#ENDPOINTS[@]} endpoints from $SOURCE_SCRIPT" >&2

PIDS=()
trap cleanup INT TERM EXIT

for endpoint in "${ENDPOINTS[@]}"; do
    worker_loop "$endpoint" &
    PIDS+=("$!")
done

wait
