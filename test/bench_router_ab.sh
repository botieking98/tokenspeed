#!/bin/bash
set -euo pipefail

# A/B benchmark: LMetric router vs round-robin across 2 ts serve instances.
# Instance 0: GPU 0-3, port 30000
# Instance 1: GPU 4-7, port 30001
# Router: port 9000

MODEL="/root/models/Qwen/Qwen3.6-27B"
ROUNDS=3
RESULT_DIR="/tmp/bench_router_results"
rm -rf "$RESULT_DIR"
mkdir -p "$RESULT_DIR"

INST0_PORT=30000
INST1_PORT=30001
ROUTER_PORT=9000

INST0_PID= INST1_PID= ROUTER_PID=

cleanup() {
    echo "Cleaning up..."
    for pid in $ROUTER_PID $INST1_PID $INST0_PID; do
        if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
            kill -TERM -"$pid" 2>/dev/null || true
            sleep 2
            kill -KILL -"$pid" 2>/dev/null || true
        fi
    done
    pkill -9 -f 'smg.*router' 2>/dev/null || true
}
trap cleanup EXIT

launch_instances() {
    echo "Launching instance 0 (GPU 0-3, port $INST0_PORT)..."
    bash -c 'export CUDA_VISIBLE_DEVICES=0,1,2,3; exec ts serve \
        --model "'"$MODEL"'" --port '"$INST0_PORT"' --attn-tp-size 4 \
        --enable-prefix-caching --prometheus-port 8413' \
        > /tmp/ts_inst0.log 2>&1 &
    INST0_PID=$!

    echo "Launching instance 1 (GPU 4-7, port $INST1_PORT)..."
    bash -c 'export CUDA_VISIBLE_DEVICES=4,5,6,7; exec ts serve \
        --model "'"$MODEL"'" --port '"$INST1_PORT"' --attn-tp-size 4 \
        --enable-prefix-caching --prometheus-port 8414' \
        > /tmp/ts_inst1.log 2>&1 &
    INST1_PID=$!

    echo "Waiting for instances..."
    local timeout=600 start=$SECONDS
    local inst0_ready=false inst1_ready=false
    while (( SECONDS - start < timeout )); do
        if ! $inst0_ready && curl -sf -o /dev/null http://127.0.0.1:$INST0_PORT/health 2>/dev/null; then
            inst0_ready=true; echo "  Instance 0 ready"
        fi
        if ! $inst1_ready && curl -sf -o /dev/null http://127.0.0.1:$INST1_PORT/health 2>/dev/null; then
            inst1_ready=true; echo "  Instance 1 ready"
        fi
        if $inst0_ready && $inst1_ready; then
            echo "  Both instances ready in $((SECONDS - start))s"
            return 0
        fi
        sleep 5
    done
    echo "  Timeout waiting for instances"
    return 1
}

launch_router() {
    local mode=$1
    echo "  Launching router (mode=$mode, port $ROUTER_PORT)..."
    setsid ts router \
        --instance-urls http://127.0.0.1:$INST0_PORT http://127.0.0.1:$INST1_PORT \
        --port $ROUTER_PORT \
        --mode "$mode" \
        > /tmp/ts_router_${mode}.log 2>&1 &
    ROUTER_PID=$!
    # Wait for router to start and pass health check
    local timeout=30 start=$SECONDS
    while (( SECONDS - start < timeout )); do
        if curl -sf http://127.0.0.1:$ROUTER_PORT/health >/dev/null 2>&1; then
            echo "  Router ready"
            return 0
        fi
        sleep 1
    done
    echo "  Router failed to start. Log:"
    cat /tmp/ts_router_${mode}.log 2>/dev/null | tail -10
    return 1
}

stop_router() {
    if [[ -n "${ROUTER_PID:-}" ]] && kill -0 "$ROUTER_PID" 2>/dev/null; then
        kill -TERM "$ROUTER_PID" 2>/dev/null || true
        sleep 2
        kill -KILL "$ROUTER_PID" 2>/dev/null || true
    fi
    ROUTER_PID=
}

run_sweep() {
    local label="$1" round="$2"
    for conc in 1 4 8 16 32; do
        local rlabel="${label}_r${round}_c${conc}"
        echo -n "    c=$conc ... "
        tokenspeed bench serve \
            --backend openai \
            --model "$MODEL" \
            --port $ROUTER_PORT \
            --dataset-name random \
            --random-prefix-len 512 \
            --random-input-len 1024 \
            --random-output-len 100 \
            --num-prompts 64 \
            --max-concurrency $conc \
            --num-warmups 4 \
            --ignore-eos \
            --label "$rlabel" \
            --save-result \
            --result-dir "$RESULT_DIR" \
            --disable-tqdm \
            --trust-remote-code \
            --seed 42 \
            2>&1 | grep -E "Mean TTFT|Mean TPOT|Output token throughput" | \
            awk '{printf "%s  ", $0}'
        echo ""
    done
}

echo "======================================================================"
echo "LMetric Router A/B Benchmark ($ROUNDS rounds)"
echo "Model: $MODEL"
echo "2 instances × 4 GPU, prefix=512, input=1024, output=100"
echo "======================================================================"

launch_instances

for round in $(seq 1 $ROUNDS); do
    for mode in round_robin lmetric; do
        echo ""
        echo "--- Round $round/$ROUNDS  $(echo $mode | tr a-z A-Z) ---"
        launch_router "$mode"
        run_sweep "$mode" "$round"
        # Print router stats
        curl -s http://127.0.0.1:$ROUTER_PORT/router/stats 2>/dev/null | python3 -m json.tool 2>/dev/null || true
        stop_router
        sleep 3
    done
done

echo ""
echo "======================================================================"
echo "Aggregating..."
echo "======================================================================"

python3 -u - "$RESULT_DIR" <<'PYEOF'
import json, glob, sys, statistics, os

result_dir = sys.argv[1]
data = {}

for f in sorted(glob.glob(os.path.join(result_dir, "*.json"))):
    with open(f) as fh:
        r = json.load(fh)
    label = r.get("label", "")
    parts = label.split("_")
    if len(parts) < 3: continue
    arm = "_".join(parts[:-2])  # round_robin or lmetric
    conc = int(parts[-1].replace("c", ""))
    if arm not in data: data[arm] = {}
    if conc not in data[arm]: data[arm][conc] = {"ttft": [], "tpot": [], "tput": []}
    data[arm][conc]["ttft"].append(r.get("mean_ttft_ms", 0))
    data[arm][conc]["tpot"].append(r.get("mean_tpot_ms", 0))
    data[arm][conc]["tput"].append(r.get("output_throughput", 0))

if "round_robin" not in data or "lmetric" not in data:
    print("Missing data"); sys.exit(1)

print()
print("=" * 105)
print("  MULTI-INSTANCE ROUTER — AGGREGATED RESULTS")
print("=" * 105)
print(f"{'conc':>5} | {'TTFT RR':>10} {'±':>5} {'TTFT LM':>10} {'±':>5} {'Δ':>8} | {'TPOT RR':>10} {'±':>5} {'TPOT LM':>10} {'±':>5} {'Δ':>8} | {'tput Δ':>8}")
print("-" * 105)

for c in [1, 4, 8, 16, 32]:
    b = data.get("round_robin", {}).get(c)
    t = data.get("lmetric", {}).get(c)
    if not b or not t: continue
    def s(v): return (statistics.mean(v), statistics.stdev(v) if len(v)>1 else 0)
    bm, bs = s(b["ttft"]); tm, ts = s(t["ttft"])
    td = (tm - bm) / bm * 100 if bm else 0
    bpm, bps = s(b["tpot"]); tpm, tps = s(t["tpot"])
    tpd = (tpm - bpm) / bpm * 100 if bpm else 0
    bt = statistics.mean(b["tput"]); tt = statistics.mean(t["tput"])
    tputd = (tt - bt) / bt * 100 if bt else 0
    print(f"{c:>5} | {bm:>8.1f}ms {bs:>4.1f} {tm:>8.1f}ms {ts:>4.1f} {td:>+7.1f}% | {bpm:>8.2f}ms {bps:>4.2f} {tpm:>8.2f}ms {tps:>4.2f} {tpd:>+7.1f}% | {tputd:>+7.1f}%")

print("=" * 105)
PYEOF
