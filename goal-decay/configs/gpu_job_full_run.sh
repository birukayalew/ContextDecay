#!/bin/bash
#SBATCH --job-name=goaldecay-full-run
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/share/agenticsystems/%u/project/logs/goaldecay_full_run_%j.out
#SBATCH --error=/share/agenticsystems/%u/project/logs/goaldecay_full_run_%j.err

# Full Phase 0 corpus generation (protocol §2.4). Runs unattended via
# sbatch (NOT srun) since --qos gpu supports up to 3-day wall time,
# unlike --qos short_gpu's 2-hour interactive cap.
#
# Time budget: estimated ~6.2hr from a small retail/telecom timing
# sample (~85s/task avg at concurrency 8), but the frozen telecom sample
# (configs/sampled_task_ids.json) skews toward complex multi-issue tasks
# (7-8 stacked conditions) not represented in that timing sample --
# requested 2 days generously rather than re-testing on the real sample.
# The job exits on its own once generation finishes; the time limit is
# just a ceiling, not a target.
#
# Usage: sbatch gpu_job_full_run.sh
# Monitor: squeue -u $USER ; tail -f /share/agenticsystems/$USER/project/logs/goaldecay_full_run_<jobid>.out
# After: sacct -j <jobid> --format=JobID,Elapsed,State,ExitCode

set -euo pipefail

PROJECT_ROOT="/share/agenticsystems/${USER}/project"
REPO_ROOT="${PROJECT_ROOT}/ContextDecay/goal-decay"
TAU2_PATH="${PROJECT_ROOT}/tau2-bench"

mkdir -p "${PROJECT_ROOT}/logs"

source "${PROJECT_ROOT}/miniforge3/bin/activate" goaldecay
module load cuda/13.2
export CUDA_HOME=/usr/local/apps/cuda/cuda-13.2
export LD_LIBRARY_PATH="${PROJECT_ROOT}/miniforge3/envs/goaldecay/lib:${LD_LIBRARY_PATH:-}"

cd "${PROJECT_ROOT}"

echo "[job] node: $(hostname)"
echo "[job] started: $(date -Iseconds)"

# --- Start vLLM server in the background ---
vllm serve models/Qwen3.8-27B --served-model-name Qwen/Qwen3.8-27B \
    --host 0.0.0.0 --port 8000 --dtype bfloat16 \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    > "${PROJECT_ROOT}/logs/vllm_server_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!
echo "[job] vllm server pid: ${VLLM_PID}"

# --- Wait for the server to actually be ready, don't just sleep a guess ---
echo "[job] waiting for vllm server to become ready..."
for i in $(seq 1 60); do
    if curl -s -m 5 "http://localhost:8000/v1/models" > /dev/null 2>&1; then
        echo "[job] vllm server ready after ${i}0s (approx)"
        break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[job] ERROR: vllm server process died during startup. Check vllm_server_${SLURM_JOB_ID}.log"
        exit 1
    fi
    sleep 10
done
if ! curl -s -m 5 "http://localhost:8000/v1/models" > /dev/null 2>&1; then
    echo "[job] ERROR: vllm server did not become ready within 600s"
    kill "${VLLM_PID}" 2>/dev/null || true
    exit 1
fi

export HOSTED_VLLM_API_BASE="http://localhost:8000/v1"

cd "${REPO_ROOT}"

# --- Sample task IDs once, if not already done (idempotent: script
#     refuses to overwrite an existing sample, matching the "frozen
#     splits" philosophy -- protocol §2.5) ---
if [ ! -f configs/sampled_task_ids.json ]; then
    echo "[job] sampling task ids..."
    python -m src.rollout.sample_tasks \
        --tau2-repo-path "${TAU2_PATH}" \
        --n-per-domain 50 \
        --seed 42 \
        --out configs/sampled_task_ids.json
else
    echo "[job] configs/sampled_task_ids.json already exists, reusing it."
fi

# --- Run the full generation ---
echo "[job] starting trajectory generation: $(date -Iseconds)"
python -m src.rollout.generate \
    --config configs/rollout_full.yaml \
    --tau2-repo-path "${TAU2_PATH}"
GEN_EXIT=$?
echo "[job] generation finished: $(date -Iseconds), exit code ${GEN_EXIT}"

# --- Clean shutdown of the vLLM server ---
kill "${VLLM_PID}" 2>/dev/null || true
wait "${VLLM_PID}" 2>/dev/null || true

echo "[job] done: $(date -Iseconds)"
exit ${GEN_EXIT}
