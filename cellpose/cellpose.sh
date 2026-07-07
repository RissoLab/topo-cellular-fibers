#!/bin/bash
#SBATCH -p gpu
#SBATCH --time=1-00:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=512G
#SBATCH --gres=gpu:1
#SBATCH -o logs/cellpose_%j.out
#SBATCH -e logs/cellpose_%j.err

set -euo pipefail

module load micromamba

ENV_NAME="env-cellpose"
INPUT_DIR=""
OUTPUT_DIR=""

SEG_DIR="${OUTPUT_DIR}/segmentations"
BENCH_DIR="${OUTPUT_DIR}/benchmarks/job_${SLURM_JOB_ID}"

mkdir -p "$SEG_DIR" "$BENCH_DIR"/{time_logs,gpu_logs}

SUMMARY="${BENCH_DIR}/cellpose_summary.tsv"
DETAILS="${BENCH_DIR}/cellpose_details.tsv"

echo -e "job_id\timage\tstart_time\tend_time\telapsed_seconds\tstatus\ttime_log\tgpu_log" > "$SUMMARY"

{
  echo "Job ID: ${SLURM_JOB_ID}"
  echo "Node: ${SLURMD_NODENAME}"
  echo "CPUs: ${SLURM_CPUS_PER_TASK:-NA}"
  echo "Memory requested: ${SLURM_MEM_PER_NODE:-NA} MB"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-NA}"
  echo "Environment: ${ENV_NAME}"
  echo "Input directory: ${INPUT_DIR}"
  echo "Output directory: ${OUTPUT_DIR}"
  echo "Start: $(date)"
  echo
  nvidia-smi
} > "${BENCH_DIR}/job_info.txt"

run_cellpose() {
  local image="$1"
  /usr/bin/time -v -o "$2" \
    micromamba run -n "$ENV_NAME" python -m cellpose \
      --image_path "$image" \
      --pretrained_model cpsam \
      --use_gpu \
      --channel_axis -1 \
      --diameter 60 \
      --batch_size 128 \
      --flow_threshold 0.6 \
      --niter 2000 \
      --min_size 100 \
      --save_tif \
      --savedir "$SEG_DIR"
}

for image in "$INPUT_DIR"/*.tif; do
  [ -e "$image" ] || continue

  name="$(basename "${image%.tif}" | tr ' /' '__')"
  time_log="${BENCH_DIR}/time_logs/${name}.txt"
  gpu_log="${BENCH_DIR}/gpu_logs/${name}.csv"

  echo "Processing: $image"
  start_epoch="$(date +%s)"
  start_iso="$(date --iso-8601=seconds)"

  nvidia-smi \
    --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
    --format=csv \
    -l 2 \
    > "$gpu_log" &
  gpu_monitor_pid="$!"

  status="OK"
  run_cellpose "$image" "$time_log" || status="FAILED"

  if ps -p "$gpu_monitor_pid" > /dev/null 2>&1; then
    kill "$gpu_monitor_pid" || true
    wait "$gpu_monitor_pid" 2>/dev/null || true
  fi

  end_epoch="$(date +%s)"
  end_iso="$(date --iso-8601=seconds)"
  elapsed="$((end_epoch - start_epoch))"

  echo -e "${SLURM_JOB_ID}\t${image}\t${start_iso}\t${end_iso}\t${elapsed}\t${status}\t${time_log}\t${gpu_log}" >> "$SUMMARY"
  echo "Done: $image (${elapsed}s, ${status})"

  [ "$status" = "OK" ] || exit 1
done

if command -v sacct >/dev/null 2>&1; then
  sacct -j "$SLURM_JOB_ID" \
    --format=JobID,JobName,Partition,AllocCPUS,Elapsed,TotalCPU,MaxRSS,ReqMem,State,ExitCode \
    > "${BENCH_DIR}/slurm_sacct.txt" || true
fi

echo -e "image\telapsed_seconds\tstatus\tmax_ram_kb\tcpu_percent\tmax_gpu_util_percent\tmax_gpu_mem_used_mib\tmax_gpu_temp_c\tmax_gpu_power_w" > "$DETAILS"

tail -n +2 "$SUMMARY" | while IFS=$'\t' read -r _ image _ _ elapsed status time_log gpu_log; do
  max_ram_kb="NA"
  cpu_percent="NA"
  max_gpu_util="NA"
  max_gpu_mem="NA"
  max_gpu_temp="NA"
  max_gpu_power="NA"

  if [ -f "$time_log" ]; then
    max_ram_kb="$(grep "Maximum resident set size" "$time_log" | awk '{print $NF}' || echo "NA")"
    cpu_percent="$(grep "Percent of CPU this job got" "$time_log" | awk '{print $NF}' | tr -d '%' || echo "NA")"
  fi

  if [ -f "$gpu_log" ]; then
    max_gpu_util="$(awk -F',' 'NR > 1 {gsub(/ %/, "", $4); if ($4+0 > max) max=$4+0} END {print max == "" ? "NA" : max}' "$gpu_log")"
    max_gpu_mem="$(awk -F',' 'NR > 1 {gsub(/ MiB/, "", $6); if ($6+0 > max) max=$6+0} END {print max == "" ? "NA" : max}' "$gpu_log")"
    max_gpu_power="$(awk -F',' 'NR > 1 {gsub(/ W/, "", $8); if ($8+0 > max) max=$8+0} END {print max == "" ? "NA" : max}' "$gpu_log")"
    max_gpu_temp="$(awk -F',' 'NR > 1 {gsub(/ C/, "", $9); if ($9+0 > max) max=$9+0} END {print max == "" ? "NA" : max}' "$gpu_log")"
  fi

  echo -e "${image}\t${elapsed}\t${status}\t${max_ram_kb}\t${cpu_percent}\t${max_gpu_util}\t${max_gpu_mem}\t${max_gpu_temp}\t${max_gpu_power}" >> "$DETAILS"
done

echo "End: $(date)"
echo "Benchmark files written to: $BENCH_DIR"
