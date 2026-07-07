#!/bin/bash
#SBATCH -p gpu
#SBATCH --time=1-00:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=512G
#SBATCH --gres=gpu:1
#SBATCH -o logs/tda_%j.out
#SBATCH -e logs/tda_%j.err

set -euo pipefail

module load micromamba

ENV_NAME="env-tda"
INPUT_DIR=""
OUTPUT_DIR=""
RUNNER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_segmentation.py"

SEG_DIR="${OUTPUT_DIR}/segmentations"
PREVIEW_DIR="${OUTPUT_DIR}/preview"
BENCH_DIR="${OUTPUT_DIR}/benchmarks/job_${SLURM_JOB_ID}"

mkdir -p "$SEG_DIR" "$PREVIEW_DIR" "$BENCH_DIR/time_logs"

SUMMARY="${BENCH_DIR}/tda_summary.tsv"
DETAILS="${BENCH_DIR}/tda_details.tsv"

echo -e "job_id\timage\tstart_time\tend_time\telapsed_seconds\tstatus\tlabels_out\tpreview_out\ttime_log" > "$SUMMARY"

{
  echo "Job ID: ${SLURM_JOB_ID}"
  echo "Node: ${SLURMD_NODENAME}"
  echo "CPUs: ${SLURM_CPUS_PER_TASK:-NA}"
  echo "Memory requested: ${SLURM_MEM_PER_NODE:-NA} MB"
  echo "Environment: ${ENV_NAME}"
  echo "Input directory: ${INPUT_DIR}"
  echo "Output directory: ${OUTPUT_DIR}"
  echo "Runner: ${RUNNER}"
  echo "Start: $(date)"
} > "${BENCH_DIR}/job_info.txt"

run_tda() {
  /usr/bin/time -v -o "$4" \
    micromamba run -n "$ENV_NAME" python "$RUNNER" \
      --image "$1" \
      --labels-out "$2" \
      --preview-out "$3"
}

for image in "$INPUT_DIR"/*.tif; do
  [ -e "$image" ] || continue

  filename="$(basename "$image")"
  name="$(basename "${image%.tif}" | tr ' /' '__')"
  labels_out="${SEG_DIR}/${filename}"
  preview_out="${PREVIEW_DIR}/preview_${filename}"
  time_log="${BENCH_DIR}/time_logs/${name}.txt"

  echo "Processing: $image"
  start_epoch="$(date +%s)"
  start_iso="$(date --iso-8601=seconds)"

  status="OK"
  run_tda "$image" "$labels_out" "$preview_out" "$time_log" || status="FAILED"

  end_epoch="$(date +%s)"
  end_iso="$(date --iso-8601=seconds)"
  elapsed="$((end_epoch - start_epoch))"

  echo -e "${SLURM_JOB_ID}\t${image}\t${start_iso}\t${end_iso}\t${elapsed}\t${status}\t${labels_out}\t${preview_out}\t${time_log}" >> "$SUMMARY"
  echo "Done: $image (${elapsed}s, ${status})"

  [ "$status" = "OK" ] || exit 1
done

if command -v sacct >/dev/null 2>&1; then
  sacct -j "$SLURM_JOB_ID" \
    --format=JobID,JobName,Partition,AllocCPUS,Elapsed,TotalCPU,MaxRSS,ReqMem,State,ExitCode \
    > "${BENCH_DIR}/slurm_sacct.txt" || true
fi

echo -e "image\telapsed_seconds\tstatus\tmax_ram_kb\tmax_ram_gb\tcpu_percent\tuser_seconds\tsystem_seconds\tlabels_out\tpreview_out" > "$DETAILS"

tail -n +2 "$SUMMARY" | while IFS=$'\t' read -r _ image _ _ elapsed status labels_out preview_out time_log; do
  max_ram_kb="NA"
  max_ram_gb="NA"
  cpu_percent="NA"
  user_seconds="NA"
  system_seconds="NA"

  if [ -f "$time_log" ]; then
    max_ram_kb="$(grep "Maximum resident set size" "$time_log" | awk '{print $NF}' || echo "NA")"
    cpu_percent="$(grep "Percent of CPU this job got" "$time_log" | awk '{print $NF}' | tr -d '%' || echo "NA")"
    user_seconds="$(grep "User time" "$time_log" | awk '{print $NF}' || echo "NA")"
    system_seconds="$(grep "System time" "$time_log" | awk '{print $NF}' || echo "NA")"
    if [ "$max_ram_kb" != "NA" ]; then
      max_ram_gb="$(awk -v kb="$max_ram_kb" 'BEGIN {printf "%.3f", kb / 1024 / 1024}')"
    fi
  fi

  echo -e "${image}\t${elapsed}\t${status}\t${max_ram_kb}\t${max_ram_gb}\t${cpu_percent}\t${user_seconds}\t${system_seconds}\t${labels_out}\t${preview_out}" >> "$DETAILS"
done

echo "End: $(date)"
echo "Benchmark files written to: $BENCH_DIR"
