#!/usr/bin/env bash
set -euo pipefail

action="${1:-start}"
container_name="${SGLANG_CONTAINER_NAME:-gist-tabletop-sglang}"
image="${SGLANG_IMAGE:-lmsysorg/sglang:latest-cu130}"
cache_dir="${HF_HUB_CACHE_DIR:-${HOME}/.cache/huggingface/hub}"
profile="${SGLANG_PROFILE:-gemma4-26b-a4b}"

profile_total_tokens="16384"
profile_extra_args=()

case "${profile}" in
  gemma4-26b-a4b)
    profile_model="nvidia/Gemma-4-26B-A4B-NVFP4"
    profile_reasoning_parser="gemma4"
    profile_tool_parser="gemma4"
    profile_attention_backend="triton"
    profile_quantization="modelopt_fp4"
    profile_context_length="65536"
    profile_total_tokens="131072"
    profile_memory_fraction="0.40"
    profile_max_running="2"
    profile_disable_cuda_graph="true"
    ;;
  gemma4-31b)
    profile_model="nvidia/Gemma-4-31B-IT-NVFP4"
    profile_reasoning_parser="gemma4"
    profile_tool_parser="gemma4"
    profile_attention_backend="triton"
    profile_quantization="modelopt_fp4"
    profile_context_length="16384"
    profile_memory_fraction="0.34"
    profile_max_running="2"
    profile_disable_cuda_graph="true"
    ;;
  qwen3.8-27b)
    profile_model="nvidia/Qwen3.8-27B-NVFP4"
    profile_reasoning_parser="qwen3"
    profile_tool_parser="qwen3_coder"
    profile_attention_backend="flashinfer"
    profile_quantization=""
    profile_context_length="65536"
    profile_total_tokens="65536"
    profile_memory_fraction="0.40"
    profile_max_running="1"
    profile_disable_cuda_graph="true"
    profile_extra_args=(--kv-cache-dtype fp8_e4m3 --mamba-full-memory-ratio 0.5 --max-mamba-cache-size 8)
    ;;
  qwen3-8b)
    profile_model="nvidia/Qwen3-8B-FP4"
    profile_reasoning_parser="qwen3"
    profile_tool_parser="qwen"
    profile_attention_backend="flashinfer"
    profile_quantization="modelopt_fp4"
    profile_context_length="4096"
    profile_memory_fraction="0.20"
    profile_max_running="4"
    profile_disable_cuda_graph="false"
    ;;
  *)
    echo "Unknown SGLANG_PROFILE: ${profile}" >&2
    echo "Profiles: gemma4-26b-a4b, gemma4-31b, qwen3.8-27b, qwen3-8b" >&2
    exit 2
    ;;
esac

model="${SGLANG_MODEL:-${profile_model}}"
reasoning_parser="${SGLANG_REASONING_PARSER:-${profile_reasoning_parser}}"
tool_parser="${SGLANG_TOOL_PARSER:-${profile_tool_parser}}"
attention_backend="${SGLANG_ATTENTION_BACKEND:-${profile_attention_backend}}"
quantization="${SGLANG_QUANTIZATION:-${profile_quantization}}"
context_length="${SGLANG_CONTEXT_LENGTH:-${profile_context_length}}"
memory_fraction="${SGLANG_MEMORY_FRACTION:-${profile_memory_fraction}}"
max_running="${SGLANG_MAX_RUNNING_REQUESTS:-${profile_max_running}}"
max_total_tokens="${SGLANG_MAX_TOTAL_TOKENS:-${profile_total_tokens}}"

served_model() {
  curl -fsS http://127.0.0.1:30000/v1/models 2>/dev/null \
    | python3 -c 'import json,sys; data=json.load(sys.stdin).get("data", []); print(data[0].get("id", "") if data else "")' \
    2>/dev/null || true
}

start_server() {
  mkdir -p "${cache_dir}"
  if docker ps --format '{{.Names}}' | rg -qx "${container_name}"; then
    current_model="$(served_model)"
    if [[ "${current_model}" == "${model}" ]]; then
      if ! curl -fsS http://127.0.0.1:30000/get_server_info | python3 -c '
import json,sys
d=json.load(sys.stdin)
expected=(int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),float(sys.argv[4]))
actual=(d.get("context_length"),d.get("max_total_tokens"),d.get("max_running_requests"),d.get("mem_fraction_static"))
sys.exit(0 if actual==expected else 1)
' "${context_length}" "${max_total_tokens}" "${max_running}" "${memory_fraction}"; then
        echo "The model is running with different context/KV settings. Use '$0 switch' to apply the requested profile." >&2
        return 1
      fi
      echo "SGLang is already running: ${container_name} (${current_model})"
      return 0
    fi
    echo "SGLang is running ${current_model:-an unknown model}; requested ${model}." >&2
    echo "Use '$0 switch' to replace the running model." >&2
    return 1
  fi
  if docker ps -a --format '{{.Names}}' | rg -qx "${container_name}"; then
    docker rm "${container_name}" >/dev/null
  fi

  server_args=(
    sglang serve
    --model-path "${model}"
    --served-model-name "${model}"
    --host 0.0.0.0
    --port 30000
    --context-length "${context_length}"
    --mem-fraction-static "${memory_fraction}"
    --max-running-requests "${max_running}"
    --max-total-tokens "${max_total_tokens}"
    --chunked-prefill-size 4096
    --attention-backend "${attention_backend}"
    --enable-cache-report
    --trust-remote-code
    --reasoning-parser "${reasoning_parser}"
    --tool-call-parser "${tool_parser}"
  )
  server_args+=("${profile_extra_args[@]}")
  if [[ -n "${quantization}" ]]; then
    server_args+=(--quantization "${quantization}")
  fi
  if [[ "${profile_disable_cuda_graph}" == "true" ]]; then
    server_args+=(--cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled)
  else
    server_args+=(--disable-prefill-cuda-graph --cuda-graph-max-bs-decode 4)
  fi

  docker run -d \
    --name "${container_name}" \
    --gpus all \
    --ipc host \
    --cap-add SYS_NICE \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -p 30000:30000 \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -v "${cache_dir}:/root/.cache/huggingface/hub" \
    "${image}" \
    "${server_args[@]}"
  echo "SGLang container started: ${container_name}"
  echo "Profile: ${profile} | model: ${model} | vision: auto"
  echo "Watch startup: docker logs -f ${container_name}"
}

case "${action}" in
  start)
    start_server
    ;;
  switch)
    docker stop "${container_name}" >/dev/null 2>&1 || true
    docker rm "${container_name}" >/dev/null 2>&1 || true
    start_server
    ;;
  stop)
    docker stop "${container_name}" >/dev/null 2>&1 || true
    echo "SGLang stopped: ${container_name}"
    ;;
  status)
    docker ps -a --filter "name=^/${container_name}$"
    echo "Configured profile: ${profile} | requested model: ${model}"
    echo "Served model: $(served_model)"
    curl -fsS http://127.0.0.1:30000/health || true
    ;;
  logs)
    exec docker logs -f "${container_name}"
    ;;
  *)
    echo "Usage: $0 {start|switch|stop|status|logs}" >&2
    exit 2
    ;;
esac
