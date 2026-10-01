#!/usr/bin/env bash
# Download the Hub-hosted checkpoints named in docs/ASSETS.md.
# The three .pth files (BLIP-2 Q-Former, EVA ViT-G, MiniGPT-4) are released by
# LAVIS / EVA / MiniGPT-4 rather than the Hub; fetch those from the links in
# docs/ASSETS.md and drop them in $PMRL_MODEL_DIR yourself.
set -euo pipefail

MODEL_DIR="${PMRL_MODEL_DIR:-$(cd "$(dirname "$0")/../.." && pwd)/hugging_cache}"
mkdir -p "$MODEL_DIR"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli is not on PATH. Install huggingface_hub and retry." >&2
  exit 1
fi

declare -A REPOS=(
  [opt-2.7b]=facebook/opt-2.7b
  [opt-125m]=facebook/opt-125m
  [bert-base-uncased]=bert-base-uncased
  [vicuna-7b-v1.5]=lmsys/vicuna-7b-v1.5
  [Qwen2-VL-7B-Instruct]=Qwen/Qwen2-VL-7B-Instruct
  [Qwen2-0.5B-Instruct]=Qwen/Qwen2-0.5B-Instruct
  [llava-onevision-qwen2-7b-ov-hf]=llava-hf/llava-onevision-qwen2-7b-ov-hf
  [all-MiniLM-L6-v2]=sentence-transformers/all-MiniLM-L6-v2
  [distilbert-base-cased]=distilbert-base-cased
)

for name in "${!REPOS[@]}"; do
  dest="$MODEL_DIR/$name"
  if [ -d "$dest" ] && [ -n "$(ls -A "$dest" 2>/dev/null || true)" ]; then
    echo "skip $name (already present)"
    continue
  fi
  echo "download ${REPOS[$name]} -> $dest"
  huggingface-cli download "${REPOS[$name]}" --local-dir "$dest"
done

echo "Hub checkpoints are in $MODEL_DIR."
echo "Still required, see docs/ASSETS.md: blip2_pretrained_opt2.7b.pth, blip2_pretrained_flant5xxl.pth, eva_vit_g.pth, pretrained_minigpt4_7b.pth, and the MMEdit image folders."
