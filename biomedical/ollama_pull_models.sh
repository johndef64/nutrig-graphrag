#!/bin/bash

MODELS=(
  # "deepseek-v2"
  # "gemma2"
  "gemma2:27b"
  "qwen2:7b"
  "llama3.1:8b"
  "qwen2.5:14b"
)

for model in "${MODELS[@]}"; do
  echo "Pulling model: $model"
  ollama pull "$model"
  if [ $? -ne 0 ]; then
    echo "Failed to pull $model"
  fi
done