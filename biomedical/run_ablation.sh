#!/bin/bash

EMBEDDERS=("all-mpnet-base-v2" "dmis-lab/biobert-v1.1")
LLMS=("deepseek-v2" "gemma2" "gemma2:27b" "qwen2:7b" "llama3.1:8b" "qwen2.5:14b")

for llm in "${LLMS[@]}"; do
  for embedder in "${EMBEDDERS[@]}"; do
    echo "Running: python main_graph_build.py --embedder \"$embedder\" --llm \"$llm\""
    python main_graph_build.py --embedder "$embedder" --llm "$llm"
    if [ $? -ne 0 ]; then
      echo "WARNING: Error with embedder=$embedder and llm=$llm, continuing..."
      continue
    fi
  done
  echo "Freeing Ollama memory for model: $llm"
  ollama stop "$llm"
done