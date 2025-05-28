import os
import time
import logging
import argparse
import numpy as np
import pandas as pd
from huggingface_hub import login
from nano_graphrag import GraphRAG
from sentence_transformers import SentenceTransformer
from nano_graphrag._utils import wrap_embedding_func_with_attrs

from biomedical.utils import *
from biomedical.graphml_visualize import CreateGraphVisualization

# logging.basicConfig(level=logging.WARNING)
# logging.getLogger("nano-graphrag").setLevel(logging.INFO)

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def setup_logging(log_file):
	logger = logging.getLogger()
	logger.setLevel(logging.INFO)  # Livello globale

	# Rimuovi eventuali handler già esistenti
	if logger.hasHandlers():
			logger.handlers.clear()

	# File handler
	fh = logging.FileHandler(log_file)
	fh.setLevel(logging.INFO)
	fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))

	# Console handler
	ch = logging.StreamHandler()
	ch.setLevel(logging.INFO)
	ch.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))

	logger.addHandler(fh)
	logger.addHandler(ch)

def remove_if_exist(file):
	if os.path.exists(file):
		os.remove(file)

def insert_doc(doc, path, llm_fun, embed_fun):
	rag = GraphRAG(
		working_dir=path,
		enable_llm_cache=True,
		best_model_func=llm_fun,
		cheap_model_func=llm_fun,
		embedding_func=embed_fun,
	)
	start = time.time()
	rag.insert(doc)
	print("indexing time:", time.time() - start)

def get_dataset():
	# df = pd.read_csv("datasets/fulltext_dataset.zip")
	df = pd.read_csv("datasets/chunks/0-3000pmc_fulltext.csv")

	# Filter Dataset
	df = df[~((df['INTRO'].isna() & df['METHODS'].isna() & df['RESULTS'].isna() & df['DISCUSS'].isna()))]
	df = df[~((df['RESULTS'].isna() & df['DISCUSS'].isna()))].reset_index(drop=True)
	df = df[~(df['RESULTS'].isna())].reset_index(drop=True)
	df.fillna("", inplace=True)

	df.text = df["RESULTS"] + "\n\n\n\n" + df["DISCUSS"]
	df.text = df["RESULTS"]
	df.text = df.text.str.replace("<SEP>","\n\n")
	return df

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="GraphRAG Biomedical Main Graph Build")
	parser.add_argument("--start_id", type=int, default=0, help="Starting ID for processing")
	parser.add_argument("--batch_size", type=int, default=100, help="Number of entries to process in a batch")
	parser.add_argument("--embedder", type=str, default="all-mpnet-base-v2", choices=["all-mpnet-base-v2","dmis-lab/biobert-v1.1"], help="")
	parser.add_argument("--llm", type=str, default="deepseek-v2" , choices=["deepseek-v2","gemma2", "gemma2:27b", "qwen2:7b", "llama3.1:8b", "qwen2.5:14b"], help="")
	parser.add_argument("--ip", type=str, default=None, help="IP address for the LLM server")
	parser.add_argument("--port", type=int, default=None, help="Port for the LLM server")

	args = parser.parse_args()
	start_id = args.start_id
	batch_size = args.batch_size
	embedder = args.embedder
	llm = args.llm
	ip = args.ip
	port = args.port

	# LOG FILE
	log_file = f"log_{llm}_{embedder}.log".replace(":", "_").replace("/", "_")
	log_path = os.path.join(os.getcwd(), "ablation_logs", log_file)
	setup_logging(log_path)
	
	print(f"[i] Working directory: {os.getcwd()}")
	print(f"[i] Starting from ID: {start_id}, Batch Size: {batch_size}")

	llm_fun = get_ollama_model_fun(llm)
	if ip and port:
		print(f"[i] Connecting to LLM server at {ip}:{port}")
		ollama_client = connect_ollama_server(ip, port)
		llm_fun = get_ollama_model_server_fun(ollama_client)

	working_dir = f"cache_{llm}_{embedder}"
	working_dir = working_dir.replace(":", "_").replace("/", "_")
	print(f"[i] Dataset save path: {working_dir}")

	if embedder == "dmis-lab/biobert-v1.1":
		login("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")

	embed_model = SentenceTransformer(
		embedder, cache_folder=working_dir, device="cpu"
	)

	@wrap_embedding_func_with_attrs(
		embedding_dim=embed_model.get_sentence_embedding_dimension(),
		max_token_size=embed_model.max_seq_length,
	)
	async def embed_fun(texts: list[str]) -> np.ndarray:
		return embed_model.encode(texts, normalize_embeddings=True)
	
	df = get_dataset()
	start_time = time.time()
	if True:
		for i in range(start_id, start_id+batch_size):
			if len(df.text[i]) > 10: 
				insert_doc(df.text[i], working_dir, llm_fun, embed_fun)
				print(f"""\n\n<<<<<<<<<<<<<<<<<<<<<  {i+1}/{batch_size+start_id}  >>>>>>>>>>>>>>>>>>\n\n """)
			else:
				print("\n\n\n\n\nNo text...\n\n\n\n\n")

	end_time = time.time() - start_time
	print(f"Total time: {end_time:.2f} seconds")

	# Generate GraphML file
	graph_save_path = working_dir.split("/")[-1]
	CreateGraphVisualization(graph_save_path)

