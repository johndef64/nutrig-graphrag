#%%
import os
import logging
import ollama
import numpy as np
import pandas as pd
from openai import AsyncOpenAI
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash, wrap_embedding_func_with_attrs
from sentence_transformers import SentenceTransformer

from biomedical.llm_utils import *

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)


######################################################
# Set WORKING_DIR
# os.chdir(r"/root/projects/nano-graphrag/biomedical")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())



#### Choose the model to use for the RAG #####
OLLAMA_MODELS = {
    0: "deepseek-v2",
    1: "gemma2",
    2: "gemma2:27b",
    3: "qwen2:7b",
    4: "llama3.1:8b",
    5: "qwen2.5:14b"
}

GROQ_MODELS = {
    0: "gemma2-9b-it",
    1: "llama-3.3-70b-versatile",
    2: "llama-3.1-8b-instant",
    3: "llama-guard-3-8b",
    4: "llama3-70b-8192",
    5: "llama3-8b-8192",
    6: "deepseek-r1-distill-llama-70b",
    7: "meta-llama/llama-4-maverick-17b-128e-instruct",
    8: "meta-llama/llama-4-scout-17b-16e-instruct",
    9: "mistral-saba-24b",
    10: "qwen-qwq-32b"
}

DEEP_MODELS = {
    0: "deepseek-chat"
}

# Choose a model from the GROQ_MODELS dictionary
os.environ['MODEL'] =  GROQ_MODELS[0]  # <===== Change this to select a different model
# os.environ['MODEL'] =  OLLAMA_MODELS[0] 

print(f"Using model: {os.environ['MODEL']}")

######################################################


##### Choose the embedding model #####
BERT_MODELS = {
    0: "dmis-lab/biobert-v1.1",
    1: "all-MiniLM-L6-v2",
    2: "all-mpnet-base-v2"
}
OPENAI_EMBEDDER = "text-embedding-3-small"  
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"

EMBEDDER = BERT_MODELS[2]  # <===== Change this to select a different embedding model

######################################################

#%%


# Set working Directory
project = "nutrig-graphrag"
model_name = os.environ['MODEL'].replace("/", "_").replace(":", "_")
WORKING_DIR = f"./{project}_{model_name}_{EMBEDDER}_1"  
# WORKING_DIR = "./ablation_study/cache_gemma2_dmis-lab_biobert-v1.1" 
# WORKING_DIR = "/root/projects/nano-graphrag/biomedical/answer_evaluation/cache_gemma2_all-mpnet-base-v2"  # For testing purposes, use a dummy cache directory
WORKING_DIR = "/root/projects/nano-graphrag/biomedical/nutrig-graphrag_gemma2_all-mpnet-base-v2"

print(f"Working Directory: {WORKING_DIR}")
print(f"LLM Model: {os.environ['MODEL']}")
print(f"Embedding Model: {EMBEDDER}")
#%%
######################################################


def remove_if_exist(file):
    if os.path.exists(file):
        os.remove(file)

def query_global(question, **kwargs):
    """
    Default parameters for global query:
    global_min_community_rating: float = 0
    global_max_consider_community: float = 512
    global_max_token_for_community_report: int = 16384
    """
    # rag = GraphRAG(
    #     working_dir=WORKING_DIR,
    #     best_model_func=USE_LLM,
    #     cheap_model_func=USE_LLM,
    #     embedding_func=local_embedding,
    # )
    print(f"Usint {EMBEDDER} embedding model")
    print(f"Using {os.environ['MODEL']} model for LLM")
    rag = NutrigGraphRAG(GraphRAG,
        working_dir=WORKING_DIR,
        llm_model=os.environ['MODEL'],
        embedding_model=EMBEDDER,
        )
    response = rag.query(
            question, param=QueryParam(mode="global", 
                                       **kwargs
                                       )
        )
    print(response)
    return response


def query_local(question):

    rag = NutrigGraphRAG(GraphRAG,
        working_dir=WORKING_DIR,
        llm_model=os.environ['MODEL'],
        embedding_model=EMBEDDER,
        )
    print(f"Usint {EMBEDDER} embedding model")
    print(f"Using {os.environ['MODEL']} model for LLM")

    response = rag.query(
            question, param=QueryParam(mode="local")
        )
    print(response)
    return response



def query_naive(question, **kwargs):
    """
    top_k: int = 20
    # naive search
    naive_max_token_for_text_unit = 12000
    """
    # rag = GraphRAG(
    #     working_dir=WORKING_DIR,
    #     best_model_func=USE_LLM,
    #     cheap_model_func=USE_LLM,
    #     enable_naive_rag =True,
    #     embedding_func=embedder,
    # )
    print(f"Usint {EMBEDDER} embedding model")
    print(f"Using {os.environ['MODEL']} model for LLM")
    rag = NutrigGraphRAG(GraphRAG,
        working_dir=WORKING_DIR,
        llm_model=os.environ['MODEL'],
        embedding_model=EMBEDDER,
        enable_naive_rag = True,
        )

    response = rag.query(
            question, param=QueryParam(mode="naive", **kwargs)
        )
    print(response)
    return response

# Get Questions
import pickle
os.chdir("/root/projects/nano-graphrag/biomedical/")
# root = os.path.dirname(os.path.abspath(__file__))
root = "question_generation"
# questions_filename = "questions_cache_gemma2_all-mpnet-base-v2.pkl"
questions_filename = "question_dataframe_v4.pkl"


questions_file = os.path.join(root, questions_filename)
questions = pickle.load(open(questions_file, "rb"))
#questions.columns = ['Category', 'User', 'Task', 'Questions']
questions.columns = ['Category', 'User', 'Task', 'Questions','UserID']

question = "Tell me hypertension-associated Mutations and genes"
question = "What is SOD and what are his main relationships in nutrition?"
# question = "tell me to what condition are Genetic variants in DLG5 associated"
question="What gene is associated with rs45500793 and what disease?"
# questions = questions[:10]

questions
print( f"Number of questions: {len(questions)*5}")
answer_df = pd.DataFrame(columns=["User","Task", "Question","GraphRAG-Answer","Naive-RAG-Answer","No-RAG-Answer"])
# add n 
answer_df["User"] = questions.UserID
answer_filename = "answer_evaluation/answers_V4_nutrig-graphrag_gemma2_all-mpnet-base-v2_V3.pkl"
if os.path.exists(answer_filename):
    answer_df = pd.read_pickle(answer_filename)
else:
    answer_df.to_pickle(answer_filename)

answer_df.Question.to_list()

questions#.columns
answer_df
#%%
GROQ_MODELS = {
    0: "gemma2-9b-it",
    1: "llama-3.3-70b-versatile",
    2: "llama-3.1-8b-instant",
    3: "llama-guard-3-8b",
    4: "llama3-70b-8192",
    5: "llama3-8b-8192",
    6: "deepseek-r1-distill-llama-70b",
    7: "meta-llama/llama-4-maverick-17b-128e-instruct",
    8: "meta-llama/llama-4-scout-17b-16e-instruct",
    9: "mistral-saba-24b",
    10: "qwen-qwq-32b"
}
os.environ['MODEL'] = GROQ_MODELS[1] 
# MOdel context window
# gemma2-9b-it 8,192
# llama-3.3-70b-versatile 32,768
# llama-3.1-8b-instant 8,192


import re
import time
from groq import Groq
no_rag = False  # Set to True if you want to test the No-RAG query

time1 = time.time()
counter = 0
# Loop through each question and run the queries
for n in tqdm(range(len(questions))):

    user = questions.UserID[n]
    task = questions.Task[n]
    print(f"User: {user}, Task: {task}")

    question = questions.Questions[n]

    #for question in questions.Questions[n]:

    answer_df = pd.read_pickle(answer_filename)
    in_df = question in answer_df.Question.to_list()
    try:
        #is_answered = answer_df[answer_df["Question"] == question]["GraphRAG-Answer"].values[0] != None
        #is_answered = answer_df["GraphRAG-Answer"][n] != None
        is_answered = not pd.isna(answer_df["GraphRAG-Answer"][n])
    except KeyError:
        is_answered = True

    # if not in_df or not is_answered:
    if not is_answered:
        counter += 1
        if counter % 1 == 0 and counter > 1:
            sec = 12
            print(f"Sleeping for {sec} seconds to avoid rate limiting...")
            time.sleep(sec)

        # usa una regex per estrarre la domanda
        # question = re.split(r'Q\d+\. ', question, 1)[1]

        print(f"Question: {question}")

        ########## RUN THE Queries ##########
        
        
        try:
            print("\n\n<<<<<<<<<<<<< GraphRAG Answer >>>>>>>>>>>>>>>")
            graphrag_response = query_global(question,
                                            global_max_consider_community = 512,
                                            global_min_community_rating  =0,
                                            global_max_token_for_community_report = 16384)
            print("<<< ----------------- >>>")
        except Exception as e:
            print(f"Error in GraphRAG query: {e}")
            graphrag_response = None
        
        try:
            # Naive RAG Query
            print("\n\n<<<<<<<<<<<<< Naive-RAG Answer >>>>>>>>>>>>>>>")
            naive_response = query_naive(question,  
                                        top_k=10, 
                                        naive_max_token_for_text_unit = 12000) # threshold=0.2
            # naive_response = QueryNaiveGraphRAG(
            #                 question,llm_model=os.environ["MODEL"], 
            #                 working_dir=WORKING_DIR, k=10, threshold=0.6, 
            #                 BERT_MODEL= EMBEDDER
            #                 )
            print("<<< ----------------- >>>")
        except Exception as e:
            print(f"Error in NaiveRAG query: {e}")
            graphrag_response = None

        
        if no_rag:
            print("\n\n<<<<<<<<<<<<< No-RAG Answer >>>>>>>>>>>>>>>")
            GROQ_API_KEY = api_keys["groq"]
            MODEL = os.environ['MODEL'] 
            
            # Test Native LLM response
            client = Groq(api_key=GROQ_API_KEY)

            norag_response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": question},
                ],
                stream=False
            )
            norag_response = norag_response.choices[0].message.content
        else:
            norag_response = None

        print("<<< ----------------- >>>")

        # answer_df["User"][n] = user
        answer_df["Task"][n] = task
        answer_df["Question"][n] = question
        answer_df["GraphRAG-Answer"][n] = graphrag_response
        answer_df["Naive-RAG-Answer"][n] = naive_response
        answer_df["No-RAG-Answer"][n] = norag_response

        # # Dati da aggiungere
        # new_row = {
        #     "User": user,
        #     "Task": task,
        #     "Question": question,
        #     "GraphRAG-Answer": graphrag_response,
        #     "Naive-RAG-Answer": naive_response,
        #     "No-RAG-Answer": norag_response
        # }

        # new_df = pd.DataFrame([new_row])
        # answer_df = pd.concat([answer_df, new_df], ignore_index=True)
        answer_df.to_pickle(answer_filename)

        #time.sleep(1)  # Sleep to avoid rate limiting
    else:

        print(f"Question already answered: {question}")
        continue    

print(f"Time taken for question: {time.time() - time1:.2f} seconds")
# %%

# Save Answer df
answer_df.to_csv(f"answer_evaluation/answers_V2_nutrig-graphrag_gemma2_all-mpnet-base-v2_PARTIAL.csv")
answer_df
# tempo: 15 domande in 6 minuti con time sleep 10 ongni 2 ed una skippata 

# %%
2
# %%
GROQ_MODELS = {
    0: "gemma2-9b-it",
    1: "llama-3.3-70b-versatile",
    2: "llama-3.1-8b-instant",
    3: "llama-guard-3-8b",
    4: "llama3-70b-8192",
    5: "llama3-8b-8192",
    6: "deepseek-r1-distill-llama-70b",
    7: "meta-llama/llama-4-maverick-17b-128e-instruct",
    8: "meta-llama/llama-4-scout-17b-16e-instruct",
    9: "mistral-saba-24b",
    10: "qwen-qwq-32b"
}
os.environ['MODEL'] = GROQ_MODELS[4] 

# try naive
replyes = []
for n in range(1):
    user = questions.User[n]
    task = questions.Task[n]
    print(f"User: {user}, Task: {task}")
    for question in questions.Questions[n][:1]:
        print(f"Question: {question}")

        ########## RUN THE Queries ##########

        print("<<< ----------------- >>>")
        
        # Naive RAG Query
        print("\n\n<<<<<<<<<<<<< Naive-RAG Answer >>>>>>>>>>>>>>>")
        print(f"Using {EMBEDDER} embedding model")
        print(f"Using {os.environ['MODEL']} model for LLM")
        naive_response = query_naive(question, 
                                     top_k=8)
        # naive_response = QueryNaiveGraphRAG(
        #         question,llm_model=os.environ["MODEL"], 
        #         working_dir=WORKING_DIR, k=10, threshold=0.6, 
        #         BERT_MODEL= EMBEDDER
        #         )
        replyes.append(naive_response)
# %%
set(replyes)
    
# %%



import re
import time
from groq import Groq
no_rag = False  # Set to True if you want to test the No-RAG query

time1 = time.time()
counter = 0
# Loop through each question and run the queries
for n in tqdm(range(len(questions))):

    user = questions.UserID[n]
    task = questions.Task[n]
    question = questions.Questions[n]

    answer_df = pd.read_pickle(answer_filename)
    in_df = question in answer_df.Question.to_list()
    try:
        is_answered = not pd.isna(answer_df["GraphRAG-Answer"][n])
    except KeyError:
        is_answered = True

    # if not in_df or not is_answered:
    if not is_answered:
        print(f"Question {n}: not answered yet, running queries...")

# %%
# answer_df["GraphRAG-Answer"][42] != nan
# pd.isna(answer_df["GraphRAG-Answer"][42])
