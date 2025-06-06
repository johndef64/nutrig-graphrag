import os
import re
import sys
import time
import json
import logging
import argparse
import requests
from groq import Groq
from tqdm import tqdm
from pathlib import Path
from openai import OpenAI

# MODEL =  "meta-llama/llama-4-scout-17b-16e-instruct"
# MODEL = "llama-3.1-8b-instant"
GROQ_MODEL      = "llama-3.3-70b-versatile"
OPENAI_MODEL    = "gpt-4.1"
MAX_TOKENS      = 32_768
SLEEP_TIME      = 5 # sleep time between API queries

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('graph_comparison_results_openai_4_1.log'),
        logging.StreamHandler(),
    ]
)

def load_api_keys(file_path='api_keys.json'):
    with open(file_path, 'r') as fin:
        return json.load(fin)
    return None

def extract_json_from_response(response):
    """Extract valid JSON array from LLM response, handling malformed or partial content."""
    if not response:
        return None

    # Remove markdown code blocks
    response = re.sub(r'```json\s*', '', response)
    response = re.sub(r'```\s*', '', response)

    # Try direct parsing first
    try:
        parsed = json.loads(response.strip())
        return parsed
    except Exception:
        pass

    # Find first JSON array
    start_idx = response.find('[')
    if start_idx == -1:
        return None

    # Try to find matching closing bracket
    bracket_count = 0
    for end_idx in range(start_idx, len(response)):
        if response[end_idx] == '[':
            bracket_count += 1
        elif response[end_idx] == ']':
            bracket_count -= 1
            if bracket_count == 0:
                candidate = response[start_idx:end_idx+1]
                try:
                    return json.loads(candidate)
                except Exception:
                    break  # Stop at the first plausible array

    # Fallback: extract entries inside {...}
    entries = []
    brace_count = 0
    current_entry = ""
    in_string = False
    escape_next = False
    i = start_idx + 1

    while i < len(response):
        char = response[i]
        if escape_next:
            current_entry += char
            escape_next = False
        elif char == '\\':
            current_entry += char
            escape_next = True
        elif char == '"' and not escape_next:
            in_string = not in_string
            current_entry += char
        elif not in_string:
            if char == '{':
                brace_count += 1
                current_entry += char
            elif char == '}':
                brace_count -= 1
                current_entry += char
                if brace_count == 0:
                    try:
                        entry = json.loads(current_entry.strip())
                        entries.append(entry)
                    except Exception:
                        pass
                    current_entry = ""
            elif char == ',' and brace_count == 0:
                pass  # Ignore commas between top-level entries
            elif char == ']':
                break
            else:
                current_entry += char
        else:
            current_entry += char
        i += 1

    return entries if entries else None

def query_ollama(prompt, model="gemma2:27b"):
    """Query local Ollama instance, expect JSON reply."""
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': model,
                'prompt': prompt,
                'stream': False
            },
            timeout=360
        )
        # Extract the generated text
        result = response.json()['response']
        return result
    except Exception as e:
        logging.error(f"Ollama query failed: {e}")
        return None

def query_remote(client_type, client, prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL if client_type == 'groq' else OPENAI_MODEL,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a biomedical entity classification expert. Always respond with valid JSON only."
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,      
                max_tokens=MAX_TOKENS,      
                top_p=0.9,           
                stream=False
            )
            
            result = response.choices[0].message.content
            logging.info(result)
            return result
                
        except Exception as e:
            logging.warning(f"Remote attempt {attempt+1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)  # Pausa prima del retry
            
    logging.error('All remote attempts failed')
    return None

def validate_entities_with_llm(client_type, client, extracted_types, max_retries=5):
    """Validate entity types with LLM. Return JSON-decoded result."""
    expected_types = [
        "Gene", "Genetic Variant", "Transcript", "Protein", "Nutrient",
        "Food", "Dietary Pattern", "Physiological Process", "Metabolic Pathway",
        "Molecular Interaction", "Environmental Factor", "Disease"
    ]

    # prompt = f"""You are a biomedical entity type classification expert.

    #     TASK: For each extracted entity type, classify it as one of:
    #     - "EXACT": exact match to expected type
    #     - "SYNONYM": synonym of an expected type  
    #     - "INVALID": not a biomedical entity type

    #     RULES:
    #     - If an extracted type could map to multiple expected types, choose the most specific match
    #     - Consider common biomedical abbreviations and alternative names as SYNONYM
    #     - Mark as INVALID only if clearly not biomedical (e.g., "Car", "Building")

    #     EXPECTED TYPES:
    #     {expected_types}

    #     EXTRACTED TYPES TO CLASSIFY:
    #     {extracted_types}

    #     OUTPUT FORMAT: Return a valid JSON array ONLY. No explanations, comments, or additional text.

    #     SCHEMA:
    #     [
    #     {{
    #         "type": "<extracted_type>",
    #         "classification": "<EXACT|SYNONYM|INVALID>", 
    #         "maps_to": "<expected_type or none>"
    #     }}
    #     ]

    #     EXAMPLE:
    #     Input: ["Gene", "SNP", "Vehicle"]
    #     Output:
    #     [
    #     {{"type": "Gene", "classification": "EXACT", "maps_to": "Gene"}},
    #     {{"type": "SNP", "classification": "SYNONYM", "maps_to": "Genetic Variant"}},
    #     {{"type": "Vehicle", "classification": "INVALID", "maps_to": "none"}}
    #     ]

    #     CRITICAL: Your JSON must contain exactly {len(extracted_types)} objects, one for each extracted type.

    #     JSON OUTPUT:"""

    prompt = f"""You are a biomedical expert. For each extracted entity type, classify it as one of:
            - "EXACT": exact match to expected type
            - "SYNONYM": synonym of expected type
            - "INVALID": not biomedical

            Expected types: {expected_types}
            Extracted types: {extracted_types}

            Respond **ONLY** as valid JSON list, using this schema:
            [
            {{"type": "extracted_type", "classification": "EXACT/SYNONYM/INVALID", "maps_to": "expected_type or none"}}
            ]
            """

    for attempt in range(max_retries):
        if client_type == 'local':
          response = query_ollama(prompt)
        else:
          response = query_remote(client_type, client, prompt)
        
        parsed = extract_json_from_response(response)
        if parsed:
            try:
                # parsed = json.loads(response)
                # Ensure each type from extracted_types is processed
                processed_types = set([ent['type'] for ent in parsed if 'type' in ent])
                unprocessed = set(extracted_types) - processed_types
                # Add any missing types as INVALID
                for up in unprocessed:
                    parsed.append({
                        "type": up,
                        "classification": "INVALID",
                        "maps_to": "none"
                    })
                return parsed
            except Exception as e:
                logging.error(f"Failed to parse LLM response as JSON (attempt {attempt+1}): {e}\nResponse was:\n{response}")
        logging.warning(f"Validation attempt {attempt+1} failed, retrying...")
    logging.error("All LLM validation attempts failed.")
    return None

def summarize_validation(parsed, extracted_types):
    """Given parsed LLM results, return summary dict (matches, synonyms, invalids)."""
    exact_matches = []
    synonym_matches = {}
    invalid_types = []

    if parsed is None:
        return {
            'exact_matches': [],
            'synonym_matches': {},
            'invalid_types': extracted_types
        }

    for entry in parsed:
        typ = entry.get("type")
        cls = entry.get("classification", "").upper()
        maps_to = entry.get("maps_to", "none")
        if cls == "EXACT":
            exact_matches.append(typ)
        elif cls == "SYNONYM":
            synonym_matches[typ] = maps_to
        elif cls == "INVALID":
            invalid_types.append(typ)
    return {
        'exact_matches': exact_matches,
        'synonym_matches': synonym_matches,
        'invalid_types': invalid_types
    }

def calculate_composite_score(stats, validation_summary):
    """Calculate composite and component scores."""
    connectivity = stats['lcc_size'] / stats['num_nodes'] if stats['num_nodes'] > 0 else 0
    isolation_penalty = stats['num_isolated_nodes'] / stats['num_nodes'] if stats['num_nodes'] > 0 else 0
    structure_quality = max(0, 1 - isolation_penalty)
    if validation_summary and stats.get('entity_types'):
        valid_entities = len(validation_summary['exact_matches']) + len(validation_summary['synonym_matches'])
        entity_coherence = valid_entities / len(stats['entity_types'])
    else:
        entity_coherence = 0

    composite = (connectivity * 0.4) + (structure_quality * 0.3) + (entity_coherence * 0.3)
    return {
        'composite_score': composite,
        'connectivity_score': connectivity,
        'structure_quality': structure_quality,
        'entity_coherence': entity_coherence
    }

def analyze_all_graphs(client_type, client):
    """Analyze all graph stats files."""
    stats_dir = Path('graphs_stats')
    if not stats_dir.exists():
        logging.error(f"Directory {stats_dir} not found")
        return [], []
    files = list(stats_dir.glob('*.json'))
    logging.info(f"Found {len(files)} graph stats files")
    results = []
    failed_graphs = []

    for file in tqdm(files, desc="Analyzing graphs"):
        try:
            with open(file, 'r') as f:
                stats = json.load(f)
            model_name = file.stem.replace('graph_stats_', '')
            logging.info(f"Processing {model_name}")

            validation_json = None
            validation_summary = None
            if stats.get('entity_types'):
                validation_json = validate_entities_with_llm(client_type, client, stats['entity_types'])
                validation_summary = summarize_validation(validation_json, stats['entity_types'])
                if validation_summary:
                    logging.info(f"Entity validation completed for {model_name}")
                else:
                    logging.warning(f"Entity validation failed for {model_name}")
            else:
                validation_summary = None

            scores = calculate_composite_score(stats, validation_summary)
            result = {
                'model': model_name,
                'num_nodes': stats['num_nodes'],
                'num_edges': stats['num_edges'],
                'lcc_size': stats['lcc_size'],
                'lcc_ratio': stats['lcc_size'] / stats['num_nodes'] if stats['num_nodes'] > 0 else 0,
                'num_entity_types': len(stats.get('entity_types', [])),
                'isolated_nodes': stats['num_isolated_nodes'],
                'graph_density': stats['graph_density'],
                **scores
            }
            if validation_summary:
                result.update({
                    'valid_entities': len(validation_summary['exact_matches']) + len(validation_summary['synonym_matches']),
                    'invalid_entities': len(validation_summary['invalid_types']),
                    'exact_matches': len(validation_summary['exact_matches']),
                    'synonym_matches': len(validation_summary['synonym_matches']),
                    'invalid_types_list': validation_summary['invalid_types']
                })
            results.append(result)
            if client_type != 'local':
                time.sleep(SLEEP_TIME)
        except Exception as e:
            logging.error(f"Error processing {file.name}: {e}")
            failed_graphs.append(file.name)
    return results, failed_graphs

def print_results(results, failed_graphs):
    """Print and log all model scores and summaries."""
    if not results:
        print("No results to display.")
        return

    results_sorted = sorted(results, key=lambda x: x['composite_score'], reverse=True)

    print("\n" + "="*80)
    print("GRAPH MODEL COMPARISON RESULTS")
    print("="*80)

    print(f"\n{'Rank':<4} {'Model':<35} {'Score':<8} {'LCC%':<7} {'Entities':<8} {'Valid':<6} {'Invalid':<7}")
    print("-" * 80)
    for i, r in enumerate(results_sorted, 1):
        print(f"{i:<4} {r['model']:<35} {r['composite_score']:.3f} "
              f"{r['lcc_ratio']:.3f} {r['num_entity_types']:<8} {r.get('valid_entities', 'N/A'):<6} {r.get('invalid_entities', 'N/A'):<7}")
    print("\nDetailed per-model breakdown:")
    print(f"{'Model':<35} {'Nodes':<8} {'Edges':<8} {'LCC':<8} {'Density':<8} {'Isolated':<8} {'Entity coherence':<15}")
    print("-" * 110)
    for r in results_sorted:
        print(f"{r['model']:<35} {r['num_nodes']:<8} {r['num_edges']:<8} {r['lcc_size']:<8} "
              f"{r['graph_density']:.4f} {r['isolated_nodes']:<8} {r['entity_coherence']:.3f}")

    if failed_graphs:
        print("\nWARNING: The following graphs failed and were skipped:")
        for fg in failed_graphs:
            print(f"  - {fg}")

    # Save results to JSON
    with open('graph_comparison_results_openai_4_1.json', 'w') as f:
        json.dump(results_sorted, f, indent=2)
    print("\nResults saved to and 'graph_comparison_results_openai_4_1.json'.")

def main(client_type, client):
    logging.info("Starting graph comparison analysis")
    results, failed_graphs = analyze_all_graphs(client_type, client)
    print_results(results, failed_graphs)
    if results:
        logging.info("Analysis completed successfully")
    else:
        logging.error("No results generated.")
    if failed_graphs:
        logging.error(f"Some graphs failed: {failed_graphs}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--client', '-c', type=str, choices=['local', 'groq', 'openai'])
    args = parser.parse_args()
    client_type = args.client

    api_keys = load_api_keys()
    GROQ_API_KEY = api_keys["groq"]
    OPENAI_API_KEY = api_keys["openai"] 

    client = None
    if client_type == 'groq':
        client = Groq(api_key=GROQ_API_KEY)
    elif client_type == 'openai':
        client = OpenAI(api_key=OPENAI_API_KEY)
    elif client_type == 'local':
        client = None
    else:
        logging.error(f'Wrong client type {client_type}')
        
    main(client_type, client)