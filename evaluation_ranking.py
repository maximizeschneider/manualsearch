import requests
import json
from ir_measures import calc_aggregate, nDCG, P, ScoredDoc, R, MRR
from enum import Enum
from typing import List, Dict, Optional
from pandas import DataFrame
import argparse
import os
from datetime import datetime

class RModel(Enum):
    SPARSE = 1
    DENSE = 2
    HYBRID = 3

def parse_vespa_response(response: dict, qid: str) -> List[ScoredDoc]:
    result = []
    seen_combinations = set()  # Track unique title + page_number combinations
    hits = response['root'].get('children', [])

    for hit in hits:
        doc_id = hit['fields']['doc_id']
        relevance = hit.get('relevance', 0)
        title = hit['fields']['title']
        page_number = hit['fields']['page_number']

        # Create tuple for title + page_number combination
        combination = (title, page_number)

        # Only add if we haven't seen this combination before
        if combination not in seen_combinations:
            seen_combinations.add(combination)

            # Transform doc_id to strip away the chunk number and the format 
            parts = doc_id.split('-')
            trans_doc_id = '-'.join(parts[1:-1])

            result.append(ScoredDoc(qid, trans_doc_id, relevance))

    return result

def search(query: str, qid: str, ranking: str,
           hits: int = 30, language: str = "en",
           mode: RModel = RModel.SPARSE, format: str = "ocr",
           log_entry: Optional[Dict] = None) -> List[ScoredDoc]:
    format_filter = 'text_format contains "{}"'.format(format)
    if mode == RModel.SPARSE:
        yql = "select * from doc where {} and ({{targetHits:100}}userInput(@query))".format(format_filter)
    elif mode == RModel.DENSE:
        yql = "select * from doc where {} and ({{targetHits:100}}nearestNeighbor(embedding, e))".format(format_filter)
    elif mode == RModel.HYBRID:
        yql = "select * from doc where {} and (({{targetHits:100}}nearestNeighbor(embedding,e)) or (userInput(@query)))".format(format_filter)

    query_request = {
        'yql': yql,
        'query': query,
        'ranking': ranking,
        'hits': hits,
        'language': language
    }

    if mode in {RModel.DENSE, RModel.HYBRID}:
        query_request['input.query(e)'] = "embed(@query)"

    # Log the request
    if log_entry is not None:
        log_entry['request'] = query_request

    try:
        response = requests.post("http://localhost:8080/search/", json=query_request)
    except requests.exceptions.RequestException as e:
        print("Search request exception for QID {}: {}".format(qid, e))
        if log_entry is not None:
            log_entry['error'] = str(e)
        return []

    if response.ok:
        response_json = response.json()
        # Log the raw response
        if log_entry is not None:
            log_entry['raw_response'] = response_json

        # Parse the response
        scored_docs = parse_vespa_response(response_json, qid)

        # Log the transformed results
        if log_entry is not None:
            log_entry['transformed_results'] = [
                {'qid': doc.query_id, 'doc_id': doc.doc_id, 'relevance': doc.score}
                for doc in scored_docs
            ]

        return scored_docs
    else:
        error_message = "Search request failed for QID {} with response {}".format(qid, response.text)
        print(error_message)
        if log_entry is not None:
            log_entry['error'] = response.text
        return []

def load_queries_and_qrels(json_file: str) -> tuple[Dict, DataFrame]:
    """
    Load queries and relevance judgments from a JSON file.
    Expected format:
    {
        "queries": {
            "q1": "query text1",
            "q2": "query text2"
        },
        "qrels": {
            "q1": {
                "doc1": 2,
                "doc2": 1
            }
        }
    }
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    if not isinstance(data, dict) or 'queries' not in data or 'qrels' not in data:
        raise ValueError("JSON file must contain 'queries' and 'qrels' objects")

    return data['queries'], DataFrame.from_dict(data['qrels'])

def main():
    parser = argparse.ArgumentParser(description='Evaluate ranking models with logging')
    parser.add_argument('--ranking', type=str, required=True, help='Vespa ranking profile')
    parser.add_argument('--mode', type=str, default="sparse",
                        choices=["sparse", "dense", "hybrid"],
                        help='Retrieval mode: sparse, dense, hybrid')
    parser.add_argument('--input', type=str, required=True,
                        help='Input JSON file containing queries and qrels')
    parser.add_argument('--format', type=str, default="ocr",
                        help='Input the text_format you want to query')
    parser.add_argument('--log', type=str, default="search_logs.json",
                        help='Output JSON file to save logs')

    args = parser.parse_args()

    # Determine retrieval mode
    mode_mapping = {
        "sparse": RModel.SPARSE,
        "dense": RModel.DENSE,
        "hybrid": RModel.HYBRID
    }
    mode = mode_mapping.get(args.mode.lower(), RModel.SPARSE)

    # Load queries and qrels from JSON file
    try:
        queries, qrels = load_queries_and_qrels(args.input)
    except Exception as e:
        print("Error loading input file: {}".format(e))
        return

    # Initialize log entries list
    log_entries = []

    # Define metrics
    metrics = [nDCG@5, P(rel=1)@1, R(rel=1)@5, R(rel=1)@10, MRR@5, R(rel=1)@8]

    # Initialize results list for evaluation
    all_results = []

    # Process each query
    for qid, query_text in queries.items():
        print("Processing Query ID: {}".format(qid))
        log_entry = {"qid": qid, "query": query_text}

        # Perform search with logging
        scored_docs = search(query_text, qid, args.ranking,
                            hits=10, language="en",
                            mode=mode, format=args.format,
                            log_entry=log_entry)
        if log_entry is not None:
            log_entry["Optimal Document"] = qrels.loc[qrels['query_id'] == qid].to_dict('records')[0]
        # Append scored documents for evaluation
        all_results.extend(scored_docs)

        # Append log entry
        log_entries.append(log_entry)

    # Calculate aggregate metrics
    metrics_scores = calc_aggregate(metrics, qrels, all_results)

    # Print metrics
    print("\nRanking Metrics:")
    for metric in metrics:
        metric_name = str(metric)
        metric_value = metrics_scores.get(metric, 0.0)
        print("{} for rank profile '{}': {:.4f}".format(metric_name, args.ranking, metric_value))
    print(metrics_scores)    
    # Save logs to JSON file
    try:
        with open(args.log, 'w') as log_file:
            json.dump(log_entries, log_file, indent=2)
        print("\nLogs have been saved to {}".format(args.log))
    except Exception as e:
        print("Failed to save logs to {}: {}".format(args.log, e))

if __name__ == "__main__":
    main()
