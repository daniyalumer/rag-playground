import json
import os
from datetime import datetime

def construct_simple_knn_query(parsed_job_description):
    # Initialize array for all queries
    queries = []

    # Define field settings
    nested_fields = {
        "age_indicators": {"boost": 1.0, "score_mode": "avg"},
        "contact_information": {"boost": 0.7, "score_mode": "max"},
        "personal_summary": {"boost": 1.5, "score_mode": "max"},
        "work_experience": {"boost": 2.0, "score_mode": "sum"},
        "education": {"boost": 2.0, "score_mode": "sum"},
        "skills": {"boost": 2.5, "score_mode": "max"},
        "projects": {"boost": 1.5, "score_mode": "avg"},
        "certifications": {"boost": 1.3, "score_mode": "avg"},
        "publications": {"boost": 1.2, "score_mode": "avg"},
        "languages": {"boost": 1.0, "score_mode": "max"},
        "awards_and_honors": {"boost": 1.0, "score_mode": "max"},
        "volunteer_experience": {"boost": 1.0, "score_mode": "avg"}
    }

    # Process each field
    for field_name, settings in nested_fields.items():
        field_data = parsed_job_description.get(field_name, [])
        if isinstance(field_data, list):
            for item in field_data:
                if "semantic" in item:
                    for embedding_field, embedding in item["semantic"].items():
                        if embedding:
                            queries.append({
                                "nested": {
                                    "path": f"{field_name}.semantic",
                                    "query": {
                                        "knn": {
                                            "field": f"{field_name}.semantic.{embedding_field}",
                                            "query_vector": embedding,
                                            "k": 3,
                                            "num_candidates": 10
                                        }
                                    },
                                    "score_mode": settings["score_mode"],
                                    "boost": settings["boost"]
                                }
                            })
        elif isinstance(field_data, dict) and "semantic" in field_data:
            for embedding_field, embedding in field_data["semantic"].items():
                if embedding:
                    queries.append({
                        "nested": {
                            "path": f"{field_name}.semantic",
                            "query": {
                                "knn": {
                                    "field": f"{field_name}.semantic.{embedding_field}",
                                    "query_vector": embedding,
                                    "k": 3,
                                    "num_candidates": 10
                                }
                            },
                            "score_mode": settings["score_mode"],
                            "boost": settings["boost"]
                        }
                    })

    return queries

def knn_search3(client, index_name, parsed_job_description):
    queries = construct_simple_knn_query(parsed_job_description)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join("data/generated_query3", f"query_{timestamp}.json")
    
    # Ensure directory exists
    os.makedirs("data/generated_query3", exist_ok=True)
    
    # Save queries with pretty formatting
    with open(filepath, 'w') as f:
        json.dump(queries, f, indent=2)

    response = client.search(
        index=index_name,
        body={
            "size": 3,
            "query": {
                "bool": {
                    "should": queries,
                    "minimum_should_match": 1
                }
            },
            "_source": ["document_id", "is_teenage", "teenage_confidence", "teenage_indicators"],
            "track_scores": True,
            "explain": True,
            "min_score": 3.0
        }
    )
    
    return response