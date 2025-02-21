import json
import os
from datetime import datetime

def construct_knn_query(parsed_job_description):
    # Initialize a dictionary to hold queries by field
    field_queries = {}

    # Add semantic search for nested fields with customized settings
    nested_fields = {
        "age_indicators": {"boost": 1.0, "score_mode": "avg"},       # Average for age relevance
        "contact_information": {"boost": 0.7, "score_mode": "max"},  # Use max for best matching contact
        "personal_summary": {"boost": 1.5, "score_mode": "max"},     # Average for overall summary match
        "work_experience": {"boost": 2.0, "score_mode": "sum"},      # Sum for cumulative experience
        "education": {"boost": 2.0, "score_mode": "sum"},            # Average for education level
        "skills": {"boost": 2.5, "score_mode": "max"},               # Max for best skill matches
        "projects": {"boost": 1.5, "score_mode": "avg"},             # Average for project relevance
        "certifications": {"boost": 1.3, "score_mode": "avg"},       # Max for most relevant cert
        "publications": {"boost": 1.2, "score_mode": "avg"},         # Average for publication match
        "languages": {"boost": 1.0, "score_mode": "max"},            # Max for best language match
        "awards_and_honors": {"boost": 1.0, "score_mode": "max"},    # Max for best award match
        "volunteer_experience": {"boost": 1.0, "score_mode": "avg"}   # Average for experience relevance
    }

    # Initialize query arrays for each field
    for field_name in nested_fields.keys():
        field_queries[field_name] = []

    # Populate queries for each field
    for field_name, settings in nested_fields.items():
        field_data = parsed_job_description.get(field_name, [])
        if isinstance(field_data, list):
            for item in field_data:
                if "semantic" in item:
                    for embedding_field, embedding in item["semantic"].items():
                        if embedding:
                            field_queries[field_name].append({
                                "nested": {
                                    "path": f"{field_name}.semantic",
                                    "query": {
                                        "bool": {
                                            "should": [
                                                {
                                                    "knn": {
                                                        "field": f"{field_name}.semantic.{embedding_field}",
                                                        "query_vector": embedding,
                                                        "k": 3,
                                                        "num_candidates": 10,
                                                        "boost": settings["boost"]
                                                    }
                                                }
                                            ]
                                        }
                                    },
                                    "score_mode": settings["score_mode"]
                                }
                            })
        elif isinstance(field_data, dict) and "semantic" in field_data:
            for embedding_field, embedding in field_data["semantic"].items():
                if embedding:
                    field_queries[field_name].append({
                        "nested": {
                            "path": f"{field_name}.semantic",
                            "query": {
                                "bool": {
                                    "should": [
                                        {
                                            "knn": {
                                                "field": f"{field_name}.semantic.{embedding_field}",
                                                "query_vector": embedding,
                                                "k": 3,
                                                "num_candidates": 10,
                                                "boost": settings["boost"]
                                            }
                                        }
                                    ]
                                }
                            },
                            "score_mode": settings["score_mode"]
                        }
                    })

    # Create bool queries for each field
    field_bool_queries = []
    for field_name, queries in field_queries.items():
        if queries:  # Only add fields that have queries
            field_bool_queries.append({
                "bool": {
                    "should": queries,
                    "minimum_should_match": 1
                }
            })

    # Combine all field bool queries
    return {
        "bool": {
            "should": field_bool_queries,
            "minimum_should_match": 1
        }
    }

def knn_search2(client, index_name, parsed_job_description):
    query = construct_knn_query(parsed_job_description)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join("data/generated_query2", f"query_{timestamp}.json")
    
    # Ensure directory exists
    os.makedirs("data/generated_query2", exist_ok=True)
    
    # Save only the query with pretty formatting
    with open(filepath, 'w') as f:
        json.dump(query, f, indent=2)

    response = client.search(
        index=index_name,
        body={
            "size": 3,  # Limit number of results
            "query": query,
            "_source": ["document_id", "is_teenage", "teenage_confidence", "teenage_indicators"],
            "track_scores": True,
            "explain": True,
            "min_score": 3.0
        }
    )
    
    return response