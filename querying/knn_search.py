def construct_knn_query(parsed_job_description):
    # Initialize an array to hold all queries
    queries = []

    # Add semantic search for age_indicators
    age_indicators = parsed_job_description.get("age_indicators", {})
    if "semantic" in age_indicators:
        for field, embedding in age_indicators["semantic"].items():
            if embedding:
                queries.append({
                    "nested": {
                        "path": "age_indicators.semantic",
                        "query": {
                            "knn": {
                                "field": f"age_indicators.semantic.{field}",
                                "query_vector": embedding,
                                "k": 3,
                                "num_candidates": 10
                            }
                        },
                        "score_mode": "avg"
                    }
                })

    # Add semantic search for other nested fields
    nested_fields = {
        "contact_information": 0.7,
        "personal_summary": 1.5,
        "work_experience": 2.0,
        "education": 2.0,
        "skills": 1.8,
        "projects": 1.5,
        "certifications": 1.3,
        "publications": 1.2,
        "languages": 1.0,
        "awards_and_honors": 1.0,
        "volunteer_experience": 1.0
    }

    for field_name, boost in nested_fields.items():
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
                                    "score_mode": "avg",
                                    "boost": boost
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
                            "score_mode": "avg",
                            "boost": boost
                        }
                    })

    # Combine queries with bool should
    return {
        "bool": {
            "should": queries,
            "minimum_should_match": 1
        }
    }

def knn_search(client, index_name, parsed_job_description):
    query = construct_knn_query(parsed_job_description)
    response = client.search(
        index=index_name,
        body={
            "size": 5,  # Limit number of results
            "query": query,
            "_source": ["document_id", "is_teenage", "teenage_confidence"],
            "track_scores": True,
            "explain": True,
            "min_score": 0.1
        }
    )
    return response