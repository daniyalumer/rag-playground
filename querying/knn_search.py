import json
import os
from datetime import datetime

def construct_knn_query(embedded_query):
    """
    Construct KNN query from pre-embedded query vectors.
    
    Args:
        embedded_query (dict): Dictionary containing embedded vectors for each field
    
    Returns:
        dict: Elasticsearch query object with knn queries
    """
    queries = []
    
    # Define which fields are nested in the index
    nested_fields = {
        "education", "work_experience", "projects", "certifications",
        "publications", "languages", "awards_and_honors", "volunteer_experience"
    }
    
    # Process each field in the embedded query
    for field_name in embedded_query:
        field_data = embedded_query.get(field_name)
        if isinstance(field_data, dict):
            # Check if this is a nested field in the index
            if field_name in nested_fields:
                # Construct nested KNN query
                for key, value in field_data.items():
                    if value and key.endswith('_embedding'):
                        queries.append({
                            "nested": {
                                "path": field_name,
                                "query": {
                                    "knn": {
                                        "query_vector": value,
                                        "field": f"{field_name}.{key}",
                                        "num_candidates": 10,
                                        "k": 3
                                    }
                                }
                            }
                        })
            else:
                # Construct regular KNN query for non-nested fields
                for key, value in field_data.items():
                    if value and key.endswith('_embedding'):
                        queries.append({
                            "knn": {
                                "query_vector": value,
                                "field": f"{field_name}.{key}",
                                "num_candidates": 10,
                                "k": 3
                            }
                        })
    
    if not queries:
        return {}  # Return empty query if no embeddings found
        
    return {
        "bool": {
            "should": queries,
            "minimum_should_match": 1
        }
    }

def knn_search(client, index_name, embedded_query):
    """
    Perform KNN search using pre-embedded query vectors.
    
    Args:
        client: Elasticsearch client
        index_name (str): Name of the index to search
        embedded_query (dict): Pre-embedded query vectors
    
    Returns:
        dict: Elasticsearch search response
    """
    query = construct_knn_query(embedded_query)

    # Save query for debugging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join("data/generated_query", f"query_{timestamp}.json")
    
    os.makedirs("data/generated_query", exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(query, f, indent=2)
    print(f"Generated query saved to {filepath}")

    response = client.search(
        index=index_name,
        body={
            "size": 10,
            "query": query,
            "_source": ["document_id", "is_teenage", "teenage_confidence"],
            "track_scores": True,
            "explain": True
        }
    )
    
    return response