import json
import os
from datetime import datetime

def construct_knn_query(embedded_query):
    """
    Construct KNN query from pre-embedded query vectors with field boosting
    and hybrid search (vector + text).
    """
    knn_queries = []
    keyword_queries = set()  # Use set to prevent duplicates
    
    # Define field weights for boosting
    field_weights = {
        "work_experience": 4.0,
        "skills": 3.5,
        "education": 2.0,
        "personal_summary": 1.5,
        "projects": 1.5,
        "certifications": 1.2
    }
    
    # Weight multipliers for semantic vs text search
    SEMANTIC_WEIGHT = 1.5  # Semantic search gets higher weight
    TEXT_WEIGHT = 0.5     # Text search gets lower weight
    
    # Define which fields are nested in the index
    nested_fields = {
        "education", "work_experience", "projects", "certifications",
        "publications", "languages", "awards_and_honors", "volunteer_experience"
    }
    
    # Process each field in the embedded query
    for field_name in embedded_query:
        if field_name == "age_indicators":
            continue
            
        field_data = embedded_query.get(field_name)
        if isinstance(field_data, dict):
            field_boost = field_weights.get(field_name, 1.0)
            
            # Check if this is a nested field in the index
            if field_name in nested_fields:
                # Process each field in the nested object
                for key, value in field_data.items():
                    # Handle semantic search (vector fields)
                    if value and key.endswith('_embedding'):
                        knn_queries.append({
                            "nested": {
                                "path": field_name,
                                "query": {
                                    "knn": {
                                        "query_vector": value,
                                        "field": f"{field_name}.{key}",
                                        "num_candidates": 100,
                                        "k": 10,
                                        "boost": field_boost * SEMANTIC_WEIGHT
                                    }
                                }
                            }
                        })
                    
                    # Handle text search for all non-embedding fields
                    elif isinstance(value, str) and not key.endswith('_embedding'):
                        query_str = json.dumps({
                            "nested": {
                                "path": field_name,
                                "query": {
                                    "match": {
                                        f"{field_name}.{key}": {
                                            "query": value,
                                            "boost": field_boost * TEXT_WEIGHT,
                                            "fuzziness": "AUTO"
                                        }
                                    }
                                }
                            }
                        })
                        keyword_queries.add(query_str)
            else:
                # Process non-nested fields
                for key, value in field_data.items():
                    # Handle semantic search (vector fields)
                    if value and key.endswith('_embedding'):
                        knn_queries.append({
                            "knn": {
                                "query_vector": value,
                                "field": f"{field_name}.{key}",
                                "num_candidates": 100,
                                "k": 10,
                                "boost": field_boost * SEMANTIC_WEIGHT
                            }
                        })
                    
                    # Handle text search for non-embedding fields
                    elif isinstance(value, str) and not key.endswith('_embedding'):
                        query_str = json.dumps({
                            "match": {
                                f"{field_name}.{key}": {
                                    "query": value,
                                    "boost": field_boost * TEXT_WEIGHT,
                                    "fuzziness": "AUTO"
                                }
                            }
                        })
                        keyword_queries.add(query_str)
    
    if not knn_queries and not keyword_queries:
        return {}  # Return empty query if no queries generated
    
    # Convert string queries back to dictionaries
    keyword_queries_list = [json.loads(q) for q in keyword_queries]
        
    return {
        "bool": {
            "should": knn_queries + keyword_queries_list,
            "minimum_should_match": "30%"
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
    