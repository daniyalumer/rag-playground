import os
import pandas as pd
import json
from processing.extract import process_pdfs
from processing.parse_cv import process_cvs
from processing.calculate_embeddings import embed_json_files, embed_json_file
from clients.elasticsearch import create_es_client
from indexing import create_index, index_documents, delete_index
from processing.parse_jd import process_job_description
from querying.knn_search import knn_search

def main():

    extracted_file = "data/extracted/resumes2.csv"
    
    if os.path.exists(extracted_file):
        print("Loading previously extracted data...")
        df = pd.read_csv(extracted_file)
        print(f"Loaded {len(df)} previously processed resumes")
    else:
        print("Starting PDF extraction...")
        input_dir = "data/resumepdf/data/data/data"
        df = process_pdfs(input_dir)
        print(f"Extracted text from {len(df)} PDF files")
    
    print(f"DataFrame shape: {df.shape}")

    csv_path = "data/extracted/resumes2.csv"
    batch_size = 10  # You can adjust the batch size as needed
    process_cvs(csv_path, batch_size)

    #input_directory = 'data/parsed_data/28998957.json'
    input_directory = 'data/parsed_data'
    output_directory = 'data/parsed_data_embeddings'
    embed_json_files(input_directory, output_directory)

    client = create_es_client()
    print(client.info())
    

    # Create index (only first time)
    index_name = "rag-playground"
    #print(f'index info:', client.indices.get(index=index_name))
    print("--------------------------DELETING INDEX-------------------------")
    #delete_index(client, index_name)
    try:
        print("------------------------CREATING INDEX-------------------------")
    #    create_index(client, index_name)
    except Exception as e:
        print(f"Index might already exist: {e}")

    print("--------------------------INDEXING DOCUMENTS-------------------------")

    # Index documents
    #data_directory = "data/parsed_data_embeddings"
    #index_documents(client, index_name, data_directory)

    print("---------------------------Indexing complete--------------------------")

    # Load job description

    print("---------------------------Parsing job description---------------------")

    # Commented to save tokens

    print("-------------------Calculating embeddings for job description-------------------")

    # input_file_path = 'data/parsed_jd/parsed_job_description.json'
    output_file_path = 'data/parsed_jd_embeddings/parsed_job_description_embeddings.json'
    # embed_json_file(input_file_path, output_file_path)


    # Perform KNN search
    print("-------------------------Performing KNN search----------------------------------")
    with open(output_file_path, 'r') as file:
        parsed_job_description_embeddings = json.load(file)
    results = knn_search(client, index_name, parsed_job_description_embeddings)
    for hit in results['hits']['hits']:
        if results.get('error'):
            print("Error:", results['error'])
        print(f"Score: {hit['_score']}")
        print(f"Document ID: {hit['_source']}")
        print(f"Hit: {hit}")
        print("-"*50)

    # Check index mapping
    # mapping = client.indices.get_mapping(index=index_name)
    # print("Index mapping:", json.dumps(mapping.body, indent=2))  # Use .body to get serializable dict
    

    #    # Initialize search
    #search_engine = SearchEngine()
    
    # Example search
    #results = search_engine.semantic_search("your query", index_name)
    #for hit in results:
    #    print(f"Score: {hit['_score']}")
    #    print(f"Content: {hit['_source']['content'][:200]}...")
    #    print("---")

if __name__ == "__main__":
    main()
