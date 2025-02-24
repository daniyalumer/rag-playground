import os
import pandas as pd
import json
import datetime
from processing.extract import process_pdfs
from processing.parse_cv import process_cvs
from processing.calculate_embeddings import embed_json_files, embed_json_file
from processing.user_input import collect_user_input, save_and_embed_query
from clients.elasticsearch import create_es_client
from indexing import create_index, index_documents, delete_index
from querying.knn_search import knn_search
from querying.knn_search2 import knn_search2
from querying.knn_search3 import knn_search3

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
    batch_size = 100  # You can adjust the batch size as needed
    process_cvs(csv_path, batch_size)

    #input_directory = 'data/parsed_data/28998957.json'
    input_directory = 'data/parsed_data'
    output_directory = 'data/parsed_data_embeddings'
    embed_json_files(input_directory, output_directory)

    client = create_es_client()
    print(client.info())
    

    # Create index (only first time)
    index_name = "rag-playground-data-change"
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

    print("---------------------------Parsing job description---------------------")

    #user_input = input("Please enter the job description: ")

    # Take user input for all fields
    print("---------------------------Collecting User Input---------------------")
    user_data = collect_user_input()
    print("User data collected:")
    print(user_data)

    # Save and embed user query
    embedded_query_file, embedded_query = save_and_embed_query(user_data)


    # Perform KNN search
    # print("-------------------------Performing KNN search----------------------------------")
    # with open('data/user_query_embeddings/query_20250224_124831_embedding.json', 'r') as file:
    #     embedded_query = json.load(file)
    results = knn_search(client, index_name, embedded_query)
    
    # Display results
    if results and 'hits' in results and 'hits' in results['hits']:
        for hit in results['hits']['hits']:
            print(f"Score: {hit['_score']}")
            print(f"Document ID: {hit['_source']}")
            print("-"*50)
    else:
        print("No results found")

    # Perform KNN search
    # print("-------------------------Performing KNN search1----------------------------------")
    # with open(output_file_path, 'r') as file:
    #     parsed_job_description_embeddings = json.load(file)
    # results = knn_search(client, index_name, parsed_job_description_embeddings)
    # for hit in results['hits']['hits']:
    #     if results.get('error'):
    #         print("Error:", results['error'])
    #     print(f"Score: {hit['_score']}")
    #     print(f"Document ID: {hit['_source']}")
    #     print(f"Hit: {hit}")
    #     print("-"*50)

    # print("-------------------------Performing KNN search2----------------------------------")
    # with open(output_file_path, 'r') as file:
    #     parsed_job_description_embeddings = json.load(file) 
    # results = knn_search2(client, index_name, parsed_job_description_embeddings)
    # for hit in results['hits']['hits']:
    #     if results.get('error'):
    #         print("Error:", results['error'])
    #     print(f"Score: {hit['_score']}")
    #     print(f"Document ID: {hit['_source']}")
    #     print(f"Hit: {hit}")
    #     print("-"*50)

    # print("-------------------------Performing KNN search3----------------------------------")
    # with open(output_file_path, 'r') as file:
    #     parsed_job_description_embeddings = json.load(file)
    # results = knn_search3(client, index_name, parsed_job_description_embeddings)
    # for hit in results['hits']['hits']:
    #     if results.get('error'):
    #         print("Error:", results['error'])
    #     print(f"Score: {hit['_score']}")
    #     print(f"Document ID: {hit['_source']}")
    #     print(f"Hit: {hit}")
    #     print("-"*50)
        

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
