import os
import json
import logging
from tqdm import tqdm
from config.logging_config import setup_logging  
from clients.openai_embeddings import create_openai_embeddings

# Initialize OpenAI embeddings
embeddings_model = create_openai_embeddings()

def calculate_embeddings(data):
    # Calculate embeddings for relevant fields based on the mapping
    fields_to_embed = {
        'age_indicators': [
            'education_timeline',
            'work_experience_type',
            'extracurricular_focus',
            'email_style',
            'certification_level',
            'writing_style'
        ],
        'contact_information': [
            'address'
        ],
        'personal_summary': 'personal_summary',
        'education': [
            'degree',
            'institution',
            'location',
            'honors',
            'description'
        ],
        'work_experience': [
            'job_title',
            'employer',
            'location',
            'description',
            'achievements'
        ],
        'skills': 'skills',
        'projects': [
            'title',
            'description',
            'role',
            'technologies'
        ],
        'certifications': [
            'name',
            'description',
            'issuing_organization'
        ],
        'publications': [
            'title',
            'publisher',
            'description'
        ],
        'languages': [
            'language',
            'proficiency'
        ],
        'awards_and_honors': [
            'title',
            'issuing_organization',
            'description'
        ],
        'volunteer_experience': [
            'role',
            'organization',
            'description'
        ]
    }
    
    for field, subfields in fields_to_embed.items():
        if isinstance(subfields, list):
            if data.get(field):
                for item in data[field] if isinstance(data[field], list) else [data[field]]:
                    if isinstance(item, dict):
                        for subfield in subfields:
                            if subfield in item and item[subfield] is not None and isinstance(item[subfield], str):
                                # Store embeddings directly in the object
                                item[f"{subfield}_embedding"] = embeddings_model.embed_query(item[subfield])
        else:
            if data.get(field) is not None and isinstance(data[field], dict):
                # Add the embedding directly to the object
                embedding = embeddings_model.embed_query(data[field]['value'])
                data[field] = {
                    f"value": data[field]['value'],
                    f"{field}_embedding": embedding,
                    f"value_boost": data.get(field).get(f"value_boost"),
                    f"{field}_embedding_boost": data.get(field).get(f"{field}_embedding_boost")
                }
    
    return data


def embed_query_file(input_file_path, output_file_path):
    setup_logging('embeddings_calculation')
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    # Skip processing if the output file already exists
    if os.path.exists(output_file_path):
        logging.info(f"Skipping {os.path.basename(input_file_path)} - already processed")
        return
    
    with open(input_file_path, 'r') as f:
        data = json.load(f)
    
    logging.info(f"Calculating embeddings for {os.path.basename(input_file_path)}")
    # Calculate embeddings
    updated_data = calculate_embeddings(data)
    
    # Save updated JSON with embeddings
    with open(output_file_path, 'w') as f:
        json.dump(updated_data, f, indent=2)
    logging.info(f"Successfully saved {os.path.basename(output_file_path)}")


