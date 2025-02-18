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
            'full_name',
            'phone_number',
            'email',
            'linkedin',
            'portfolio_website',
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
            'issuing_organization'
        ],
        'publications': [
            'title',
            'publisher',
            'url'
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
                                # Store embeddings in the expected nested structure
                                if 'semantic' not in item:
                                    item['semantic'] = {}
                                item['semantic'][f"{subfield}_embedding"] = embeddings_model.embed_query(item[subfield])
        else:
            if data.get(field) is not None and isinstance(data[field], str):
                # Replace the original string with a new nested dictionary
                data[field] = {
                    'value': data[field],  # Store the original string value
                    'semantic': {
                        f"{field}_embedding": embeddings_model.embed_query(data[field])
                    }
                }
    
    return data

def process_json_files(input_dir, output_dir):
    setup_logging('embeddings_calculation')
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith('.json'):
            input_file_path = os.path.join(input_dir, filename)
            output_file_path = os.path.join(output_dir, filename)
            
            # Skip processing if the output file already exists
            if os.path.exists(output_file_path):
                logging.info(f"Skipping {filename} - already processed")
                continue
            
            with open(input_file_path, 'r') as f:
                data = json.load(f)
            
            logging.info(f"Calculating embeddings for {filename}")
            # Calculate embeddings
            updated_data = calculate_embeddings(data)
            
            # Save updated JSON with embeddings
            with open(output_file_path, 'w') as f:
                json.dump(updated_data, f, indent=2)
            logging.info(f"Successfully saved {filename}")
