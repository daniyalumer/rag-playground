import os
import json
from datetime import datetime
from processing.calculate_embeddings import embed_json_file


def collect_user_input():
    """Collect and structure user input for resume matching."""
    user_data = {
        "age_indicators": {
            "education_timeline": input("Enter age_indicators.education_timeline: ") or None,
            "work_experience_type": input("Enter age_indicators.work_experience_type: ") or None,
            "extracurricular_focus": input("Enter age_indicators.extracurricular_focus: ") or None,
            "email_style": input("Enter age_indicators.email_style: ") or None,
            "certification_level": input("Enter age_indicators.certification_level: ") or None,
            "writing_style": input("Enter age_indicators.writing_style: ") or None
        },
        "contact_information": {
            "address": input("Enter contact_information.address: ") or None
        },
        "personal_summary": input("Enter personal_summary: ") or None,
        "education": {
            "degree": input("Enter education.degree: ") or None,
            "institution": input("Enter education.institution: ") or None,
            "location": input("Enter education.location: ") or None,
            "honors": input("Enter education.honors: ") or None,
            "description": input("Enter education.description: ") or None
        },
        "work_experience": {
            "job_title": input("Enter work_experience.job_title: ") or None,
            "employer": input("Enter work_experience.employer: ") or None,
            "location": input("Enter work_experience.location: ") or None,
            "description": input("Enter work_experience.description: ") or None,
            "achievements": input("Enter work_experience.achievements: ") or None
        },
        "skills": input("Enter skills: ") or None,
        "projects": {
            "title": input("Enter projects.title: ") or None,
            "description": input("Enter projects.description: ") or None,
            "role": input("Enter projects.role: ") or None,
            "technologies": input("Enter projects.technologies: ") or None
        },
        "certifications": {
            "name": input("Enter certifications.name: ") or None,
            "description": input("Enter certifications.description: ") or None,
            "issuing_organization": input("Enter certifications.issuing_organization: ") or None
        },
        "publications": {
            "title": input("Enter publications.title: ") or None,
            "publisher": input("Enter publications.publisher: ") or None,
            "description": input("Enter publications.description: ") or None
        },
        "languages": {
            "language": input("Enter languages.language: ") or None,
            "proficiency": input("Enter languages.proficiency: ") or None
        },
        "awards_and_honors": {
            "title": input("Enter awards_and_honors.title: ") or None,
            "issuing_organization": input("Enter awards_and_honors.issuing_organization: ") or None,
            "description": input("Enter awards_and_honors.description: ") or None
        },
        "volunteer_experience": {
            "role": input("Enter volunteer_experience.role: ") or None,
            "organization": input("Enter volunteer_experience.organization: ") or None,
            "description": input("Enter volunteer_experience.description: ") or None
        }
    }
    return user_data

def save_and_embed_query(user_data):
    """
    Save user query to JSON and calculate its embeddings.
    
    Args:
        user_data (dict): The user input data to save and embed
        
    Returns:
        tuple: (embedded_query_file, embedded_query) containing the path to embedded file 
        and the loaded embedded query
    """
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    query_dir = "data/user_query"
    os.makedirs(query_dir, exist_ok=True)
    
    # Save raw query
    query_file = os.path.join(query_dir, f"query_{timestamp}.json")
    with open(query_file, 'w') as f:
        json.dump(user_data, f, indent=2)
    print(f"User query saved to {query_file}")

    # Calculate and save embeddings
    query_embeddings_dir = "data/user_query_embeddings"
    embedded_query_file = os.path.join(query_embeddings_dir, f"query_{timestamp}_embedding.json")
    embed_json_file(query_file, embedded_query_file)
    print(f"Query embeddings saved to {embedded_query_file}")

    # Load and return the embedded query
    with open(embedded_query_file, 'r') as f:
        embedded_query = json.load(f)
    
    return embedded_query_file, embedded_query