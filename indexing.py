import json
import os
from elasticsearch import helpers

def create_index(client, index_name):
    mapping = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "document_id": {"type": "keyword"},
                "is_teenage": {"type": "boolean"},
                "teenage_confidence": {"type": "float"},
                "age_indicators": {
                    "properties": {
                        "education_timeline": {"type": "text"},
                        "work_experience_type": {"type": "text"},
                        "extracurricular_focus": {"type": "text"},
                        "email_style": {"type": "text"},
                        "certification_level": {"type": "text"},
                        "writing_style": {"type": "text"},
                        "education_timeline_embedding": {
                            "type": "dense_vector",
                            "dims": 3072,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "work_experience_type_embedding": {
                            "type": "dense_vector",
                            "dims": 3072,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "extracurricular_focus_embedding": {
                            "type": "dense_vector",
                            "dims": 3072,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "email_style_embedding": {
                            "type": "dense_vector",
                            "dims": 3072,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "certification_level_embedding": {
                            "type": "dense_vector",
                            "dims": 3072,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "writing_style_embedding": {
                            "type": "dense_vector",
                            "dims": 3072,
                            "index": True,
                            "similarity": "cosine"
                        }
                    }
                },
                "contact_information": {
                    "properties": {
                        "full_name": {"type": "text"},
                        "phone_number": {"type": "text"},
                        "email": {"type": "text"},
                        "linkedin": {"type": "text"},
                        "portfolio_website": {"type": "text"},
                        "address": {"type": "text"},
                        "address_embedding": {
                            "type": "dense_vector",
                            "dims": 3072,
                            "index": True,
                            "similarity": "cosine"
                        }
                    }
                },
                "personal_summary": {
                    "properties": {
                        "value": {"type": "text"},
                        "personal_summary_embedding": {
                            "type": "dense_vector",
                            "dims": 3072,
                            "index": True,
                            "similarity": "cosine"
                        }
                    }
                },
                "education": {
                    "type": "nested",
                    "properties": {
                        "degree": {"type": "text"},
                        "institution": {"type": "keyword"},
                        "location": {"type": "keyword"},
                        "start_date": {"type": "date"},
                        "end_date": {"type": "date"},
                        "gpa": {"type": "float"},
                        "honors": {"type": "text"},
                        "description": {"type": "text"},
                        "degree_embedding": {
                            "type": "dense_vector",
                            "dims": 3072,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "institution_embedding": {
                            "type": "dense_vector",
                            "dims": 3072,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "location_embedding": {
                            "type": "dense_vector",
                            "dims": 3072,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "honors_embedding": {
                            "type": "dense_vector",
                            "dims": 3072,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "description_embedding": {
                            "type": "dense_vector",
                            "dims": 3072,
                            "index": True,
                            "similarity": "cosine"
                        }
                    }
                },
                "work_experience": {
                    "type": "nested",
                    "properties": {
                        "job_title": {"type": "text"},
                        "employer": {"type": "keyword"},
                        "location": {"type": "keyword"},
                        "start_date": {"type": "date"},
                        "end_date": {"type": "date"},
                        "description": {"type": "text"},
                        "achievements": {"type": "text"},
                        "job_title_embedding": {
                            "type": "dense_vector",
                            "dims": 3072,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "employer_embedding": {
                            "type": "dense_vector",
                            "dims": 3072,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "location_embedding": {
                            "type": "dense_vector",
                            "dims": 3072,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "description_embedding": {
                            "type": "dense_vector",
                            "dims": 3072,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "achievements_embedding": {
                            "type": "dense_vector",
                            "dims": 3072,
                            "index": True,
                            "similarity": "cosine"
                        }
                    }
                },
                "skills": {
                    "properties": {
                        "value": {"type": "text"},
                        "skills_embedding": {
                            "type": "dense_vector",
                            "dims": 3072,
                            "index": True,
                            "similarity": "cosine"
                        }
                    }
                },
                "projects": {
                    "type": "nested",
                    "properties": {
                        "title": {"type": "text"},
                        "description": {"type": "text"},
                        "role": {"type": "text"},
                        "technologies": {"type": "text"},
                        "start_date": {"type": "date"},
                        "end_date": {"type": "date"},
                        "title_embedding": {
                            "type": "dense_vector",
                            "dims": 3072,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "description_embedding": {
                            "type": "dense_vector",
                            "dims": 3072,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "role_embedding": {
                            "type": "dense_vector",
                            "dims": 3072,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "technologies_embedding": {
                            "type": "dense_vector",
                            "dims": 3072,
                            "index": True,
                            "similarity": "cosine"
                        }
                    }
                },
                "certifications": {
                    "type": "nested",
                    "properties": {
                        "name": {"type": "text"},
                        "description": {"type": "text"},
                        "issuing_organization": {"type": "text"},
                        "issue_date": {"type": "date"},
                        "expiration_date": {"type": "date"},
                        "name_embedding": {
                            "type": "dense_vector",
                            "dims": 3072,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "description_embedding": {
                            "type": "dense_vector",
                            "dims": 3072,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "issuing_organization_embedding": {
                            "type": "dense_vector",
                            "dims": 3072,
                            "index": True,
                            "similarity": "cosine"
                        }
                    }
                },
                "publications": {
                    "type": "nested",
                    "properties": {
                        "title": {"type": "text"},
                        "publisher": {"type": "text"},
                        "description": {"type": "text"},
                        "publication_date": {"type": "date"},
                        "url": {"type": "text"},
                        "title_embedding": {
                            "type": "dense_vector",
                            "dims": 3072,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "publisher_embedding": {
                            "type": "dense_vector",
                            "dims": 3072,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "description_embedding": {
                            "type": "dense_vector",
                            "dims": 3072,
                            "index": True,
                            "similarity": "cosine"
                        }
                    }
                },
                "languages": {
                    "type": "nested",
                    "properties": {
                        "language": {"type": "text"},
                        "proficiency": {"type": "text"},
                        "language_embedding": {
                            "type": "dense_vector",
                            "dims": 3072,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "proficiency_embedding": {
                            "type": "dense_vector",
                            "dims": 3072,
                            "index": True,
                            "similarity": "cosine"
                        }
                    }
                },
                "awards_and_honors": {
                    "type": "nested",
                    "properties": {
                        "title": {"type": "text"},
                        "issuing_organization": {"type": "text"},
                        "issue_date": {"type": "date"},
                        "description": {"type": "text"},
                        "title_embedding": {
                            "type": "dense_vector",
                            "dims": 3072,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "issuing_organization_embedding": {
                            "type": "dense_vector",
                            "dims": 3072,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "description_embedding": {
                            "type": "dense_vector",
                            "dims": 3072,
                            "index": True,
                            "similarity": "cosine"
                        }
                    }
                },
                "volunteer_experience": {
                    "type": "nested",
                    "properties": {
                        "role": {"type": "text"},
                        "organization": {"type": "text"},
                        "start_date": {"type": "date"},
                        "end_date": {"type": "date"},
                        "description": {"type": "text"},
                        "role_embedding": {
                            "type": "dense_vector",
                            "dims": 3072,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "organization_embedding": {
                            "type": "dense_vector",
                            "dims": 3072,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "description_embedding": {
                            "type": "dense_vector",
                            "dims": 3072,
                            "index": True,
                            "similarity": "cosine"
                        }
                    }
                },
                "references": {
                    "type": "nested",
                    "properties": {
                        "name": {"type": "text"},
                        "relationship": {"type": "text"},
                        "contact_information": {"type": "text"}
                    }
                }
            }
        }
    }
    # Create the index with the specified mapping
    client.indices.create(index=index_name, body=mapping)

def index_documents(client, index_name, data_directory):
    for filename in os.listdir(data_directory):
        if filename.endswith('.json'):  # adjust based on your file types
            file_path = os.path.join(data_directory, filename)
            print(f"Indexing file: {filename}")  # Print the name of the file being indexed
            with open(file_path, 'r') as f:
                content = f.read()
                doc = json.loads(content)  # Parse the JSON content
                client.index(index=index_name, document=doc)

def delete_index(client, index_name):
    if client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)
        print(f"Deleted index: {index_name}")