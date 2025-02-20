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
                # Document Identifier as a top-level field
                "document_id": {
                    "type": "text"  # Use text type for the document identifier
                },
                # Age-related fields for filtering and analysis
                "is_teenage": {"type": "boolean"},
                "teenage_confidence": {"type": "float"},
                "age_indicators": {
                    "properties": {
                        # Original fields without keyword and text types
                        "education_timeline": {"type": "text"},
                        "work_experience_type": {"type": "text"},
                        "extracurricular_focus": {"type": "text"},
                        "email_style": {"type": "text"},
                        "certification_level": {"type": "text"},
                        "writing_style": {"type": "text"},
                        # Semantic search fields
                        "semantic": {
                            "type": "nested",
                            "properties": {
                                "education_timeline_embedding": {"type": "dense_vector", "dims": 3072},
                                "work_experience_type_embedding": {"type": "dense_vector", "dims": 3072},
                                "extracurricular_focus_embedding": {"type": "dense_vector", "dims": 3072},
                                "email_style_embedding": {"type": "dense_vector", "dims": 3072},
                                "certification_level_embedding": {"type": "dense_vector", "dims": 3072},
                                "writing_style_embedding": {"type": "dense_vector", "dims": 3072}
                            }
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
                        "address": {
                            "type": "text"
                        },
                        # Semantic search field for address
                        "semantic": {
                            "type": "nested",
                            "properties": {
                                "address_embedding": {"type": "dense_vector", "dims": 3072}
                            }
                        }
                    }
                },
                
                # Personal Summary with semantic search
                "personal_summary": {
                    "properties": {
                        "value": {
                            "type": "text"
                        },
                        "semantic": {
                            "type": "nested",
                            "properties": {
                                "personal_summary_embedding": {"type": "dense_vector", "dims": 3072}
                            }
                        }
                    }
                },
                
                # Education
                "education": {
                    "type": "nested",
                    "properties": {
                        "degree": {
                            "type": "text"
                        },
                        "institution": {
                            "type": "text"
                        },
                        "location": {
                            "type": "text"
                        },
                        "start_date": {"type": "date"},
                        "end_date": {"type": "date"},
                        "gpa": {"type": "float"},
                        "honors": {
                            "type": "text"
                        },
                        "description": {
                            "type": "text"
                        },
                        # Semantic search fields
                        "semantic": {
                            "type": "nested",
                            "properties": {
                                "degree_embedding": {"type": "dense_vector", "dims": 3072},
                                "institution_embedding": {"type": "dense_vector", "dims": 3072},
                                "location_embedding": {"type": "dense_vector", "dims": 3072},
                                "honors_embedding": {"type": "dense_vector", "dims": 3072},
                                "description_embedding": {"type": "dense_vector", "dims": 3072}
                            }
                        }
                    }
                },
                
                # Work Experience with semantic search
                "work_experience": {
                    "type": "nested",
                    "properties": {
                        "job_title": {
                            "type": "text"
                        },
                        "employer": {
                            "type": "text"
                        },
                        "location": {
                            "type": "text"
                        },
                        "start_date": {"type": "date"},
                        "end_date": {"type": "date"},
                        "description": {
                            "type": "text"
                        },
                        "achievements": {
                            "type": "text"
                        },
                        # Semantic search fields
                        "semantic": {
                            "type": "nested",
                            "properties": {
                                "job_title_embedding": {"type": "dense_vector", "dims": 3072},
                                "employer_embedding": {"type": "dense_vector", "dims": 3072},
                                "location_embedding": {"type": "dense_vector", "dims": 3072},
                                "description_embedding": {"type": "dense_vector", "dims": 3072},
                                "achievements_embedding": {"type": "dense_vector", "dims": 3072}
                            }
                        }
                    }
                },
                
                # Skills with semantic search
                "skills": {
                    "properties": {
                        "value": {
                            "type": "text"
                        },
                        "semantic": {
                            "type": "nested",
                            "properties": {
                                "skills_embedding": {"type": "dense_vector", "dims": 3072}
                            }
                        }
                    }
                },
                
                # Projects with semantic search
                "projects": {
                    "type": "nested",
                    "properties": {
                        "title": {
                            "type": "text"
                        },
                        "description": {
                            "type": "text"
                        },
                        "role": {
                            "type": "text"
                        },
                        "technologies": {
                            "type": "text"
                        },
                        "start_date": {"type": "date"},
                        "end_date": {"type": "date"},
                        # Semantic search fields
                        "semantic": {
                            "type": "nested",
                            "properties": {
                                "title_embedding": {"type": "dense_vector", "dims": 3072},
                                "description_embedding": {"type": "dense_vector", "dims": 3072},
                                "role_embedding": {"type": "dense_vector", "dims": 3072},
                                "technologies_embedding": {"type": "dense_vector", "dims": 3072}
                            }
                        }
                    }
                },
                
                # Certifications with semantic search
                "certifications": {
                    "type": "nested",
                    "properties": {
                        "name": {
                            "type": "text"
                        },
                        "description": {
                            "type": "text"
                        },
                        "issuing_organization": {
                            "type": "text"
                        },
                        "issue_date": {"type": "date"},
                        "expiration_date": {"type": "date"},
                        # Semantic search fields
                        "semantic": {
                            "type": "nested",
                            "properties": {
                                "name_embedding": {"type": "dense_vector", "dims": 3072},
                                "description_embedding": {"type": "dense_vector", "dims": 3072},
                                "issuing_organization_embedding": {"type": "dense_vector", "dims": 3072}
                            }
                        }
                    }
                },
                
                # Publications with semantic search
                "publications": {
                    "type": "nested",
                    "properties": {
                        "title": {
                            "type": "text"
                        },
                        "publisher": {
                            "type": "text"
                        },
                        "description": {
                            "type": "text"
                        },
                        "publication_date": {"type": "date"},
                        "url": {
                            "type": "text"
                        },
                        # Semantic search fields
                        "semantic": {
                            "type": "nested",
                            "properties": {
                                "title_embedding": {"type": "dense_vector", "dims": 3072},
                                "publisher_embedding": {"type": "dense_vector", "dims": 3072},
                                "description_embedding": {"type": "dense_vector", "dims": 3072}
                            }
                        }
                    }
                },
                
                # Languages with semantic search
                "languages": {
                    "type": "nested",
                    "properties": {
                        "language": {
                            "type": "text"
                        },
                        "proficiency": {
                            "type": "text"
                        },
                        # Semantic search fields
                        "semantic": {
                            "type": "nested",
                            "properties": {
                                "language_embedding": {"type": "dense_vector", "dims": 3072},
                                "proficiency_embedding": {"type": "dense_vector", "dims": 3072}
                            }
                        }
                    }
                },
                
                # Awards and Honors with semantic search
                "awards_and_honors": {
                    "type": "nested",
                    "properties": {
                        "title": {
                            "type": "text"
                        },
                        "issuing_organization": {
                            "type": "text"
                        },
                        "issue_date": {"type": "date"},
                        "description": {
                            "type": "text"
                        },
                        # Semantic search fields
                        "semantic": {
                            "type": "nested",
                            "properties": {
                                "title_embedding": {"type": "dense_vector", "dims": 3072},
                                "issuing_organization_embedding": {"type": "dense_vector", "dims": 3072},
                                "description_embedding": {"type": "dense_vector", "dims": 3072}
                            }
                        }
                    }
                },
                
                # Volunteer Experience with semantic search
                "volunteer_experience": {
                    "type": "nested",
                    "properties": {
                        "role": {
                            "type": "text"
                        },
                        "organization": {
                            "type": "text"
                        },
                        "start_date": {"type": "date"},
                        "end_date": {"type": "date"},
                        "description": {
                            "type": "text"
                        },
                        # Semantic search fields
                        "semantic": {
                            "type": "nested",
                            "properties": {
                                "role_embedding": {"type": "dense_vector", "dims": 3072},
                                "organization_embedding": {"type": "dense_vector", "dims": 3072},
                                "description_embedding": {"type": "dense_vector", "dims": 3072}
                            }
                        }
                    }
                },
                
                # References
                "references": {
                    "type": "nested",
                    "properties": {
                        "name": {
                            "type": "text"
                        },
                        "relationship": {
                            "type": "text"
                        },
                        "contact_information": {
                            "type": "text"
                        }
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
            with open(os.path.join(data_directory, filename), 'r') as f:
                content = f.read()
                # Assuming the content is a JSON object, we need to parse it
                doc = json.loads(content)  # Parse the JSON content
                # Ensure the document structure matches the mapping
                client.index(index=index_name, document=doc)

def delete_index(client, index_name):
    if client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)
        print(f"Deleted index: {index_name}")


