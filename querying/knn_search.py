def construct_knn_query(parsed_job_description):
    query = {
        "bool": {
            "must": [
                {"term": {"is_teenage": parsed_job_description["is_teenage"]}}
            ],
            "should": []
        }
    }

    # Add semantic search for age_indicators
    age_indicators = parsed_job_description.get("age_indicators", {})
    if "semantic" in age_indicators:
        for field, embedding in age_indicators["semantic"].items():
            if embedding:
                query["bool"]["should"].append({
                    "knn": {
                        "field": f"age_indicators.semantic.{field}",
                        "query_vector": embedding,
                    }
                })

    # Add semantic search for contact_information
    contact_information = parsed_job_description.get("contact_information", {})
    if contact_information and "semantic" in contact_information:
        for field, embedding in contact_information["semantic"].items():
            if embedding:
                query["bool"]["should"].append({
                    "knn": {
                        "field": f"contact_information.semantic.{field}",
                        "query_vector": embedding,
                    }
                })

    # Add semantic search for personal_summary
    personal_summary = parsed_job_description.get("personal_summary", {})
    if personal_summary and "semantic" in personal_summary:
        for field, embedding in personal_summary["semantic"].items():
            if embedding:
                query["bool"]["should"].append({
                    "knn": {
                        "field": f"personal_summary.semantic.{field}",
                        "query_vector": embedding,
                    }
                })

    # Add semantic search for education
    education_list = parsed_job_description.get("education", [])
    if education_list:
        for education in education_list:
            if "semantic" in education:
                for field, embedding in education["semantic"].items():
                    if embedding:
                        query["bool"]["should"].append({
                            "knn": {
                                "field": f"education.semantic.{field}",
                                "query_vector": embedding,
                            }
                        })

    # Add semantic search for work_experience
    work_experience_list = parsed_job_description.get("work_experience", [])
    if work_experience_list:
        for work_experience in work_experience_list:
            if "semantic" in work_experience:
                for field, embedding in work_experience["semantic"].items():
                    if embedding:
                        query["bool"]["should"].append({
                            "knn": {
                                "field": f"work_experience.semantic.{field}",
                                "query_vector": embedding,
                            }
                        })

    # Add semantic search for skills
    skills = parsed_job_description.get("skills", {})
    if skills and "semantic" in skills:
        for field, embedding in skills["semantic"].items():
            if embedding:
                query["bool"]["should"].append({
                    "knn": {
                        "field": f"skills.semantic.{field}",
                        "query_vector": embedding,
                    }
                })

    # Add semantic search for projects
    projects_list = parsed_job_description.get("projects", [])
    if projects_list:
        for project in projects_list:
            if "semantic" in project:
                for field, embedding in project["semantic"].items():
                    if embedding:
                        query["bool"]["should"].append({
                            "knn": {
                                "field": f"projects.semantic.{field}",
                                "query_vector": embedding,
                            }
                        })

    # Add semantic search for certifications
    certifications_list = parsed_job_description.get("certifications", [])
    if certifications_list:
        for certification in certifications_list:
            if "semantic" in certification:
                for field, embedding in certification["semantic"].items():
                    if embedding:
                        query["bool"]["should"].append({
                            "knn": {
                                "field": f"certifications.semantic.{field}",
                                "query_vector": embedding,
                            }
                        })

    # Add semantic search for publications
    publications_list = parsed_job_description.get("publications", [])
    if publications_list:
        for publication in publications_list:
            if "semantic" in publication:
                for field, embedding in publication["semantic"].items():
                    if embedding:
                        query["bool"]["should"].append({
                            "knn": {
                                "field": f"publications.semantic.{field}",
                                "query_vector": embedding,
                            }
                        })

    # Add semantic search for languages
    languages_list = parsed_job_description.get("languages", [])
    if languages_list:
        for language in languages_list:
            if "semantic" in language:
                for field, embedding in language["semantic"].items():
                    if embedding:
                        query["bool"]["should"].append({
                            "knn": {
                                "field": f"languages.semantic.{field}",
                                "query_vector": embedding,
                            }
                        })

    # Add semantic search for awards_and_honors
    awards_list = parsed_job_description.get("awards_and_honors", [])
    if awards_list:
        for award in awards_list:
            if "semantic" in award:
                for field, embedding in award["semantic"].items():
                    if embedding:
                        query["bool"]["should"].append({
                            "knn": {
                                "field": f"awards_and_honors.semantic.{field}",
                                "query_vector": embedding,
                            }
                        })

    # Add semantic search for volunteer_experience
    volunteer_list = parsed_job_description.get("volunteer_experience", [])
    if volunteer_list:
        for volunteer in volunteer_list:
            if "semantic" in volunteer:
                for field, embedding in volunteer["semantic"].items():
                    if embedding:
                        query["bool"]["should"].append({
                            "knn": {
                                "field": f"volunteer_experience.semantic.{field}",
                                "query_vector": embedding,
                            }
                        })

    return query

def knn_search(client, index_name, parsed_job_description):
    query = construct_knn_query(parsed_job_description)
    response = client.search(
        index=index_name,
        body={
            "size": 5,  # Limit number of results
            "query": query,
            "_source": ["document_id"]  # Only return necessary fields
        }
    )
    return response