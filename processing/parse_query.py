import os
import pprint
import json
import logging
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from config.logging_config import setup_logging 
from config.config import OPENAI_API_KEY
from processing.models_query import QueryFormatter

def create_extraction_prompt_job_description(job_description_text: str) -> ChatPromptTemplate:
    system_message = SystemMessage(
        content="""
        You are an expert in semantic search optimization for job matching, specializing in deep understanding of requirements.
        Your primary focus is semantic matching with text search as a supplementary signal.

        These example just serve as a basis and you will infer the role based priority on your own expertise to optimize the querying.

        I will be querying over the following fields in defined in my elastic search index:
        - Personal Summary (semantic, text search)
        - Education
            degree (semantic, text search)
            institution (semantic, text search)
            location (semantic, text search)
            start_date (text search)
            end_date (text search)
            gpa (text search)
            honors (semantic, text search)
            description (semantic, text search)
        - Work Experience
            job_title (semantic, text search)
            employer (semantic, text search)
            location (semantic, text search)
            start_date (text search)
            end_date (text search)
            description (semantic, text search)
            achievements (semantic, text search)
        - Skills (semantic, text search)
        - Projects
            title (semantic, text search)
            description (semantic, text search)
            role (semantic, text search)
            technologies (semantic, text search)
            start_date (text search)
            end_date (text search)
        - Certifications
            name (semantic, text search)
            description (semantic, text search)
            issuing_organization (semantic, text search)
            issue_date (text search)
            expiration_date (text search)
        - Publications
            title (semantic, text search)
            publisher (semantic, text search)
            description (semantic, text search)
            publication_date (text search)
        - Languages
            language (semantic, text search)
            proficiency (semantic, text search)
        - Awards/Honors
            title (semantic, text search)
            issuing_organization (semantic, text search)
            description (semantic, text search)
            issue_date (text search)
        - Volunteer Experience
            role (semantic, text search)
            organization (semantic, text search)
            description (semantic, text search)
            start_date (text search)
            end_date (text search)
        


        Boost Value Guidelines:
        - Semantic search boost (_embedding_boost): 0.0 to 10.0
          - Critical Understanding: 8.0-10.0 (deep semantic comprehension)
          - Important Concepts: 6.0-7.9 (conceptual relevance)
          - Supporting Context: 4.0-5.9 (contextual alignment)
          - Not Mentioned: 0.0

        - Text search boost (_boost): 0.0 to 5.0
          - Exact Terms: 4.0-5.0 (specific requirements)
          - Key Terms: 2.0-3.9 (important terminology)
          - Supporting Terms: 1.0-1.9 (contextual terms)
          - Not Mentioned: 0.0

        Role-Based Priority Examples:
        1. Technical Roles:
           Skills (semantic: 9-10, text: 3-4) = Experience (semantic: 9-10, text: 3-4) > Education (semantic: 6-7, text: 2-3)
        
        2. Academic Roles:
           Education (semantic: 9-10, text: 4-5) > Publications (semantic: 8-9, text: 3-4) > Experience (semantic: 7-8, text: 2-3)
        
        3. Management Roles:
           Experience (semantic: 9-10, text: 3-4) > Skills (semantic: 8-9, text: 2-3) > Education (semantic: 6-7, text: 2-3)


        Key Analysis Rules:
        1. Prioritize semantic understanding over exact matches
        2. Consider both explicit and implicit requirements
        3. Expand abbreviated terms and acronyms
        4. Infer missing information from context
        5. Map brief mentions to comprehensive data
        6. Maintain semantic consistency across fields
        """
    )

    user_message = HumanMessage(
        content=f'''
        Extract structured information and assign boost values, following these examples. However you can infer improved boost score based on your expertise and the query requirements:
        Scoring should be optimized for semantic search with text search as a supplementary signal in elastic search.
        
        1. Technical Query Example - Note equal priority for skills and experience:
        {{
          "work_experience": {{
            "job_title": "Senior Software Engineer",
            "job_title_boost": 4.5,
            "job_title_embedding_boost": 9.5,
            "description": "Python development, cloud architecture, team leadership",
            "description_boost": 3.5,
            "description_embedding_boost": 9.5
          }},
          "skills": "Python, AWS, system design, leadership",
          "skills_boost": 4.0,
          "skills_embedding_boost": 9.5
        }}

        2. Academic Query Example:
        {{
          "education": {{
            "degree": "PhD in Computer Science",
            "degree_boost": 4.5,
            "degree_embedding_boost": 9.0,
            "description": "Machine Learning research focus",
            "description_boost": 3.0,
            "description_embedding_boost": 9.5
          }},
          "publications": {{
            "description": "Deep Learning research publications",
            "description_boost": 3.0,
            "description_embedding_boost": 9.0
          }}
        }}

        Required Fields (if mentioned in query):
        1. Contact Information (location requirements)
        2. Work Experience (roles, responsibilities)
        3. Education (qualifications, institutions)
        4. Skills (technical, soft skills)
        5. Projects (relevant work)
        6. Publications (if relevant)
        7. Certifications (required qualifications)
        8. Languages (if specified)
        9. Awards/Honors (relevant achievements)
        10. Volunteer Experience (if valuable)
        11. Personal Summary (role context)

        Schema Requirements:
        1. Return valid JSON matching QueryFormatter model
        2. Use exact field names - no modifications
        3. Format dates as YYYY-MM-DD
        4. Set null for missing fields with 0.0 boosts
        5. Include rich context for semantic matching
        6. Include relevant synonyms and related terms
        7. Maintain professional terminology
        8. Preserve context and relationships

        Input Text:
        {job_description_text}
        '''
    )

    return ChatPromptTemplate.from_messages([system_message, user_message])

def process_job_description(job_description_text: str, output_directory: str, filename: str):
    """Parse job description text using LangChain and return structured data"""
    setup_logging('job_description_parsing')
    try:
        logging.info("Starting parsing for job description")
        prompt = create_extraction_prompt_job_description(job_description_text)

        prompt = prompt.format_messages()
        
        # Call LangChain API using invoke method
        logging.debug("Sending request to LangChain for job description")
        model = ChatOpenAI(
            temperature=0,
            model="gpt-4o",
            openai_api_key=OPENAI_API_KEY
        )

        model_with_structure = model.with_structured_output(QueryFormatter)

        response = model_with_structure.invoke(prompt)

        pprint.pprint(response)
        
        # Extract JSON from response
        try:
            json_str = response.json()
            parsed_data = json.loads(json_str)  # Parse the JSON string
            
            print(parsed_data)
            logging.info("Successfully parsed and validated job description")
            
            # Ensure the output directory exists
            os.makedirs(output_directory, exist_ok=True)
            
            # Save parsed data to file
            output_file_path = os.path.join(output_directory, filename)
            with open(output_file_path, 'w') as output_file:
                json.dump(parsed_data, output_file, indent=4)
            
            return parsed_data
            
        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing error for job description: {e}")
            return None
            
    except Exception as e:
        logging.error(f"Error during job description parsing: {e}")
        return None