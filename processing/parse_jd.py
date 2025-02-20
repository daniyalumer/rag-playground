import os
import pprint
import json
import logging
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from config.logging_config import setup_logging 
from config.config import OPENAI_API_KEY
from processing.models import ResponseFormatter

def create_extraction_prompt_job_description(job_description_text: str) -> ChatPromptTemplate:
    system_message = SystemMessage(
        content="""
        You are an advanced semantic parsing expert specializing in job description analysis and vector-based matching. 
        Your role is to extract rich, semantically meaningful information that will be converted into vector embeddings for matching with candidate profiles. 
        Your task is to extract structured information from the given job description and analyze whether it belongs to a teenager. 
        Return ONLY a valid JSON object that strictly follows the predefined schema—no additional text, comments, or explanations.
        
        **Important:** The response **must** adhere to the predefined schema with exact key names and correct data types.
        
        Pay special attention to indicators of a teenage CV:
        1. Education timeline (required education level)
        2. Work experience type (required work experience)
        3. Extracurricular focus (desired extracurricular activities)
        4. Email style (formality of communication)
        5. Certification level (required certifications)
        6. Writing style (tone and formality of the job description)

        Core Responsibilities:
        1. Extract explicit requirements and implicit expectations
        2. Analyze semantic context and professional level
        3. Identify age-appropriate indicators
        4. Assess skill transferability for young applicants
        5. Determine teenage suitability with high confidence

        Guidelines for Semantic Analysis:
        - Consider both direct statements and contextual implications
        - Identify core competencies vs. preferred qualifications
        - Evaluate flexibility in requirements for young applicants
        - Assess the developmental nature of stated responsibilities
        - Map professional requirements to teenage-equivalent experiences
        """
    )

    user_message = HumanMessage(
        content=f"""
        Extract structured information from the job description text below while strictly adhering to these rules:

        **Output Requirements:**
        - Return a **valid JSON object** that strictly follows the predefined schema.
        - **Do not include extra text, explanations, or formatting.**
        - Ensure all keys are present in the JSON, even if their values are `null`.
        - **Use exact key names** as defined in the schema—do not modify, add, or remove any keys.
        - Maintain **correct data types** (e.g., `boolean`, `float`, `string`).
        - For missing data, use **`null`** (JSON equivalent of `None` in Python).
        - Format all dates as **`YYYY-MM-DD`** (e.g., "2025-02-13").

        **Teenage Analysis Requirements:**
        - Set `is_teenage` boolean based on overall assessment.
        - Provide `teenage_confidence` as a float between 0.0 and 1.0.
        - Ensure `age_indicators` object includes:
        - `education_timeline`: Note required educational stage.
        - `work_experience_type`: Describe required work experience.
        - `extracurricular_focus`: Note desired extracurricular activities.
        - `email_style`: Analyze formality of communication.
        - `certification_level`: Assess required certification complexity.
        - `writing_style`: Evaluate overall tone and formality.

        1. Teenage Suitability Analysis:
           - Evaluate 'is_teenage' based on:
            * Required education level vs. typical teenage education.
            * Experience requirements vs. possible teenage experience.
            * Responsibility level vs. teenage capability.
            * Legal working age requirements.
            * Developmental appropriateness.
           - Calculate 'teenage_confidence' considering:
            * Clarity of age-appropriate indicators.
            * Flexibility of requirements.
            * Development potential.
            * Supervision level.
            * Safety considerations.

        2. Age Indicators Extraction:
           - education_timeline: 
            * Identify the education requirements mentioned in the job description.
            * Map them to typical teenage education stages (e.g., high school, early college).
            * Consider whether the role is open to candidates currently pursuing education or in transition phases (e.g., summer jobs, internships).
           - work_experience_type:
            * Identify the nature of required work experience.
            * Determine whether equivalent teenage experiences (e.g., internships, volunteer work, school projects, part-time jobs) are acceptable.
            * Assess whether the job considers informal or seasonal/part-time experience relevant.
           - extracurricular_focus:
            * Identify any emphasis on extracurricular activities.
            * Map these to typical teenage activities such as student organizations, sports, coding bootcamps, school clubs, or online certifications.
            * Assess whether these activities contribute to skill development relevant to the job.
           - email_style:
            * Assess if the job expects a formal or professional communication style.
            * Consider whether the required level of written/verbal communication is developmentally appropriate for teenagers.
           - certification_level:
            * Identify any certification requirements in the job description.
            * Assess whether these certifications are accessible to teenagers or have age restrictions.
           - writing_style:
            * Analyze the language and tone of the job description.
            * Determine whether the role expects a highly professional writing style or allows for a more flexible, learning-oriented approach.

        3. Work Experience Section:
           Provide rich descriptions for embedding fields:
           - job_title: Include role context and alternative titles.
           - employer: Describe organization type and environment.
           - description: Detailed responsibilities with context.
           - achievements: Expected outcomes and success metrics.

        4. Skills Section:
           Create a comprehensive skills description including:
           - Technical skills with proficiency levels.
           - Soft skills with contextual examples.
           - Related and transferable skills.
           - Skill application contexts.

        5. Education Section:
           Rich descriptions for:
           - degree: Include field context and alternatives.
           - institution: Type and level of institution.
           - honors: Academic achievements and their significance.
           - description: Program details and relevant coursework.

        6. Projects Section:
           Detailed descriptions for:
           - title: Project context and scope.
           - description: Comprehensive project details.
           - role: Responsibilities and leadership aspects.
           - technologies: Technical stack and tools used.

        7. Personal Summary Guidance:
           Extract or infer:
           - Key qualifications and their relevance.
           - Professional level and experience.
           - Core competencies and strengths.
           - Career trajectory indicators.

        8. Additional Sections:
           For all remaining sections (certifications, publications, languages, awards, volunteer experience):
           - Provide detailed, context-rich descriptions.
           - Include relevance to the role.
           - Maintain semantic richness for embedding.


        Semantic Enhancement Requirements:
        - Use clear, specific language.
        - Include relevant synonyms and related terms.
        - Maintain professional terminology.
        - Preserve context and relationships.
        - Include implied skills and requirements.

        Output Requirements:
        - Follow exact schema structure.
        - Ensure all text fields are detailed enough for meaningful embeddings.
        - Include relevant synonyms and related terms.
        - Maintain professional terminology.
        - Preserve context and relationships.
        - Include implied skills and requirements.
        - Use null for missing fields.
        - Format dates as YYYY-MM-DD.
        - Maintain semantic consistency across all fields.

        **Strict Compliance Notice:**
        - The response **must** be a valid JSON object matching the schema exactly.
        - No extra commentary, markdown formatting, or explanations are allowed.

        **JOB DESCRIPTION TEXT:**
        {job_description_text}
        """
    )


    return ChatPromptTemplate.from_messages([system_message, user_message])

def process_job_description(job_description_text: str, output_directory: str):
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

        model_with_structure = model.with_structured_output(ResponseFormatter)

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
            output_file_path = os.path.join(output_directory, 'parsed_job_description.json')
            with open(output_file_path, 'w') as output_file:
                json.dump(parsed_data, output_file, indent=4)
            
            return parsed_data
            
        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing error for job description: {e}")
            return None
            
    except Exception as e:
        logging.error(f"Error during job description parsing: {e}")
        return None