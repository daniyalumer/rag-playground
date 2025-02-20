import os
import json
import logging
import pandas as pd
import pprint
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from config.config import OPENAI_API_KEY
from config.logging_config import setup_logging 
from processing.models import ResponseFormatter

def create_extraction_prompt_cv(cv_text: str) -> ChatPromptTemplate:
    system_message = SystemMessage(
        content="""
        You are a CV parsing assistant. Your task is to extract structured information from the given CV and analyze whether it belongs to a teenager. 
        Return ONLY a valid JSON object that strictly follows the predefined schema—no additional text, comments, or explanations.
        
        **Important:** The response **must** adhere to the predefined schema with exact key names and correct data types.
        
        Pay special attention to indicators of a teenage CV:
        1. Education timeline (current high school/early university status)
        2. Limited or no traditional work experience
        3. Heavy focus on extracurricular activities
        4. Email address style (casual/informal)
        5. Basic or introductory level certifications
        6. Writing style and tone (less professional/formal)
        7. Early-age internships or part-time work
        8. References to parents/guardians
        """
    )

    user_message = HumanMessage(
        content=f"""
        Extract structured information from the CV text below while strictly adhering to these rules:

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
          - `education_timeline`: Note current educational stage.
          - `work_experience_type`: Describe nature of work history.
          - `extracurricular_focus`: Note emphasis on activities.
          - `email_style`: Analyze email formality.
          - `certification_level`: Assess certification complexity.
          - `writing_style`: Evaluate overall tone and maturity.

        **Strict Compliance Notice:**
        - The response **must** be a valid JSON object matching the schema exactly.
        - No extra commentary, markdown formatting, or explanations are allowed.

        **CV TEXT:**
        {cv_text}
        """
    )

    return ChatPromptTemplate.from_messages([system_message, user_message])

def parse_cv(raw_text: str, filename: str) -> Optional[ResponseFormatter]:
    """Parse CV text using LangChain and return structured data"""
    try:
        logging.info(f"Starting parsing for {filename}")
        prompt = create_extraction_prompt_cv(raw_text)

        prompt = prompt.format_messages()
        
        # Call LangChain API using invoke method
        logging.debug(f"Sending request to LangChain for {filename}")
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
            
            # Add document identifier without .pdf extension
            document_id = os.path.splitext(filename)[0]  # Strip .pdf from filename
            parsed_data['document_id'] = document_id  # Add the document identifier
            
            print(parsed_data)
            logging.info(f"Successfully parsed and validated {filename}")
            return parsed_data
            
        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing error for {filename}: {e}")
            return None
            
    except Exception as e:
        logging.error(f"Error during CV parsing for {filename}: {e}")
        return None
    
def process_cvs(csv_path: str, batch_size: int = 10):
    """Process all CVs from CSV and save parsed results"""
    try:
        # Setup logging
        setup_logging('cv_parsing')
        logging.info(f"Starting CV processing with batch size {batch_size}")
        
        # Read CSV with raw CV text
        df = pd.read_csv(csv_path)
        logging.info(f"Loaded {len(df)} CVs from {csv_path}")
        
        # Create output directory
        output_dir = "data/parsed_data"
        os.makedirs(output_dir, exist_ok=True)
        
        # Process CVs in batches
        successful_parses = 0
        failed_parses = 0
        skipped_files = 0

        # Process only one batch of CVs
        logging.info(f"Processing batch 1/1")
        batch = df.iloc[0:batch_size]  # Get the first batch
        
        #for i in range(0, len(df), batch_size):
        #    batch = df.iloc[i:i + batch_size]
        #    logging.info(f"Processing batch {i//batch_size + 1}/{(len(df) + batch_size - 1) // batch_size}")
            
        for _, row in batch.iterrows():
            output_file = os.path.join(output_dir, f"{os.path.splitext(row['filename'])[0]}.json")
            
            # Skip if already processed
            if os.path.exists(output_file):
                logging.info(f"Skipping {row['filename']} - already processed")
                skipped_files += 1
                continue
            
            # Parse CV
            parsed_data = parse_cv(row['preprocessed_text'], row['filename'])
            if parsed_data:
                # Save to JSON file
                try:
                    with open(output_file, 'w') as f:
                        json.dump(parsed_data, f, indent=2)
                    logging.info(f"Successfully saved {row['filename']}")
                    successful_parses += 1
                except Exception as e:
                    logging.error(f"Error saving {row['filename']}: {e}")
                    failed_parses += 1
            else:
                failed_parses += 1
        
        # Log summary statistics
        logging.info("Processing completed!")
        logging.info(f"Successfully parsed: {successful_parses}")
        logging.info(f"Failed to parse: {failed_parses}")
        logging.info(f"Skipped files: {skipped_files}")
        
    except Exception as e:
        logging.error(f"Error during batch processing: {e}")
        raise


