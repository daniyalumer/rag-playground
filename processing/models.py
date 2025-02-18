from pydantic import BaseModel
from typing import Optional, List

# Pydantic models for structured output
class AgeIndicators(BaseModel):
    education_timeline: Optional[str]
    work_experience_type: Optional[str]
    extracurricular_focus: Optional[str]
    email_style: Optional[str]
    certification_level: Optional[str]
    writing_style: Optional[str]

class ContactInformation(BaseModel):
    full_name: Optional[str]
    phone_number: Optional[str]
    email: Optional[str]
    linkedin: Optional[str]
    portfolio_website: Optional[str]
    address: Optional[str]

class Education(BaseModel):
    degree: Optional[str]
    institution: Optional[str]
    location: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]
    gpa: Optional[float]
    honors: Optional[str]
    description: Optional[str]

class WorkExperience(BaseModel):
    job_title: Optional[str]
    employer: Optional[str]
    location: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]
    description: Optional[str]
    achievements: Optional[str]

class Project(BaseModel):
    title: Optional[str]
    description: Optional[str]
    role: Optional[str]
    technologies: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]

class Certification(BaseModel):
    name: Optional[str]
    description: Optional[str]
    issuing_organization: Optional[str]
    issue_date: Optional[str]
    expiration_date: Optional[str]

class Publication(BaseModel):
    title: Optional[str]
    publisher: Optional[str]
    description: Optional[str]
    publication_date: Optional[str]
    url: Optional[str]

class Language(BaseModel):
    language: Optional[str]
    proficiency: Optional[str]

class AwardAndHonor(BaseModel):
    title: Optional[str]
    issuing_organization: Optional[str]
    issue_date: Optional[str]
    description: Optional[str]

class VolunteerExperience(BaseModel):
    role: Optional[str]
    organization: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]
    description: Optional[str]

class Reference(BaseModel):
    name: Optional[str]
    relationship: Optional[str]
    contact_information: Optional[str]

class ResponseFormatter(BaseModel):
    is_teenage: bool
    teenage_confidence: float #= Field(ge=0.0, le=1.0)
    age_indicators: AgeIndicators
    contact_information: ContactInformation
    personal_summary: Optional[str]
    education: Optional[List[Education]]
    work_experience: Optional[List[WorkExperience]]
    skills: Optional[str]
    projects: Optional[List[Project]]
    certifications: Optional[List[Certification]]
    publications: Optional[List[Publication]]
    languages: Optional[List[Language]]
    awards_and_honors: Optional[List[AwardAndHonor]]
    volunteer_experience: Optional[List[VolunteerExperience]]
    references: Optional[List[Reference]]
