from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from datetime import date

def validate_date_format(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    try:
        date.fromisoformat(value)
        return value
    except ValueError:
        raise ValueError("Date must be in YYYY-MM-DD format")

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

    @field_validator('start_date', 'end_date')
    @classmethod
    def validate_dates(cls, v: Optional[str]) -> Optional[str]:
        return validate_date_format(v)

class WorkExperience(BaseModel):
    job_title: Optional[str]
    employer: Optional[str]
    location: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]
    description: Optional[str]
    achievements: Optional[str]

    @field_validator('start_date', 'end_date')
    @classmethod
    def validate_dates(cls, v: Optional[str]) -> Optional[str]:
        return validate_date_format(v)

class Project(BaseModel):
    title: Optional[str]
    description: Optional[str]
    role: Optional[str]
    technologies: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]

    @field_validator('start_date', 'end_date')
    @classmethod
    def validate_dates(cls, v: Optional[str]) -> Optional[str]:
        return validate_date_format(v)

class Certification(BaseModel):
    name: Optional[str]
    description: Optional[str]
    issuing_organization: Optional[str]
    issue_date: Optional[str]
    expiration_date: Optional[str]

    @field_validator('issue_date', 'expiration_date')
    @classmethod
    def validate_dates(cls, v: Optional[str]) -> Optional[str]:
        return validate_date_format(v)

class Publication(BaseModel):
    title: Optional[str]
    publisher: Optional[str]
    description: Optional[str]
    publication_date: Optional[str]
    url: Optional[str]

    @field_validator('publication_date')
    @classmethod
    def validate_dates(cls, v: Optional[str]) -> Optional[str]:
        return validate_date_format(v)

class Language(BaseModel):
    language: Optional[str]
    proficiency: Optional[str]

class AwardAndHonor(BaseModel):
    title: Optional[str]
    issuing_organization: Optional[str]
    issue_date: Optional[str]
    description: Optional[str]

    @field_validator('issue_date')
    @classmethod
    def validate_dates(cls, v: Optional[str]) -> Optional[str]:
        return validate_date_format(v)

class VolunteerExperience(BaseModel):
    role: Optional[str]
    organization: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]
    description: Optional[str]

    @field_validator('start_date', 'end_date')
    @classmethod
    def validate_dates(cls, v: Optional[str]) -> Optional[str]:
        return validate_date_format(v)

class Reference(BaseModel):
    name: Optional[str]
    relationship: Optional[str]
    contact_information: Optional[str]

class ResponseFormatter(BaseModel):
    is_teenage: bool
    teenage_confidence: float
    age_indicators: AgeIndicators
    contact_information: Optional[ContactInformation]
    personal_summary: Optional[str]
    education: Optional[Education]
    work_experience: Optional[WorkExperience]
    skills: Optional[str]
    projects: Optional[Project]
    certifications: Optional[Certification]
    publications: Optional[Publication]
    languages: Optional[Language]
    awards_and_honors: Optional[AwardAndHonor]
    volunteer_experience: Optional[VolunteerExperience]
    references: Optional[Reference]