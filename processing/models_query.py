from pydantic import BaseModel, field_validator
from typing import Optional
from datetime import date

def validate_date_format(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    try:
        date.fromisoformat(value)
        return value
    except ValueError:
        raise ValueError("Date must be in YYYY-MM-DD format")

class BoostMixin:
    def _set_boost_values(self):
        for field in self.__fields__:
            if not field.endswith('_boost') and not field.endswith('_embedding_boost'):
                value = getattr(self, field)
                boost_field = f"{field}_boost"
                embedding_boost_field = f"{field}_embedding_boost"
                
                # Handle text search boost
                if hasattr(self, boost_field) and value is None:
                    setattr(self, boost_field, 0)
                
                # Handle embedding boost only if field has it
                if hasattr(self, embedding_boost_field) and value is None:
                    setattr(self, embedding_boost_field, 0)

class ContactInformation(BaseModel, BoostMixin):
    full_name: Optional[str] = None
    full_name_boost: float
    phone_number: Optional[str] = None
    phone_number_boost: float
    email: Optional[str] = None
    email_boost: float
    linkedin: Optional[str] = None
    linkedin_boost: float
    portfolio_website: Optional[str] = None
    portfolio_website_boost: float
    address: Optional[str] = None
    address_boost: float
    address_embedding_boost: float

    def __init__(self, **data):
        super().__init__(**data)
        self._set_boost_values()

class Education(BaseModel, BoostMixin):
    degree: Optional[str] = None
    degree_boost: float
    degree_embedding_boost: float
    institution: Optional[str] = None
    institution_boost: float
    institution_embedding_boost: float
    location: Optional[str] = None
    location_boost: float
    location_embedding_boost: float
    start_date: Optional[str] = None
    start_date_boost: float
    end_date: Optional[str] = None
    end_date_boost: float
    gpa: Optional[float] = None
    gpa_boost: float
    honors: Optional[str] = None
    honors_boost: float
    honors_embedding_boost: float
    description: Optional[str] = None
    description_boost: float
    description_embedding_boost: float

    @field_validator('start_date', 'end_date')
    @classmethod
    def validate_dates(cls, v: Optional[str]) -> Optional[str]:
        return validate_date_format(v)

    def __init__(self, **data):
        super().__init__(**data)
        self._set_boost_values()

class WorkExperience(BaseModel, BoostMixin):
    job_title: Optional[str] = None
    job_title_boost: float
    job_title_embedding_boost: float
    employer: Optional[str] = None
    employer_boost: float
    employer_embedding_boost: float
    location: Optional[str] = None
    location_boost: float
    location_embedding_boost: float
    start_date: Optional[str] = None
    start_date_boost: float
    end_date: Optional[str] = None
    end_date_boost: float
    description: Optional[str] = None
    description_boost: float
    description_embedding_boost: float
    achievements: Optional[str] = None
    achievements_boost: float
    achievements_embedding_boost: float

    @field_validator('start_date', 'end_date')
    @classmethod
    def validate_dates(cls, v: Optional[str]) -> Optional[str]:
        return validate_date_format(v)

    def __init__(self, **data):
        super().__init__(**data)
        self._set_boost_values()

class Project(BaseModel, BoostMixin):
    title: Optional[str] = None
    title_boost: float
    title_embedding_boost: float
    description: Optional[str] = None
    description_boost: float
    description_embedding_boost: float
    role: Optional[str] = None
    role_boost: float
    role_embedding_boost: float
    technologies: Optional[str] = None
    technologies_boost: float
    technologies_embedding_boost: float
    start_date: Optional[str] = None
    start_date_boost: float
    end_date: Optional[str] = None
    end_date_boost: float

    @field_validator('start_date', 'end_date')
    @classmethod
    def validate_dates(cls, v: Optional[str]) -> Optional[str]:
        return validate_date_format(v)

    def __init__(self, **data):
        super().__init__(**data)
        self._set_boost_values()

class Certification(BaseModel, BoostMixin):
    name: Optional[str] = None
    name_boost: float
    name_embedding_boost: float
    description: Optional[str] = None
    description_boost: float
    description_embedding_boost: float
    issuing_organization: Optional[str] = None
    issuing_organization_boost: float
    issuing_organization_embedding_boost: float
    issue_date: Optional[str] = None
    issue_date_boost: float
    expiration_date: Optional[str] = None
    expiration_date_boost: float

    @field_validator('issue_date', 'expiration_date')
    @classmethod
    def validate_dates(cls, v: Optional[str]) -> Optional[str]:
        return validate_date_format(v)

    def __init__(self, **data):
        super().__init__(**data)
        self._set_boost_values()

class Publication(BaseModel, BoostMixin):
    title: Optional[str] = None
    title_boost: float
    title_embedding_boost: float
    publisher: Optional[str] = None
    publisher_boost: float
    publisher_embedding_boost: float
    description: Optional[str] = None
    description_boost: float
    description_embedding_boost: float
    publication_date: Optional[str] = None
    publication_date_boost: float
    url: Optional[str] = None
    url_boost: float

    @field_validator('publication_date')
    @classmethod
    def validate_dates(cls, v: Optional[str]) -> Optional[str]:
        return validate_date_format(v)

    def __init__(self, **data):
        super().__init__(**data)
        self._set_boost_values()

class Language(BaseModel, BoostMixin):
    language: Optional[str] = None
    language_boost: float
    language_embedding_boost: float
    proficiency: Optional[str] = None
    proficiency_boost: float
    proficiency_embedding_boost: float

    def __init__(self, **data):
        super().__init__(**data)
        self._set_boost_values()

class AwardAndHonor(BaseModel, BoostMixin):
    title: Optional[str] = None
    title_boost: float
    title_embedding_boost: float
    issuing_organization: Optional[str] = None
    issuing_organization_boost: float
    issuing_organization_embedding_boost: float
    description: Optional[str] = None
    description_boost: float
    description_embedding_boost: float
    issue_date: Optional[str] = None
    issue_date_boost: float

    @field_validator('issue_date')
    @classmethod
    def validate_dates(cls, v: Optional[str]) -> Optional[str]:
        return validate_date_format(v)

    def __init__(self, **data):
        super().__init__(**data)
        self._set_boost_values()

class VolunteerExperience(BaseModel, BoostMixin):
    role: Optional[str] = None
    role_boost: float
    role_embedding_boost: float
    organization: Optional[str] = None
    organization_boost: float
    organization_embedding_boost: float
    description: Optional[str] = None
    description_boost: float
    description_embedding_boost: float
    start_date: Optional[str] = None
    start_date_boost: float
    end_date: Optional[str] = None
    end_date_boost: float

    @field_validator('start_date', 'end_date')
    @classmethod
    def validate_dates(cls, v: Optional[str]) -> Optional[str]:
        return validate_date_format(v)

    def __init__(self, **data):
        super().__init__(**data)
        self._set_boost_values()

class Reference(BaseModel, BoostMixin):
    name: Optional[str] = None
    name_boost: float
    relationship: Optional[str] = None
    relationship_boost: float
    contact_information: Optional[str] = None
    contact_information_boost: float

    def __init__(self, **data):
        super().__init__(**data)
        self._set_boost_values()

class Skills(BaseModel):
    value: Optional[str] = None
    skills_boost: float
    skills_embedding_boost: float

class PersonalSummary(BaseModel):
    value: Optional[str] = None
    personal_summary_boost: float
    personal_summary_embedding_boost: float

class QueryFormatter(BaseModel, BoostMixin):
    contact_information: Optional[ContactInformation] = None
    personal_summary: Optional[PersonalSummary] = None
    education: Optional[Education] = None
    work_experience: Optional[WorkExperience] = None
    skills: Optional[Skills] = None
    projects: Optional[Project] = None
    certifications: Optional[Certification] = None
    publications: Optional[Publication] = None
    languages: Optional[Language] = None
    awards_and_honors: Optional[AwardAndHonor] = None
    volunteer_experience: Optional[VolunteerExperience] = None
    references: Optional[Reference] = None

    def __init__(self, **data):
        super().__init__(**data)
        self._set_boost_values()