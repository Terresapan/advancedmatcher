from typing import List, Optional
from pydantic import BaseModel, Field

# PROJECT MODE STATE
class Requirement(BaseModel):
    """A single project requirement."""
    skill: str = Field(description="The required skill (e.g., 'finance', 'marketing', etc.)")
    importance: Optional[str] = Field(None, description="The importance level of this skill")

class ProjectAnalysis(BaseModel):
    """Analysis of a project description to extract requirements."""
    project_name: str = Field(description="The name of the project")
    project_scope: str = Field(description="Brief description of the project scope")
    client_expectations: str = Field(description="What the client expects from this project")
    requirements: List[Requirement] = Field(
        default_factory=list,
        description="List of skills and expertise required for the project"
    )
    industry: Optional[str] = Field(None, description="The industry the project is in, if specified")

class Criterion(BaseModel):
    """A single search criterion for consultant search."""
    field: str = Field(description="The field to search on (e.g., 'finance', 'marketing', 'strategy', 'operations', 'entrepreneurship', 'industry', 'availability')")
    value: str = Field(description="The required value for this field (e.g., 'expertise', 'healthcare', 'available', etc)")

class ProjectRequirement(BaseModel):
    """Analysis of a consultant database query to determine if it's criteria-based."""
    is_criteria_search: bool = Field(
        description="Whether the summary of the project is matching multiple criteria"
    )
    criteria: List[Criterion] = Field(
        default_factory=list,
        description="List of criteria extracted from the summary of the project"
    )
class ProjectConsultantState(BaseModel):
    """State for the project-consultant matching flow."""
    project_text: str
    analysis: Optional[ProjectRequirement] = None
    project_summary: str = ""
    requirements_approved: bool = False 
    consultant_matches: List = Field(default_factory=list)
    context: str = ""
    response: str = ""

