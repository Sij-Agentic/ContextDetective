from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime

# --- Input Models ---
class PerceptionInput(BaseModel):
    """Input for perception tools"""
    image_path: str = Field(..., description="Path to the image file to analyze")

class SearchTermsInput(BaseModel):
    """Input for search term generation"""
    descriptions: List[str] = Field(..., description="Analysis descriptions from previous steps")

class WebSearchInput(BaseModel):
    """Input for web search"""
    query: str = Field(..., description="Search query string")

class ContextInferenceInput(BaseModel):
    """Input for context inference"""
    visual_elements: str = Field(..., description="Results from visual elements analysis")
    style_analysis: str = Field(..., description="Results from style analysis")
    scenario_analysis: str = Field(..., description="Results from scenario analysis")
    web_findings: str = Field(..., description="Results from web search findings")

class MemoryStoreInput(BaseModel):
    """Input for storing analysis in memory"""
    image_hash: str = Field(..., description="Unique hash identifier for the image")
    analysis_json: str = Field(..., description="Complete analysis JSON to store")

class MemoryRetrieveInput(BaseModel):
    """Input for retrieving analysis from memory"""
    image_hash: str = Field(..., description="Unique hash identifier for the image")

class FinalOutputInput(BaseModel):
    """Input for formatting final output"""
    context_guess: str = Field(..., description="Final context determination")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the determination (0-1)")
    explanation: str = Field(..., description="Detailed explanation")
    related_links: List[str] = Field(default_factory=list, description="Related reference links")
    search_terms: List[str] = Field(..., description="Search terms used in analysis")

# --- Output Models ---
class UserPreferences(BaseModel):
    """User preferences that influence the analysis"""
    interests: List[str] = Field(..., description="User's topics of interest")

class VisualElementsAnalysis(BaseModel):
    """Results of visual elements analysis"""
    objects: List[str] = Field(..., description="Objects detected in the image")
    colors: List[str] = Field(..., description="Dominant colors in the image")
    text: Optional[str] = Field(None, description="Text detected in the image")
    people: Optional[List[str]] = Field(default_factory=list, description="People detected in the image")

class StyleAnalysis(BaseModel):
    """Results of style analysis"""
    artistic_style: str = Field(..., description="Overall artistic style")
    composition: str = Field(..., description="Composition description")
    cultural_elements: Optional[List[str]] = Field(default_factory=list, description="Cultural elements detected")

class ScenarioAnalysis(BaseModel):
    """Results of scenario analysis"""
    possible_scenario: str = Field(..., description="Main scenario detected")
    setting: str = Field(..., description="Setting/environment description")
    activity: Optional[str] = Field(None, description="Main activity detected")

class CompleteVisualAnalysis(BaseModel):
    """Combined results of all visual analyses"""
    visual_elements: VisualElementsAnalysis
    style: StyleAnalysis
    scenario: ScenarioAnalysis

class MemoryEntry(BaseModel):
    """Storage model for previous analyses"""
    image_hash: str = Field(..., description="Unique hash of the image")
    analysis: CompleteVisualAnalysis
    user_preferences: UserPreferences
    timestamp: datetime = Field(default_factory=datetime.now)

class SearchResult(BaseModel):
    """Model for web search results"""
    title: str = Field(..., description="Title of the search result")
    snippet: str = Field(..., description="Relevant text snippet")
    url: str = Field(..., description="Source URL")

class ContextDecision(BaseModel):
    """Model for context inference results"""
    context: str = Field(..., description="Inferred context")
    explanation: str = Field(..., description="Reasoning behind the context")
    supporting_evidence: List[str] = Field(..., description="Evidence supporting the context")

class ActionOutput(BaseModel):
    """Final output model"""
    context_guess: str = Field(..., description="Final context determination")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the determination")
    explanation: str = Field(..., description="Detailed explanation")
    related_links: List[str] = Field(default_factory=list, description="Related reference links")
    search_terms_used: List[str] = Field(..., description="Search terms used in analysis")