"""
Data models for the Agentic RAG system
"""
from typing import List, Literal, Optional, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    
    datasource: Literal["vectorstore", "general_knowledge"] = Field(
        ...,
        description="Given a user question choose to route it to general knowledge or a vectorstore.",
    )


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""
    
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


class UserIntent(BaseModel):
    """Analyze user intent and extract key requirements"""
    
    product_category: str = Field(description="Main product category user is interested in")
    key_features: List[str] = Field(description="Important features mentioned by user")
    budget_mentioned: bool = Field(description="Whether user mentioned budget constraints")
    urgency_level: Literal["low", "medium", "high"] = Field(description="How urgent the purchase is")
    clarification_needed: bool = Field(description="Whether we need more information")
    follow_up_question: Optional[str] = Field(description="Question to ask for clarification")


class ProductRecommendation(BaseModel):
    """Product recommendation with reasoning"""
    
    product_name: str = Field(description="Name of recommended product")
    match_score: float = Field(description="How well product matches user needs (0-1)")
    key_benefits: List[str] = Field(description="Why this product fits user needs")
    price_range: Optional[str] = Field(description="Price range if available")


class GraphState(TypedDict):
    """
    Represents the state of our graph.
    
    Attributes:
        question: Original user question
        generation: LLM generation
        documents: List of retrieved documents
    """
    
    remaining_steps: str
    messages: Annotated[List[BaseMessage], add_messages]
    question: Annotated[List[BaseMessage], add_messages]
    generation: str
    documents: List[str]