from langgraph.graph import StateGraph, START, END
from llm import llm_handler
from agentic_rag import rag_workflow
from models import GraphState
from langgraph.checkpoint.memory import MemorySaver
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4.1-mini",  
    temperature=0.0,
    max_tokens=1000,
    streaming=False
)


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def route_question(state: GraphState) -> str:
    """Route question to appropriate data source"""
    question = state["question"]
    source = llm_handler.route_question(question)
    if source == "vectorstore":
        return "vectorstore"
    else:
        return "general_knowledge"

def general_knowledge_answer(state: GraphState) -> GraphState:
    """Answer using LLM's general knowledge"""
    question = state["question"]
    # Use LLM to answer based on its training knowledge
    generation = llm_handler.generate_general_answer(question)
    return {
        "documents": [],
        "question": question,
        "generation": generation
    }
    
    
def rag_retrieve(question: str) -> str:
    """Retrieve relevant documents relating to products such as laptops, phones, headphones"""
    try:
        config = {"configurable": {"thread_id": "thread-1"}}
        print("question in retrieve node: ", question)
        inputs = {"question": question}
        result = None
        for output in rag_workflow.app.stream(inputs, config=config):
            result = output
        # Process results
        if result:
            # Return the final result
            final_key = list(result.keys())[-1]
            documents = llm_handler.format_docs(result[final_key]["documents"])
            return documents
        else:
            return {"generation": "I apologize, but I couldn't process your question properly."}
    except Exception as e:
        logging.error(f"Error running RAG workflow: {e}")
        return {"generation": "I encountered an error processing your request. Please try again."}

# Define proper Pydantic schema for the tool
class RAGToolInput(BaseModel):
    question: str = Field(description="The question to be answered using RAG workflow")
