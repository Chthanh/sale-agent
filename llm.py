"""
Language model handler for the RAG system
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
import logging

from config import config
from models import (
    RouteQuery, GradeDocuments, GradeHallucinations, 
    GradeAnswer, UserIntent, ProductRecommendation
)

logger = logging.getLogger(__name__)


class LLMHandler:
    """Handles all LLM operations and chains"""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=config.LLM_MODEL_NAME,
            temperature=config.LLM_TEMPERATURE
        )
        self.web_search_tool = TavilySearchResults(k=config.WEB_SEARCH_RESULTS)
        self._setup_chains()
    
    def _setup_chains(self):
        """Setup all the LLM chains"""
        self._setup_router()
        self._setup_graders()
        self._setup_generators()
        self._setup_chatbot_chains()
    
    def _setup_router(self):
        """Setup query routing chain"""
        system = """You are an expert at routing a user question to a vectorstore or web search.
        The vectorstore contains documents related to smartphones, laptops, and headphones.
        Use the vectorstore for questions on these topics. Otherwise, use web search."""
        
        route_prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "{question}"),
        ])
        
        self.question_router = route_prompt | self.llm.with_structured_output(RouteQuery)
    
    def _setup_graders(self):
        """Setup document and answer grading chains"""
        
        # Document relevance grader
        grade_system = """You are a grader assessing relevance of a retrieved document to a user question.
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
        The goal is to filter out erroneous retrievals. Give a binary score 'yes' or 'no'."""
        
        grade_prompt = ChatPromptTemplate.from_messages([
            ("system", grade_system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ])
        
        self.retrieval_grader = grade_prompt | self.llm.with_structured_output(GradeDocuments)
        
        # Hallucination grader
        hallucination_system = """You are a grader assessing whether an LLM generation is grounded in retrieved facts.
        Give a binary score 'yes' or 'no'. 'Yes' means the answer is grounded in the facts."""
        
        hallucination_prompt = ChatPromptTemplate.from_messages([
            ("system", hallucination_system),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ])
        
        self.hallucination_grader = hallucination_prompt | self.llm.with_structured_output(GradeHallucinations)
        
        # Answer grader
        answer_system = """You are a grader assessing whether an answer addresses/resolves a question.
        Give a binary score 'yes' or 'no'. 'Yes' means the answer resolves the question."""
        
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", answer_system),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
        ])
        
        self.answer_grader = answer_prompt | self.llm.with_structured_output(GradeAnswer)
    
    def _setup_generators(self):
        """Setup generation chains"""
        
        # RAG chain
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system" ,"""You are a professional sales assistant. Use the following pieces of retrieved context to answer the question. 
            Use three sentences maximum and keep the answer concise. 
            Always generate a helpful response and end with a EXACTLY ONE question to ask the user for clarification if needed or getting more information about cuscomer's demand to sell products."""),
                ("human", "\nQuestion: {question} \nContext: {context} \nAnswer: "),
            ])
        except:
            # Fallback prompt if hub is not available
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Answer the question based on the provided context. Be helpful and accurate."),
                ("human", "Context: {context}\n\nQuestion: {question}")
            ])
        
        self.rag_chain = prompt | self.llm | StrOutputParser()
        
        # Question rewriter
        rewrite_system = """You are a question re-writer that converts an input question to a better version
        optimized for vectorstore retrieval. Look at the input and reason about the underlying semantic intent."""
        
        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", rewrite_system),
            ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
        ])
        
        self.question_rewriter = rewrite_prompt | self.llm | StrOutputParser()
    
    def _setup_chatbot_chains(self):
        """Setup chatbot-specific chains for user interaction"""
        
        # Intent analyzer
        intent_system = """You are an AI assistant that analyzes user intent for product purchases.
        Extract key information about what the user wants to buy and determine if you need more information.
        If clarification is needed, suggest ONE specific question to better understand their needs."""
        
        intent_prompt = ChatPromptTemplate.from_messages([
            ("system", intent_system),
            ("human", "User message: {question}")
        ])
        
        self.intent_analyzer = intent_prompt | self.llm.with_structured_output(UserIntent)
        
        # Product recommender
        recommender_system = """You are a product recommendation expert. Based on user requirements and available products,
        provide personalized recommendations with clear reasoning for why each product fits their needs."""
        
        recommender_prompt = ChatPromptTemplate.from_messages([
            ("system", recommender_system),
            ("human", "User requirements: {requirements}\n\nAvailable products: {products}\n\nProvide recommendations:")
        ])
        
        self.product_recommender = recommender_prompt | self.llm.with_structured_output(ProductRecommendation)
    
    def format_docs(self, docs):
        """Format documents for context"""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def route_question(self, question: str) -> str:
        """Route question to appropriate data source"""
        try:
            result = self.question_router.invoke({"question": question})
            return result.datasource
        except Exception as e:
            logger.error(f"Error routing question: {e}")
            return "vectorstore"  # Default fallback
    
    def grade_documents(self, question: str, document: str) -> str:
        """Grade document relevance"""
        try:
            result = self.retrieval_grader.invoke({"question": question, "document": document})
            return result.binary_score
        except Exception as e:
            logger.error(f"Error grading document: {e}")
            return "yes"  # Default to including document
    
    def generate_answer(self, context: str, question: str) -> str:
        """Generate answer using RAG"""
        try:
            return self.rag_chain.invoke({"context": context, "question": question})
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I apologize, but I couldn't generate a proper response at this time."
    
    def analyze_intent(self, question: str) -> UserIntent:
        """Analyze user intent and determine if clarification is needed"""
        try:
            return self.intent_analyzer.invoke({"question": question})
        except Exception as e:
            logger.error(f"Error analyzing intent: {e}")
            # Return default intent
            return UserIntent(
                product_category="general",
                key_features=[],
                budget_mentioned=False,
                urgency_level="medium",
                clarification_needed=True,
                follow_up_question="Could you tell me more about what specific features are most important to you?"
            )
    
    def web_search(self, query: str) -> str:
        """Perform web search"""
        try:
            docs = self.web_search_tool.invoke({"query": query})
            return "\n".join([d["content"] for d in docs])
        except Exception as e:
            logger.error(f"Error in web search: {e}")
            return ""


# Global LLM handler instance
llm_handler = LLMHandler()