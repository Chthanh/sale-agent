"""
RAG workflow implementation using LangGraph
"""
from langgraph.graph import END, StateGraph, START
from langchain.schema import Document
from langgraph.checkpoint.memory import MemorySaver

import logging

from models import GraphState
from llm import llm_handler
from database import vector_db

logger = logging.getLogger(__name__)


class RAGWorkflow:
    """RAG workflow implementation"""
    
    def __init__(self):
        self.app = self._build_workflow()
    
    def _build_workflow(self):
        """Build the workflow graph"""
        workflow = StateGraph(GraphState)
        
        # Define nodes
        workflow.add_node("web_search", self.web_search)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("generate", self.generate)
        workflow.add_node("transform_query", self.transform_query)
        
        # Build graph with conditional edges
        workflow.add_conditional_edges(
            START,
            self.route_question,
            {
                "web_search": "web_search",
                "vectorstore": "retrieve",
            },
        )
        
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("retrieve", "grade_documents")
        
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        
        workflow.add_edge("transform_query", "retrieve")
        
        workflow.add_edge("generate", END)
        # workflow.add_conditional_edges(
        #     "generate",
        #     self.grade_generation_v_documents_and_question,
        #     {
        #         "not supported": "generate",
        #         "useful": END,
        #         "not useful": "transform_query",
        #     },
        # )
        
        return workflow.compile(checkpointer=MemorySaver())
    
    def retrieve(self, state: GraphState) -> GraphState:
        """Retrieve documents from vector database"""
        logger.info("---RETRIEVE---")
        question = state["question"]
        
        documents = vector_db.retrieve_documents(question)
        return {"documents": documents, "question": question}
    
    def generate(self, state: GraphState) -> GraphState:
        """Generate answer using RAG"""
        logger.info("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        
        # Format documents for context
        context = llm_handler.format_docs(documents)
        generation = llm_handler.generate_answer(context, question)
        
        return {"documents": documents, "question": question, "generation": generation}
    
    def grade_documents(self, state: GraphState) -> GraphState:
        """Grade retrieved documents for relevance"""
        logger.info("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]
        
        filtered_docs = []
        for doc in documents:
            score = llm_handler.grade_documents(question, doc.page_content)
            if score == "yes":
                logger.info("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(doc)
            else:
                logger.info("---GRADE: DOCUMENT NOT RELEVANT---")
        
        return {"documents": filtered_docs, "question": question}
    
    def transform_query(self, state: GraphState) -> GraphState:
        """Transform query for better retrieval"""
        logger.info("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]
        
        better_question = llm_handler.question_rewriter.invoke({"question": question})
        return {"documents": documents, "question": better_question}
    
    def web_search(self, state: GraphState) -> GraphState:
        """Perform web search"""
        logger.info("---WEB SEARCH---")
        question = state["question"]
        
        web_results = llm_handler.web_search(question)
        web_document = Document(page_content=web_results)
        
        return {"documents": [web_document], "question": question}
    
    def route_question(self, state: GraphState) -> str:
        """Route question to appropriate data source"""
        logger.info("---ROUTE QUESTION---")
        question = state["question"]
        
        source = llm_handler.route_question(question)
        if source == "web_search":
            logger.info("---ROUTE QUESTION TO WEB SEARCH---")
            return "web_search"
        else:
            logger.info("---ROUTE QUESTION TO RAG---")
            return "vectorstore"
    
    def decide_to_generate(self, state: GraphState) -> str:
        """Decide whether to generate or retry with transformed query"""
        logger.info("---ASSESS GRADED DOCUMENTS---")
        filtered_documents = state["documents"]
        
        if not filtered_documents:
            logger.info("---DECISION: ALL DOCUMENTS NOT RELEVANT, TRANSFORM QUERY---")
            return "transform_query"
        else:
            logger.info("---DECISION: GENERATE---")
            return "generate"
    
    def grade_generation_v_documents_and_question(self, state: GraphState) -> str:
        """Grade generation against documents and question"""
        logger.info("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        import time
        time.sleep(10)  # Simulate processing delay
        
        # Check for hallucinations
        hallucination_score = llm_handler.hallucination_grader.invoke({
            "documents": documents, 
            "generation": generation
        })
        
        if hallucination_score.binary_score == "yes":
            logger.info("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            
            # Check if generation addresses the question
            answer_score = llm_handler.answer_grader.invoke({
                "question": question, 
                "generation": generation
            })
            
            if answer_score.binary_score == "yes":
                logger.info("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                logger.info("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            logger.info("---DECISION: GENERATION NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"
    
    def run(self, question: str) -> dict:
        """Run the RAG workflow with a question"""
        import json
        
        config = {"configurable": {"thread_id": "thread-1"}}
        
        try:
            inputs = {"question": question}
            result = None
            
            for output in self.app.stream(inputs, config=config):
                result = output
            
            
            # Show results
            for i, doc in enumerate(result['generate']['documents']):
                result['generate']['documents'][i] = {"metadata": doc.metadata, "page_content": doc.page_content}

            with open("output.json", "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
                
                
            # Return the final result
            if result:
                final_key = list(result.keys())[-1]
                return result[final_key]
            else:
                return {"generation": "I apologize, but I couldn't process your question properly."}
                
        except Exception as e:
            logger.error(f"Error running RAG workflow: {e}")
            return {"generation": "I encountered an error processing your request. Please try again."}

# Global workflow instance
rag_workflow = RAGWorkflow()

while True:
    try:
        # Example usage
        question = input("Enter your question: ")
        if question.lower() in ["exit", "quit"]:
            break
        
        result = rag_workflow.run(question)
        print("result:", result)
        print(f"Generated answer: {result.get('generation', 'No answer generated')}")
    
    except KeyboardInterrupt:
        print("\nExiting...")
        break
    except Exception as e:
        print(f"Error: {e}")
        
        