"""
RAG workflow implementation using LangGraph with Gradio Interface
"""
from langgraph.graph import END, StateGraph, START
from langchain.schema import Document
from langgraph.checkpoint.memory import InMemorySaver

import logging
import gradio as gr
import json
from datetime import datetime

from models import GraphState
from llm import llm_handler
from database import vector_db
from proactive import proactive_agent

logger = logging.getLogger(__name__)

class RAGWorkflow:
    """RAG workflow implementation"""
    
    def __init__(self):
        self.app = self._build_workflow()
    
    def _build_workflow(self):
        """Build the workflow graph"""
        workflow = StateGraph(GraphState)
        
        # Define nodes
        #workflow.add_node("general_knowledge", self.general_knowledge_answer)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("grade_documents", self.grade_documents)
        #workflow.add_node("generate", self.generate)
        workflow.add_node("transform_query", self.transform_query)
        
        # Build graph with conditional edges
        workflow.add_edge(START, "retrieve")
        
        #workflow.add_edge("general_knowledge", END)
        workflow.add_edge("retrieve", "grade_documents")
        
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": END,
            },
        )
        
        workflow.add_edge("transform_query", "retrieve")
        #workflow.add_edge("generate", END)
        
        memory = InMemorySaver()
        return workflow.compile(checkpointer=memory)
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(level=logging.INFO)
    
    def retrieve(self, state: GraphState) -> GraphState:
        """Retrieve documents from vector database"""
        logger.info("---RETRIEVE---")
        question = state["question"][-1].content
        
        documents = vector_db.retrieve_documents(question)
        return {"documents": documents, "question": question}
    
    def generate(self, state: GraphState) -> GraphState:
        """Generate answer using RAG"""
        logger.info("---GENERATE---")
        question = state["question"]#[-1].content
        documents = state["documents"]
        
        logging.info("Generating answer for question: %s", question)
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
    
    def general_knowledge_answer(self, state: GraphState) -> GraphState:
        """Answer using LLM's general knowledge"""
        logger.info("---GENERAL KNOWLEDGE ANSWER---")
        question = state["question"] #[-1].content if isinstance(state["question"], list) else state["question"]
        
        # Use LLM to answer based on its training knowledge
        generation = llm_handler.generate_general_answer(question)
        
        #logging.debug("Generated answer from general knowledge: %s", generation)
        return {
            "documents": [], 
            "question": question, 
            "generation": generation
        }
    
    def route_question(self, state: GraphState) -> str:
        """Route question to appropriate data source"""
        logger.info("---ROUTE QUESTION---")
        question = state["question"]
        
        source = llm_handler.route_question(question)
        if source == "vectorstore":
            logger.info("---ROUTE QUESTION TO RAG---")
            return "vectorstore"
        else:
            logger.info("---ROUTE QUESTION TO GENERAL KNOWLEDGE---")
            return "general_knowledge"
    
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
    
    def run(self, question: str) -> dict:
        """Run the RAG workflow with a question"""
        config = {"configurable": {"thread_id": "thread-1"}}
        
        try:
            inputs = {"question": question}
            result = None
            
            for output in self.app.stream(inputs, config=config):
                result = output
                
            # # Process results
            if result:
                # Return the final result
                final_key = list(result.keys())[-1]
                return result[final_key]
            else:
                return {"generation": "I apologize, but I couldn't process your question properly."}
                
        except Exception as e:
            logger.error(f"Error running RAG workflow: {e}")
            return {"generation": "I encountered an error processing your request. Please try again."}


# Global workflow instance
rag_workflow = RAGWorkflow()
agent = rag_workflow.app

def process_question(question, history):
    """Process question through RAG workflow and return chat history"""
    if not question.strip():
        return history, ""
    
    # Update activity timestamp
    proactive_agent.update_activity()
    
    try:
        # Convert to messages format for new Gradio
        if not history:
            history = []
        
        # Process through RAG workflow
        print(f"Processing question: {question}")  # Console log
        result = rag_workflow.run(question)
        answer = result.get('generation', 'No answer generated')
        print(f"Generated answer: {answer[:100]}...")  # Console log
        
        # Add to history in messages format
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        
        return history, ""
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"Error occurred: {error_msg}")  # Console log
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": error_msg})
        return history, ""

def send_proactive_message(history):
    """Send proactive engagement message when user is inactive"""
    if proactive_agent.should_engage():
        message = proactive_agent.get_proactive_message()
        history.append({"role": "assistant", "content": message})
        print(f"Proactive message sent: {message}")
    return history

def clear_chat():
    """Clear chat history"""
    return [], ""

# Create Gradio interface
def create_gradio_app():
    """Create and configure Gradio interface"""
    
    with gr.Blocks(
        title="RAG AI Assistant",
        theme=gr.themes.Monochrome(),
        css="""
        .gradio-container {
            max-width: 2500px !important;
        }
        .chat-container {
            height: 600px !important;
        }
        """
    ) as app:
        
        gr.Markdown(
            """
            # 🤖 RAG AI Assistant
            
            Ask me anything! I'll search through documents and the web to provide accurate answers.
            
            **Features:**
            - Document retrieval from vector database
            - Web search integration
            - Query transformation and optimization
            - Document relevance grading
            """
        )
        
        with gr.Row():
            with gr.Column(scale=4):
                # Chat interface
                chatbot = gr.Chatbot(
                    label="Chat with RAG Assistant",
                    height=500,
                    show_copy_button=True,
                    container=True,
                    type="messages"  # Fix deprecation warning
                )
                
                with gr.Row():
                    question_input = gr.Textbox(
                        placeholder="Enter your question here... (Press Enter to submit)",
                        label="Your Question",
                        lines=2,
                        scale=4,
                        autofocus=True
                    )
                    
                with gr.Row():
                    submit_btn = gr.Button("Submit", variant="primary", scale=1)
                    clear_btn = gr.Button("Clear Chat", variant="secondary", scale=1)
            
            with gr.Column(scale=1):
                # Status and information panel
                gr.Markdown("### 📊 Status")
                status_display = gr.Textbox(
                    label="Current Status",
                    value="Ready to answer questions",
                    interactive=False,
                    lines=2
                )
                
                gr.Markdown("### ℹ️ Instructions")
                gr.Markdown(
                    """
                    1. Type your question in the text box
                    2. Click Submit or press Enter
                    3. Wait for the AI to process your request
                    4. View the response in the chat
                    
                    **Proactive Features:**
                    - Agent will reach out if you're inactive for 45 seconds
                    - Up to 3 proactive messages per session
                    - Toggle engagement on/off as needed
                    
                    **Tips:**
                    - Be specific with your questions
                    - Ask follow-up questions for clarification
                    - Use the Clear Chat button to start fresh
                    """
                )
        
        def handle_submit(question, history):
            if not question.strip():
                return history, "", "Please enter a question"
            
            # Update activity timestamp
            proactive_agent.update_activity()

            try:
                new_history, cleared_input = process_question(question, history)
                final_status = "Ready for next question"
                return new_history, cleared_input, final_status
            except Exception as e:
                error_status = f"Error occurred: {str(e)}"
                return history, question, error_status
        
        def check_and_send_proactive(history):
            """Check for inactivity and send proactive message if needed"""
            return send_proactive_message(history)
        
        # Connect events
        submit_btn.click(
            fn=handle_submit,
            inputs=[question_input, chatbot],
            outputs=[chatbot, question_input, status_display]
        )
        
        clear_btn.click(
            fn=lambda: ([], "", "Chat cleared - Ready for questions"),
            outputs=[chatbot, question_input, status_display]
        )
        
        # Proactive engagement timer - check every 10 seconds
        timer = gr.Timer(value=45, active=True)
        timer.tick(
            fn=check_and_send_proactive,
            inputs=[chatbot],
            outputs=[chatbot]
        )
        
        # Update activity on any interaction
        question_input.change(
            fn=lambda x: proactive_agent.update_activity(),
            inputs=[question_input]
        )
    
    return app

if __name__ == "__main__":
    # Create and launch the Gradio app
    app = create_gradio_app()
    
    print("🚀 Starting RAG AI Assistant...")
    print("📖 Loading models and databases...")
    
    # Launch the app
    app.launch(
        server_name="127.0.0.1",  # Use localhost instead of 0.0.0.0 for development
        server_port=7861,         # Your specified port
        share=False,              # Set to True to create public link
        debug=False,              # Disable debug to reduce console noise
        show_error=True,          # Show detailed errors
        inbrowser=False,          # Don't auto-open browser
        quiet=False               # Set to True to reduce startup messages
    )