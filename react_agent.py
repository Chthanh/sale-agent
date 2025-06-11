"""
Refactored Sales Assistant with RAG Workflow
Enhanced version with better structure, error handling, and maintainability
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import logging
import time
import threading
import uuid
from datetime import datetime, timedelta
from contextlib import contextmanager

from agentic_rag import rag_workflow
import gradio as gr
from models import GraphState
from llm import llm_handler
from langchain.schema import Document
from langchain.agents import Tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage
from langgraph_supervisor import create_supervisor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


@dataclass
class CustomerActivity:
    """Data class to track customer activity"""
    thread_id: str
    last_message_time: datetime
    last_message_content: str
    message_count: int = 0
    
    def get_silence_duration(self) -> timedelta:
        """Get how long the customer has been silent"""
        return datetime.now() - self.last_message_time
    
    def get_silence_seconds(self) -> float:
        """Get silence duration in seconds"""
        return self.get_silence_duration().total_seconds()
    
    def is_silent_for(self, seconds: int) -> bool:
        """Check if customer has been silent for at least the specified seconds"""
        return self.get_silence_seconds() >= seconds


@dataclass
class ProactiveMessage:
    """Data class for proactive messages"""
    thread_id: str
    message: str
    wait_time: int
    scheduled_time: datetime
    sent: bool = False
    cancelled: bool = False
    cancel_event: threading.Event = None
    
    def __post_init__(self):
        if self.cancel_event is None:
            self.cancel_event = threading.Event()


class SalesAssistantConfig:
    """Configuration class for the sales assistant"""
    
    # LLM Configuration
    MODEL_NAME = "gpt-4o-mini"  # Fixed: Changed from "gpt-4.1-mini" to valid model
    TEMPERATURE = 0.0
    MAX_TOKENS = 1000
    STREAMING = False
    
    # Proactive messaging configuration
    DEFAULT_WAIT_TIME = 45
    PRICE_SENSITIVE_WAIT_TIME = 60
    COMPARISON_WAIT_TIME = 120
    UNDECIDED_WAIT_TIME = 90
    ACTIVE_SHOPPING_WAIT_TIME = 45
    MEDIUM_WAIT_TIME = 75
    
    # Thread configuration
    DEFAULT_THREAD_ID = "thread-1"


class ConversationAnalyzer:
    """Analyzes conversation history for context and patterns"""
    
    PRICE_KEYWORDS = ["price", "cost", "budget", "cheap", "expensive"]
    COMPARISON_KEYWORDS = ["compare", "versus", "vs", "difference"]
    UNCERTAINTY_KEYWORDS = ["think", "consider", "maybe", "not sure"]
    INTENT_KEYWORDS = ["looking for", "need", "want", "buy"]
    
    PRODUCT_KEYWORDS = {
        "laptop": ["gaming", "work", "office"],
        "phone": ["camera"],
        "headphones": ["wireless", "bluetooth"]
    }
    
    @classmethod
    def estimate_wait_time(cls, history: str) -> int:
        """Estimate wait time based on conversation context"""
        if not history or history == "No conversation history available.":
            return SalesAssistantConfig.DEFAULT_WAIT_TIME
        
        history_lower = history.lower()
        
        if any(word in history_lower for word in cls.PRICE_KEYWORDS):
            return SalesAssistantConfig.PRICE_SENSITIVE_WAIT_TIME
        elif any(word in history_lower for word in cls.COMPARISON_KEYWORDS):
            return SalesAssistantConfig.COMPARISON_WAIT_TIME
        elif any(word in history_lower for word in cls.UNCERTAINTY_KEYWORDS):
            return SalesAssistantConfig.UNDECIDED_WAIT_TIME
        elif any(word in history_lower for word in cls.INTENT_KEYWORDS):
            return SalesAssistantConfig.ACTIVE_SHOPPING_WAIT_TIME
        else:
            return SalesAssistantConfig.MEDIUM_WAIT_TIME
    
    @classmethod
    def create_proactive_message(cls, history: str) -> str:
        """Create a proactive follow-up message based on conversation history"""
        if not history or history == "No conversation history available.":
            return "Hi! I'm here to help you find the perfect product. What are you looking for today?"
        
        history_lower = history.lower()
        
        # Generate contextual proactive messages based on products mentioned
        for product, contexts in cls.PRODUCT_KEYWORDS.items():
            if product in history_lower:
                return cls._get_product_specific_message(product, history_lower, contexts)
        
        # Generate messages based on customer behavior
        if any(word in history_lower for word in cls.PRICE_KEYWORDS):
            return "I understand budget is important to you. Would you like me to show you our best value products that offer great features at competitive prices?"
        
        return "Is there anything else I can help you with? I'm here to assist you in finding the perfect product for your needs!"
    
    @classmethod
    def _get_product_specific_message(cls, product: str, history_lower: str, contexts: List[str]) -> str:
        """Get product-specific proactive message"""
        messages = {
            "laptop": {
                "gaming": "I noticed you were interested in gaming laptops. Would you like me to show you our latest high-performance gaming models with the best graphics cards?",
                "work": "Since you're looking for a work laptop, would you like to see our business-grade models with excellent battery life and productivity features?",
                "default": "I see you were browsing laptops. Do you have any specific requirements like screen size, performance needs, or budget in mind?"
            },
            "phone": {
                "camera": "You mentioned phone cameras earlier. Would you like to see our phones with the best camera systems for photography?",
                "default": "I noticed you were interested in phones. Are you looking for any specific features like battery life, storage, or brand preferences?"
            },
            "headphones": {
                "wireless": "You were looking at wireless headphones. Would you like to compare our top wireless models with noise cancellation features?",
                "default": "I see you were interested in headphones. Are you looking for something specific like noise cancellation, bass quality, or comfort for long use?"
            }
        }
        
        product_messages = messages.get(product, {})
        
        for context in contexts:
            if context in history_lower:
                return product_messages.get(context, product_messages.get("default", ""))
        
        return product_messages.get("default", "")


class ProactiveMessageManager:
    """Manages proactive messaging functionality with cancellation support"""
    
    def __init__(self, activity_tracker: 'CustomerActivityTracker'):  # Fixed: Added forward reference
        self.scheduled_messages: Dict[str, ProactiveMessage] = {}
        self.activity_tracker = activity_tracker
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()
    
    def schedule_message(self, thread_id: str, history: str) -> None:
        """Schedule a proactive message for a thread"""
        try:
            # Cancel any existing scheduled message for this thread
            self.cancel_scheduled_message(thread_id)
            
            wait_time = ConversationAnalyzer.estimate_wait_time(history)
            message = ConversationAnalyzer.create_proactive_message(history)
            scheduled_time = datetime.now() + timedelta(seconds=wait_time)
            
            proactive_msg = ProactiveMessage(
                thread_id=thread_id,
                message=message,
                wait_time=wait_time,
                scheduled_time=scheduled_time
            )
            
            with self._lock:
                self.scheduled_messages[thread_id] = proactive_msg
            
            # Start background thread
            threading.Thread(
                target=self._send_after_delay,
                args=(proactive_msg,),
                daemon=True
            ).start()
            
            # Log current silence duration
            silence_seconds = self.activity_tracker.get_silence_seconds(thread_id)
            silence_info = f" (customer silent for {silence_seconds:.1f}s)" if silence_seconds else ""
            
            self.logger.info(f"Scheduled proactive message for thread {thread_id} in {wait_time} seconds{silence_info}")
            
        except Exception as e:
            self.logger.error(f"Error scheduling proactive message: {e}")
    
    def cancel_scheduled_message(self, thread_id: str) -> bool:
        """Cancel a scheduled proactive message for a thread"""
        try:
            with self._lock:
                if thread_id in self.scheduled_messages:
                    proactive_msg = self.scheduled_messages[thread_id]
                    if not (proactive_msg.sent or proactive_msg.cancelled):
                        proactive_msg.cancelled = True
                        proactive_msg.cancel_event.set()
                        
                        # Log silence duration when cancelling
                        silence_seconds = self.activity_tracker.get_silence_seconds(thread_id)
                        silence_info = f" (was silent for {silence_seconds:.1f}s)" if silence_seconds else ""
                        
                        self.logger.info(f"Cancelled scheduled proactive message for thread {thread_id}{silence_info}")
                        return True
            return False
        except Exception as e:
            self.logger.error(f"Error cancelling proactive message: {e}")
            return False
    
    def should_send_proactive_message(self, thread_id: str, min_silence_seconds: int = 30) -> bool:
        """Check if we should send a proactive message based on silence duration"""
        return self.activity_tracker.is_customer_silent_for(thread_id, min_silence_seconds)
    
    def _send_after_delay(self, proactive_msg: ProactiveMessage) -> None:
        """Send proactive message after delay, checking for cancellation"""
        try:
            # Wait for the specified time, but check for cancellation
            if proactive_msg.cancel_event.wait(timeout=proactive_msg.wait_time):
                # Event was set, meaning message was cancelled
                self.logger.info(f"Proactive message for thread {proactive_msg.thread_id} was cancelled")
                return
            
            # Check if not cancelled after wait and if customer is still silent
            with self._lock:
                if not proactive_msg.cancelled:
                    # Double-check silence duration before sending
                    silence_seconds = self.activity_tracker.get_silence_seconds(proactive_msg.thread_id)
                    
                    if silence_seconds and silence_seconds >= proactive_msg.wait_time:
                        proactive_msg.sent = True
                        print(f"\n🤖 Proactive Assistant: {proactive_msg.message}")
                        self.logger.info(f"Proactive message ready for thread {proactive_msg.thread_id} (silent for {silence_seconds:.1f}s): {proactive_msg.message}")
                    else:
                        self.logger.info(f"Proactive message cancelled - customer became active (silent for only {silence_seconds:.1f}s)")
                        proactive_msg.cancelled = True
                else:
                    self.logger.info(f"Proactive message for thread {proactive_msg.thread_id} was cancelled during wait")
                    
        except Exception as e:
            self.logger.error(f"Error in proactive message delay: {e}")
    
    def get_pending_message(self, thread_id: str) -> Optional[str]:
        """Get pending proactive message for a thread"""
        with self._lock:
            if thread_id in self.scheduled_messages:
                msg = self.scheduled_messages[thread_id]
                if msg.sent and not msg.cancelled:
                    return msg.message
        return None
    
    def clear_message(self, thread_id: str) -> None:
        """Clear scheduled message for a thread"""
        with self._lock:
            if thread_id in self.scheduled_messages:
                del self.scheduled_messages[thread_id]
    
    def get_message_status(self, thread_id: str) -> Dict[str, Any]:
        """Get status of scheduled message for debugging"""
        with self._lock:
            if thread_id in self.scheduled_messages:
                msg = self.scheduled_messages[thread_id]
                silence_seconds = self.activity_tracker.get_silence_seconds(thread_id)
                
                return {
                    "scheduled": True,
                    "sent": msg.sent,
                    "cancelled": msg.cancelled,
                    "scheduled_time": msg.scheduled_time.isoformat(),
                    "wait_time": msg.wait_time,
                    "current_silence_seconds": silence_seconds,
                    "should_send_soon": silence_seconds >= msg.wait_time if silence_seconds else False
                }
        return {"scheduled": False}


class CustomerActivityTracker:
    """Tracks customer activity and silence duration"""
    
    def __init__(self):
        self.activities: Dict[str, CustomerActivity] = {}
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()
    
    def update_activity(self, thread_id: str, message_content: str) -> None:
        """Update customer activity when they send a message"""
        with self._lock:
            current_time = datetime.now()
            
            if thread_id in self.activities:
                activity = self.activities[thread_id]
                activity.last_message_time = current_time
                activity.last_message_content = message_content
                activity.message_count += 1
            else:
                self.activities[thread_id] = CustomerActivity(
                    thread_id=thread_id,
                    last_message_time=current_time,
                    last_message_content=message_content,
                    message_count=1
                )
            
            self.logger.info(f"Updated activity for thread {thread_id}")
    
    def get_silence_duration(self, thread_id: str) -> Optional[timedelta]:
        """Get how long the customer has been silent"""
        with self._lock:
            if thread_id in self.activities:
                return self.activities[thread_id].get_silence_duration()
        return None
    
    def get_silence_seconds(self, thread_id: str) -> Optional[float]:
        """Get silence duration in seconds"""
        with self._lock:
            if thread_id in self.activities:
                return self.activities[thread_id].get_silence_seconds()
        return None
    
    def is_customer_silent_for(self, thread_id: str, seconds: int) -> bool:
        """Check if customer has been silent for at least the specified seconds"""
        with self._lock:
            if thread_id in self.activities:
                return self.activities[thread_id].is_silent_for(seconds)
        return False
    
    def get_last_message_time(self, thread_id: str) -> Optional[datetime]:
        """Get the timestamp of the last customer message"""
        with self._lock:
            if thread_id in self.activities:
                return self.activities[thread_id].last_message_time
        return None
    
    def get_activity_summary(self, thread_id: str) -> Dict[str, Any]:
        """Get a summary of customer activity"""
        with self._lock:
            if thread_id in self.activities:
                activity = self.activities[thread_id]
                silence_duration = activity.get_silence_duration()
                
                return {
                    "thread_id": thread_id,
                    "last_message_time": activity.last_message_time.isoformat(),
                    "last_message_content": activity.last_message_content[:100] + "..." if len(activity.last_message_content) > 100 else activity.last_message_content,
                    "message_count": activity.message_count,
                    "silence_duration_seconds": activity.get_silence_seconds(),
                    "silence_duration_formatted": self._format_duration(silence_duration),
                    "is_active": silence_duration.total_seconds() < 30  # Consider active if silent < 30 seconds
                }
        return {"thread_id": thread_id, "status": "No activity recorded"}
    
    def _format_duration(self, duration: timedelta) -> str:
        """Format duration in a human-readable way"""
        total_seconds = int(duration.total_seconds())
        
        if total_seconds < 60:
            return f"{total_seconds} seconds"
        elif total_seconds < 3600:
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            return f"{minutes} minutes, {seconds} seconds"
        else:
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            return f"{hours} hours, {minutes} minutes"
    
    def cleanup_old_activities(self, max_age_hours: int = 24) -> None:
        """Clean up old activity records"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        with self._lock:
            threads_to_remove = [
                thread_id for thread_id, activity in self.activities.items()
                if activity.last_message_time < cutoff_time
            ]
            
            for thread_id in threads_to_remove:
                del self.activities[thread_id]
                self.logger.info(f"Cleaned up old activity for thread {thread_id}")


class ConversationHistoryManager:
    """Manages conversation history and retrieval"""
    
    def __init__(self, memory_saver: MemorySaver):
        self.memory_saver = memory_saver
        self.logger = logging.getLogger(__name__)
    
    def get_history(self, thread_id: str) -> str:
        """Get conversation history for a specific thread"""
        try:
            config = {"configurable": {"thread_id": thread_id}}
            checkpoint = self.memory_saver.get(config)
            
            if not (checkpoint and checkpoint.get("channel_values")):
                return "No previous conversation history."
            
            messages = checkpoint["channel_values"].get('messages', [])
            conversation_text = []
            
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    conversation_text.append(f"Customer: {msg.content}")
                elif isinstance(msg, AIMessage) and msg.content.strip():
                    conversation_text.append(f"Assistant: {msg.content}")
            
            return "\n".join(conversation_text) if conversation_text else "No previous conversation history."
            
        except Exception as e:
            self.logger.error(f"Error getting conversation history: {e}")
            return "No conversation history available."


class RAGToolInput(BaseModel):
    """Pydantic schema for the RAG tool input"""
    question: str = Field(description="The question to be answered using RAG workflow")


class RAGHandler:
    """Handles RAG operations and document retrieval"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def retrieve_documents(self, question: str) -> str:
        """Retrieve relevant documents relating to products"""
        try:
            config = {"configurable": {"thread_id": SalesAssistantConfig.DEFAULT_THREAD_ID}}
            self.logger.info(f"Processing question in retrieve node: {question}")
            
            inputs = {"question": question}
            result = None
            
            for output in rag_workflow.app.stream(inputs, config=config):
                result = output
            
            if result:
                final_key = list(result.keys())[-1]
                documents = llm_handler.format_docs(result[final_key]["documents"])
                return documents
            else:
                return "I apologize, but I couldn't process your question properly."
                
        except Exception as e:
            self.logger.error(f"Error running RAG workflow: {e}")
            return "I encountered an error processing your request. Please try again."


class SalesAssistant:
    """Main sales assistant class"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.llm = self._initialize_llm()
        self.shared_memory = MemorySaver()
        self.activity_tracker = CustomerActivityTracker()
        self.history_manager = ConversationHistoryManager(self.shared_memory)
        self.proactive_manager = ProactiveMessageManager(self.activity_tracker)
        self.rag_handler = RAGHandler()
        self.assistant = self._create_assistant()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _initialize_llm(self) -> ChatOpenAI:
        """Initialize the language model"""
        return ChatOpenAI(
            model=SalesAssistantConfig.MODEL_NAME,
            temperature=SalesAssistantConfig.TEMPERATURE,
            max_tokens=SalesAssistantConfig.MAX_TOKENS,
            streaming=SalesAssistantConfig.STREAMING
        )
    
    def _create_rag_tool(self) -> Tool:
        """Create the RAG tool"""
        return Tool(
            name="rag_retrieve",
            func=self.rag_handler.retrieve_documents,
            description="Retrieve relevant documents and information about products such as laptops, phones, headphones. Use this when customer asks about specific products or needs product recommendations."
        )
    
    def _create_proactive_tool(self) -> Tool:
        """Create the proactive messaging tool"""
        def schedule_proactive_message(thread_id: str = SalesAssistantConfig.DEFAULT_THREAD_ID):
            history = self.history_manager.get_history(thread_id)
            self.proactive_manager.schedule_message(thread_id, history)
            return f"Proactive message scheduled for thread {thread_id}"
        
        return Tool(
            name="schedule_proactive_message",
            func=schedule_proactive_message,
            description="Schedule proactive message. Use this when customer has been inactive for a while."
        )
    
    def _create_assistant(self) -> Any:
        """Create the sales assistant agent"""
        tools = [self._create_rag_tool()]
        
        prompt = """
        You are a professional sales assistant that can answer questions about products such as laptops, phones, and headphones. 
        Use the tools provided to retrieve information when necessary. Use three sentences maximum and keep the answer concise. 
        You can use the RAG tool to retrieve relevant documents related to the user's question. 
        Always generate a helpful response and end with EXACTLY ONE question to ask the user for clarification if needed or getting more information about customer's demand to sell products.
        
        Remember previous conversations with this customer to provide personalized recommendations.
        """
        
        return create_react_agent(
            model=self.llm,
            tools=tools,
            prompt=prompt,
            name="sale_assistant",
            state_schema=GraphState,
            checkpointer=self.shared_memory
        )
    
    def chat(self, query: str, thread_id: str = SalesAssistantConfig.DEFAULT_THREAD_ID) -> Dict[str, Any]:
        """Process a chat query and cancel any pending proactive messages"""
        try:
            # Update customer activity first
            self.activity_tracker.update_activity(thread_id, query)
            
            # Cancel any scheduled proactive message since customer is now active
            cancelled = self.proactive_manager.cancel_scheduled_message(thread_id)
            if cancelled:
                silence_duration = self.activity_tracker.get_silence_seconds(thread_id)
                self.logger.info(f"Cancelled proactive message for thread {thread_id} due to customer activity (was silent for {silence_duration:.1f}s)")
            
            config = {"configurable": {"thread_id": thread_id}}
            response = self.assistant.invoke(
                {"messages": [HumanMessage(content=query)]}, 
                config=config
            )
            
            # Schedule a new proactive message for future inactivity
            history = self.history_manager.get_history(thread_id)
            self.proactive_manager.schedule_message(thread_id, history)
            
            return response
        except Exception as e:
            self.logger.error(f"Error processing chat query: {e}")
            return {
                "messages": [AIMessage(content="I apologize, but I encountered an error. Please try again.")]
            }
    
    def get_customer_silence_duration(self, thread_id: str = SalesAssistantConfig.DEFAULT_THREAD_ID) -> Optional[timedelta]:
        """Get how long the customer has been silent"""
        return self.activity_tracker.get_silence_duration(thread_id)
    
    def get_customer_activity_summary(self, thread_id: str = SalesAssistantConfig.DEFAULT_THREAD_ID) -> Dict[str, Any]:
        """Get a summary of customer activity including silence duration"""
        return self.activity_tracker.get_activity_summary(thread_id)
    
    def is_customer_silent_for(self, seconds: int, thread_id: str = SalesAssistantConfig.DEFAULT_THREAD_ID) -> bool:
        """Check if customer has been silent for at least the specified seconds"""
        return self.activity_tracker.is_customer_silent_for(thread_id, seconds)

    def check_proactive_messages(self, thread_id: str = SalesAssistantConfig.DEFAULT_THREAD_ID) -> Optional[str]:
        """Check and return any pending proactive messages"""
        return self.proactive_manager.get_pending_message(thread_id)
    
    def cancel_proactive_message(self, thread_id: str = SalesAssistantConfig.DEFAULT_THREAD_ID) -> bool:
        """Manually cancel a proactive message"""
        return self.proactive_manager.cancel_scheduled_message(thread_id)
    
    def get_proactive_status(self, thread_id: str = SalesAssistantConfig.DEFAULT_THREAD_ID) -> Dict[str, Any]:
        """Get the status of proactive messaging for debugging"""
        return self.proactive_manager.get_message_status(thread_id)
    
    def run_interactive_session(self):  # Fixed: Added the missing method
        """Run an interactive chat session"""
        self.logger.info("Starting interactive sales assistant session...")
        thread_id = SalesAssistantConfig.DEFAULT_THREAD_ID
        
        while True:
            try:
                # Check for pending proactive messages
                pending_message = self.proactive_manager.get_pending_message(thread_id)
                if pending_message:
                    print(f"\n🤖 Proactive Assistant: {pending_message}")
                    self.proactive_manager.clear_message(thread_id)
                
                query = input("\nEnter your question (or 'exit' to quit, 'status' to check message status): ").strip()
                
                if query.lower() == 'exit':
                    # Cancel any pending messages before exit
                    self.proactive_manager.cancel_scheduled_message(thread_id)
                    print("Thank you for using the Sales Assistant. Goodbye!")
                    break
                
                if query.lower() == 'status':
                    status = self.proactive_manager.get_message_status(thread_id)
                    activity = self.get_customer_activity_summary(thread_id)
                    print(f"📊 Proactive Message Status: {status}")
                    print(f"👤 Customer Activity: {activity}")
                    continue
                
                if query.lower() == 'silence':
                    silence_duration = self.get_customer_silence_duration(thread_id)
                    if silence_duration:
                        print(f"⏰ Customer has been silent for: {silence_duration}")
                        print(f"📏 Silence in seconds: {silence_duration.total_seconds():.1f}")
                    else:
                        print("❌ No activity recorded yet")
                    continue
                
                if not query:
                    continue
                
                print("\nProcessing your request...")
                response = self.chat(query, thread_id)
                
                print("\nAssistant Response:")
                print("-" * 50)
                for message in response["messages"]:
                    if hasattr(message, 'pretty_print'):
                        message.pretty_print()
                    else:
                        print(f"{type(message).__name__}: {message.content}")
                        
            except KeyboardInterrupt:
                print("\n\nSession interrupted. Goodbye!")
                # Cancel any pending messages before exit
                self.proactive_manager.cancel_scheduled_message(thread_id)
                break
            except Exception as e:
                self.logger.error(f"Error in interactive session: {e}")
                print("An error occurred. Please try again.")


def main():
    """Main function to run the sales assistant"""
    try:
        assistant = SalesAssistant()
        assistant.run_interactive_session()
    except Exception as e:
        logging.error(f"Failed to start sales assistant: {e}")
        print("Failed to start the sales assistant. Please check the logs for details.")


if __name__ == "__main__":
    main()