import os
from typing import TypedDict, List, Dict, Any, Optional, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import google.generativeai as genai
from dataclasses import dataclass
import json
import re
import os 
from pprint import pprint
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Sample product catalog
PRODUCT_CATALOG = {
    "smartphones": [
        {
            "id": "phone1", "name": "iPhone 15 Pro", "price": 999, "brand": "Apple", 
            "storage": "128GB", "camera": "48MP Pro", "battery": "3274mAh", 
            "features": ["5G", "Face ID", "Wireless Charging", "Water Resistant"]
        },
        {
            "id": "phone2", "name": "Samsung Galaxy S24 Ultra", "price": 1199, "brand": "Samsung",
            "storage": "256GB", "camera": "200MP", "battery": "5000mAh",
            "features": ["5G", "S Pen", "Wireless Charging", "AI Photography"]
        },
        {
            "id": "phone3", "name": "Google Pixel 8 Pro", "price": 899, "brand": "Google",
            "storage": "128GB", "camera": "50MP", "battery": "5050mAh",
            "features": ["5G", "AI Features", "Night Photography", "Fast Charging"]
        },
        {
            "id": "phone4", "name": "OnePlus 12", "price": 699, "brand": "OnePlus",
            "storage": "256GB", "camera": "50MP", "battery": "5400mAh",
            "features": ["5G", "Fast Charging", "Gaming Mode", "OxygenOS"]
        }
    ],
    "laptops": [
        {
            "id": "laptop1", "name": "MacBook Air M3 13\"", "price": 1299, "brand": "Apple",
            "processor": "M3 Chip", "ram": "16GB", "storage": "512GB SSD", "screen": "13.6\"",
            "features": ["All-day battery", "Lightweight", "Retina Display", "Touch ID"]
        },
        {
            "id": "laptop2", "name": "Dell XPS 13 Plus", "price": 1399, "brand": "Dell",
            "processor": "Intel i7", "ram": "16GB", "storage": "1TB SSD", "screen": "13.4\"",
            "features": ["4K Display", "Premium Design", "Thunderbolt 4", "Windows 11"]
        },
        {
            "id": "laptop3", "name": "ThinkPad X1 Carbon Gen 11", "price": 1599, "brand": "Lenovo",
            "processor": "Intel i7", "ram": "32GB", "storage": "1TB SSD", "screen": "14\"",
            "features": ["Business Grade", "Durable", "Long Battery", "4G LTE Option"]
        },
        {
            "id": "laptop4", "name": "ASUS ROG Zephyrus G14", "price": 1199, "brand": "ASUS",
            "processor": "AMD Ryzen 9", "ram": "16GB", "storage": "1TB SSD", "screen": "14\"",
            "features": ["Gaming Laptop", "RTX 4060", "High Refresh Rate", "RGB Keyboard"]
        }
    ],
    "headphones": [
        {
            "id": "headphone1", "name": "AirPods Pro 2", "price": 249, "brand": "Apple",
            "type": "wireless earbuds", "noise_canceling": True, "battery": "30 hours",
            "features": ["Spatial Audio", "Transparency Mode", "MagSafe Case", "Water Resistant"]
        },
        {
            "id": "headphone2", "name": "Sony WH-1000XM5", "price": 399, "brand": "Sony",
            "type": "over-ear wireless", "noise_canceling": True, "battery": "30 hours",
            "features": ["Premium ANC", "LDAC Audio", "Touch Controls", "Comfortable"]
        },
        {
            "id": "headphone3", "name": "Bose QuietComfort 45", "price": 329, "brand": "Bose",
            "type": "over-ear wireless", "noise_canceling": True, "battery": "24 hours",
            "features": ["Legendary ANC", "Comfortable Fit", "Clear Calls", "App Control"]
        },
        {
            "id": "headphone4", "name": "Sennheiser Momentum 4", "price": 379, "brand": "Sennheiser",
            "type": "over-ear wireless", "noise_canceling": True, "battery": "60 hours",
            "features": ["Audiophile Sound", "Adaptive ANC", "Smart Controls", "Premium Materials"]
        }
    ]
}

class AgentState(TypedDict):
    messages: List[Dict[str, str]]
    conversation_context: Dict[str, Any]
    customer_preferences: Dict[str, Any]
    current_products: List[Dict[str, Any]]
    conversation_stage: str
    next_question_intent: str

class GeminiSalesAgent:
    def __init__(self, api_key: Optional[str] = None):
        if api_key:
            genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.memory = MemorySaver()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("analyze_input", self.analyze_customer_input)
        workflow.add_node("generate_response", self.generate_response_with_question)
        workflow.add_node("update_context", self.update_conversation_context)
        
        # Set entry point
        workflow.set_entry_point("analyze_input")
        
        # Add edges
        workflow.add_edge("analyze_input", "generate_response")
        workflow.add_edge("generate_response", "update_context")
        workflow.add_edge("update_context", END)
        
        return workflow.compile(checkpointer=self.memory)
    
    def analyze_customer_input(self, state: AgentState) -> AgentState:
        """Analyze customer input to understand intent and extract information"""
        if not state["messages"]:
            # First interaction
            state["conversation_stage"] = "greeting"
            state["next_question_intent"] = "discover_category"
            return state
        
        last_message = state["messages"][-1]["content"]
        conversation_history = self._format_conversation_history(state["messages"])
        current_context = state["conversation_context"]
        
        analysis_prompt = f"""
        Analyze this customer message in a sales conversation context:
        
        Customer Message: "{last_message}"
        
        Conversation History: {conversation_history}
        
        Current Context: {json.dumps(current_context, indent=2)}
        
        Extract and determine:
        1. What new information did the customer provide?
        2. What is their current need or concern?
        3. What stage are they in: greeting, exploring, comparing, deciding, objecting, or ready_to_buy?
        4. What should be the intent of the next question: discover_category, get_budget, understand_usage, clarify_preferences, show_products, address_concerns, or close_sale?
        
        Respond in JSON format:
        {{
            "extracted_info": {{}},
            "customer_need": "",
            "conversation_stage": "",
            "next_question_intent": "",
            "customer_sentiment": "positive/neutral/negative"
        }}
        """
        
        try:
            response = self.model.generate_content(analysis_prompt)
            # Clean the response text to ensure it's valid JSON
            response_text = response.text.strip()
            print("response_text:", response_text)
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]
            
            analysis = json.loads(response_text)
            
            # Update state with analysis
            state["conversation_stage"] = analysis.get("conversation_stage", "exploring")
            state["next_question_intent"] = analysis.get("next_question_intent", "discover_category")
            
            # Merge extracted info into customer preferences
            extracted_info = analysis.get("extracted_info", {})
            state["customer_preferences"].update(extracted_info)
            
            # Update conversation context
            state["conversation_context"].update({
                "last_customer_need": analysis.get("customer_need", ""),
                "customer_sentiment": analysis.get("customer_sentiment", "neutral")
            })
            
            print("state after analysis:", state)
            
        except Exception as e:
            print(f"Analysis error: {e}")
            # Fallback to simple analysis
            state["conversation_stage"] = "exploring"
            state["next_question_intent"] = "discover_category"
        
        return state
    
    def generate_response_with_question(self, state: AgentState) -> AgentState:
        """Generate response with exactly one strategic question"""
        conversation_history = self._format_conversation_history(state["messages"])
        customer_prefs = state["customer_preferences"]
        context = state["conversation_context"]
        stage = state["conversation_stage"]
        question_intent = state["next_question_intent"]
        
        # Get relevant products if customer has indicated category
        relevant_products = self._get_relevant_products(customer_prefs)
        print("relevant_products:", relevant_products)
        state["current_products"] = relevant_products
        
        products_info = ""
        if relevant_products:
            products_info = f"\nAvailable products that might match: {json.dumps(relevant_products[:3], indent=2)}"
        
        response_prompt = f"""
        You are a professional sales assistant. Generate a helpful response with EXACTLY ONE strategic question.
        
        Conversation History: {conversation_history}
        
        Customer Preferences So Far: {json.dumps(customer_prefs, indent=2)}
        
        Conversation Context: {json.dumps(context, indent=2)}
        
        Current Stage: {stage}
        Next Question Intent: {question_intent}
        {products_info}
        
        Guidelines:
        1. Always end with exactly ONE question
        2. Make the question strategic and purposeful based on the intent: {question_intent}
        3. Keep response concise and conversational
        4. If showing products, briefly mention 1-2 key options then ask a specific question
        5. The question should move the conversation forward toward a sale
        
        Question Intent Meanings:
        - discover_category: Find out what type of product they want
        - get_budget: Understand their price range
        - understand_usage: Learn how they'll use the product
        - clarify_preferences: Get specific feature preferences
        - show_products: Present options and ask for preference
        - address_concerns: Handle objections with a helpful question
        - close_sale: Move toward purchase decision
        
        Response format: [Helpful response] [One strategic question]?
        """
        
        try:
            response = self.model.generate_content(response_prompt)
            assistant_response = response.text.strip()
            
            # Ensure response ends with exactly one question
            assistant_response = self._ensure_single_question(assistant_response)
            
            state["messages"].append({
                "role": "assistant",
                "content": assistant_response
            })
            
        except Exception as e:
            print(f"Response generation error: {e}")
            # Fallback response
            fallback_response = "I'd love to help you find the perfect product. What type of item are you looking for today?"
            state["messages"].append({
                "role": "assistant",
                "content": fallback_response
            })
        
        return state
    
    def update_conversation_context(self, state: AgentState) -> AgentState:
        """Update conversation context for next interaction"""
        # Track conversation flow
        state["conversation_context"]["total_exchanges"] = len([m for m in state["messages"] if m["role"] == "user"])
        state["conversation_context"]["last_stage"] = state["conversation_stage"]
        state["conversation_context"]["products_shown"] = len(state["current_products"]) > 0
        
        return state
    
    def _format_conversation_history(self, messages: List[Dict[str, str]]) -> str:
        """Format conversation history for prompts"""
        if not messages:
            return "No previous conversation"
        
        formatted = []
        for msg in messages[-6:]:  # Last 6 messages for context
            role = "Customer" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['content']}")
        
        return "\n".join(formatted)
    
    def _get_relevant_products(self, preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get products that match customer preferences"""
        category = preferences.get("category") or preferences.get("product_type")
        
        if not category:
            # Try to infer category from other preferences
            for cat in PRODUCT_CATALOG.keys():
                if cat in str(preferences).lower():
                    category = cat
                    break
        
        if category not in PRODUCT_CATALOG:
            return []
        
        products = PRODUCT_CATALOG[category]
        
        # Simple filtering based on budget
        budget = preferences.get("budget") or preferences.get("price_range")
        if budget:
            budget_num = self._extract_budget_number(str(budget))
            if budget_num:
                products = [p for p in products if p["price"] <= budget_num * 1.1]  # 10% flexibility
        
        # Filter by brand preference
        brand = preferences.get("brand") or preferences.get("preferred_brand")
        if brand:
            products = [p for p in products if brand.lower() in p["brand"].lower()]
        
        return products
    
    def _extract_budget_number(self, budget_str: str) -> Optional[float]:
        """Extract numeric budget from string"""
        # Look for numbers in the string
        numbers = re.findall(r'\d+', budget_str.lower())
        if numbers:
            return float(numbers[0])
        return None
    
    def _ensure_single_question(self, response: str) -> str:
        """Ensure response ends with exactly one question"""
        # Count question marks
        question_count = response.count('?')
        
        if question_count == 0:
            # Add a question if none exists
            response += " What would you like to know more about?"
        elif question_count > 1:
            # Find the last question and use only that
            sentences = response.split('.')
            last_question = ""
            for sentence in reversed(sentences):
                if '?' in sentence:
                    last_question = sentence.strip()
                    break
            
            if last_question:
                # Rebuild response with content before the last question + the question
                response_parts = response.split(last_question)
                if len(response_parts) > 1:
                    response = response_parts[0].rstrip() + " " + last_question
        
        return response
    
    def chat(self, user_message: str, session_id: str = "default") -> str:
        """Main chat interface"""
        config = {"configurable": {"thread_id": session_id}}
        
        # Get or initialize state
        try:
            state_snapshot = self.graph.get_state(config)
            print("state_snapshot:", state_snapshot)
            if state_snapshot and state_snapshot.values:
                current_state = state_snapshot.values
            else:
                current_state = None
        except:
            current_state = None
        
        # Initialize state if needed
        if current_state is None:
            current_state = {
                "messages": [],
                "conversation_context": {},
                "customer_preferences": {},
                "current_products": [],
                "conversation_stage": "greeting",
                "next_question_intent": "discover_category"
            }
        
        # Ensure all required keys exist
        if "messages" not in current_state:
            current_state["messages"] = []
        if "conversation_context" not in current_state:
            current_state["conversation_context"] = {}
        if "customer_preferences" not in current_state:
            current_state["customer_preferences"] = {}
        if "current_products" not in current_state:
            current_state["current_products"] = []
        if "conversation_stage" not in current_state:
            current_state["conversation_stage"] = "greeting"
        if "next_question_intent" not in current_state:
            current_state["next_question_intent"] = "discover_category"
        
        # Add user message
        current_state["messages"].append({"role": "user", "content": user_message})
        
        # Process through the graph
        result = self.graph.invoke(current_state, config)
        
        # Return the latest assistant message
        assistant_messages = [msg for msg in result["messages"] if msg["role"] == "assistant"]
        return assistant_messages[-1]["content"] if assistant_messages else "How can I help you today?"

# Demo function
def demo_sales_agent():
    """Demo the sales agent without requiring API key"""
    print("=== Gemini Sales Agent Demo ===\n")
    
    # Create agent (will use mock model if no API key)
    agent = GeminiSalesAgent(os.getenv("GOOGLE_API_KEY"))
    
    # Simulate conversation
    conversations = [
        "Hi, I'm looking for a new phone",
        "I have around $800 to spend",
        "I mostly use it for taking photos and social media",
        "I prefer Android phones",
        "What's the difference between the Samsung and Google options?"
    ]
    
    session_id = "demo_session"
    
    # for user_msg in conversations:
    #     print(f"Customer: {user_msg}")
    #     response = agent.chat(user_msg, session_id)
    #     print(f"Sales Agent: {response}\n")

    while True:
        user_msg = input("Customer: ")
        if user_msg.lower() in ["exit", "quit"]:
            print("Ending demo.")
            break
        response = agent.chat(user_msg, session_id)
        print(f"Sales Agent: {response}\n")
if __name__ == "__main__":
    # Uncomment to run demo
    demo_sales_agent()
    
    # To use with real Gemini API:
    # agent = GeminiSalesAgent(api_key="your_gemini_api_key_here")
    # response = agent.chat("I'm looking for a laptop for work")
    # print(response)