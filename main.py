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
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from data.product import PRODUCT_CATALOG

# Load environment variables from .env file
load_dotenv()

class ProductRAG:
    """Agentic RAG system for intelligent product retrieval"""
    
    def __init__(self, product_catalog: Dict[str, List[Dict]], model):
        self.product_catalog = product_catalog
        self.model = model
        self.products_flat = self._flatten_products()
        self.vectorizer = None
        self.product_vectors = None
        self._build_product_index()
    
    def _flatten_products(self) -> List[Dict]:
        """Flatten the product catalog into a single list"""
        products = []
        for category, product_list in self.product_catalog.items():
            for product in product_list:
                product_copy = product.copy()
                product_copy['category'] = category
                products.append(product_copy)

        return products
    
    def _build_product_index(self):
        """Build TF-IDF index for semantic search"""
        # Create searchable text for each product
        product_texts = []
        for product in self.products_flat:
            # Combine all relevant text fields
            text_parts = [
                product.get('name', ''),
                product.get('brand', ''),
                product.get('category', ''),
                product.get('description', ''),
                ' '.join(product.get('features', [])),
                str(product.get('price', '')),
                product.get('processor', ''),
                product.get('storage', ''),
                product.get('ype', '')
            ]
            searchable_text = ' '.join(filter(None, text_parts)).lower()
            product_texts.append(searchable_text)
        print(f"product_texts: {product_texts[:5]}")  # Debug: show first 5 texts
        
        # Build TF-IDF vectors
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=1000
        )
        self.product_vectors = self.vectorizer.fit_transform(product_texts)
    
    def _semantic_search(self, query: str, top_k: int = 5) -> List[tuple]:
        """Perform semantic search using TF-IDF similarity"""
        if not self.vectorizer or self.product_vectors is None:
            return []
        
        # Vectorize the query
        query_vector = self.vectorizer.transform([query.lower()])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.product_vectors).flatten()
        
        # Get top matches
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                results.append((self.products_flat[idx], similarities[idx]))
        
        return results
    
    def _analyze_customer_intent(self, customer_preferences: Dict[str, Any], conversation_context: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to analyze customer intent and requirements"""
        analysis_prompt = f"""
        Analyze the customer's requirements and preferences to understand what they're looking for:
        
        Customer Preferences: {json.dumps(customer_preferences, indent=2)}
        Conversation Context: {json.dumps(conversation_context, indent=2)}
        
        Based on this information, determine:
        1. What product category are they interested in? (smartphones, laptops, headphones, or unclear)
        2. What are their key requirements? (budget, features, use cases, brand preferences)
        3. What is their primary use case? (work, gaming, photography, travel, etc.)
        4. What's their experience level? (beginner, intermediate, expert)
        5. Any specific constraints or deal-breakers?
        
        Create a search query that would help find the most relevant products.
        
        Respond in JSON format:
        {{
            "category": "smartphones/laptops/headphones/unclear",
            "key_requirements": [],
            "primary_use_case": "",
            "experience_level": "beginner/intermediate/expert",
            "constraints": [],
            "search_query": "detailed search query for finding relevant products",
            "max_budget": null,
            "preferred_brands": []
        }}
        """
        
        try:
            response = self.model.generate_content(analysis_prompt)
            response_text = response.text.strip()
            
            # Clean JSON response
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]
            
            return json.loads(response_text)
        
        except Exception as e:
            print(f"Intent analysis error: {e}")
            # Fallback analysis
            return {
                "category": "unclear",
                "key_requirements": [],
                "primary_use_case": "general",
                "experience_level": "intermediate",
                "constraints": [],
                "search_query": " ".join(str(v) for v in customer_preferences.values()),
                "max_budget": None,
                "preferred_brands": []
            }
    
    def _filter_and_rank_products(self, products: List[Dict], intent_analysis: Dict[str, Any]) -> List[Dict]:
        """Filter and rank products based on intent analysis"""
        filtered_products = products.copy()
        
        # Filter by category if specified
        if intent_analysis["category"] != "unclear":
            filtered_products = [p for p in filtered_products if p.get("category") == intent_analysis["category"]]
        
        # Filter by budget
        max_budget = intent_analysis.get("max_budget")
        if max_budget:
            filtered_products = [p for p in filtered_products if p.get("price", 0) <= max_budget * 1.15]  # 15% flexibility
        
        # Filter by preferred brands
        preferred_brands = intent_analysis.get("preferred_brands", [])
        if preferred_brands:
            filtered_products = [
                p for p in filtered_products 
                if any(brand.lower() in p.get("brand", "").lower() for brand in preferred_brands)
            ]
        
        # Rank products using LLM
        if len(filtered_products) > 1:
            filtered_products = self._llm_rank_products(filtered_products, intent_analysis)
        
        return filtered_products
    
    def _llm_rank_products(self, products: List[Dict], intent_analysis: Dict[str, Any]) -> List[Dict]:
        """Use LLM to rank products based on customer requirements"""
        if len(products) <= 1:
            return products
        
        ranking_prompt = f"""
        Rank these products based on how well they match the customer's requirements:
        
        Customer Analysis: {json.dumps(intent_analysis, indent=2)}
        
        Products to rank:
        {json.dumps(products, indent=2)}
        
        Consider:
        1. How well each product matches the primary use case
        2. Value for money given the budget
        3. Feature alignment with requirements
        4. Brand preference match
        5. Overall suitability for the customer's experience level
        
        Return only the product IDs in order from best match to least match:
        ["product_id_1", "product_id_2", "product_id_3", ...]
        """
        
        try:
            response = self.model.generate_content(ranking_prompt)
            response_text = response.text.strip()
            
            # Clean response
            if response_text.startswith('```'):
                response_text = response_text[3:-3]
            
            ranked_ids = json.loads(response_text)
            
            # Reorder products based on ranking
            id_to_product = {p["id"]: p for p in products}
            ranked_products = []
            
            for product_id in ranked_ids:
                if product_id in id_to_product:
                    ranked_products.append(id_to_product[product_id])
            
            # Add any products that weren't ranked
            for product in products:
                if product["id"] not in ranked_ids:
                    ranked_products.append(product)
            
            return ranked_products
        
        except Exception as e:
            print(f"Ranking error: {e}")
            return products
    
    def get_relevant_products(self, customer_preferences: Dict[str, Any], conversation_context: Dict[str, Any], max_results: int = 3) -> List[Dict[str, Any]]:
        """Main method to get relevant products using agentic RAG"""
        
        # Step 1: Analyze customer intent
        intent_analysis = self._analyze_customer_intent(customer_preferences, conversation_context)
        print(f"Intent Analysis: {intent_analysis}")
        
        # Step 2: Perform semantic search
        search_results = self._semantic_search(intent_analysis["search_query"], top_k=8)
        candidate_products = [result[0] for result in search_results]
        
        # Step 3: Filter and rank products
        final_products = self._filter_and_rank_products(candidate_products, intent_analysis)
        
        # Step 4: Return top results
        return final_products[:max_results]

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
        
        # Initialize RAG system
        self.product_rag = ProductRAG(PRODUCT_CATALOG, self.model)
        
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
        1. What new information did the customer provide? (budget, preferences, use cases, constraints)
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
            response_text = response.text.strip()
            
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
        
        # Get relevant products using RAG system
        relevant_products = self.product_rag.get_relevant_products(
            customer_prefs,
            context,
            max_results=3
        )
        
        print(f"RAG Retrieved Products: {[p['name'] for p in relevant_products]}")
        state["current_products"] = relevant_products
        
        products_info = ""
        if relevant_products:
            # Create concise product info for the prompt
            product_summaries = []
            for p in relevant_products:
                summary = f"{p['name']} (${p['price']}) - {p.get('description', '')[:100]}..."
                product_summaries.append(summary)
            products_info = f"\nRelevant products found: {'; '.join(product_summaries)}"
        
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
    
    def _ensure_single_question(self, response: str) -> str:
        """Ensure response ends with exactly one question"""
        question_count = response.count('?')
        
        if question_count == 0:
            response += " What would you like to know more about?"
        elif question_count > 1:
            sentences = response.split('.')
            last_question = ""
            for sentence in reversed(sentences):
                if '?' in sentence:
                    last_question = sentence.strip()
                    break
            
            if last_question:
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
        for key in ["messages", "conversation_context", "customer_preferences", "current_products", "conversation_stage", "next_question_intent"]:
            if key not in current_state:
                current_state[key] = [] if key in ["messages", "current_products"] else {} if key in ["conversation_context", "customer_preferences"] else "greeting" if key == "conversation_stage" else "discover_category"
        
        # Add user message
        current_state["messages"].append({"role": "user", "content": user_message})
        
        # Process through the graph
        result = self.graph.invoke(current_state, config)
        
        # Return the latest assistant message
        assistant_messages = [msg for msg in result["messages"] if msg["role"] == "assistant"]
        return assistant_messages[-1]["content"] if assistant_messages else "How can I help you today?"

# Demo function
def demo_sales_agent():
    """Demo the enhanced sales agent with RAG"""
    print("=== Enhanced Gemini Sales Agent with Agentic RAG ===\n")
    
    # Create agent
    agent = GeminiSalesAgent(os.getenv("GOOGLE_API_KEY"))
    
    session_id = "demo_session"
    
    print("Sales Agent: Hello! I'm here to help you find the perfect product. What can I help you with today?")
    
    while True:
        user_msg = input("\nCustomer: ")
        if user_msg.lower() in ["exit", "quit"]:
            print("Ending demo.")
            break
        response = agent.chat(user_msg, session_id)
        print(f"Sales Agent: {response}")

if __name__ == "__main__":
    demo_sales_agent()