from workflow import rag_workflow
import gradio as gr

def process_question(question, history):
    """Process question through RAG workflow and return chat history"""
    if not question.strip():
        return history, ""
    
    try:
        # Add user question to history
        history.append([question, None])
        
        # Process through RAG workflow
        result = rag_workflow.run(question)
        answer = result.get('generation', 'No answer generated')
        
        # Update history with bot response
        history[-1][1] = answer
        
        return history, ""
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        history[-1][1] = error_msg
        return history, ""

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
            # Sale Chatbot
            
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
                    label="Chat with Bot",
                    height=1000,
                    show_copy_button=True,
                    bubble_full_width=True,
                    container=True
                )
                
                with gr.Row():
                    question_input = gr.Textbox(
                        placeholder="Enter your question here...",
                        label="Your Question",
                        lines=2,    
                        scale=4
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
                    
                    **Tips:**
                    - Be specific with your questions
                    - Ask follow-up questions for clarification
                    - Use the Clear Chat button to start fresh
                    """
                )
        
        # Event handlers
        def update_status(status_msg):
            return status_msg
        
        def handle_submit(question, history):
            if not question.strip():
                return history, "", "Please enter a question"
            
            # Update status
            status = "Processing your question..."
            try:
                new_history, cleared_input = process_question(question, history)
                final_status = "Ready for next question"
                return new_history, cleared_input, final_status
            except Exception as e:
                error_status = f"Error occurred: {str(e)}"
                return history, question, error_status
        
        # Connect events
        submit_btn.click(
            fn=handle_submit,
            inputs=[question_input, chatbot],
            outputs=[chatbot, question_input, status_display]
        )
        
        question_input.submit(
            fn=handle_submit,
            inputs=[question_input, chatbot],
            outputs=[chatbot, question_input, status_display]
        )
        
        clear_btn.click(
            fn=lambda: ([], "", "Chat cleared - Ready for questions"),
            outputs=[chatbot, question_input, status_display]
        )
    
    return app

if __name__ == "__main__":
    # Create and launch the Gradio app
    app = create_gradio_app()
    
    print("🚀 Starting RAG AI Assistant...")
    print("📖 Loading models and databases...")
    
    # Launch the app
    app.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7861,       # Default Gradio port
        share=False,            # Set to True to create public link
        debug=True,             # Enable debug mode
        show_error=True         # Show detailed errors
    )