import gradio as gr
import os
from langchain_community.document_loaders import UnstructuredPDFLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

def process_pdf(file_path):
    """Process PDF with fallback strategies"""
    try:
        # Try different loaders with fallback
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
        except:
            loader = UnstructuredPDFLoader(file_path, strategy="ocr_only")
            documents = loader.load()
        
        # Create embeddings and vector store
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return Chroma.from_documents(documents, embeddings)
    except Exception as e:
        raise gr.Error(f"Error processing PDF: {str(e)}")

def setup_conversation_chain(vector_store, api_key):
    """Initialize conversation chain with memory"""
    try:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        llm = ChatOllama(
            model="gpt-oss:20b"
        )

        return ConversationalRetrievalChain.from_llm(
            llm,
            vector_store.as_retriever(search_kwargs={"k": 3}),
            memory=memory
        )
    except Exception as e:
        raise gr.Error(f"Error initializing chat: {str(e)}")

def upload_file(file, api_key, chat_history):
    """Handle PDF upload and initialization"""
    
    if not file.name.endswith('.pdf'):
        raise gr.Error("Only PDF files are supported")
    
    vector_store = process_pdf(file.name)
    if not vector_store:
        raise gr.Error("Failed to process PDF")
    
    conversation_chain = setup_conversation_chain(vector_store, api_key)
    return conversation_chain, [("System", "PDF processed successfully! Ask me anything about the document.")]

def respond(query, chat_history, conversation_chain):
    """Handle user queries"""
    if not conversation_chain:
        raise gr.Error("Please upload a PDF first")
    
    try:
        result = conversation_chain({"question": query})
        chat_history.append((query, result["answer"]))
        return "", chat_history
    except Exception as e:
        raise gr.Error(f"Error processing query: {str(e)}")

with gr.Blocks(title="PDF Chatbot", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 📄 DocuBuddy - Ask Me Questions About Your Document")
    
    # State variables
    conversation_chain = gr.State(None)
    
    with gr.Row():
        with gr.Column(scale=1):
            api_key = gr.Textbox(
                label="OpenAI API Key",
                type="password",
                placeholder="Enter your OpenAI API key (sk-...)"
            )
            upload_btn = gr.UploadButton(
                "📁 Upload PDF",
                file_types=[".pdf"],
                file_count="single"
            )
    
    chatbot = gr.Chatbot(label="Conversation", height=500)
    query = gr.Textbox(label="Your Question", placeholder="Type your question here...")
    clear_btn = gr.ClearButton([query, chatbot])
    
    # Event handlers
    upload_btn.upload(
        upload_file,
        [upload_btn, api_key, chatbot],
        [conversation_chain, chatbot]
    )
    
    query.submit(
        respond,
        [query, chatbot, conversation_chain],
        [query, chatbot]
    )

if __name__ == "__main__":
    app.launch(share=True)