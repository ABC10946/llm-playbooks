import os
import traceback
from langchain_community.document_loaders import UnstructuredPDFLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

import gradio as gr


def process_pdf(file_path):
    """Process PDF with fallback strategies"""
    # Function to process the PDF and create a vector store
    try:
        # Try loading with PyPDFLoader first
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            # If PyPDFLoader fails, use UnstructuredPDFLoader with OCR
        except:
            loader = UnstructuredPDFLoader(file_path, strategy="ocr_only")
            documents = loader.load()

        # Create embeddings using HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # Build a Chroma vector store from the documents and embeddings
        return Chroma.from_documents(documents, embeddings)
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return None

def main():
    # Main function to run the chatbot
    # Set OpenAI API key (replace with your actual key)
    # api_key = input("Enter your OpenAI API key (sk-...): ").strip()
    # os.environ["OPENAI_API_KEY"] = "PUT YOUR OPEN AI KEY HERE"

    # Upload PDF file using Google Colab's files.upload()
    print("\nUpload a PDF file:")

    # Get the filename of the uploaded PDF
    file_name = input()
    # Process the PDF and create the vector store
    vector_store = process_pdf(file_name)

    # Exit if PDF processing fails
    if not vector_store:
        print("Failed to process PDF. Exiting.")
        return

    # Setup conversation chain with memory and retriever
    llm = ChatOllama(
        model="gpt-oss:20b"
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        vector_store.as_retriever(search_kwargs={"k": 3}),
        memory=memory
    )

    # Start the chat loop
    print("\nChat with your PDF (type 'exit' to quit)")
    while True:
        # Get user's query
        query = input("\nYour question: ").strip()
        # Exit if the user types 'exit' or 'quit'
        if query.lower() in ['exit', 'quit']:
            break

        # Get the chatbot's answer and print it
        try:
            result = qa({"question": query})
            print(f"\nAnswer: {result['answer']}")
        # Handle exceptions during chat
        except Exception as e:
            print(f"Error: {str(e)}")

# Run the main function if the script is executed
if __name__ == "__main__":
    main()
