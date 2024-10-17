import os
import time
import PyPDF2
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Check if the API key was loaded
if not google_api_key:
    st.error("API key not found. Please add it to your .env file as GOOGLE_API_KEY.")
else:
    # Streamlit app title
    st.title("Chatbot with Gemini Model and PDF-based Question Answering")

    # File uploader for PDF input
    pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

    # Process the PDF and create a knowledge base
    if pdf_file:
        # Load PDF file content
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        pdf_text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            pdf_text += page.extract_text()
        
        # Text splitting and embedding
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        text_chunks = text_splitter.split_text(pdf_text)

        # Embeddings and VectorStore
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",  # Added required model parameter
            api_key=google_api_key
        )
        vector_store = FAISS.from_texts(text_chunks, embeddings)

        # Initialize a retriever using the vector store
        retriever = vector_store.as_retriever()

        # Initialize the language model with the API key
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None, api_key=google_api_key)

        # System prompt for the chatbot
        system_prompt = (
            "You are an assistant for answering questions based on the content of a PDF document. "
            "Use the retrieved information from the document to answer the question."
        )
        system_message = SystemMessage(content=system_prompt)

        # User input
        query = st.chat_input("Ask a question about the PDF: ")

        # Check if there is a query, then proceed with the chatbot response
        if query:
            # Retrieve relevant content from PDF
            docs = retriever.get_relevant_documents(query)
            context = " ".join([doc.page_content for doc in docs])
            query_with_context = f"Context: {context} \nQuestion: {query}"

            human_message = HumanMessage(content=query_with_context)
            messages = [system_message, human_message]

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = llm.invoke(messages)

                    # Extract only the content field from the response
                    if hasattr(response, 'content'):  # Check if response has 'content' attribute
                        chatbot_reply = response.content
                    else:
                        chatbot_reply = "Sorry, the response format was not as expected."

                    # Display only the chatbot's reply content
                    st.write(chatbot_reply)
                    break  # Exit the loop if successful

                except Exception as e:
                    st.write(f"Attempt {attempt + 1} failed: {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        st.write("Sorry, I couldn't complete your request due to a server error.")