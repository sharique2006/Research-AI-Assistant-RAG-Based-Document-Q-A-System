# Research-AI-Assistant-RAG-Based-Document-Q-A-System
An AI-powered document question-answering application built using Retrieval-Augmented Generation (RAG). It allows users to ask questions about research papers in PDF format and receive context-aware answers based on the document content.

The application combines LangChain, Groq LLM, OpenAI embeddings, and FAISS to retrieve relevant passages and generate answers through a clean Streamlit interface.

✨ Features

📄 PDF Ingestion: Load research papers from a local folder.

✂️ Text Splitting: Split documents into manageable chunks with overlapping context.

🧠 Embeddings: Generate document embeddings using OpenAI Embeddings.

🔎 Semantic Retrieval: Retrieve relevant document passages using FAISS.

🤖 AI-Powered Answers: Generate context-aware responses using Groq.

📖 Source Context: View retrieved passages to understand the basis of an answer.

⚡ Response Timing: Display the time taken to generate a response.

🎨 Interactive UI: Modern dark-themed interface built with Streamlit.

🛠️ Tech Stack

Technology

Purpose

Python

Core programming language

Streamlit

Web application interface

LangChain

RAG pipeline and document processing

Groq

LLM inference

OpenAI Embeddings

Text embeddings

FAISS

Vector storage and similarity search

PyPDF

PDF document loading

python-dotenv

Environment variable management

🏗️ How It Works

Load PDFs: Read research papers from the research_papers/ directory.

Split Documents: Divide document text into chunks of 1,000 characters with 200 characters of overlap.

Create Embeddings: Convert document chunks into vector representations.

Store Vectors: Save the vectors in a FAISS vector store.

Retrieve Context: Search for relevant passages when the user asks a question.

Generate Answer: Pass the retrieved context and question to the Groq-powered language model.

Display Results: Show the answer, response time, and retrieved passages in the UI.

📁 Project Structure

research-ai-rag-assistant/
│
├── app.py
├── research_papers/
│   └── your_research_paper.pdf
├── .env
├── .gitignore
└── README.md

Note: The structure above represents the expected layout. Add your own PDF files to the research_papers/ folder.

⚙️ Installation and Setup

1. Clone the Repository

git clone https://github.com/YOUR_USERNAME/research-ai-rag-assistant.git
cd research-ai-rag-assistant

Replace YOUR_USERNAME with your GitHub username.

2. Create a Virtual Environment

Windows:

python -m venv venv
.\venv\Scripts\Activate.ps1

macOS/Linux:

python3 -m venv venv
source venv/bin/activate

3. Install Dependencies

python -m pip install streamlit langchain-groq langchain-openai langchain-community langchain-text-splitters langchain-classic faiss-cpu pypdf python-dotenv

4. Configure API Keys

Create a .env file in the project root:

GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key

Obtain a Groq API key from Groq Console.

Obtain an OpenAI API key from OpenAI Platform.

Important: Keep your API keys private. Never commit your .env file to GitHub.

5. Add Research Papers

Create a folder named research_papers in the project root and place your PDF research papers inside it.

The application loads PDFs from this folder when you create document embeddings.

6. Run the Application

python -m streamlit run app.py

Streamlit will display a local URL, usually:

http://localhost:8501

Open the URL in your browser to use the application.

💡 How to Use

Start the Streamlit application.

Click Create Document Embeddings.

Wait until the vector database is ready.

Enter a question related to your research papers.

Click Get Answer.

Read the generated answer.

Expand View document similarity search to inspect retrieved passages.

🔐 Environment Variables

Variable

Description

GROQ_API_KEY

Authenticates requests to the Groq language model

OPENAI_API_KEY

Authenticates requests to OpenAI Embeddings

The Groq key is used for LLM inference, while the OpenAI key is used to generate document embeddings.

⚠️ Notes and Limitations

The application requires valid API keys and an internet connection for the hosted AI services.

OpenAI embedding usage may incur charges according to your account and current pricing.

Answers depend on the content and quality of the uploaded PDFs.

The application uses a local FAISS vector store created during the session.

The current implementation processes up to the first 50 loaded documents when creating embeddings.

The vector store is held in Streamlit session state; it is not automatically persisted between sessions.

The application is designed for local development and learning.

🔮 Future Improvements

Support uploading PDFs directly through the UI.

Persist and reload the FAISS vector database.

Display page numbers and richer source citations.

Add chat history and multi-turn conversations.

Support multiple document collections.

Add configurable retrieval settings and evaluation metrics.

Deploy the application to a cloud platform.

🎯 Learning Outcomes

This project explores:

Retrieval-Augmented Generation (RAG)

Document ingestion and text chunking

Text embeddings and vector databases

Semantic similarity search

LLM integration with LangChain

Building interactive AI applications with Streamli
