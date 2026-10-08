# HR Handbook RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions using information retrieved from an HR handbook.

## Overview

This project loads an HR handbook PDF, splits the content into smaller chunks, converts the chunks into embeddings, and stores them in ChromaDB.

When a user asks a question, the system retrieves the most relevant chunks and provides them as context to an Azure OpenAI model to generate a grounded answer.

## RAG Pipeline

```text
HR Handbook PDF
       ↓
Document Loading
       ↓
Text Chunking
       ↓
Azure OpenAI Embeddings
       ↓
ChromaDB
       ↓
Similarity Search
       ↓
Relevant Context
       ↓
Azure OpenAI
       ↓
Generated Answer
```

## Technologies Used

* Python
* LangChain
* ChromaDB
* Azure OpenAI
* PyPDF
* python-dotenv

## Project Structure

```text
HR-Handbook-RAG-Chatbot/
│
├── rag.py
├── requirements.txt
├── .gitignore
├── .env.example
└── README.md
```

## Setup

1. Clone the repository.

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file using `.env.example` as a template and add your Azure OpenAI configuration.

4. Place the required handbook PDF in the project folder.

5. Run the application:

```bash
python rag.py
```

## Note

The original HR handbook PDF and API credentials are intentionally excluded from this repository because they contain private/company-specific information.

You must provide your own compatible document and Azure OpenAI credentials to run the application.
