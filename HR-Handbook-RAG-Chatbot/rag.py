#Stage1: Loadinding the Handbook pdf

from langchain_community.document_loaders import PyPDFLoader
# Place the required PDF in this folder before running the application.
# The original company handbook is not included in this repository.
pdf_loader = PyPDFLoader("Saxon_Handbook 2.pdf")
documents = pdf_loader.load()
print(len(documents))

#Stage2: Chunking

from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 800,
    chunk_overlap = 100
)

chunks = text_splitter.split_documents(documents)

print(len(chunks))

#Stage 3: Embedding
#Embedding model client → Connects our application to the Azure OpenAI embedding model so our chunks can be converted into vectors.

import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

client = AzureOpenAI(
    azure_endpoint=os.getenv("CORE_AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("CORE_AZURE_OPENAI_API_KEY"),
    api_version="2024-12-01-preview"
)

test_vector = client.embeddings.create(
    model=os.getenv("CORE_AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
    input="test"
)




# # Stage 4: ChromaDB / Vector Store

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings


class AzureEmbeddingFunction(Embeddings):

    def embed_documents(self, texts):
        response = client.embeddings.create(
            model=os.getenv("CORE_AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
            input=texts
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text):
        response = client.embeddings.create(
            model=os.getenv("CORE_AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
            input=text
        )
        return response.data[0].embedding


embedding_model = AzureEmbeddingFunction()

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)

print("ChromaDB created successfully!") 



# Stage 6: GPT-5.1 Generation

from openai import AzureOpenAI

gpt_client = AzureOpenAI(
    azure_endpoint=os.getenv("CORE_AZURE_OPENAI_GPT_ENDPOINT"),
    api_key=os.getenv("CORE_AZURE_OPENAI_GPT_API_KEY"),
    api_version="2025-04-01-preview"
)

while True:

    query = input("\nAsk a question (type 'exit' to quit): ")

    if query.lower() == "exit":
        break

    results = vectorstore.similarity_search(
        query,
        k=3
    )

    context = "\n\n".join(
        result.page_content for result in results
    )

    response = gpt_client.responses.create(
        model=os.getenv("CORE_AZURE_OPENAI_GPT_DEPLOYMENT_NAME"),
        instructions=(
            "You are a helpful assistant. "
            "Answer the user's question using only the provided handbook context. "
            "If the answer is not present in the context, say you don't know."
        ),
        input=f"""
Handbook context:

{context}

Question:
{query}
""",
        max_output_tokens=1000
    )

    print("\nAnswer:")
    print(response.output_text)
