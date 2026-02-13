**RAG Implementation with Qdrant and Groq (GPT-OSS-120B)**
This repository contains a Google Colab notebook demonstrating a high-performance Retrieval-Augmented Generation (RAG) pipeline. It leverages Qdrant as a vector database, Sentence-Transformers for semantic embeddings, and the ultra-fast Groq inference engine using the OpenAI GPT-OSS 120B model.

🚀 **Tech Stack**
Vector Database: Qdrant (Cloud or Local)

Embedding Model: sentence-transformers/all-MiniLM-L6-v2 (384-dimensional vectors)

LLM Provider: Groq Cloud

LLM Model: openai/gpt-oss-120b (State-of-the-art open-weight reasoning model)

Environment: Google Colab

🛠️ **Features**
Fast Embeddings: Uses the lightweight all-MiniLM-L6-v2 model for real-time document indexing.

High-Speed Inference: Utilizes Groq's LPU (Language Processing Unit) for near-instant responses from the 120B parameter model.

Secure Credential Handling: Designed to use Google Colab userdata (Secrets) to prevent API key exposure.

Scalable Vector Search: Direct integration with Qdrant for efficient semantic retrieval.

📋 **Prerequisites**
Before running the notebook, ensure you have the following:

Qdrant Endpoint & API Key: Get them from Qdrant Cloud.

Groq API Key: Generate one at the Groq Console.

Python Libraries: The notebook handles the installation of qdrant-client, sentence-transformers, and groq.

⚙️ **Setup Instructions**
Open in Colab: Upload the .ipynb file to your Google Colab environment.

Add Secrets: Click on the Key Icon (Secrets) in the left sidebar and add:

QDRANT_ENDPOINT

QDRANT_API_KEY

GROQ_API_KEY

Run All: Execute the cells sequentially to initialize the client, index your documents, and run the RAG loop.

📖 **How it Works**
Ingestion: Documents are processed and converted into 384-dimensional vectors using all-MiniLM-L6-v2.

Storage: These vectors are upserted into a Qdrant collection.

Retrieval: When a user asks a question, the query is embedded, and Qdrant performs a similarity search to find the most relevant context.

Generation: The context and the user query are sent to the gpt-oss-120b model via Groq, which generates a grounded, accurate response.
