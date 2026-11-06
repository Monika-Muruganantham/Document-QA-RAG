# Document Q&A Assistant using RAG

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions using natural language. The application retrieves relevant document chunks and generates answers grounded in the retrieved content, with source references.

## Features

- Upload and process multiple PDF documents
- Split documents into smaller chunks for retrieval
- Generate embeddings using HuggingFace Sentence Transformers
- Store document embeddings in ChromaDB
- Perform semantic search to retrieve relevant information
- Generate context-grounded answers using Google Gemini
- Display source document names, page numbers, and relevant excerpts
- Persist vector embeddings for reuse
- Evaluate retrieval performance using a custom test set

## Tech Stack

- **Language:** Python
- **Frontend:** Streamlit
- **RAG Framework:** LangChain
- **LLM:** Google Gemini API
- **Embeddings:** HuggingFace Sentence Transformers
- **Vector Database:** ChromaDB
- **PDF Processing:** PyPDFLoader

## How It Works

1. The user uploads one or more PDF documents.
2. The application extracts text from the PDFs.
3. Text is divided into smaller chunks with overlapping content.
4. HuggingFace generates vector embeddings for each chunk.
5. Embeddings and metadata are stored in ChromaDB.
6. The user's question is converted into an embedding.
7. ChromaDB retrieves semantically relevant chunks.
8. Gemini generates an answer using the retrieved context.
9. The application displays the answer with source references.

## Project Structure

```text
Document-QA-RAG/
├── app.py
├── evaluate.py
├── evaluation_questions.json
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
└── screenshots/
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Document-QA-RAG.git
cd Document-QA-RAG
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the API key

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

Get your API key from [Google AI Studio](https://aistudio.google.com/).

**Never commit your actual API key to GitHub.**

### 5. Run the application

```bash
streamlit run app.py
```

## Evaluation

The project includes a custom retrieval evaluation script and a set of test questions.

Run:

```bash
python evaluate.py
```

The evaluation checks whether relevant information is retrieved for the defined test questions. The current test set achieved **100% keyword-based retrieval success across 10 questions**.

This result applies to the project's test dataset and does not represent universal answer accuracy.

## Future Improvements

- Support additional document formats
- Improve retrieval quality with advanced reranking
- Expand the evaluation dataset

## Author

**Monika Muruganantham**

Computer Science and Engineering Student

[GitHub](https://github.com/monika-muruganantham) | [LinkedIn](https://www.linkedin.com/in/monikamuruganantham/)
