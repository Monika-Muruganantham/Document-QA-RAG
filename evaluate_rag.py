import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# Paths relative to this Python file
BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_db"
EVALUATION_FILE = BASE_DIR / "evaluation_questions.json"


# Load embeddings
print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Connect to the SAME database and collection as app.py
print("Connecting to ChromaDB...")
print("Database path:", CHROMA_DIR)

vectorstore = Chroma(
    collection_name="document_chunks",
    embedding_function=embeddings,
    persist_directory=str(CHROMA_DIR)
)


# Diagnostics
print("Collection:", vectorstore._collection.name)
print("Stored chunks:", vectorstore._collection.count())


if vectorstore._collection.count() == 0:
    print("\nERROR: ChromaDB collection is empty.")
    print("Process your PDF in app.py first.")
    raise SystemExit


# Load test dataset
with open(EVALUATION_FILE, "r", encoding="utf-8") as file:
    test_questions = json.load(file)


passed = 0
total = len(test_questions)

print("\n========== RAG EVALUATION ==========\n")


for index, test in enumerate(test_questions, start=1):

    question = test["question"]
    expected_keywords = test["expected_keywords"]

    retrieved_docs = vectorstore.similarity_search(
        question,
        k=4
    )

    context = " ".join(
        doc.page_content.lower()
        for doc in retrieved_docs
    )

    matched_keywords = [
        keyword
        for keyword in expected_keywords
        if keyword.strip().lower() in context
    ]

    success = (
        len(expected_keywords) > 0
        and len(matched_keywords) == len(expected_keywords)
    )

    if success:
        passed += 1

    print(f"Test {index}")
    print(f"Question: {question}")
    print(f"Retrieved documents: {len(retrieved_docs)}")
    print(f"Matched keywords: {matched_keywords}")
    print(f"Result: {'PASS' if success else 'FAIL'}")
    print("-" * 50)


score = (passed / total * 100) if total else 0

print("\n========== SUMMARY ==========")
print(f"Total: {total}")
print(f"Passed: {passed}")
print(f"Failed: {total - passed}")
print(f"Keyword retrieval score: {score:.2f}%")