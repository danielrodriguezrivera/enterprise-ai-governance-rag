import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

DATA_FOLDER = "./data"
DB_PATH = "./vector_db"

def ingest_data():
    if not os.path.exists(DATA_FOLDER):
        print("Data folder not found.")
        return

    pdf_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF files found.")
        return

    print(f"Found {len(pdf_files)} PDF(s).")

    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        print("Old vector DB removed.")

    all_docs = []
    
    for pdf_file in pdf_files:
        file_path = os.path.join(DATA_FOLDER, pdf_file)
        print(f"Loading {pdf_file}...")
        
        loader = PyPDFLoader(file_path)
        loaded_docs = loader.load()
        
        for doc in loaded_docs:
            doc.metadata["source_file"] = pdf_file
                
        all_docs.extend(loaded_docs)

    print(f"Loaded {len(all_docs)} pages.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )

    splits = text_splitter.split_documents(all_docs)
    print(f"Created {len(splits)} chunks.")

    embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
    
    Chroma.from_documents(
        documents=splits,
        embedding=embedding_function,
        persist_directory=DB_PATH
    )
    
    print("Knowledge Base updated successfully.")

if __name__ == "__main__":
    ingest_data()
