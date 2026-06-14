import os
from dotenv import load_dotenv
from config.settings import Chroma_Persist_Dir
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma

load_dotenv()

class RAG_Logic:

    def __init__(self):
        self.embeddings=HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )
    
    def Load_Doc(self,file_path):
        if file_path.endswith("pdf"):
            self.Loader=PyPDFLoader(file_path)
        else:
            self.Loader=WebBaseLoader(file_path) 
        self.doc=self.Loader.load()
        return self.doc
    
    def Split_Chunks(self):
        self.splitter=RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", ""]
        )
        self.chunks=self.splitter.split_documents(self.doc)
        
        # Add basic sequential IDs to metadata for structured deduplication
        for i, chunk in enumerate(self.chunks):
            chunk.metadata["id"] = f"doc_{i}"
            
        return self.chunks
    
    def Vector_store(self):
        self.db=Chroma.from_documents(
            documents=self.chunks,
            embedding=self.embeddings,
            persist_directory=Chroma_Persist_Dir
        )
        return self.db

    def builder(self, file_path):
        self.Load_Doc(file_path)
        self.Split_Chunks()
        
        if os.path.exists(Chroma_Persist_Dir):
            print("Loading existing vector store...")
            self.db=Chroma(persist_directory=Chroma_Persist_Dir,embedding_function=self.embeddings)
        else:
            print("Creating new vector store...")
            os.makedirs(Chroma_Persist_Dir, exist_ok=True)
            self.db=self.Vector_store()

        bm25_retriever = BM25Retriever.from_documents(self.chunks)
        bm25_retriever.k=3
        
        return self.db,bm25_retriever
    



