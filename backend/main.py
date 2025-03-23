# backend.py
import os
from typing import List, Dict, Any, Optional
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class Rag:
    def __init__(self,
                 pdf_directory: str = "./data",
                 persist_directory: str = "./chroma_db",
                 embedded_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
                 huggingface_api_key: str = os.environ.get("HUGGINGFACE_API_KEY"),
                 llm_model_id: str = "mistralai/Mistral-7B-Instruct-v0.3",
                 ):
        self.pdf_directory = pdf_directory
        self.persist_directory = persist_directory
        self.huggingface_api_key = huggingface_api_key
        self.embedded_model = HuggingFaceInferenceAPIEmbeddings( api_key=huggingface_api_key, model_name=embedded_model_id)
        self.llm_model = HuggingFaceEndpoint( huggingfacehub_api_token=huggingface_api_key, repo_id=llm_model_id)
    
    def load_and_split_documents(self) -> List[Document]:
        loader = DirectoryLoader(self.pdf_directory, glob= "*.pdf", loader_cls = PyPDFLoader)
        documents = loader.load()
        # documents = RecursiveCharacterTextSplitter().split(documents)
        return documents
    
    def create_chroma(self):
        documents = self.load_and_split_documents()
        chroma = Chroma(embedding_function = self.embedded_model, persist_directory = self.persist_directory)
        chroma.add_documents(documents)
        return chroma.persist()
    
    def query(self, question: str) -> str:
        db = Chroma(embedding_function = self.embedded_model, persist_directory = self.persist_directory)
        results = db.similarity_search(
                question, 
                k=3
            )
        if results:
            context_text = "\n\n".join([doc.page_content for doc in results])
            return context_text
        else:
            return "cannot find the answer"

    def generate_answer(self, question: str):
        context_text = self.query(question)
        template = """
        You are a helpful assistant that can answer questions based on the provided context.
        If you cannot find the answer, just say "cannot find the answer".
        Please answer the question based on the context.
        Question: {question}
        Context: {context_text}
        Answer:
        """
        prompt = PromptTemplate.from_template(template)
        chain = LLMChain(llm=self.llm_model, prompt=prompt)  
        response = chain.run(context_text=context_text, question=question)

        return response

        
# test_RAG = Rag(
#     pdf_directory="./data",
#     persist_directory="./chroma_db"
# )

# question = "Predicting and preventing heat"
# print(test_RAG.generate_answer(question))


# check checkout
