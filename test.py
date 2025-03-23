from langchain_huggingface import HuggingFaceEndpoint
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
import os
from langchain_community.vectorstores import Chroma

embedded_model = HuggingFaceInferenceAPIEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", api_key=os.getenv("HUGGINGFACE_API_KEY"))
persist_directory = "./chroma_db"
question = "Predicting and preventing heat "
db = Chroma(embedding_function = embedded_model, persist_directory = persist_directory)
results = db.similarity_search(
        question, 
        k=3
    )
print(results)


