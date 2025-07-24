"""Loading LLMs and Embeddings."""
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")


chat_model = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

store = LocalFileStore("./cache/")

underlying_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
)
# Avoiding unnecessary costs by caching the embeddings.
EMBEDDINGS = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings, store, namespace=underlying_embeddings.model
)