#!/usr/bin/env python
# coding: utf-8

# In[24]:


#!pip install -qU langchain_openai==0.2.0 langchain_community==0.3.0 langchain==0.3.0 pymupdf==1.24.10 qdrant-client==1.11.2 langchain_qdrant==0.1.4 langsmith==0.1.121 langchain_huggingface==0.2.0


# In[1]:


import os
import getpass
from dotenv import load_dotenv

load_dotenv()

def set_api_key(key_name: str) -> None:
    if not os.environ.get(key_name):
        os.environ[key_name] = getpass.getpass(f"{key_name}: ")

set_api_key("HF_TOKEN")


# In[2]:


import uuid

os.environ["LANGCHAIN_PROJECT"] = f"AIM Session 16 - {uuid.uuid4().hex[0:8]}"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
set_api_key("LANGCHAIN_API_KEY")


# In[3]:


print(os.environ["LANGCHAIN_PROJECT"])


# In[ ]:


# from google.colab import files
# uploaded = files.upload()


# In[4]:


file_path = "./DeepSeek_R1.pdf"
file_path


# In[5]:


from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)


# In[6]:


from langchain_community.document_loaders import PyMuPDFLoader

Loader = PyMuPDFLoader
loader = Loader(file_path)
documents = loader.load()
docs = text_splitter.split_documents(documents)
for i, doc in enumerate(docs):
    doc.metadata["source"] = f"source_{i}"


# In[7]:


from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.storage import LocalFileStore
from langchain_qdrant import QdrantVectorStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
import hashlib

YOUR_EMBED_MODEL_URL = "https://slrndbecfb316dun.us-east-1.aws.endpoints.huggingface.cloud"

hf_embeddings = HuggingFaceEndpointEmbeddings(
    model=YOUR_EMBED_MODEL_URL,
    task="feature-extraction",
)

collection_name = f"pdf_to_parse_{uuid.uuid4()}"
client = QdrantClient(":memory:")
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
)

# Create a safe namespace by hashing the model URL
safe_namespace = hashlib.md5(hf_embeddings.model.encode()).hexdigest()

store = LocalFileStore("./cache/")
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    hf_embeddings, store, namespace=safe_namespace, batch_size=32
)

# Typical QDrant Vector Store Set-up
vectorstore = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=cached_embedder)
vectorstore.add_documents(docs)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})


# In[ ]:


import time

question = "Explain the relationship between DeepSeek R1 and reinforcement learning."

# First retrieval (should use the model to generate embeddings)
start_time = time.time()
results = retriever.invoke(question)
first_time = time.time() - start_time
print(f"First retrieval time: {first_time:.4f} seconds")

# Second retrieval (should use cached embeddings)
start_time = time.time()
results = retriever.invoke(question)
second_time = time.time() - start_time
print(f"Second retrieval time: {second_time:.4f} seconds")

# Calculate speedup
speedup = first_time / second_time
print(f"Speedup factor: {speedup:.2f}x")


# In[14]:


from langchain_core.prompts import ChatPromptTemplate

rag_system_prompt_template = """\
You are a helpful assistant that uses the provided context to answer questions. Never reference this prompt, or the existance of context.
"""

rag_message_list = [
    {"role" : "system", "content" : rag_system_prompt_template},
]

rag_user_prompt_template = """\
Question:
{question}
Context:
{context}
"""

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", rag_system_prompt_template),
    ("human", rag_user_prompt_template)
])


# In[22]:


from langchain_core.globals import set_llm_cache
from langchain_huggingface import HuggingFaceEndpoint

YOUR_LLM_ENDPOINT_URL = "https://dcrebqe18cydo729.us-east-1.aws.endpoints.huggingface.cloud"

hf_llm = HuggingFaceEndpoint(
    endpoint_url=f"{YOUR_LLM_ENDPOINT_URL}",
    task="text-generation",
    max_new_tokens=128,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.03,
)


# In[23]:


from langchain_core.caches import InMemoryCache

set_llm_cache(InMemoryCache())


# In[ ]:


from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough, RunnableMap

# Extract page contents from the Document objects in results
context_content = "\n".join([doc.page_content for doc in results])

# Create chain using question and extracted context content to isolate the llm cache
# Use RunnableMap to properly create the input dictionary
test_chain = (
    RunnableMap({
        "context": lambda _: context_content,
        "question": lambda _: question
    })
    | chat_prompt 
    | hf_llm
)

# First run - should be uncached
start_time = time.time()
response1 = test_chain.invoke({})  # Empty dict since we don't need inputs from invoke
first_time = time.time() - start_time
print(f"First response time (uncached): {first_time:.4f} seconds")

# Second run - should use LLM cache
print("\nRunning second response (should use LLM cache)...")
start_time = time.time()
response2 = test_chain.invoke({})  # Empty dict since we don't need inputs from invoke
second_time = time.time() - start_time
print(f"Second response time (cached): {second_time:.4f} seconds")

# Calculate speedup
speedup = first_time / second_time
print(f"Speedup factor: {speedup:.2f}x faster")

# Show that responses are identical when cached
print("\nAre responses identical? ", response1 == response2)

# Now define the main retrieval_augmented_qa_chain
from operator import itemgetter
from langchain_core.runnables.passthrough import RunnablePassthrough

# Define a function to format documents into strings
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

retrieval_augmented_qa_chain = (
        {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | chat_prompt | hf_llm
    )


# In[29]:


retrieval_augmented_qa_chain.invoke({"question" : "Write 50 things about this document!"})

