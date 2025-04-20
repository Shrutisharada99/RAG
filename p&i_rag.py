from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import pandas as pd
import sys
import json
import xlsxwriter

#Split the document into Chunks & Store them in Vector Store
def ingest():
    # Get the documents
    loader = PyPDFLoader("STS clearance BOT/STS clearance BOT/CAPTAI IN SPIRO STS DOCS/CAPTAIN SPIRO - P&I CONFIRMATION OF ENTRY (TWIMC).pdf")
    pages = loader.load_and_split()
    # Split the pages by char
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(pages)
    print(f"Split {len(pages)} documents into {len(chunks)} chunks.")
    #
    embedding = FastEmbedEmbeddings()
    #Create vector store
    Chroma.from_documents(documents=chunks,  embedding=embedding, persist_directory="./sql_chroma_db")

# only run this once to generate vector store
ingest()

from huggingface_hub import login
access_token_read = "" # Read token from Huggingface hub
access_token_write = "" # Write token from Huggingface hub
login(token = access_token_read)

#writer = pd.ExcelWriter("Shipping_validation.xlsx", engine= "xlsxwriter")

#Create a RAG chain that retreives relevent chunks and prepares a response
model = ChatOllama(model="llama2")
#
prompt = PromptTemplate.from_template(
    """
    <s> [Instructions] You are a helpful assistant which returns only JSON outputs for all prompts. 
    Return the following as per the question based on the provided context only.

    Extract and return the following five values in valid JSON format. 
    Your output must be a **valid JSON object**, formatted as follows:

    {{ 
        "IMO_number": "{{IMO_number}}",
        "P_and_I_org": "{{P_and_I_organization}}",
        "Oil_pollution_liability_limit": "{{Oil_pollution_liability_limit}}",
        "Insurance_validity_date": "{{Insurance_validity_date}}",
        "Technical_managers": "{{Technical_managers_co_assured}}"
    }}

    Ensure that the output is a properly formatted JSON object with actual values for each field, without any additional text, explanations, or headers.

    Context: {context}
    Question: {input}
    Answer: [/Instructions]
    """
)
#Load vector store
embedding = FastEmbedEmbeddings()
vector_store = Chroma(persist_directory="./sql_chroma_db", embedding_function=embedding)

#Create chain
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 1,
        "score_threshold": 0.5,
    },
)

document_chain = create_stuff_documents_chain(model, prompt)
chain = create_retrieval_chain(retriever, document_chain)


# invoke chain
result = chain.invoke({"input": "Return the IMO number of the vessel, The Protection & Indemnity organization the vessel is a member of, The total limit of liability for Oil Pollution, The date until which the insurace certificate for the vessel is valid and The technical managers who have co-assured the certificate"})
#print(result)
output = result["answer"]

#print(output)

# Convert JSON to dictionary
data_dict = json.loads(output)

#print(data_dict)

# Convert dictionary to DataFrame
df = pd.DataFrame([data_dict])

#print(df)

df.to_excel("P&I_rag.xlsx")

import shutil

# Delete the existing ChromaDB directory
shutil.rmtree("./sql_chroma_db", ignore_errors=True)