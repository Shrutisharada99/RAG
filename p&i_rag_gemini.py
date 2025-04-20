from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
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
from langchain_google_genai import ChatGoogleGenerativeAI
import shutil
import re

# Google Gemini API Key
GEMINI_API_KEY = "" # Gemini API Key

# Split the document into Chunks & Store them in Vector Store
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
    
    embedding = FastEmbedEmbeddings()
    # Create vector store
    Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory="./sql_chroma_db")

# Only run this once to generate vector store
ingest()

# Initialize Gemini AI model
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GEMINI_API_KEY)

# Define prompt
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

# Load vector store
embedding = FastEmbedEmbeddings()
vector_store = Chroma(persist_directory="./sql_chroma_db", embedding_function=embedding)

# Create chain
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 1,
        "score_threshold": 0.5,
    },
)

document_chain = create_stuff_documents_chain(model, prompt)
chain = create_retrieval_chain(retriever, document_chain)

# Invoke chain
result = chain.invoke({"input": "Return the IMO number of the vessel, The Protection & Indemnity organization the vessel is a member of, The total limit of liability for Oil Pollution, The date until which the insurance certificate for the vessel is valid and The technical managers who have co-assured the certificate"})

# Extract output
output = result["answer"]

match = re.search(r'\{.*\}', output, re.DOTALL)

if match:
    extracted_json = match.group()
    print(extracted_json)  # Output the extracted JSON
else:
    print("No JSON found in the text.")

# Convert JSON to dictionary
data_dict = json.loads(extracted_json)

# Convert dictionary to DataFrame
df = pd.DataFrame([data_dict])

# Save to Excel
df.to_excel("P&I_rag.xlsx")

# Delete the existing ChromaDB directory
shutil.rmtree("./sql_chroma_db", ignore_errors=True)
