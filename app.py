import os
import torch
from transformers import (
  pipeline
)
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline

from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.chains import LLMChain
from ctransformers import AutoModelForCausalLM, AutoTokenizer
from langchain.text_splitter import CharacterTextSplitter

import transformers
from transformers import pipeline

import transformers
from huggingface_hub import login
login(token=st.secrets["HF_TOKEN"])

loader = PyPDFLoader("https://github.com/Meryembadaoui55/methodology-assistant/blob/main/test-1.pdf")

data = loader.load()

text_splitter1 = CharacterTextSplitter(chunk_size=512, chunk_overlap=0,separator="\n\n")
texts = text_splitter1.split_documents(data)
# Load chunked documents into the FAISS index
db = FAISS.from_documents(texts,
                          HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L12-v2'))


# Connect query to FAISS index using a retriever
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={'k': 1}
)
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

text_generation_pipeline = transformers.pipeline(
   repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    tokenizer=tokenizer,
    task="text-generation",

    temperature=0.02,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=512,
)

prompt_template = """
### [INST]
Instruction: You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on the instructions and context provided without using prior knowledge.You answer in FRENCH
        Analyse carefully the context and provide a direct answer based on the context.
Answer in french only
{context}
Vous devez répondre aux questions en français.
### QUESTION:
{question}
[/INST]
Answer in french only
 Vous devez répondre aux questions en français.
 """

mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# Create prompt from prompt template
prompt = PromptTemplate(
    input_variables=["question"],
    template=prompt_template,
)

# Create llm chain
llm_chain = LLMChain(llm=mistral_llm, prompt=prompt)
from langchain.chains import RetrievalQA


retriever.search_kwargs = {'k':1}
qa = RetrievalQA.from_chain_type(
    llm=mistral_llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
)

import streamlit as st

# Streamlit interface
st.title("Chatbot Interface")

# Define function to handle user input and display chatbot response
def chatbot_response(user_input):
    response = qa.get_answer(user_input)
    return response

# Streamlit components
user_input = st.text_input("You:", "")
submit_button = st.button("Send")

# Handle user input
if submit_button:
    if user_input.strip() != "":
        bot_response = chatbot_response(user_input)
        st.text_area("Bot:", value=bot_response, height=200)
    else:
        st.warning("Please enter a message.")
