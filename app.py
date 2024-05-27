import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_huggingface import HuggingFaceEndpoint

from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.chains import LLMChain

from huggingface_hub import login
login(token=st.secrets["HF_TOKEN"])

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

# Montez Google Drive
loader = PyPDFLoader("test-1.pdf")
data = loader.load()
# split the documents into chunks
text_splitter1 = CharacterTextSplitter(chunk_size=512, chunk_overlap=0,separator="\n\n")
texts = text_splitter1.split_documents(data)
db = FAISS.from_documents(texts,
                          HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L12-v2'))



retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={'k': 1}
)


prompt_template = """
### [INST]
Instruction: You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on the instructions and context provided without using prior knowledge.You answer in FRENCH
        Analyse carefully the context and provide a direct answer based on the context. if he said bonjour oe hello or Hi you answer with Hi! How can I help you?
Answer in french only
{context}
Vous devez répondre aux questions en français.
### QUESTION:
{question}
[/INST]
Answer in french only
 Vous devez répondre aux questions en français.
 """

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

mistral_llm = HuggingFaceEndpoint(
    repo_id=repo_id, max_length=512, temperature=0.05, huggingfacehub_api_token=st.secrets["HF_TOKEN"]
)

# Create prompt from prompt template
prompt = PromptTemplate(
    input_variables=["question"],
    template=prompt_template,
)

# Create llm chain
llm_chain = LLMChain(llm=mistral_llm, prompt=prompt)


retriever.search_kwargs = {'k':1}
qa = RetrievalQA.from_chain_type(
    llm=mistral_llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
)
import streamlit as st

# Streamlit interface with improved aesthetics
st.set_page_config(page_title="Chatbot Interface", page_icon="🤖")

# Define function to handle user input and display chatbot response
def chatbot_response(user_input):
    response = qa.run(user_input)
    return response

# Streamlit components
st.markdown("# 🤖 **Your Friendly Methodo Assistant**")
st.markdown("## \"Votre Réponse à Chaque Défi Méthodologique\" 📈")

user_input = st.text_input("You:", "")
submit_button = st.button("Send 📨")

# Handle user input
if submit_button:
    if user_input.strip() != "":
        bot_response = chatbot_response(user_input)
        st.markdown("### You:")
        st.markdown(f"> {user_input}")
        st.markdown("### Bot:")
        st.markdown(f"> {bot_response}")
    else:
        st.warning("⚠️ Please enter a message.")

# Motivational quote at the bottom
st.markdown("---")
st.markdown("*La collaboration est la clé du succès. Chaque question trouve sa réponse, chaque défi devient une opportunité.*")
