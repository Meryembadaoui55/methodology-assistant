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

db = FAISS.load_local("faiss_index", HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L12-v2'),allow_dangerous_deserialization=True)



retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 1}
)


prompt_template = """
### [INST]
Instruction: You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on the instructions and context provided without using prior knowledge.You answer in FRENCH
        Analyse carefully the context and provide a direct answer based on the context. If the user said Bonjour you answer with Hi! comment puis-je vous aider?
Answer in french only
{context}
Vous devez répondre aux questions en français.

### QUESTION:
{question}
[/INST]
Answer in french only
 Vous devez répondre aux questions en français.

 """

repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

mistral_llm = HuggingFaceEndpoint(
    repo_id=repo_id, max_length=1024, temperature=0.05, huggingfacehub_api_token=st.secrets["HF_TOKEN"]
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
st.set_page_config(page_title="Alter-IA Chat", page_icon="🤖")

# Define function to handle user input and display chatbot response
def chatbot_response(user_input):
    response = qa.run(user_input)
    return response


# Create columns for logos
col1, col2, col3 = st.columns([2, 3, 2])

with col1:
    st.image("Design 3_22.png", width=150, use_column_width=True)  # Adjust image path and size as needed

with col3:
    st.image("Altereo logo 2023 original - eau et territoires durables.png", width=150, use_column_width=True)  # Adjust image path and size as needed
# Streamlit components
# Ajouter un peu de CSS pour centrer le texte
# Ajouter un peu de CSS pour centrer le texte et le colorer en orange foncé
st.markdown("""
    <style>
    .centered-text {
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Utiliser la classe CSS pour centrer et colorer le texte
st.markdown('<h3 class="centered-text">🤖 AlteriaChat 🤖 </h3>', unsafe_allow_html=True)
st.markdown("""
    <style>
    .centered-orange-text {
        text-align: center;
        color: darkorange;
    }
    </style>
    """, unsafe_allow_html=True)

# Centrer le texte principal
# Centrer et colorer en orange foncé le texte spécifique
st.markdown('<p class="centered-orange-text">"Votre Réponse à Chaque Défi Méthodologique "</p>', unsafe_allow_html=True)
# Input and button for user interaction
user_input = st.text_input("### You:", "")
submit_button = st.button("Ask 📨")

# Handle user input
if submit_button:
    if user_input.strip() != "":
        bot_response = chatbot_response(user_input)
        st.markdown("### Bot:")
        st.text_area("", value=bot_response, height=600)
    else:
        st.warning("⚠️ Please enter a message.")

# Motivational quote at the bottom
st.markdown("---")
st.markdown("*La collaboration est la clé du succès. Chaque question trouve sa réponse, chaque défi devient une opportunité.*")
