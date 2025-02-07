import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# Importa le funzionalità di LangChain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Carica le variabili d'ambiente (in particolare, OPENAI_API_KEY)
load_dotenv()  # Carica il file .env
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("La variabile OPENAI_API_KEY non è stata caricata correttamente.")

def load_pdf_to_tempfile(uploaded_file):
    """
    Salva il file PDF caricato in un file temporaneo e ne restituisce il percorso.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    return tmp_path

def create_qa_chain(pdf_path):
    """
    Dato il percorso di un file PDF, esegue le seguenti operazioni:
      - Carica il PDF tramite PyPDFLoader.
      - Divide il contenuto in chunk tramite TokenTextSplitter.
      - Crea un vector store (usando FAISS) in combinazione con le embeddings di OpenAI.
      - Istanzia una catena di RetrievalQA che userà il modello ChatOpenAI.
    """
    # Carica il PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    # Dividi il testo in porzioni (modifica chunk_size e chunk_overlap in base alle tue necessità)
    text_splitter = TokenTextSplitter(model_name="gpt-4o-mini", chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(docs)
    
    # Crea il vector store con le embeddings
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(docs, embeddings)
    
    # Inizializza il modello di chat (puoi sostituire "gpt-4o-mini" con un modello a te disponibile)
    llm = ChatOpenAI(temperature=0.1, model="gpt-4o-mini")
    
    # Crea la catena di RetrievalQA
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())
    return qa_chain

# Configurazione iniziale della pagina Streamlit
st.set_page_config(page_title="Chat with PDF", layout="wide")
st.title("Chatta con il tuo documento PDF")

# Caricamento del file PDF
uploaded_file = st.file_uploader("Carica il tuo file PDF", type=["pdf"])

if uploaded_file is not None:
    # Se il chain non è stato ancora creato (ovvero, la session_state non lo contiene) lo creiamo una sola volta
    if "qa_chain" not in st.session_state:
        with st.spinner("Elaborazione del PDF in corso..."):
            pdf_path = load_pdf_to_tempfile(uploaded_file)
            st.session_state.qa_chain = create_qa_chain(pdf_path)
            st.session_state.chat_history = []  # Lista per salvare la cronologia della chat
        st.success("PDF elaborato con successo!")
    
    st.subheader("Fai una domanda sul documento")
    
    # Campo di input per la domanda dell'utente
    user_input = st.text_input("Inserisci la tua domanda qui")
    
    # Quando l'utente preme il pulsante "Invia" e il campo non è vuoto
    if st.button("Invia") and user_input:
        with st.spinner("Generazione della risposta..."):
            # Usa la RetrievalQA chain per ottenere una risposta
            answer = st.session_state.qa_chain.run(user_input)
        # Aggiungi la domanda e la risposta alla cronologia della chat
        st.session_state.chat_history.append({"question": user_input, "answer": answer})
    
    # Visualizza la cronologia della chat, se presente
    if st.session_state.chat_history:
        st.subheader("Cronologia della conversazione")
        for chat in st.session_state.chat_history:
            st.markdown(f"**Q:** {chat['question']}")
            st.markdown(f"**A:** {chat['answer']}")
else:
    st.info("Carica un file PDF per iniziare.")
