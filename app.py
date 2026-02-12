import sys
import os
import streamlit as st
import hmac

# --- 1. STREAMLIT CLOUD COMPATIBILITY FIX ---
# This allows ChromaDB to run on Streamlit Cloud by swapping the system sqlite with pysqlite3.
# It is wrapped in a try/except block so it doesn't crash your local development environment.
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

# --- IMPORTS ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# -----------------------------
# 2. PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Enterprise Knowledge Base",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# 3. SECURITY LAYER
# -----------------------------
def check_password():
    """Returns `True` if the user had the correct password."""

    def login_form():
        """Form with widgets to collect user information"""
        st.title("🔒 Enterprise Access")
        with st.form("credentials"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["username"] in st.secrets["passwords"]:
            if hmac.compare_digest(
                st.session_state["password"],
                st.secrets["passwords"][st.session_state["username"]]
            ):
                st.session_state["password_correct"] = True
                del st.session_state["password"]  # Don't store password
                del st.session_state["username"]
                return
        st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    login_form()
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("😕 User not known or password incorrect")
    return False

# Stop execution if not logged in
if not check_password():
    st.stop()

# -----------------------------
# 4. ENVIRONMENT SETUP
# -----------------------------
# Ensure OpenAI Key is set from Streamlit Secrets
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.error("OPENAI_API_KEY not found in secrets.toml")
    st.stop()

DB_PATH = "./vector_db"
DATA_FOLDER = "./data"

# -----------------------------
# 5. SIDEBAR - CONTROLS
# -----------------------------
with st.sidebar:
    st.header("Author")
    st.markdown("Daniel Edgardo Rodríguez Rivera")
    st.markdown("El Salvador")
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.link_button("LinkedIn", "https://www.linkedin.com/in/daniel-rodriguez-sv")
    with col2:
        st.link_button("GitHub Repo", "https://github.com/danielrodriguezrivera/")
    st.markdown("---")

    st.subheader("Knowledge Scope")
    
    # --- DOCUMENT SELECTOR ---
    available_files = []
    if os.path.exists(DATA_FOLDER):
        available_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.pdf')]
    
    selected_files = st.multiselect(
        "Select Documents to Analyze:",
        options=available_files,
        placeholder="Select files or leave empty for all"
    )

    if selected_files:
        st.info(f"🔍 Searching in {len(selected_files)} specific document(s).")
    else:
        st.info("🌐 Searching across ALL documents.")

    st.markdown("---")
    
    st.subheader("System Architecture")
    if os.path.exists("architecture_diagram.png"):
        st.image("architecture_diagram.png", caption="RAG Pipeline Architecture", width="stretch")

    st.markdown("### Technologies Used")
    st.markdown("""
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/)
    - [ChromaDB](https://docs.trychroma.com/)
    - [Streamlit](https://docs.streamlit.io/)
    """)

# -----------------------------
# 6. MAIN HEADER
# -----------------------------
st.title("Enterprise Document Assistant")
st.markdown("### Multi-Document RAG System")
st.markdown("""
This system allows for **Individual Document Analysis** and **Cross-Document Comparisons**.
Select specific files in the sidebar to narrow your search, or leave it empty to query the entire knowledge base.
""")
st.divider()

# -----------------------------
# 7. BACKEND LOGIC (RAG)
# -----------------------------

if not os.path.exists(DB_PATH):
    st.error("System Error: Knowledge Base not found. Please run ingestion script locally and commit the 'vector_db' folder.")
    st.stop()

# Cache the Vector DB connection ONLY (not the retriever/chain, as those are dynamic)
@st.cache_resource
def get_vector_store():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

def get_rag_chain(vector_db, selected_files):
    # --- DYNAMIC FILTERING LOGIC ---
    search_kwargs = {
        "k": 10,
        "fetch_k": 50
    }

    # Apply ChromaDB metadata filter if files are selected
    if selected_files:
        if len(selected_files) == 1:
            search_kwargs["filter"] = {"source_file": selected_files[0]}
        else:
            search_kwargs["filter"] = {"source_file": {"$in": selected_files}}

    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_template("""
You are a Corporate Knowledge Assistant.

Instructions:
1. Answer the question using ONLY the provided Context.
2. The Context may contain information from specific selected files.
3. If comparing documents, explicitly mention the differences or similarities found.
4. If the answer is not present, strictly state: "Information not found in the provided documents."

Context:
{context}

Question:
{question}
""")

    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever

try:
    vector_db = get_vector_store()
    # Re-generate chain based on current sidebar selection
    rag_chain, retriever = get_rag_chain(vector_db, selected_files)
except Exception as e:
    st.error(f"System Error: {e}")
    st.stop()

# -----------------------------
# 8. CHAT INTERFACE
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "System ready. Select documents in the sidebar or ask a question about the whole database."}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Enter your query..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Processing query..."):
            try:
                answer = rag_chain.invoke(prompt)

                # --- FIX: Only show sources if the answer was actually found ---
                if "Information not found" in answer:
                    footer = ""
                else:
                    # Fetch source documents for citation
                    docs = retriever.invoke(prompt)

                    sources_list = []
                    for doc in docs:
                        source_name = doc.metadata.get("source_file", "Unknown File")
                        raw_page = doc.metadata.get("page", 0)
                        human_page = int(raw_page) + 1
                        sources_list.append(f"- {source_name} (Page {human_page})")

                    unique_sources = sorted(list(set(sources_list)))

                    if unique_sources:
                        footer = "\n\n---\n**Source References:**\n" + "\n".join(unique_sources)
                    else:
                        footer = ""

                full_response = answer + footer

                st.markdown(full_response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )

            except Exception as e:
                st.error(f"Error: {e}")