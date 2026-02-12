import streamlit as st
import os
from dotenv import load_dotenv

# --- IMPORTS ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load Environment Variables
load_dotenv()

# -----------------------------
# 1. PROFESSIONAL UI CONFIG
# -----------------------------
st.set_page_config(
    page_title="Enterprise Knowledge Base",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# 2. SIDEBAR - PROFILE & DATA
# -----------------------------
DB_PATH = "./vector_db"
DATA_FOLDER = "./data"

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
    
    # --- NEW FEATURE: DOCUMENT SELECTOR ---
    available_files = []
    if os.path.exists(DATA_FOLDER):
        available_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.pdf')]
    
    # Multiselect allows: 
    # 1. Empty = Search All
    # 2. One file = Specific Query
    # 3. Two+ files = Comparison Mode
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
# 3. MAIN HEADER
# -----------------------------
st.title("Enterprise Document Assistant")
st.markdown("### Multi-Document RAG System")
st.markdown("""
This system allows for **Individual Document Analysis** and **Cross-Document Comparisons**.
Select specific files in the sidebar to narrow your search, or leave it empty to query the entire knowledge base.
""")
st.divider()

# -----------------------------
# 4. BACKEND LOGIC
# -----------------------------

if not os.path.exists(DB_PATH):
    st.error("System Error: Knowledge Base not found. Please run ingestion script.")
    st.stop()

# Cache the Vector DB connection ONLY (not the retriever/chain, as those are now dynamic)
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
        "k": 10, # Good balance for comparisons
        "fetch_k": 50
    }

    # Apply ChromaDB metadata filter if files are selected
    if selected_files:
        if len(selected_files) == 1:
            # Filter for exactly one file
            search_kwargs["filter"] = {"source_file": selected_files[0]}
        else:
            # Filter for ANY of the selected files ($in operator)
            search_kwargs["filter"] = {"source_file": {"$in": selected_files}}

    # Create retriever with dynamic filters
    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Prompt tweaked to encourage comparisons if multiple docs are present
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
# 5. CHAT INTERFACE
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

                # Fetch source documents for citation
                docs = retriever.invoke(prompt)

                sources_list = []
                for doc in docs:
                    source_name = doc.metadata.get("source_file", "Unknown File")
                    raw_page = doc.metadata.get("page", 0)
                    human_page = int(raw_page) + 1
                    sources_list.append(f"{source_name} (Page {human_page})")

                unique_sources = sorted(list(set(sources_list)))

                if unique_sources:
                    footer = "\n\n---\n**Source References:**\n" + "\n".join(
                        [f"- {s}" for s in unique_sources]
                    )
                else:
                    footer = ""

                full_response = answer + footer

                st.markdown(full_response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )

            except Exception as e:
                st.error(f"Error: {e}")