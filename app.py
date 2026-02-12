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

    st.subheader("Data Sources")
    data_folder = "./data"
    
    if os.path.exists(data_folder):
        files = [f for f in os.listdir(data_folder) if f.endswith('.pdf')]
        if files:
            st.info(f"{len(files)} Document(s) Loaded")
            for f in files:
                st.text(f"- {f}")
        else:
            st.warning("No PDFs found in /data directory")
    
    st.markdown("---")
    
    # ARCHITECTURE DIAGRAM
    st.subheader("System Architecture")
    if os.path.exists("architecture_diagram.png"):
        st.image("architecture_diagram.png", caption="RAG Pipeline Architecture", width="stretch")
    else:
        st.info("Architecture diagram not found.")

    # TECH STACK LINKS
    st.markdown("### Technologies Used")
    st.markdown("""
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [OpenAI API Reference](https://platform.openai.com/docs/introduction)
- [ChromaDB Docs](https://docs.trychroma.com/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Python 3.10+](https://www.python.org/)
""")

# -----------------------------
# 3. MAIN HEADER
# -----------------------------
st.title("Enterprise Document Assistant")
st.markdown("### Multi-Document RAG System")
st.markdown("""
This system ingests PDF documents from the secure data repository and allows for cross-document querying using retrieval-augmented generation.
""")
st.divider()

# -----------------------------
# 4. BACKEND LOGIC
# -----------------------------
DB_PATH = "./vector_db"

if not os.path.exists(DB_PATH):
    st.error("System Error: Knowledge Base not found. Please run ingestion script.")
    st.stop()

@st.cache_resource
def load_chain():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 12,
            "fetch_k": 50
        }
    )


    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_template("""
You are a Corporate Knowledge Assistant.

Instructions:
1. Answer the question using ONLY the provided Context.
2. The Context may contain information from multiple different files.
3. If the answer is not present, strictly state:
   "Information not found in the provided documents."

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
    rag_chain, retriever = load_chain()
except Exception as e:
    st.error(f"System Error: {e}")
    st.stop()

# -----------------------------
# 5. CHAT INTERFACE
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "System ready. Please enter your query regarding the loaded documents."}
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
