from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# Load model and tokenizer (using your Mistral RAG setup)
@st.cache_resource
def load_model():

  model_name = "mistralai/Mistral-7B-Instruct-v0.2"
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForCausalLM.from_pretrained(
      model_name,
      device_map="auto",          # Automatically use GPU if available
      load_in_4bit=True,          # 4-bit quantization to fit into T4 VRAM
      torch_dtype="auto"          # Let PyTorch decide precision
      )
  mistral_pipeline = pipeline(
      "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=300
      )
  return mistral_pipeline

mistral_pipeline = load_model()

# Load your retriever/vector store
@st.cache_resource
def load_retriever():
  loader = PyPDFLoader("THE-REFUGEES-GENERAL-REGULATIONS-2024.pdf")
  docs = loader.load()
  splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
  chunks = splitter.split_documents(docs)
  embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
  vectorstore = FAISS.from_documents(chunks, embeddings)
  retriever = vectorstore.as_retriever()
  return vectorstore.as_retriever()

retriever = load_retriever()

# Build Streamlit UI
st.title("Refugee Camp Policy Assistant")
st.write("Ask me anything about refugee policies and rights.")

user_input = st.text_input("Enter your question:")
if st.button("Get Answer"):
    if user_input.strip():
        docs = retriever.get_relevant_documents(user_input)
        context = "\n".join([doc.page_content for doc in docs])
        prompt = f"Use the context to answer:\nContext:\n{context}\n\nQuestion: {user_input}\nAnswer:"
        output_text = mistral_pipeline(prompt)[0]["generated_text"]
        # Keep only the part after "Answer:"
        if "Answer:" in output_text:
          output_text = output_text.split("Answer:", 1)[1].strip()

        st.subheader("Answer")
        st.write(output_text)
