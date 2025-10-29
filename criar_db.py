import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Carrega variáveis do .env (GOOGLE_API_KEY)
load_dotenv()

PASTA_BASE = "base"      # pasta com seus PDFs
DB_DIR = "db"            # pasta onde o Chroma vai persistir o índice


def criar_db():
    documentos = carregar_documentos()
    print(f"Documentos carregados: {len(documentos)}")

    chunks = dividir_chunks(documentos)
    vetorizar_chunks(chunks)


def carregar_documentos():
    """
    Lê todos os PDFs em PASTA_BASE (recursivo) e retorna uma lista de Document.
    """
    loader = PyPDFDirectoryLoader(PASTA_BASE, glob="**/*.pdf")
    documentos = loader.load()
    return documentos


def dividir_chunks(documentos):
    """
    Divide os documentos em chunks com sobreposição para manter contexto.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documentos)
    print(f"Documentos divididos em {len(chunks)} chunks.")
    return chunks


def vetorizar_chunks(chunks):
    """
    Gera embeddings com Google Generative AI e cria/atualiza a base Chroma persistida.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY não encontrada. Verifique seu .env")

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",   # modelo de embeddings do Gemini
        google_api_key=api_key,
    )

    os.makedirs(DB_DIR, exist_ok=True)

    _ = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
    )

    print(f"Vector DB criado/atualizado com sucesso em: {DB_DIR}")


if __name__ == "__main__":
    criar_db()
