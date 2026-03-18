from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv() -> bool:
        return False


load_dotenv()

BASE_DIR = Path("base")
DB_DIR = Path("db")
EMBEDDING_MODEL = "models/embedding-001"
CHAT_MODEL = "gemini-1.5-flash"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 500
DEFAULT_K = 4


class RAGPDFError(Exception):
    """Erro esperado da aplicação."""


@dataclass(frozen=True)
class IndexResult:
    documents_loaded: int
    chunks_created: int
    db_directory: Path


@dataclass(frozen=True)
class RetrievedSource:
    file_name: str
    page: int | None
    chunk_index: int
    preview: str


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    sources: list[RetrievedSource]


def get_google_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RAGPDFError(
            "GOOGLE_API_KEY não encontrada. Configure a variável no arquivo .env antes de continuar."
        )
    return api_key


def ensure_base_dir(base_dir: Path = BASE_DIR) -> None:
    if not base_dir.exists() or not base_dir.is_dir():
        raise RAGPDFError(f"Pasta de documentos não encontrada: {base_dir}")

    pdf_files = list(base_dir.rglob("*.pdf"))
    if not pdf_files:
        raise RAGPDFError(
            f"Nenhum PDF foi encontrado em {base_dir}. Adicione arquivos antes de rodar a indexação."
        )


def ensure_vector_db_exists(db_dir: Path = DB_DIR) -> None:
    if not db_dir.exists():
        raise RAGPDFError(
            f"Base vetorial não encontrada em {db_dir}. Rode 'python main.py index' primeiro."
        )

    has_files = any(db_dir.rglob("*"))
    if not has_files:
        raise RAGPDFError(
            f"Base vetorial vazia em {db_dir}. Rode 'python main.py index' primeiro."
        )


def _load_langchain_components():
    try:
        from langchain_chroma import Chroma
        from langchain_community.document_loaders import PyPDFDirectoryLoader
        from langchain_google_genai import (
            ChatGoogleGenerativeAI,
            GoogleGenerativeAIEmbeddings,
        )
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as exc:
        raise RAGPDFError(
            "Dependências do RAG não estão instaladas. Rode 'pip install -r requirements.txt'."
        ) from exc

    return Chroma, PyPDFDirectoryLoader, ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings, RecursiveCharacterTextSplitter


def load_documents(base_dir: Path = BASE_DIR):
    ensure_base_dir(base_dir)
    _, PyPDFDirectoryLoader, _, _, _ = _load_langchain_components()
    loader = PyPDFDirectoryLoader(str(base_dir), glob="**/*.pdf")
    return loader.load()


def split_documents(documents: list, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    _, _, _, _, RecursiveCharacterTextSplitter = _load_langchain_components()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
    return chunks


def build_embeddings():
    _, _, _, GoogleGenerativeAIEmbeddings, _ = _load_langchain_components()
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=get_google_api_key(),
    )


def reset_vector_db(db_dir: Path = DB_DIR) -> None:
    if db_dir.exists():
        shutil.rmtree(db_dir)


def index_documents(base_dir: Path = BASE_DIR, db_dir: Path = DB_DIR) -> IndexResult:
    documents = load_documents(base_dir)
    chunks = split_documents(documents)

    reset_vector_db(db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)

    Chroma, _, _, _, _ = _load_langchain_components()
    Chroma.from_documents(
        documents=chunks,
        embedding=build_embeddings(),
        persist_directory=str(db_dir),
    )

    return IndexResult(
        documents_loaded=len(documents),
        chunks_created=len(chunks),
        db_directory=db_dir,
    )


def load_vector_store(db_dir: Path = DB_DIR):
    ensure_vector_db_exists(db_dir)
    Chroma, _, _, _, _ = _load_langchain_components()
    return Chroma(
        persist_directory=str(db_dir),
        embedding_function=build_embeddings(),
    )


def retrieve_context(question: str, db_dir: Path = DB_DIR, k: int = DEFAULT_K):
    vector_store = load_vector_store(db_dir)
    return vector_store.similarity_search(question, k=k)


def _format_context(documents: Iterable) -> str:
    parts: list[str] = []
    for index, document in enumerate(documents, start=1):
        source = Path(document.metadata.get("source", "desconhecido")).name
        page = document.metadata.get("page")
        page_label = f", página {page + 1}" if isinstance(page, int) else ""
        parts.append(
            f"[Trecho {index} | {source}{page_label}]\n{document.page_content.strip()}"
        )
    return "\n\n".join(parts)


def _build_prompt(question: str, context: str) -> str:
    return f"""
Você é um assistente de perguntas e respostas sobre documentos PDF.
Responda sempre em português do Brasil.
Use apenas o contexto fornecido.
Se o contexto não contiver informação suficiente para responder com segurança, diga claramente:
"Não encontrei base suficiente nos documentos para responder com segurança."
Não invente fatos, páginas ou detalhes fora do contexto.

Contexto:
{context}

Pergunta:
{question}
""".strip()


def _build_sources(documents: Iterable) -> list[RetrievedSource]:
    sources: list[RetrievedSource] = []
    for document in documents:
        preview = " ".join(document.page_content.split())
        sources.append(
            RetrievedSource(
                file_name=Path(document.metadata.get("source", "desconhecido")).name,
                page=document.metadata.get("page"),
                chunk_index=document.metadata.get("chunk_index", -1),
                preview=preview[:180],
            )
        )
    return sources


def ask_question(question: str, db_dir: Path = DB_DIR, k: int = DEFAULT_K) -> AnswerResult:
    normalized_question = question.strip()
    if not normalized_question:
        raise RAGPDFError("Informe uma pergunta não vazia para usar o comando 'ask'.")

    documents = retrieve_context(normalized_question, db_dir=db_dir, k=k)
    if not documents:
        return AnswerResult(
            answer="Não encontrei base suficiente nos documentos para responder com segurança.",
            sources=[],
        )

    _, _, ChatGoogleGenerativeAI, _, _ = _load_langchain_components()
    llm = ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        google_api_key=get_google_api_key(),
        temperature=0,
    )
    response = llm.invoke(_build_prompt(normalized_question, _format_context(documents)))
    content = getattr(response, "content", response)
    answer = content.strip() if isinstance(content, str) else str(content).strip()
    if not answer:
        answer = "Não encontrei base suficiente nos documentos para responder com segurança."

    return AnswerResult(answer=answer, sources=_build_sources(documents))
