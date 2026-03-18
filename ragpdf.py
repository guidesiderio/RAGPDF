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
COLLECTION_NAME = "ragpdf"
EMBEDDING_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = "gemini-1.5-flash"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 500
DEFAULT_K = 4


class RAGPDFError(Exception):
    """Erro esperado da aplicação."""


@dataclass(frozen=True)
class IndexedChunk:
    chunk_id: str
    text: str
    metadata: dict


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


def _load_runtime_dependencies():
    try:
        import chromadb
        import google.generativeai as genai
        from pypdf import PdfReader
    except ImportError as exc:
        raise RAGPDFError(
            "Dependências do RAG não estão instaladas. Rode 'pip install -r requirements.txt'."
        ) from exc

    return chromadb, genai, PdfReader


def _configure_genai():
    _, genai, _ = _load_runtime_dependencies()
    genai.configure(api_key=get_google_api_key())
    return genai


def _get_chroma_collection(db_dir: Path = DB_DIR):
    chromadb, _, _ = _load_runtime_dependencies()
    client = chromadb.PersistentClient(path=str(db_dir))
    return client.get_or_create_collection(name=COLLECTION_NAME)


def load_documents(base_dir: Path = BASE_DIR) -> list[dict]:
    ensure_base_dir(base_dir)
    _, _, PdfReader = _load_runtime_dependencies()

    documents: list[dict] = []
    for pdf_path in sorted(base_dir.rglob("*.pdf")):
        reader = PdfReader(str(pdf_path))
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            documents.append(
                {
                    "source": pdf_path.name,
                    "path": str(pdf_path),
                    "page": page_number,
                    "text": text,
                }
            )
    return documents


def _split_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    chunks: list[str] = []
    start = 0
    step = max(chunk_size - chunk_overlap, 1)
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        chunks.append(cleaned[start:end])
        if end >= len(cleaned):
            break
        start += step
    return chunks


def split_documents(
    documents: list[dict], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
) -> list[IndexedChunk]:
    chunks: list[IndexedChunk] = []
    chunk_counter = 0
    for document_index, document in enumerate(documents):
        text_chunks = _split_text(document["text"], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for local_chunk_index, chunk_text in enumerate(text_chunks):
            chunks.append(
                IndexedChunk(
                    chunk_id=f"{document_index}-{local_chunk_index}",
                    text=chunk_text,
                    metadata={
                        "source": document["source"],
                        "path": document["path"],
                        "page": document["page"],
                        "chunk_index": chunk_counter,
                    },
                )
            )
            chunk_counter += 1
    return chunks


def _embed_texts(texts: list[str]) -> list[list[float]]:
    genai = _configure_genai()
    embeddings: list[list[float]] = []
    for text in texts:
        response = genai.embed_content(model=EMBEDDING_MODEL, content=text, task_type="retrieval_document")
        embeddings.append(response["embedding"])
    return embeddings


def _embed_query(text: str) -> list[float]:
    genai = _configure_genai()
    response = genai.embed_content(model=EMBEDDING_MODEL, content=text, task_type="retrieval_query")
    return response["embedding"]


def reset_vector_db(db_dir: Path = DB_DIR) -> None:
    if db_dir.exists():
        shutil.rmtree(db_dir)


def index_documents(base_dir: Path = BASE_DIR, db_dir: Path = DB_DIR) -> IndexResult:
    documents = load_documents(base_dir)
    chunks = split_documents(documents)
    if not chunks:
        raise RAGPDFError("Os PDFs foram lidos, mas nenhum texto utilizável foi extraído para indexação.")

    reset_vector_db(db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)

    collection = _get_chroma_collection(db_dir)
    collection.add(
        ids=[chunk.chunk_id for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
        embeddings=_embed_texts([chunk.text for chunk in chunks]),
    )

    return IndexResult(
        documents_loaded=len(documents),
        chunks_created=len(chunks),
        db_directory=db_dir,
    )


def retrieve_context(question: str, db_dir: Path = DB_DIR, k: int = DEFAULT_K) -> list[dict]:
    ensure_vector_db_exists(db_dir)
    collection = _get_chroma_collection(db_dir)
    results = collection.query(
        query_embeddings=[_embed_query(question)],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved: list[dict] = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        if not document or not metadata:
            continue
        retrieved.append({"text": document, "metadata": metadata, "distance": distance})
    return retrieved


def _format_context(documents: Iterable[dict]) -> str:
    parts: list[str] = []
    for index, document in enumerate(documents, start=1):
        metadata = document["metadata"]
        source = Path(metadata.get("source", "desconhecido")).name
        page = metadata.get("page")
        page_label = f", página {page}" if isinstance(page, int) else ""
        parts.append(f"[Trecho {index} | {source}{page_label}]\n{document['text'].strip()}")
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


def _build_sources(documents: Iterable[dict]) -> list[RetrievedSource]:
    sources: list[RetrievedSource] = []
    for document in documents:
        metadata = document["metadata"]
        preview = " ".join(document["text"].split())
        sources.append(
            RetrievedSource(
                file_name=Path(metadata.get("source", "desconhecido")).name,
                page=metadata.get("page"),
                chunk_index=metadata.get("chunk_index", -1),
                preview=preview[:180],
            )
        )
    return sources


def _generate_answer(prompt: str) -> str:
    genai = _configure_genai()
    model = genai.GenerativeModel(CHAT_MODEL)
    response = model.generate_content(prompt)
    text = getattr(response, "text", "") or ""
    return text.strip()


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

    answer = _generate_answer(_build_prompt(normalized_question, _format_context(documents)))
    if not answer:
        answer = "Não encontrei base suficiente nos documentos para responder com segurança."

    return AnswerResult(answer=answer, sources=_build_sources(documents))
