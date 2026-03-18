from ragpdf import RAGPDFError, index_documents


def criar_db() -> None:
    result = index_documents()
    print("Indexação concluída com sucesso.")
    print(f"Documentos carregados: {result.documents_loaded}")
    print(f"Chunks gerados: {result.chunks_created}")
    print(f"Base persistida em: {result.db_directory}")


if __name__ == "__main__":
    try:
        criar_db()
    except RAGPDFError as exc:
        print(f"Erro: {exc}")
