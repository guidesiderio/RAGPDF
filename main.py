from __future__ import annotations

import argparse
import sys

from ragpdf import AnswerResult, IndexResult, RAGPDFError, ask_question, index_documents


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLI local para indexar PDFs e consultar a base vetorial do projeto."
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("index", help="Recria a base vetorial a partir da pasta base/.")

    ask_parser = subparsers.add_parser(
        "ask", help="Faz uma pergunta usando a base vetorial já indexada."
    )
    ask_parser.add_argument("question", help='Pergunta, por exemplo: "Sobre o que fala o PDF?"')
    return parser


def print_index_result(result: IndexResult) -> None:
    print("Indexação concluída com sucesso.")
    print(f"Documentos carregados: {result.documents_loaded}")
    print(f"Chunks gerados: {result.chunks_created}")
    print(f"Base persistida em: {result.db_directory}")


def print_answer_result(result: AnswerResult) -> None:
    print("Resposta:")
    print(result.answer)

    print("\nFontes:")
    if not result.sources:
        print("- Nenhuma fonte relevante foi recuperada.")
        return

    for source in result.sources:
        page_info = f", página {source.page + 1}" if isinstance(source.page, int) else ""
        print(f"- {source.file_name}{page_info}, chunk {source.chunk_index}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    try:
        if args.command == "index":
            print_index_result(index_documents())
            return 0
        if args.command == "ask":
            print_answer_result(ask_question(args.question))
            return 0
    except RAGPDFError as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover
        print(f"Erro inesperado: {exc}", file=sys.stderr)
        return 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
