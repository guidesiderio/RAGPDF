from __future__ import annotations

import argparse
import sys

from ragpdf import AnswerResult, IndexResult, RAGPDFError, ask_question, index_documents


def echo(message: str = "") -> None:
    print(message, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLI local para indexar PDFs e consultar a base vetorial do projeto."
    )
    subparsers = parser.add_subparsers(dest="command")

    index_parser = subparsers.add_parser("index", help="Indexa PDFs da pasta base/ (incremental por padrão).")
    index_parser.add_argument(
        "--force",
        action="store_true",
        help="Apaga e recria a base vetorial completa.",
    )

    ask_parser = subparsers.add_parser(
        "ask", help="Faz uma pergunta usando a base vetorial já indexada."
    )
    ask_parser.add_argument("question", help='Pergunta, por exemplo: "Sobre o que fala o PDF?"')
    return parser


def print_index_result(result: IndexResult) -> None:
    echo("Indexação concluída com sucesso.")
    echo(f"Documentos carregados: {result.documents_loaded}")
    echo(f"Chunks gerados: {result.chunks_created}")
    echo(f"Base persistida em: {result.db_directory}")


def print_answer_result(result: AnswerResult) -> None:
    echo("Resposta:")
    echo(result.answer)
    echo()
    echo("Fontes:")
    if not result.sources:
        echo("- Nenhuma fonte relevante foi recuperada.")
        return

    for source in result.sources:
        page_info = f", página {source.page}" if isinstance(source.page, int) else ""
        echo(f"- {source.file_name}{page_info}, chunk {source.chunk_index}")


def run_index(force: bool = False) -> int:
    echo("Iniciando indexação dos PDFs...")
    result = index_documents(force=force)
    print_index_result(result)
    return 0


def run_ask(question: str) -> int:
    echo("Consultando a base vetorial...")
    result = ask_question(question)
    print_answer_result(result)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    try:
        if args.command == "index":
            return run_index(force=args.force)
        if args.command == "ask":
            return run_ask(args.question)
    except RAGPDFError as exc:
        print(f"Erro: {exc}", file=sys.stderr, flush=True)
        return 1
    except Exception as exc:  # pragma: no cover
        print(f"Erro inesperado: {exc}", file=sys.stderr, flush=True)
        return 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
