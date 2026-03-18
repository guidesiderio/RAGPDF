from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from ragpdf import RAGPDFError, ask_question, ensure_base_dir, ensure_vector_db_exists


class RagPdfValidationTests(unittest.TestCase):
    def test_ensure_base_dir_fails_when_directory_is_missing(self):
        with self.assertRaises(RAGPDFError) as context:
            ensure_base_dir(Path("base-inexistente"))

        self.assertIn("Pasta de documentos não encontrada", str(context.exception))

    def test_ensure_base_dir_fails_when_no_pdf_exists(self):
        base_dir = Path("base-vazia")
        with patch.object(Path, "exists", return_value=True), patch.object(
            Path, "is_dir", return_value=True
        ), patch.object(Path, "rglob", return_value=[]):
            with self.assertRaises(RAGPDFError) as context:
                ensure_base_dir(base_dir)

        self.assertIn("Nenhum PDF foi encontrado", str(context.exception))

    def test_ensure_vector_db_exists_fails_when_directory_is_missing(self):
        with self.assertRaises(RAGPDFError) as context:
            ensure_vector_db_exists(Path("db-inexistente"))

        self.assertIn("Base vetorial não encontrada", str(context.exception))

    def test_ask_question_rejects_blank_input(self):
        with self.assertRaises(RAGPDFError) as context:
            ask_question("   ")

        self.assertIn("Informe uma pergunta não vazia", str(context.exception))


if __name__ == "__main__":
    unittest.main()
