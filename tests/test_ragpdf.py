from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

from ragpdf import RAGPDFError, _embed_texts, _file_hash, ask_question, ensure_base_dir, ensure_vector_db_exists


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


class EmbedTextsTests(unittest.TestCase):
    def test_embed_texts_calls_api_once_for_multiple_texts(self):
        """_embed_texts deve fazer uma única chamada batch, não N chamadas."""
        texts = ["texto um", "texto dois", "texto três"]
        mock_response = {"embedding": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]}

        with patch("ragpdf._configure_genai") as mock_configure:
            mock_genai = MagicMock()
            mock_genai.embed_content.return_value = mock_response
            mock_configure.return_value = mock_genai

            result = _embed_texts(texts)

        mock_genai.embed_content.assert_called_once()
        self.assertEqual(result, [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

    def test_embed_texts_passes_full_list_to_api(self):
        """_embed_texts deve passar a lista inteira para embed_content."""
        texts = ["a", "b"]
        mock_response = {"embedding": [[1.0], [2.0]]}

        with patch("ragpdf._configure_genai") as mock_configure:
            mock_genai = MagicMock()
            mock_genai.embed_content.return_value = mock_response
            mock_configure.return_value = mock_genai

            _embed_texts(texts)

        args, kwargs = mock_genai.embed_content.call_args
        self.assertEqual(kwargs.get("content", args[1] if len(args) > 1 else None), texts)


class FileHashTests(unittest.TestCase):
    def test_file_hash_returns_consistent_value(self):
        """Hash SHA-256 deve ser determinístico para o mesmo conteúdo."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(b"conteudo de teste")
            tmp_path = Path(f.name)

        try:
            hash1 = _file_hash(tmp_path)
            hash2 = _file_hash(tmp_path)
            self.assertEqual(hash1, hash2)
            self.assertEqual(len(hash1), 64)  # SHA-256 hex = 64 chars
        finally:
            tmp_path.unlink()

    def test_file_hash_differs_for_different_content(self):
        """Arquivos com conteúdo diferente devem ter hashes distintos."""
        with tempfile.NamedTemporaryFile(delete=False) as f1, tempfile.NamedTemporaryFile(delete=False) as f2:
            f1.write(b"conteudo A")
            f2.write(b"conteudo B")
            path1, path2 = Path(f1.name), Path(f2.name)

        try:
            self.assertNotEqual(_file_hash(path1), _file_hash(path2))
        finally:
            path1.unlink()
            path2.unlink()


if __name__ == "__main__":
    unittest.main()
