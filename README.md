# RAGPDF

CLI local para indexar PDFs e fazer perguntas sobre o conteúdo usando Chroma + Google Gemini.

## Requisitos

- Python 3.10+
- Chave `GOOGLE_API_KEY`

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Crie um arquivo `.env` na raiz do projeto:

```env
GOOGLE_API_KEY=sua_chave_aqui
```

## Uso

Adicione os PDFs na pasta `base/` e rode:

```bash
python main.py index
```

Para fazer uma pergunta:

```bash
python main.py ask "Sobre o que fala o documento?"
```

## Comportamento atual

- A indexação recria a base vetorial do zero em `db/`.
- A consulta usa busca por similaridade com `k=4`.
- As respostas são geradas em português e restritas ao contexto recuperado.
- Se o contexto não sustentar a resposta, a CLI informa que não encontrou base suficiente.
