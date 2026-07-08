# Desafio MBA Engenharia de Software com IA - Full Cycle

Ingestão e busca semântica de um PDF usando LangChain, PostgreSQL + pgVector e Google Gemini.

## Pré-requisitos

- Python 3.10+
- Docker & Docker Compose
- Chave de API do Google Gemini (`GOOGLE_API_KEY`)

## Configuração

1. Crie e ative o ambiente virtual:

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Instale as dependências:

```bash
pip install -r requirements.txt
```

3. Configure as variáveis de ambiente:

```bash
cp .env.example .env
```

Preencha o `.env`:

```
GOOGLE_API_KEY=sua_chave_aqui
GOOGLE_EMBEDDING_MODEL='models/embedding-001'
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag
PG_VECTOR_COLLECTION_NAME=documents
PDF_PATH=document.pdf
```

## Execução

1. Suba o banco de dados:

```bash
docker compose up -d
```

2. Execute a ingestão do PDF (split em chunks de 1000 caracteres com overlap de 150, geração de embeddings e armazenamento no pgVector):

```bash
python src/ingest.py
```

3. Rode o chat:

```bash
python src/chat.py
```

## Exemplo de uso

```
Faça sua pergunta (digite 'sair' para encerrar):

PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento foi de 10 milhões de reais.

PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

## Como funciona

- `src/ingest.py`: carrega o `document.pdf` com PyPDFLoader, divide em chunks (1000/150) com RecursiveCharacterTextSplitter, gera embeddings com Gemini e grava no PostgreSQL via PGVector.
- `src/search.py`: vetoriza a pergunta, busca os 10 chunks mais relevantes (`similarity_search_with_score(query, k=10)`), monta o prompt com o contexto e chama a LLM (`gemini-2.5-flash`). Respostas fora do contexto retornam a mensagem padrão.
- `src/chat.py`: CLI que lê perguntas no terminal em loop e exibe as respostas.
