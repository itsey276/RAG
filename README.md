# PDF RAG

A small retrieval-augmented question-answering pipeline for PDF documents. It extracts PDF text, splits the extracted Markdown into overlapping chunks, stores those chunks and their embeddings in SQLite, retrieves the most similar chunks for a question, and asks an OpenAI-compatible chat model to answer using the retrieved context.

## How it works

1. `pdf_to_md.py` extracts text from each PDF page into Markdown.
2. `md_to_chunks.py` splits Markdown into 2,000-character chunks with 50 characters of overlap and stores them in a SQLite `chunks` table.
3. `embed_chunks.py` generates embeddings with `nomic-embed-text-v1.5` and saves the vectors as SQLite BLOBs.
4. `semantic_search.py` embeds a query and ranks stored chunks by cosine similarity.
5. `rag_query.py` supplies the top matching chunks to a chat model and requests an answer grounded in those passages, with source and chunk citations.

`pipeline.py` runs the PDF conversion, chunking, and embedding steps together. Search computes cosine similarity over the embedded chunks in SQLite; it does not require a separate vector database.

## Requirements

- Python 3.10 or newer
- An API key for the University of Florida AI service used by this project
- Python packages: `PyPDF2` and `openai`

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install PyPDF2 openai
```

Set the API key in your shell. The key is read from `UF_API_KEY` and should not be committed to the repository:

```bash
export UF_API_KEY="your-api-key"  # Windows PowerShell: $env:UF_API_KEY="your-api-key"
```

The API-compatible endpoint is configured in `embed_chunks.py` as `https://api.ai.it.ufl.edu/v1`.

## Build the database

Put one or more PDFs in `sources/` and run:

```bash
python pipeline.py
```

By default, the pipeline reads PDFs from `./sources` beside `pipeline.py` and writes the database to `./chunks.db`. Override either path or the embedding batch size with:

```bash
python pipeline.py --sources ./sources --database ./chunks.db --batch-size 64
```

The pipeline replaces chunks for each PDF it processes and embeds any chunks that do not yet have embeddings. If you change the embedding model or want to regenerate existing vectors, use `--force` with the standalone embedding script after building the chunks:

```bash
python embed_chunks.py ./chunks.db --force
```

## Ask a question

```bash
python rag_query.py "What are the main findings?"
```

Options include selecting a database, number of retrieved chunks, and chat model:

```bash
python rag_query.py "Summarize the evidence" \
  --database ./chunks.db \
  --top-k 5 \
  --model gpt-oss-120b
```

The default chat model is `gpt-oss-120b`; set `UF_CHAT_MODEL` to change the default, or pass `--model`. Answers include citations in the form `[source_file, chunk N]`. If retrieved context is insufficient, the model is instructed to say so rather than invent an answer.

## Individual commands

Each step can also be run independently:

```bash
# Extract a PDF to Markdown (defaults to the same filename with .md extension)
python pdf_to_md.py sources/example.pdf

# Split Markdown into chunks in a SQLite database
python md_to_chunks.py example.md --output chunks.db

# Generate embeddings for unembedded chunks
python embed_chunks.py chunks.db

# Find the most similar chunks for a query
python semantic_search.py "search phrase" --database chunks.db --top-k 5
```

All API-using commands require `UF_API_KEY`. The standalone search script prints retrieved passages and similarity scores; `rag_query.py` additionally sends retrieved content to the chat-completions API.

## Notes

- Chunk indexes start at zero. Chunking uses character windows, not token or sentence boundaries.
- PDF extraction relies on `PyPDF2` text extraction; scanned/image-only PDFs may produce little or no usable text because OCR is not included.
- Embedding and chat requests use the configured UF endpoint. `openai` is the client library for its OpenAI-compatible API.
- `chunks.db` and documents in `sources/` are runtime data, not required source code. Keep API credentials and any sensitive documents out of version control.
