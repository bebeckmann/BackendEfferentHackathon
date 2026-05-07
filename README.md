# Sepsis Research Agent API

RAG pipeline for querying indexed sepsis research papers, exposed as a FastAPI HTTP service.

## Setup

```bash
cp .env.example .env  # fill in OPENROUTER_API_KEY, QDRANT_URL, QDRANT_API_KEY
pip install -r requirements.txt
```

## Start the server

```bash
uvicorn main:app --reload
```

Runs on `http://localhost:8000`. Interactive docs available at `http://localhost:8000/docs`.

---

## API Endpoints

### Health check

```bash
curl http://localhost:8000/health
```

---

### Chat (RAG query)

```bash
curl -X POST http://localhost:8000/api/chat \
  -F "message=What are the diagnostic criteria for sepsis?"
```

---

### Index PDFs

Single file:

```bash
curl -X POST http://localhost:8000/api/index \
  -F "files=@/path/to/paper.pdf"
```

Multiple files:

```bash
curl -X POST http://localhost:8000/api/index \
  -F "files=@paper1.pdf" \
  -F "files=@paper2.pdf"
```

---

### List all indexed documents

```bash
curl http://localhost:8000/api/documents
```

---

### Delete a document by UUID

```bash
curl -X DELETE "http://localhost:8000/api/documents?doc_hash=3912037316751052580"
```

### Delete a document by name

```bash
curl -X DELETE "http://localhost:8000/api/documents?name=Besen_2016"
```

---

### Delete all documents

```bash
curl -X DELETE http://localhost:8000/api/documents/all
```
