# Sepsis Research Agent API

RAG pipeline for querying indexed sepsis research papers, exposed as a FastAPI HTTP service.

## Setup

```bash
cp .env.example .env  # fill in OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY
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

Basic query:

```bash
curl -X POST http://localhost:8000/api/chat \
  -F "message=Why was it important to test Sepsis-3 definitions outside high-income countries?"
```

With session ID (maintains conversation memory across turns):

```bash
curl -X POST http://localhost:8000/api/chat \
  -F "message=What is the main finding?" \
  -F "session_id=my-session-123"

curl -X POST http://localhost:8000/api/chat \
  -F "message=Which paper was that from?" \
  -F "session_id=my-session-123"
```

---

### Index PDFs

Single file:

```bash
curl -X POST http://localhost:8000/api/index \
  -F "files=@data/Besen_2016.pdf"
```

Multiple files:

```bash
curl -X POST http://localhost:8000/api/index \
  -F "files=@data/Besen_2016.pdf" \
  -F "files=@data/Li_2020.pdf"
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

---

### Study index

Get the full index (all papers):

```bash
curl http://localhost:8000/api/index/entries
```

Get a specific entry by number (1-based):

```bash
curl http://localhost:8000/api/index/entries/1
```
