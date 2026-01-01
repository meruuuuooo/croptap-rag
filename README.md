# CropTAP RAG

A Retrieval-Augmented Generation (RAG) system for Philippine agricultural knowledge. Ask questions about crop production, planting tips, statistics, and soil data — powered by local LLM (Ollama).

## Features

- **PDF Document Ingestion** - Automatically extracts and indexes PDF documents
- **Semantic Search** - Find relevant information using natural language
- **Local LLM** - Uses Ollama for privacy-friendly, offline AI responses
- **Category Filtering** - Filter by crop guides, statistics, planting tips, or soil data
- **FastAPI Backend** - RESTful API with interactive docs

## System Architecture

![System Architecture](/architecture-diagram.png)

## RAG Pipeline

The Retrieval-Augmented Generation pipeline processes user queries through the following stages:

![RAG Pipeline](/rag-diagram.png)

### Pipeline Steps

| Step | Component         | Description                                                 |
| ---- | ----------------- | ----------------------------------------------------------- |
| 1    | **Embed Query**   | Convert user question to vector using sentence-transformers |
| 2    | **Vector Search** | Find similar document chunks in ChromaDB                    |
| 3    | **Build Context** | Combine retrieved chunks with source metadata               |
| 4    | **Create Prompt** | Format system + context + question for LLM                  |
| 5    | **Generate**      | Ollama generates grounded response                          |
| 6    | **Return**        | Response with cited sources                                 |

## Document Ingestion Flow

![Document Ingestion Flow](/Ingestion-Flow.png)

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/download) installed and running

### Installation

```bash
# Clone and navigate to project
cd croptap-rag

# Create virtual environment
python -m venv .venv
source .venv/Scripts/activate  # Windows
# source .venv/bin/activate    # Linux/Mac

# Install dependencies
pip install -r app/requirements.txt

# Pull an LLM model
ollama pull llama3.2
```

### Configuration

Copy the environment template:

```bash
cp app/.env.example app/.env
```

Default settings work out of the box with Ollama.

### Ingest Documents

Place your PDF files in `data/raw/` organized by category, then run:

```bash
python -m ingestion.load_documents
```

### Start the Server

```bash
cd app
uvicorn main:app --reload
```

Visit **http://localhost:8000/docs** for interactive API documentation.

## API Endpoints

| Method | Endpoint             | Description                        |
| ------ | -------------------- | ---------------------------------- |
| `POST` | `/api/v1/query`      | Ask a question (full RAG pipeline) |
| `GET`  | `/api/v1/search`     | Search documents without LLM       |
| `POST` | `/api/v1/ingest`     | Trigger document ingestion         |
| `GET`  | `/api/v1/categories` | List available categories          |
| `GET`  | `/api/v1/stats`      | Collection statistics              |
| `GET`  | `/api/v1/health`     | Health check                       |

### Example Query

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I grow bananas?", "top_k": 3}'
```

### Example Response

```json
{
  "answer": "To grow bananas successfully, follow these key steps:\n\n1. **Site Selection**: Choose a location with full sunlight and well-drained soil. Bananas prefer slightly acidic soil with pH 5.5-7.0.\n\n2. **Planting**: Use suckers or tissue-cultured plantlets. Plant in holes 30cm deep and 30cm wide, spaced 2-3 meters apart.\n\n3. **Watering**: Bananas need consistent moisture. Water regularly, especially during dry periods. Avoid waterlogging.\n\n4. **Fertilization**: Apply organic matter and complete fertilizer (14-14-14) every 2-3 months.\n\n5. **Maintenance**: Remove dead leaves and excess suckers. Support heavy bunches with props.\n\n6. **Harvesting**: Harvest when fingers are plump and ridges rounded, typically 75-80 days after flowering.\n\nSource: Banana Production Guide",
  "sources": [
    {
      "content": "Banana requires a warm and humid climate. It grows best in areas with temperatures ranging from 25-30°C...",
      "source": "C:\\data\\raw\\crop_production_guide\\Banana-Production-Guide.pdf",
      "category": "crop_production_guide",
      "filename": "Banana-Production-Guide.pdf",
      "score": 0.8234
    },
    {
      "content": "Planting materials include sword suckers, water suckers, and tissue-cultured plantlets...",
      "source": "C:\\data\\raw\\crop_production_guide\\Banana-Production-Guide.pdf",
      "category": "crop_production_guide",
      "filename": "Banana-Production-Guide.pdf",
      "score": 0.7891
    }
  ],
  "question": "How do I grow bananas?",
  "category_filter": null,
  "documents_retrieved": 3
}
```

## Project Structure

```text
croptap-rag/
├── api/                # API routes and schemas
├── app/                # FastAPI application
│   ├── main.py         # Entry point
│   ├── config.py       # Configuration
│   └── requirements.txt
├── data/
│   └── raw/            # PDF documents by category
├── embeddings/         # Text embedding generation
├── ingestion/          # Document processing pipeline
├── llm/                # LLM client (Ollama)
├── prompt/             # Prompt templates
├── retrieval/          # Vector search
└── vector_store/       # ChromaDB persistence
```

## Data Categories

| Category                | Description                   |
| ----------------------- | ----------------------------- |
| `crop_production_guide` | Farming and production guides |
| `crops_statistics`      | Agricultural statistics       |
| `planting_tips`         | Planting recommendations      |
| `soil_data`             | Soil properties data          |

## Tech Stack

- **Framework**: FastAPI
- **Vector DB**: ChromaDB
- **Embeddings**: `sentence-transformers` (all-MiniLM-L6-v2)
- **LLM**: Ollama (`llama3.2`)
- **PDF Processing**: PyMuPDF

## License

MIT
