# Manim Generator Studio

A modern web application for generating and rendering Manim animations using AI.

## Architecture

```mermaid
graph TD
    subgraph Client
        UI["Frontend (Next.js)"]
    end

    subgraph Server
        API["Backend API (FastAPI)"]
        Gen["Generator Service"]
        Render["Manim Renderer"]
        Ingest["Ingestion Service"]
    end

    subgraph External Services
        Gemini["Google Gemini AI"]
        Pinecone["Pinecone Vector DB"]
        Supabase["Supabase DB"]
        Storage["Video Storage"]
    end

    UI -->|Generate Request| API
    UI -->|Poll Status| API
    
    API -->|Job Created| Supabase
    API -->|Process Job| Gen
    
    Gen -->|Retrieve Context| Pinecone
    Gen -->|Generate Code| Gemini
    Gen -->|Code| Render
    
    Render -->|Render Video| Storage
    Render -->|Update Status| Supabase
    
    Ingest -->|Embed Docs| Pinecone
```

## Quick Start (Docker)

The fastest way to run the full stack:

```bash
# 1. Setup environment
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env

# 2. Run with Docker Compose
docker compose up --build
```

- **Frontend:** http://localhost:3000
- **Backend:** http://localhost:8000
- **Docs:** http://localhost:8000/docs

## Local Development

### Backend
```bash
cd backend
./dev.sh setup   # Install dependencies
./dev.sh run     # Start server
```

### Frontend
```bash
cd frontend
pnpm install
pnpm dev
```

## Features
- **AI Generation:** Text-to-animation using Gemini.
- **Real-time Rendering:** Instant feedback loop.
- **RAG Search:** Context-aware documentation search.
- **Job History:** Track and manage generation jobs.

## Tech Stack
- **Backend:** FastAPI, Manim, LangChain, Pinecone, Supabase
- **Frontend:** Next.js 14, Tailwind CSS, shadcn/ui
- **Infra:** Docker

## License
MIT
