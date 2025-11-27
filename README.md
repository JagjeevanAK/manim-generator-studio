# Manim Generator Studio

A modern web application for generating and rendering Manim animations using AI.

## Project Structure

```
manim-generator-studio/
├── backend/          # FastAPI backend with uv
└── frontend/         # Next.js frontend
```

## Quick Start

### Backend Setup

```bash
cd backend

# Option 1: Local development with uv (recommended for development)
./dev.sh setup    # Install dependencies and setup environment
./dev.sh run      # Start development server

# Option 2: Docker (recommended for production)
./dev.sh docker-run   # Build and run with Docker
```

The backend API will be available at `http://localhost:8000`

See [backend/README.md](backend/README.md) for detailed documentation.

### Frontend Setup

```bash
cd frontend

# Install dependencies
pnpm install

# Run development server
pnpm dev
```

The frontend will be available at `http://localhost:3000`

## Requirements

### For Local Development
- **Backend**: [uv](https://docs.astral.sh/uv/) (Python package manager)
- **Frontend**: [pnpm](https://pnpm.io/) or npm
- **System**: ffmpeg, Cairo, Pango, LaTeX (see backend README)

### For Docker Development
- Docker & Docker Compose

## Environment Variables

Copy the example environment files and fill in your credentials:

```bash
# Backend
cp backend/.env.example backend/.env

# Frontend  
cp frontend/.env.example frontend/.env
```

## Features

- 🎨 Generate Manim animations from natural language
- 🤖 AI-powered code generation using Google Gemini
- 📚 RAG-based documentation search
- 🎬 Real-time rendering with quality options
- 💾 Cloud storage with Supabase
- 📊 Job history and status tracking

## Tech Stack

### Backend
- **Framework**: FastAPI
- **Package Manager**: uv (fast Python package installer)
- **Animation**: Manim Community
- **AI/LLM**: LangChain, Google Gemini, Cohere
- **Vector DB**: Pinecone
- **Storage**: Supabase
- **Containerization**: Docker

### Frontend
- **Framework**: Next.js 14 (App Router)
- **Styling**: Tailwind CSS
- **UI Components**: shadcn/ui
- **State Management**: React Hooks
- **Type Safety**: TypeScript

## Development Scripts

### Backend
```bash
./dev.sh setup         # Initial setup
./dev.sh run           # Start dev server
./dev.sh docker-run    # Docker development
./dev.sh install <pkg> # Add new package
./dev.sh clean         # Clean build artifacts
```

### Frontend
```bash
pnpm dev              # Start dev server
pnpm build            # Build for production
pnpm start            # Start production server
pnpm lint             # Run linter
```

## API Documentation

Once the backend is running:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT

## Support

For issues and questions, please open a GitHub issue.
