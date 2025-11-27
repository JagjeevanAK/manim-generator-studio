# Manim Generator Studio - Backend

Backend API for generating Manim visualizations using FastAPI and uv.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) - Fast Python package installer
- [Docker](https://www.docker.com/) (optional, for containerized deployment)
- Python 3.10+

## Local Development Setup

### 1. Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Install Dependencies

```bash
cd backend
uv sync
```

This will:
- Create a virtual environment in `.venv`
- Install all dependencies from `pyproject.toml`
- Generate `uv.lock` for reproducible installs

### 3. Environment Variables

Create a `.env` file in the backend directory:

```env
# Supabase Configuration
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
SUPABASE_BUCKET=your_bucket_name

# API Keys
GEMINI_API_KEY=your_gemini_key
GOOGLE_API_KEY=your_google_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=your_index_name

# LangChain (optional)
LANGCHAIN_API_KEY=your_langchain_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=your_project_name

# Local settings
RENDER_DIR=./renders
MANIM_QUALITY=m
```

### 4. Create Renders Directory

```bash
mkdir -p renders
```

### 5. Run the Development Server

```bash
# Using uv
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or activate the virtual environment first
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## Docker Deployment

### Build and Run with Docker Compose

```bash
cd backend
docker-compose up --build
```

This will:
- Build the Docker image with uv
- Start the API on port 8000
- Mount the renders directory for persistent storage

### Docker Commands

```bash
# Start in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild after changes
docker-compose up --build
```

## API Documentation

Once running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI application
│   ├── router.py         # API routes
│   ├── config.py         # Configuration settings
│   ├── generator.py      # Manim code generation
│   ├── renderer.py       # Manim rendering
│   ├── scheduler.py      # Background tasks
│   ├── schemas.py        # Pydantic models
│   └── supabase_client.py
├── pyproject.toml        # Project dependencies (uv)
├── uv.lock              # Locked dependencies
├── Dockerfile           # Container configuration
├── docker-compose.yml   # Docker orchestration
└── README.md
```

## Development Commands

```bash
# Install new package
uv add package-name

# Install dev dependency
uv add --dev package-name

# Update dependencies
uv lock --upgrade

# Run tests (if configured)
uv run pytest

# Format code
uv run ruff format .

# Lint code
uv run ruff check .
```

## Troubleshooting

### Port Already in Use

```bash
# Find and kill process on port 8000
lsof -ti:8000 | xargs kill -9
```

### Renders Directory Permission Issues

```bash
chmod 777 renders
```

### Docker Build Issues

```bash
# Clean rebuild
docker-compose down -v
docker-compose build --no-cache
docker-compose up
```

## System Dependencies

The application requires these system packages (automatically installed in Docker):
- ffmpeg
- libcairo2-dev
- libpango1.0-dev
- texlive (with fonts-extra, latex-recommended, science)
- tipa

For local development on macOS:
```bash
brew install ffmpeg cairo pango
brew install --cask mactex  # For LaTeX support
```

For local development on Ubuntu/Debian:
```bash
sudo apt-get update && sudo apt-get install -y \
    ffmpeg libcairo2-dev libpango1.0-dev \
    texlive texlive-fonts-extra texlive-latex-recommended texlive-science tipa
```

## Production Deployment

For production, consider:
1. Setting proper environment variables
2. Using a production ASGI server configuration
3. Setting up reverse proxy (nginx)
4. Enabling HTTPS
5. Configuring proper logging
6. Setting up monitoring

## License

MIT
