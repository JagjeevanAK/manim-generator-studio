#!/bin/bash

# Manim Generator Studio - Development Script
# This script helps with common development tasks

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper function to print colored messages
print_message() {
    echo -e "${GREEN}==>${NC} $1"
}

print_error() {
    echo -e "${RED}Error:${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}Warning:${NC} $1"
}

# Check if uv is installed
check_uv() {
    if ! command -v uv &> /dev/null; then
        print_error "uv is not installed. Please install it first:"
        echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
}

# Setup environment
setup() {
    print_message "Setting up development environment..."
    
    check_uv
    
    # Create .env if it doesn't exist
    if [ ! -f .env ]; then
        if [ -f .env.example ]; then
            print_message "Copying .env.example to .env"
            cp .env.example .env
            print_warning "Please edit .env with your actual configuration"
        else
            print_error ".env.example not found"
        fi
    fi
    
    # Create renders directory
    mkdir -p renders
    
    # Install dependencies
    print_message "Installing dependencies with uv..."
    uv sync
    
    print_message "Setup complete!"
    print_message "To start the server, run: ./dev.sh run"
}

# Run the development server
run() {
    check_uv
    
    if [ ! -f .env ]; then
        print_error ".env file not found. Run './dev.sh setup' first"
        exit 1
    fi
    
    print_message "Starting development server..."
    uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
}

# Build and run with Docker
docker_run() {
    print_message "Building and starting Docker container..."
    docker-compose up --build
}

# Stop Docker containers
docker_stop() {
    print_message "Stopping Docker containers..."
    docker-compose down
}

# Clean up
clean() {
    print_message "Cleaning up..."
    rm -rf .venv
    rm -rf __pycache__
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    print_message "Clean complete!"
}

# Install a new package
install() {
    if [ -z "$1" ]; then
        print_error "Please specify a package name"
        exit 1
    fi
    
    check_uv
    print_message "Installing $1..."
    uv add "$1"
}

# Show help
help() {
    echo "Manim Generator Studio - Development Script"
    echo ""
    echo "Usage: ./dev.sh [command]"
    echo ""
    echo "Commands:"
    echo "  setup          - Set up development environment"
    echo "  run            - Run the development server"
    echo "  docker-run     - Build and run with Docker"
    echo "  docker-stop    - Stop Docker containers"
    echo "  install <pkg>  - Install a new package"
    echo "  clean          - Clean up generated files"
    echo "  help           - Show this help message"
    echo ""
}

# Main script
case "$1" in
    setup)
        setup
        ;;
    run)
        run
        ;;
    docker-run)
        docker_run
        ;;
    docker-stop)
        docker_stop
        ;;
    install)
        install "$2"
        ;;
    clean)
        clean
        ;;
    help|--help|-h|"")
        help
        ;;
    *)
        print_error "Unknown command: $1"
        help
        exit 1
        ;;
esac
