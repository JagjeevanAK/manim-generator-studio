import os
import time
import glob
from pathlib import Path
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from app.config import settings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from markdownify import markdownify as md
from pydantic import SecretStr

load_dotenv()
embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=SecretStr(settings.GEMINI_API_KEY) if settings.GEMINI_API_KEY else None,
)


SOURCE_DIR = "manim-docs/docs.manim.community/en/stable"
MD_OUTPUT_DIR = "manim-docs-md"

def convert_html_to_markdown():
    """Converts HTML documentation to Markdown and saves it."""
    print(f"Converting HTML from {SOURCE_DIR} to Markdown in {MD_OUTPUT_DIR}...")
    
    if not os.path.exists(MD_OUTPUT_DIR):
        os.makedirs(MD_OUTPUT_DIR)

    html_files = glob.glob(f"{SOURCE_DIR}/**/*.html", recursive=True)
    print(f"Found {len(html_files)} HTML files.")

    for file_path in html_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            
            # Convert to Markdown
            # heading_style="ATX" gives # Header
            markdown_content = md(html_content, heading_style="ATX")
            
            rel_path = os.path.relpath(file_path, SOURCE_DIR)
            md_rel_path = os.path.splitext(rel_path)[0] + ".md"
            output_path = os.path.join(MD_OUTPUT_DIR, md_rel_path)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)
                
        except Exception as e:
            print(f"Error converting {file_path}: {e}")

    print("Conversion complete.")

def ingest_docs():
    
    if not os.path.exists(MD_OUTPUT_DIR) or not os.listdir(MD_OUTPUT_DIR):
        convert_html_to_markdown()
    else:
        print(f"Using existing Markdown files in {MD_OUTPUT_DIR}")

    md_files = glob.glob(f"{MD_OUTPUT_DIR}/**/*.md", recursive=True)
    print(f"Loading {len(md_files)} Markdown files...")
    
    documents = []
    for file_path in md_files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

            metadata = {"source": file_path}
            documents.append(Document(page_content=content, metadata=metadata))

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    md_header_splits = []
    for doc in documents:
        splits = markdown_splitter.split_text(doc.page_content)
        for split in splits:
            split.metadata.update(doc.metadata)
            md_header_splits.append(split)

    print(f"Split into {len(md_header_splits)} header-based chunks")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    
    final_splits = text_splitter.split_documents(md_header_splits)
    print(f"Final split count: {len(final_splits)}")

    batch_size = 100
    max_retries = 5

    for i in range(0, len(final_splits), batch_size):
        batch = final_splits[i : i + batch_size]
        attempt = 0
        while attempt < max_retries:
            try:
                print(
                    f"Processing batch {i // batch_size + 1} with {len(batch)} documents, attempt {attempt + 1}"
                )
                PineconeVectorStore.from_documents(
                    batch,
                    embedding,
                    index_name=settings.PINECONE_INDEX_NAME,
                )
                print(f"Batch {i // batch_size + 1} added to Pinecone")
                break
            except Exception as e:
                if "ResourceExhausted" in str(e):
                    wait_time = (
                        2**attempt * 5
                    )
                    print(f"Quota exceeded, retrying after {wait_time} seconds...")
                    time.sleep(wait_time)
                    attempt += 1
                else:
                    raise
        else:
            print(
                f"Failed to process batch {i // batch_size + 1} after {max_retries} retries. Skipping."
            )

        time.sleep(0.25)

    print("****Loading to vectorstore done ***")


if __name__ == "__main__":
    ingest_docs()
