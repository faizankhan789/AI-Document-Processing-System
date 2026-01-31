#!/usr/bin/env python3
"""
AI Document Processing System - Web UI
Gradio-based interface for document classification, extraction, and search
"""

import os
import sys
import json
import gradio as gr

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from document_processor import load_documents_from_folder
from classifier import DocumentClassifier
from extractor import extract_data
from search import SemanticSearch

# Global state
documents = {}
classifier = None
search_engine = None
results = {}


def initialize_models():
    """Initialize ML models (lazy loading)."""
    global classifier, search_engine
    if classifier is None:
        classifier = DocumentClassifier()
    if search_engine is None:
        search_engine = SemanticSearch()
    return "Models initialized successfully!"


def process_documents(folder_path: str) -> tuple:
    """Process documents from folder."""
    global documents, results, search_engine

    if not folder_path or not os.path.isdir(folder_path):
        return "Error: Please provide a valid folder path", "", ""

    # Initialize models
    initialize_models()

    # Load documents
    documents = load_documents_from_folder(folder_path)
    if not documents:
        return "No PDF files found in the folder", "", ""

    # Classify and extract
    results = {}
    log_messages = []

    for filename, text in documents.items():
        doc_class, confidence = classifier.classify(text)
        extracted_data = extract_data(text, doc_class)

        result = {"class": doc_class}
        result.update(extracted_data)
        results[filename] = result

        log_messages.append(f"✓ {filename}: {doc_class} (confidence: {confidence:.2f})")

    # Index for search
    search_engine.index_documents(documents)

    # Format outputs
    log_output = "\n".join(log_messages)
    json_output = json.dumps(results, indent=2)

    # Create summary table
    summary = create_summary_table(results)

    return log_output, json_output, summary


def create_summary_table(results: dict) -> str:
    """Create a markdown summary table."""
    if not results:
        return "No results yet"

    # Count by class
    class_counts = {}
    for r in results.values():
        c = r.get("class", "Unknown")
        class_counts[c] = class_counts.get(c, 0) + 1

    table = "| Document Type | Count |\n|--------------|-------|\n"
    for doc_class, count in sorted(class_counts.items()):
        table += f"| {doc_class} | {count} |\n"

    table += f"\n**Total: {len(results)} documents processed**"
    return table


def semantic_search(query: str, top_k: int = 5) -> str:
    """Perform semantic search."""
    global search_engine, documents

    if not query:
        return "Please enter a search query"

    if search_engine is None or not documents:
        return "Please process documents first"

    try:
        results = search_engine.search(query, top_k=int(top_k))

        output = f"## Search Results for: '{query}'\n\n"
        for i, (doc_name, score, snippet) in enumerate(results, 1):
            output += f"### {i}. {doc_name}\n"
            output += f"**Relevance Score:** {score:.3f}\n\n"
            output += f"```\n{snippet}\n```\n\n"

        return output
    except Exception as e:
        return f"Search error: {str(e)}"


def get_document_details(filename: str) -> str:
    """Get details for a specific document."""
    global results, documents

    if not filename:
        return "Select a document to view details"

    if filename in results:
        result = results[filename]
        output = f"## {filename}\n\n"
        output += f"**Classification:** {result.get('class', 'Unknown')}\n\n"

        # Show extracted fields
        output += "### Extracted Data\n"
        for key, value in result.items():
            if key != "class" and value is not None:
                output += f"- **{key}:** {value}\n"

        # Show raw text
        if filename in documents:
            output += f"\n### Raw Text\n```\n{documents[filename]}\n```"

        return output

    return "Document not found"


def create_ui():
    """Create the Gradio interface."""

    with gr.Blocks(title="AI Document Processor") as app:
        gr.Markdown("""
        # 🔍 AI Document Processing System
        **Local AI system for document classification, data extraction, and semantic search**

        *Powered by open-source models - no external APIs required*
        """)

        with gr.Tabs():
            # Tab 1: Process Documents
            with gr.TabItem("📁 Process Documents"):
                gr.Markdown("### Step 1: Load and Process Documents")

                with gr.Row():
                    folder_input = gr.Textbox(
                        label="Folder Path",
                        placeholder="Enter path to folder containing PDF files",
                        value="ai_engineer_dataset_generated (1) (2)"
                    )
                    process_btn = gr.Button("🚀 Process Documents", variant="primary")

                with gr.Row():
                    with gr.Column():
                        log_output = gr.Textbox(
                            label="Processing Log",
                            lines=10,
                            interactive=False
                        )
                    with gr.Column():
                        summary_output = gr.Markdown(label="Summary")

                json_output = gr.Code(
                    label="Output JSON (output.json)",
                    language="json",
                    lines=15
                )

                save_btn = gr.Button("💾 Save to output.json")

                def save_json(json_str):
                    if json_str:
                        with open("output.json", "w") as f:
                            f.write(json_str)
                        return "Saved to output.json!"
                    return "Nothing to save"

                save_status = gr.Textbox(label="Save Status", interactive=False)
                save_btn.click(save_json, inputs=[json_output], outputs=[save_status])

                process_btn.click(
                    process_documents,
                    inputs=[folder_input],
                    outputs=[log_output, json_output, summary_output]
                )

            # Tab 2: Semantic Search
            with gr.TabItem("🔎 Semantic Search"):
                gr.Markdown("""
                ### Search Documents by Meaning
                Enter natural language queries to find relevant documents.
                """)

                with gr.Row():
                    search_input = gr.Textbox(
                        label="Search Query",
                        placeholder="e.g., 'Find documents about electricity usage'",
                        scale=4
                    )
                    top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Number of Results",
                        scale=1
                    )

                search_btn = gr.Button("🔍 Search", variant="primary")

                search_results = gr.Markdown(label="Search Results")

                gr.Markdown("### Example Queries")
                gr.Examples(
                    examples=[
                        ["Find all documents mentioning payments due"],
                        ["Documents about electricity usage"],
                        ["Professional with software engineering experience"],
                        ["Invoice with high amount"],
                        ["Resume with email contact"]
                    ],
                    inputs=[search_input]
                )

                search_btn.click(
                    semantic_search,
                    inputs=[search_input, top_k_slider],
                    outputs=[search_results]
                )
                search_input.submit(
                    semantic_search,
                    inputs=[search_input, top_k_slider],
                    outputs=[search_results]
                )

            # Tab 3: Document Viewer
            with gr.TabItem("📄 Document Viewer"):
                gr.Markdown("### View Individual Document Details")

                def get_doc_list():
                    return list(results.keys()) if results else []

                refresh_btn = gr.Button("🔄 Refresh Document List")
                doc_dropdown = gr.Dropdown(
                    label="Select Document",
                    choices=[],
                    interactive=True
                )
                doc_details = gr.Markdown()

                refresh_btn.click(
                    lambda: gr.update(choices=list(results.keys()) if results else []),
                    outputs=[doc_dropdown]
                )
                doc_dropdown.change(
                    get_document_details,
                    inputs=[doc_dropdown],
                    outputs=[doc_details]
                )

            # Tab 4: About
            with gr.TabItem("ℹ️ About"):
                gr.Markdown("""
                ## About This System

                This is a local AI document processing system built with open-source tools.

                ### Features
                - **Document Classification**: Categorizes documents into Invoice, Resume, Utility Bill, Other, or Unclassifiable
                - **Data Extraction**: Extracts structured fields based on document type
                - **Semantic Search**: Find documents by meaning using natural language queries

                ### Technologies Used
                | Component | Library |
                |-----------|---------|
                | PDF Extraction | pdfplumber |
                | Embeddings | SentenceTransformers (all-MiniLM-L6-v2) |
                | Vector Search | FAISS |
                | Web UI | Gradio |

                ### Classification Categories
                | Type | Extracted Fields |
                |------|-----------------|
                | Invoice | invoice_number, date, company, total_amount |
                | Resume | name, email, phone, experience_years |
                | Utility Bill | account_number, date, usage_kwh, amount_due |
                | Other | None |
                | Unclassifiable | None |

                ---
                *All processing runs locally - no data is sent to external servers.*
                """)

    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
