# AI Document Processing System

A local AI system for document classification, data extraction, and semantic search using only open-source tools.

## Features

1. **Document Ingestion**: Reads PDF files and extracts text content
2. **Classification**: Categorizes documents into Invoice, Resume, Utility Bill, Other, or Unclassifiable
3. **Data Extraction**: Extracts structured fields based on document type
4. **Semantic Search**: Find documents by meaning using natural language queries
5. **Question-Answering (Bonus)**: Local QA using open-source LLM

## Requirements

- Python 3.10+
- All processing runs locally (no paid APIs)

## Installation

### 1. Create Virtual Environment

```bash
# Using venv
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n ai_engineer python=3.10 -y
conda activate ai_engineer
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Web UI

Launch the Gradio web interface:

```bash
python app.py
```

Then open http://localhost:7860 in your browser.

The web UI provides:
- Document processing with visual feedback
- Interactive semantic search
- Document viewer with extracted data
- Export to JSON

### Option 2: Command Line Interface

#### Basic Usage (Classification + Extraction)

```bash
python main.py path/to/pdf/folder
```

This will:
- Process all PDF files in the folder
- Classify each document
- Extract relevant fields
- Save results to `output.json`

#### With Semantic Search Demo

```bash
python main.py path/to/pdf/folder --search
```

#### Search Only Mode

```bash
python main.py path/to/pdf/folder --search-only
```

#### With QA Demo

```bash
python main.py path/to/pdf/folder --qa
```

#### Custom Output File

```bash
python main.py path/to/pdf/folder -o results.json
```

## Output Format

The system generates `output.json` with the following structure:

```json
{
  "invoice_1.pdf": {
    "class": "Invoice",
    "invoice_number": "INV-1234",
    "date": "2025-01-01",
    "company": "ACME Ltd.",
    "total_amount": 350.5
  },
  "resume_1.pdf": {
    "class": "Resume",
    "name": "John Doe",
    "email": "john@example.com",
    "phone": "123-456-7890",
    "experience_years": 5
  },
  "utilitybill_1.pdf": {
    "class": "Utility Bill",
    "account_number": "ACC-12345",
    "date": "2025-01-15",
    "usage_kwh": 450,
    "amount_due": 125.50
  },
  "other_1.pdf": {
    "class": "Other"
  }
}
```

## Libraries Used

Library  and Purpose

`pdfplumber` --> PDF text extraction 
`sentence-transformers` --> Document embeddings for classification and search
`faiss-cpu` --> Vector similarity search
`transformers` --> Local LLM for QA (bonus)
`torch` --> Deep learning backend
`numpy` --> Numerical operations


## Classification Method

Documents are classified using semantic similarity:
1. Each document is embedded using SentenceTransformers (`all-MiniLM-L6-v2`)
2. Category descriptions are pre-computed as embeddings
3. Cosine similarity determines the best matching category
4. A confidence threshold filters uncertain classifications

## Extraction Patterns

| Document Type | Extracted Fields |
|--------------|------------------|
| Invoice --> invoice_number, date, company, total_amount |
| Resume --> name, email, phone, experience_years |
| Utility Bill --> account_number, date, usage_kwh, amount_due |
| Other/Unclassifiable --> None |

## Semantic Search

The search system uses:
- **Embeddings**: `all-MiniLM-L6-v2` from SentenceTransformers
- **Index**: FAISS IndexFlatIP for cosine similarity
- **Query**: Natural language queries are embedded and matched against document embeddings

Example queries:
- "Find all documents mentioning payments due in January"
- "Documents about electricity usage"
- "Professional with software engineering experience"

## Technical Notes

- All processing runs locally without internet access (after initial model download)
- Models are cached in `~/.cache/huggingface/`
- FAISS index can be saved/loaded for persistence
- Memory usage is optimized for CPU execution

## Project Structure

```
.
├── main.py                 # Main entry point
├── src/
│   ├── __init__.py
│   ├── document_processor.py  # PDF text extraction
│   ├── classifier.py          # Document classification
│   ├── extractor.py           # Data extraction
│   ├── search.py              # Semantic search
│   └── qa_system.py           # QA system (bonus)
├── requirements.txt        # Dependencies
├── output.json            # Generated output
└── README.md              # This file
```