#!/usr/bin/env python3
"""
AI Document Processing System
Main entry point for document classification, extraction, and search

Author: AI Engineer Assessment
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from document_processor import load_documents_from_folder
from classifier import DocumentClassifier
from extractor import extract_data
from search import SemanticSearch, demo_search


def process_documents(input_folder: str, output_file: str = "output.json") -> dict:
    """
    Main processing pipeline:
    1. Load and extract text from PDFs
    2. Classify each document
    3. Extract structured data
    4. Save results to JSON

    Args:
        input_folder: Path to folder containing PDF documents
        output_file: Path to output JSON file

    Returns:
        Dictionary of processed results
    """
    print("="*60)
    print("AI DOCUMENT PROCESSING SYSTEM")
    print("="*60)

    # Step 1: Load documents
    print("\n[1/4] Loading and extracting text from documents...")
    documents = load_documents_from_folder(input_folder)

    if not documents:
        print("No documents found. Exiting.")
        return {}

    # Step 2: Initialize classifier
    print("\n[2/4] Initializing document classifier...")
    classifier = DocumentClassifier()

    # Step 3: Classify and extract data
    print("\n[3/4] Classifying documents and extracting data...")
    results = {}

    for filename, text in documents.items():
        print(f"\nProcessing: {filename}")

        # Classify
        doc_class, confidence = classifier.classify(text)
        print(f"  Classification: {doc_class} (confidence: {confidence:.3f})")

        # Extract data based on classification
        extracted_data = extract_data(text, doc_class)

        # Build result entry
        result = {"class": doc_class}
        result.update(extracted_data)

        results[filename] = result

        # Print extracted data
        if extracted_data:
            for key, value in extracted_data.items():
                if value is not None:
                    print(f"  {key}: {value}")

    # Step 4: Save results
    print(f"\n[4/4] Saving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")
    print(f"Processed {len(results)} documents")

    return results


def run_search_demo(input_folder: str) -> None:
    """Run the semantic search demonstration."""
    print("\n[SEARCH] Loading documents for semantic search...")
    documents = load_documents_from_folder(input_folder)

    if not documents:
        print("No documents found. Exiting.")
        return

    print("\n[SEARCH] Initializing semantic search engine...")
    search_engine = SemanticSearch()
    search_engine.index_documents(documents)

    # Run interactive demo
    demo_search(search_engine)


def run_qa_demo(input_folder: str) -> None:
    """Run the question-answering demonstration (bonus feature)."""
    from qa_system import LocalQASystem, demo_qa

    print("\n[QA] Loading documents...")
    documents = load_documents_from_folder(input_folder)

    if not documents:
        print("No documents found. Exiting.")
        return

    print("\n[QA] Initializing search engine...")
    search_engine = SemanticSearch()
    search_engine.index_documents(documents)

    print("\n[QA] Initializing QA system...")
    qa_system = LocalQASystem()

    # Run interactive demo
    demo_qa(qa_system, search_engine, documents)


def main():
    parser = argparse.ArgumentParser(
        description="AI Document Processing System - Classify, Extract, and Search Documents"
    )
    parser.add_argument(
        "input_folder",
        help="Path to folder containing PDF documents"
    )
    parser.add_argument(
        "-o", "--output",
        default="output.json",
        help="Output JSON file path (default: output.json)"
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help="Run semantic search demo after processing"
    )
    parser.add_argument(
        "--qa",
        action="store_true",
        help="Run QA demo (bonus feature, requires additional download)"
    )
    parser.add_argument(
        "--search-only",
        action="store_true",
        help="Only run semantic search (skip classification/extraction)"
    )

    args = parser.parse_args()

    # Validate input folder
    if not os.path.isdir(args.input_folder):
        print(f"Error: '{args.input_folder}' is not a valid directory")
        sys.exit(1)

    if args.search_only:
        # Only run search demo
        run_search_demo(args.input_folder)
    else:
        # Run full processing pipeline
        results = process_documents(args.input_folder, args.output)

        # Optionally run search demo
        if args.search and results:
            run_search_demo(args.input_folder)

        # Optionally run QA demo
        if args.qa and results:
            run_qa_demo(args.input_folder)

    print("\n" + "="*60)
    print("Processing complete!")
    print("="*60)


if __name__ == "__main__":
    main()
