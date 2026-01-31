"""
Question-Answering System (Bonus)
Local QA using open-source LLM with RAG
"""

import torch
from typing import List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class LocalQASystem:
    """
    Local Question-Answering system using an open-source LLM.
    Uses retrieval-augmented generation (RAG) approach.
    """

    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """
        Initialize the QA system with a local LLM.

        Args:
            model_name: HuggingFace model name (small model for demo)
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.pipe = None
        self._loaded = False

    def load_model(self) -> None:
        """Load the LLM model (lazy loading to save memory)."""
        if self._loaded:
            return

        print(f"Loading QA model: {self.model_name}")
        print("This may take a moment...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )

            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.95
            )
            self._loaded = True
            print("QA model loaded successfully!")

        except Exception as e:
            print(f"Error loading QA model: {e}")
            print("QA functionality will be disabled.")
            self._loaded = False

    def answer_question(
        self,
        question: str,
        context_docs: List[Tuple[str, str]]
    ) -> str:
        """
        Answer a question using retrieved document context.

        Args:
            question: User's question
            context_docs: List of (filename, text) tuples for context

        Returns:
            Generated answer string
        """
        if not self._loaded:
            self.load_model()

        if not self._loaded or self.pipe is None:
            return "QA model not available. Please check the installation."

        # Build context from retrieved documents
        context = "\n\n".join([
            f"Document '{name}':\n{text[:500]}"
            for name, text in context_docs[:3]  # Limit context
        ])

        # Create prompt
        prompt = f"""<|system|>
You are a helpful assistant that answers questions based on the provided documents.
Only use information from the documents to answer. If the answer is not in the documents, say so.
</s>
<|user|>
Context documents:
{context}

Question: {question}
</s>
<|assistant|>
"""

        try:
            result = self.pipe(prompt)
            generated = result[0]['generated_text']

            # Extract the answer part (after the last assistant tag)
            answer = generated.split("<|assistant|>")[-1].strip()
            return answer

        except Exception as e:
            return f"Error generating answer: {e}"


def demo_qa(qa_system: LocalQASystem, search_engine, documents: dict) -> None:
    """Run interactive QA demo."""
    print("\n" + "="*60)
    print("QUESTION-ANSWERING DEMO (Bonus Feature)")
    print("="*60)
    print("Ask questions about the documents (or 'quit' to exit)")
    print("-"*60)

    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        if not question:
            continue

        # Retrieve relevant documents
        print("Searching for relevant documents...")
        results = search_engine.search(question, top_k=3)

        print(f"Found {len(results)} relevant documents:")
        for doc_name, score, _ in results:
            print(f"  - {doc_name} (relevance: {score:.3f})")

        # Get full text of retrieved documents
        context_docs = [
            (doc_name, documents.get(doc_name, ""))
            for doc_name, _, _ in results
        ]

        # Generate answer
        print("\nGenerating answer...")
        answer = qa_system.answer_question(question, context_docs)

        print(f"\nAnswer: {answer}")
