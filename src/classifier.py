"""
Document Classifier Module
Classifies documents using pattern matching + SentenceTransformers embeddings
"""

import re
import numpy as np
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer


# Category descriptions for classification
CATEGORY_DESCRIPTIONS = {
    "Invoice": "Invoice billing statement payment total amount company business transaction receipt order purchase",
    "Resume": "Resume CV curriculum vitae job application experience skills education employment candidate professional profile career email phone summary",
    "Utility Bill": "Utility bill electricity gas water account usage kWh consumption meter reading power energy provider billing",
    "Other": "General document miscellaneous information memo letter correspondence document ID",
    "Unclassifiable": "Random unclear ambiguous nonsense corrupted unreadable"
}

# Pattern-based rules for high-confidence classification
CLASSIFICATION_PATTERNS = {
    "Invoice": [
        r"Invoice\s*#?\s*:?\s*\d+",
        r"Total\s+Amount\s*:",
        r"Thank you for your business",
    ],
    "Resume": [
        r"Email\s*:\s*[\w.+-]+@[\w.-]+",
        r"Phone\s*:\s*[\+\d\-\(\)\s]+",
        r"Experience\s*:\s*\d+\s*years?",
        r"Summary\s*:",
    ],
    "Utility Bill": [
        r"Account\s*Number\s*:\s*ACC-\d+",
        r"Usage\s*:\s*\d+\s*kWh",
        r"Amount\s+Due\s*:",
        r"Utility\s+Provider",
        r"Billing\s+Date",
    ],
    "Other": [
        r"Document\s+ID\s*:",
        r"general\s+document",
        r"does\s+not\s+fit\s+into\s+any\s+specific\s+category",
    ],
}


class DocumentClassifier:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the classifier with a SentenceTransformer model."""
        print(f"Loading classification model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.categories = list(CATEGORY_DESCRIPTIONS.keys())

        # Pre-compute category embeddings
        print("Computing category embeddings...")
        category_texts = list(CATEGORY_DESCRIPTIONS.values())
        self.category_embeddings = self.model.encode(category_texts, normalize_embeddings=True)

    def _pattern_classify(self, text: str) -> Tuple[str, float]:
        """
        Classify using pattern matching rules.
        Returns (category, score) or (None, 0) if no strong match.
        """
        text_lower = text.lower()
        scores = {}

        for category, patterns in CLASSIFICATION_PATTERNS.items():
            match_count = 0
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    match_count += 1

            # Calculate score based on pattern matches
            if patterns:
                scores[category] = match_count / len(patterns)

        if scores:
            best_category = max(scores, key=scores.get)
            best_score = scores[best_category]

            # Return high confidence if multiple patterns match
            if best_score >= 0.5:  # At least half the patterns match
                return best_category, 0.9 + (best_score * 0.1)

        return None, 0.0

    def classify(self, text: str) -> Tuple[str, float]:
        """
        Classify a document based on its text content.
        Uses hybrid approach: pattern matching first, then semantic similarity.

        Returns:
            Tuple of (category, confidence_score)
        """
        if not text or len(text.strip()) < 10:
            return "Unclassifiable", 0.0

        # First try pattern-based classification
        pattern_category, pattern_confidence = self._pattern_classify(text)
        if pattern_category and pattern_confidence > 0.8:
            return pattern_category, pattern_confidence

        # Fall back to semantic similarity
        doc_embedding = self.model.encode([text], normalize_embeddings=True)[0]
        similarities = np.dot(self.category_embeddings, doc_embedding)

        # Get the best match
        best_idx = np.argmax(similarities)
        best_category = self.categories[best_idx]
        confidence = float(similarities[best_idx])

        # If pattern had a weaker match, boost confidence for that category
        if pattern_category and pattern_confidence > 0.3:
            pattern_idx = self.categories.index(pattern_category)
            similarities[pattern_idx] += 0.3  # Boost pattern match
            best_idx = np.argmax(similarities)
            best_category = self.categories[best_idx]
            confidence = min(float(similarities[best_idx]), 1.0)

        # Apply confidence threshold
        if confidence < 0.3:
            return "Unclassifiable", confidence

        return best_category, confidence

    def classify_batch(self, documents: Dict[str, str]) -> Dict[str, Tuple[str, float]]:
        """Classify multiple documents."""
        results = {}
        for filename, text in documents.items():
            category, confidence = self.classify(text)
            results[filename] = (category, confidence)
            print(f"  {filename}: {category} (confidence: {confidence:.3f})")
        return results
