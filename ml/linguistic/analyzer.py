"""Interpretable linguistic feature extraction using spaCy.

Extracts 12 descriptive structural and stylistic metrics per journal entry:
1. Token count
2. Sentence count
3. Average sentence length
4. Vocabulary diversity (Type-Token Ratio)
5. Pronoun usage
6. First-person pronoun frequency
7. Question count
8. Exclamation count
9. Negation count
10. Adjective & adverb frequency
11. Verb frequency
12. POS distribution

No personality profiling or psychological claims are performed.
"""

from typing import Any, Dict, List, Optional
import spacy

DEFAULT_SPACY_MODEL = "en_core_web_sm"

FIRST_PERSON_PRONOUNS = {
    "i", "me", "my", "mine", "myself",
    "we", "us", "our", "ours", "ourselves"
}

NEGATION_WORDS = {
    "no", "not", "n't", "never", "none", "nobody",
    "nothing", "neither", "nowhere", "hardly", "scarcely", "barely"
}


class LinguisticAnalyzer:
    """Extracts interpretable linguistic features from journal texts."""

    def __init__(self, model_name: str = DEFAULT_SPACY_MODEL) -> None:
        """Initialize spaCy English pipeline."""
        self.model_name = model_name
        self.nlp = spacy.load(model_name)

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Extract all 12 linguistic features for a single journal entry.

        Args:
            text: Raw journal entry string.

        Returns:
            Dictionary containing token counts, diversity, pronoun, syntactic, and POS metrics.
        """
        cleaned = text.strip() if text else ""
        if not cleaned:
            return {
                "token_count": 0,
                "word_count": 0,
                "sentence_count": 0,
                "avg_sentence_length": 0.0,
                "vocabulary_diversity": 0.0,
                "pronoun_count": 0,
                "pronoun_ratio": 0.0,
                "first_person_pronoun_count": 0,
                "first_person_pronoun_ratio": 0.0,
                "question_count": 0,
                "exclamation_count": 0,
                "negation_count": 0,
                "negation_ratio": 0.0,
                "adjective_count": 0,
                "adverb_count": 0,
                "modifier_ratio": 0.0,
                "verb_count": 0,
                "verb_ratio": 0.0,
                "pos_distribution": {},
            }

        doc = self.nlp(cleaned)

        # 1. Token & word counts
        total_tokens = len(doc)
        words = [token for token in doc if not token.is_punct and not token.is_space]
        word_count = len(words)
        effective_word_count = max(word_count, 1)

        # 2. Sentence count & average length
        sentences = list(doc.sents)
        sentence_count = max(len(sentences), 1)
        avg_sentence_length = round(word_count / sentence_count, 2)

        # 3. Vocabulary diversity: Type-Token Ratio (unique lower-cased lemmas / total words)
        unique_lemmas = {token.lemma_.lower() for token in words}
        vocab_diversity = round(len(unique_lemmas) / effective_word_count, 4)

        # 4. Pronoun usage & first-person pronoun frequency
        pronouns = [token for token in words if token.pos_ == "PRON"]
        pronoun_count = len(pronouns)
        pronoun_ratio = round(pronoun_count / effective_word_count, 4)

        first_person_tokens = [
            token for token in words
            if token.lower_ in FIRST_PERSON_PRONOUNS or (token.pos_ == "PRON" and token.lower_ in FIRST_PERSON_PRONOUNS)
        ]
        first_person_count = len(first_person_tokens)
        first_person_ratio = round(first_person_count / effective_word_count, 4)

        # 5. Question and exclamation marks
        question_count = sum(1 for sent in sentences if sent.text.strip().endswith("?")) + cleaned.count("?")
        # Normalize question count to avoid double-counting if sentence ends with '?'
        question_count = cleaned.count("?")
        exclamation_count = cleaned.count("!")

        # 6. Negation count
        negations = [
            token for token in words
            if token.lower_ in NEGATION_WORDS or token.dep_ == "neg"
        ]
        negation_count = len(negations)
        negation_ratio = round(negation_count / effective_word_count, 4)

        # 7. Adjectives and adverbs
        adjectives = [token for token in words if token.pos_ == "ADJ"]
        adverbs = [token for token in words if token.pos_ == "ADV"]
        adj_count = len(adjectives)
        adv_count = len(adverbs)
        modifier_ratio = round((adj_count + adv_count) / effective_word_count, 4)

        # 8. Verbs
        verbs = [token for token in words if token.pos_ in ("VERB", "AUX")]
        verb_count = len(verbs)
        verb_ratio = round(verb_count / effective_word_count, 4)

        # 9. Coarse POS distribution
        pos_counts: Dict[str, int] = {}
        for token in words:
            pos_tag = token.pos_
            pos_counts[pos_tag] = pos_counts.get(pos_tag, 0) + 1

        pos_distribution = {
            pos: {"count": count, "ratio": round(count / effective_word_count, 4)}
            for pos, count in sorted(pos_counts.items())
        }

        return {
            "token_count": total_tokens,
            "word_count": word_count,
            "sentence_count": len(sentences),
            "avg_sentence_length": avg_sentence_length,
            "vocabulary_diversity": vocab_diversity,
            "pronoun_count": pronoun_count,
            "pronoun_ratio": pronoun_ratio,
            "first_person_pronoun_count": first_person_count,
            "first_person_pronoun_ratio": first_person_ratio,
            "question_count": question_count,
            "exclamation_count": exclamation_count,
            "negation_count": negation_count,
            "negation_ratio": negation_ratio,
            "adjective_count": adj_count,
            "adverb_count": adv_count,
            "modifier_ratio": modifier_ratio,
            "verb_count": verb_count,
            "verb_ratio": verb_ratio,
            "pos_distribution": pos_distribution,
        }

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Run linguistic analysis on a batch of texts using spaCy pipe."""
        return [self.analyze_text(text) for text in texts]


# Global singleton instance
_DEFAULT_ANALYZER: Optional[LinguisticAnalyzer] = None


def get_linguistic_analyzer() -> LinguisticAnalyzer:
    """Get or create singleton LinguisticAnalyzer instance."""
    global _DEFAULT_ANALYZER
    if _DEFAULT_ANALYZER is None:
        _DEFAULT_ANALYZER = LinguisticAnalyzer()
    return _DEFAULT_ANALYZER


def extract_linguistic_features(text: str) -> Dict[str, Any]:
    """Convenience function for single-text linguistic feature extraction."""
    return get_linguistic_analyzer().analyze_text(text)
