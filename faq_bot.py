"""
FAQ Chatbot — College Helpdesk
================================
Loads a set of FAQ question/answer pairs, preprocesses the questions with
NLTK (tokenization, stopword removal, lemmatization), and matches a user's
free-text question to the most similar FAQ using TF-IDF + cosine similarity.

Usage:
    python faq_bot.py

Requirements:
    pip install nltk scikit-learn
    (first run will auto-download the small NLTK corpora it needs)
"""

import json
import string
import sys

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# One-time NLTK data setup
# ---------------------------------------------------------------------------
REQUIRED_NLTK_DATA = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
]

def ensure_nltk_data():
    for path, name in REQUIRED_NLTK_DATA:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)

ensure_nltk_data()

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()
PUNCT_TABLE = str.maketrans("", "", string.punctuation)

# Keep a few question words that matter for intent even though they're
# normally stopwords (e.g. "how", "what", "can", "where") — stripping them
# loses useful signal for short FAQ-style questions.
KEEP_WORDS = {"how", "what", "when", "where", "why", "who", "which", "can", "do", "does"}
STOP_WORDS = STOP_WORDS - KEEP_WORDS


def preprocess(text: str) -> str:
    """Lowercase, strip punctuation, tokenize, remove stopwords, lemmatize.
    Returns a cleaned string of space-separated lemmas (ready for TF-IDF)."""
    text = text.lower().translate(PUNCT_TABLE)
    tokens = word_tokenize(text)
    cleaned = [
        LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if tok not in STOP_WORDS and tok.strip()
    ]
    return " ".join(cleaned)


class FAQBot:
    def __init__(self, faqs_path: str, similarity_threshold: float = 0.25):
        with open(faqs_path, "r", encoding="utf-8") as f:
            self.faqs = json.load(f)

        self.threshold = similarity_threshold
        self.questions = [item["question"] for item in self.faqs]
        self.answers = [item["answer"] for item in self.faqs]

        # Preprocess every FAQ question once, up front.
        self.processed_questions = [preprocess(q) for q in self.questions]

        # Fit TF-IDF on the FAQ question set.
        self.vectorizer = TfidfVectorizer()
        self.faq_matrix = self.vectorizer.fit_transform(self.processed_questions)

    def match(self, user_question: str):
        """Return (answer, matched_question, score) for the best FAQ match."""
        cleaned = preprocess(user_question)
        if not cleaned:
            return None, None, 0.0

        user_vec = self.vectorizer.transform([cleaned])
        scores = cosine_similarity(user_vec, self.faq_matrix)[0]

        best_idx = scores.argmax()
        best_score = scores[best_idx]

        if best_score < self.threshold:
            return None, None, best_score

        return self.answers[best_idx], self.questions[best_idx], best_score

    def top_matches(self, user_question: str, k: int = 3):
        """Return the top-k (question, answer, score) matches — useful for
        debugging / tuning the similarity threshold."""
        cleaned = preprocess(user_question)
        user_vec = self.vectorizer.transform([cleaned])
        scores = cosine_similarity(user_vec, self.faq_matrix)[0]
        ranked = sorted(
            zip(self.questions, self.answers, scores),
            key=lambda x: x[2],
            reverse=True,
        )
        return ranked[:k]


FALLBACK_MESSAGE = (
    "I'm not confident I have a good answer for that. Could you rephrase, "
    "or try asking about admissions, fees, hostel, exams, library, or placements?"
)


def run_cli(faqs_path: str = "faqs.json"):
    bot = FAQBot(faqs_path)
    print("=" * 60)
    print("College Helpdesk FAQ Bot — type 'quit' to exit, 'debug' to")
    print("toggle showing similarity scores for the top matches.")
    print("=" * 60)

    debug = False
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("Bot: Goodbye!")
            break
        if user_input.lower() == "debug":
            debug = not debug
            print(f"Bot: Debug mode {'on' if debug else 'off'}.")
            continue

        if debug:
            print("\n[Top 3 matches]")
            for q, a, score in bot.top_matches(user_input):
                print(f"  {score:.3f}  {q}")

        answer, matched_q, score = bot.match(user_input)
        if answer:
            print(f"Bot: {answer}")
        else:
            print(f"Bot: {FALLBACK_MESSAGE}")


if __name__ == "__main__":
    faqs_file = sys.argv[1] if len(sys.argv) > 1 else "faqs.json"
    run_cli(faqs_file)
