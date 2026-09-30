# College Helpdesk FAQ Bot

A chatbot that matches free-text questions to the closest FAQ using TF-IDF + cosine similarity — built two ways:

- **`faq_bot.py`** — the "real" NLP version: NLTK for tokenization/stopword removal/lemmatization, scikit-learn for TF-IDF vectorization and cosine similarity. Runs in a terminal.
- **`index.html`** — a standalone browser chat UI with the same matching logic ported to vanilla JavaScript (no backend, so it works directly on GitHub Pages).

## Part 1 — Python (NLTK + scikit-learn)

```bash
pip install nltk scikit-learn
python faq_bot.py
```

First run auto-downloads the small NLTK corpora it needs (punkt, stopwords, wordnet). Then just chat:

```
You: how do I apply for admission?
Bot: Fill out the online application form on the college admissions portal...
```

Type `debug` to see similarity scores for the top 3 matches (useful for tuning), or `quit` to exit.

### How the matching works

1. **Preprocess** — lowercase, strip punctuation, tokenize (`nltk.word_tokenize`), remove stopwords (keeping question words like "how"/"what" since they carry intent), lemmatize (`WordNetLemmatizer`).
2. **Vectorize** — fit a `TfidfVectorizer` on all FAQ questions.
3. **Match** — preprocess the user's question the same way, transform it with the fitted vectorizer, compute cosine similarity against every FAQ question, and return the highest-scoring one — as long as it clears a similarity threshold (default `0.25`). Below that, the bot admits it doesn't know rather than guessing.

## Part 2 — Chat UI (`index.html`)

Open `index.html` in a browser (or deploy on GitHub Pages) and `faqs.json` loads automatically. The matching logic is the same TF-IDF + cosine similarity approach, reimplemented in plain JavaScript so no server or Python runtime is needed.

## Files

| File | Purpose |
|---|---|
| `faqs.json` | The FAQ dataset — 24 Q&A pairs about admissions, fees, hostel, exams, library, placements |
| `faq_bot.py` | Python CLI chatbot (NLTK + scikit-learn) |
| `index.html` | Browser chat UI (vanilla JS) |
| `README.md` | This file |

## Deploying the chat UI on GitHub Pages

Same steps as before:
1. Create a repo, upload `index.html` and `faqs.json` (both — the page fetches the JSON at runtime).
2. Settings → Pages → Deploy from branch → `main` / root → Save.
3. Visit the URL GitHub gives you.

## Using your own FAQs

Edit `faqs.json` — it's just an array of `{"question": "...", "answer": "..."}` objects. Both `faq_bot.py` and `index.html` will pick up any new set automatically; no code changes needed.

## Notes on the matching quality

- Cosine similarity over TF-IDF is a solid, transparent baseline (this is a classic technique from information retrieval), but it can't infer meaning it hasn't seen the words for — e.g. it won't recognize "money I owe the college" as related to "fees" unless a shared word appears. For a stronger match, sentence-embedding models (e.g. `sentence-transformers`) would generalize better, at the cost of a heavier dependency.
- The `similarity_threshold` (`0.25` in Python, `0.18` in JS — the two implementations tokenize slightly differently, hence the different tuning) controls how strict the bot is about admitting "I don't know." Lower it to get more (possibly wrong) answers; raise it to get more honest "I'm not sure" responses.
