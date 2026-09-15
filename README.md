# Pensieve — Privacy-First Reflective Journaling

Pensieve is a privacy-first reflective journaling platform that analyzes personal journal entries longitudinally to uncover emotional shifts, semantic themes, and linguistic patterns, grounding reflections in curated psychological and philosophical frameworks.

---

## System Architecture

```
                               ┌────────────────────────┐
                               │ React + Vite Frontend  │
                               └───────────┬────────────┘
                                           │ HTTP / JSON (CORS)
                                           ▼
                               ┌────────────────────────┐
                               │    FastAPI Backend     │
                               │  (app/main.py : 8000)  │
                               └─────┬────────────┬─────┘
                                     │            │
            ┌────────────────────────┘            └────────────────────────┐
            ▼                                                              ▼
┌────────────────────────┐                                   ┌────────────────────────┐
│  SQLite + SQLAlchemy   │                                   │   Pensieve ML Engine   │
│     (pensieve.db)      │                                   │       (ml/ package)    │
└────────────────────────┘                                   └────────────────────────┘
```

### ML Pipeline Integration (Phases 1–5)
- **Phase 1: Emotion Detection** — Fine-tuned RoBERTa on Google GoEmotions (28 emotion classes).
- **Phase 2: Semantic Theme Discovery** — Sentence-BERT (`all-MiniLM-L6-v2`) + UMAP + HDBSCAN.
- **Phase 3: Linguistic & Longitudinal Analysis** — spaCy POS and syntactic parsing, pronoun ratios, temporal windowing, trend analysis.
- **Phase 4: Reflective Knowledge Retrieval** — FAISS vector index over the curated 20-concept DEVELOPMENT/TEST knowledge base.
- **Phase 5: Grounded Reflection Generation** — Longitudinal synthesis with policy pre-checks, provider isolation (`OpenAIClient` / `MockLLMClient`), confidence capping ($\le 0.80$), and non-diagnostic safety validation.

---

## Getting Started

### Prerequisites
- Python 3.10+ (tested and verified on Python 3.13)
- PyTorch 2.0+

### 1. Installation
Clone the repository and install all dependencies:

```bash
# Clone repository
git clone https://github.com/techie-rahul/pensieve-v1.git
cd pensieve-v1

# Create and activate virtual environment (optional but recommended)
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r ml/requirements.txt
```

### 2. Environment Configuration
Copy the sample environment file and adjust if necessary:

```bash
cp .env.example .env
```

Available configuration options in `.env`:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `DATABASE_URL` | `sqlite:///./pensieve.db` | SQLAlchemy database connection string |
| `SECRET_KEY` | `development-secret-key-change-in-production` | Secret key for signing JWT tokens |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `1440` | JWT expiration duration in minutes (24h) |
| `CORS_ORIGINS` | `http://localhost:5173,http://localhost:3000` | Allowed origins for React frontend |
| `OPENAI_API_KEY` | *(optional)* | OpenAI API key for live LLM reflections (defaults to mock generator if unset) |

### 3. Run the Backend Server
Launch the FastAPI application with Uvicorn:

```bash
uvicorn app.main:app --reload --port 8000
```

The interactive API documentation is available at:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Health Check**: [http://localhost:8000/api/health](http://localhost:8000/api/health)

---

## API Endpoints

### Authentication
- `POST /api/auth/register` — Register a new user account (Argon2id password hashing).
- `POST /api/auth/login` — Authenticate credentials and acquire a JWT access token.
- `GET /api/auth/me` — Inspect profile of currently authenticated user.

### Journal Entries
- `GET /api/entries` — List entries for current user (supports `?is_draft=true/false`, pagination).
- `POST /api/entries` — Create a new journal entry.
- `GET /api/entries/{id}` — Retrieve a single journal entry (strict ownership enforcement).
- `PUT /api/entries/{id}` — Update an entry's title, content, mood, or tags.
- `DELETE /api/entries/{id}` — Delete a journal entry and its attached analysis.
- `POST /api/entries/autosave` — Autosave a draft entry without creating duplicates.

### ML Analysis & Patterns
- `POST /api/analyze/{entry_id}` — Run Phase 1–3 ML analysis on an entry (RoBERTa emotions + Sentence-BERT theme + spaCy linguistics).
- `GET /api/patterns` — Run Phase 3 longitudinal pattern analysis across chronological entries.

### Reflections
- `POST /api/reflections/suggest` — Generate a grounded reflection from longitudinal signals (Phases 4–5).
- `GET /api/reflections` — List saved reflections for current user.
- `GET /api/reflections/{id}` — Retrieve a single reflection by ID.

### Knowledge Base Concepts
- `GET /api/concepts` — Browse the curated 20-concept development knowledge base (supports `?category=...` and `?search=...`).
- `GET /api/concepts/{concept_id}` — Retrieve full concept documentation, academic/philosophical citations, and safety cautions.

---

## Running Tests

Run the complete test suite:

```bash
pytest tests/ -v
```

Run end-to-end user lifecycle verification:

```bash
python .gemini/antigravity/brain/1bd87209-1a75-4d60-bea6-af74a99f8428/scratch/verify_backend_e2e.py
```
