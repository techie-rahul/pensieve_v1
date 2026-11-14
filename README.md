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

### ML Pipeline Integration (Phases 1–5 & Enhancements)
- **Phase 1: Emotion Detection** — Fine-tuned RoBERTa on Google GoEmotions (28 emotion classes).
- **Phase 2: Semantic Theme Discovery** — Sentence-BERT (`all-MiniLM-L6-v2`) + UMAP + HDBSCAN topic clustering.
- **Phase 3: Linguistic & Longitudinal Analysis** — spaCy POS and syntactic parsing, pronoun ratios, temporal windowing, trend analysis.
- **Phase 4: Reflective Knowledge Retrieval** — FAISS vector index over the curated 20-concept developmental knowledge base.
- **Phase 5: Grounded Reflection Generation** — Longitudinal synthesis with policy pre-checks, provider isolation (`OpenAIClient` / `MockLLMClient`), confidence capping ($\le 0.80$), and non-diagnostic safety validation.
- **Personalized Longitudinal Patterns** — Multidimensional pattern extraction, circular shifts, cognitive distortion clustering, and evidence-grounded theme tracking.
- **Reflections 2.0** — Personalized emotional arc synthesis, cognitive distortion reframing, philosophical anchors, and conversational reflection prompts.

---

## Getting Started

### Prerequisites
- **Python**: 3.10+ (tested and verified on Python 3.13)
- **Node.js**: 18+ (tested and verified on Node 24)
- **PyTorch**: 2.0+

---

### 1. Backend Setup

1. **Clone repository:**
   ```bash
   git clone https://github.com/techie-rahul/pensieve_v1.git
   cd pensieve_v1
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # On Windows:
   python -m venv venv
   .\venv\Scripts\activate

   # On Linux/macOS:
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment:**
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

5. **(Optional) Seed Demo Dataset:**
   Seed 22 curated historical journal entries across a 30-day longitudinal progression:
   ```bash
   python scripts/seed_demo.py
   # Or with a custom user name:
   python scripts/seed_demo.py --name "Rahul"
   ```
   *Default Demo Credentials:* `demo@pensieve.app` / `DemoUser123!`

6. **Start the FastAPI Backend:**
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```
   The interactive API documentation is available at:
   - **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
   - **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
   - **Health Check**: [http://localhost:8000/api/health](http://localhost:8000/api/health)

---

### 2. Frontend Setup

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install frontend dependencies:**
   ```bash
   npm install
   ```

3. **Start the Vite development server:**
   ```bash
   npm run dev
   ```
   The application will be accessible at [http://localhost:5173](http://localhost:5173).

4. **Build for production:**
   ```bash
   npm run build
   ```

5. **Lint:**
   ```bash
   npm run lint
   ```

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

## Running Verification & Tests

### Backend Test Suite
Run all unit and integration tests across auth, entries, analysis, concepts, personalized patterns, and reflections:
```bash
python -m pytest tests/ -v
```

### Frontend Linting & Production Build
```bash
cd frontend
npm run lint
npm run build
```
