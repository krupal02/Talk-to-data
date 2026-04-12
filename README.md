# Talk to Data – Seamless Self-Service Intelligence

> Upload a dataset. Ask questions in plain English. Get clear answers with charts and full source transparency.

Talk to Data is a web application built for the NatWest "Code for Purpose – India Hackathon". It enables any user — regardless of technical skill — to upload a CSV or SQLite dataset and ask natural-language questions about it. The system returns plain-language insights, auto-generated visualisations, and full source provenance so that every answer is trustworthy and traceable.

---

## Features

- **Natural-language querying** – ask questions like "Why did revenue drop last month?" and receive clear, jargon-free answers
- **Four intent types** – the system automatically detects whether you want to understand a change, compare groups, break down a total, or get a summary
- **Auto-visualisation** – every answer includes a contextual chart (bar, pie, or line) powered by Recharts
- **Source transparency** – every insight cites the exact rows, columns, and metric definitions used
- **PII auto-stripping** – columns containing emails, phone numbers, SSNs, and similar PII are automatically removed before any processing
- **Semantic layer** – consistent metric definitions ensure "revenue", "sales", and "income" always compute the same way
- **Privacy-first design** – only aggregated data (never raw rows) is sent to the Gemini API
- **Confidence notes** – each response includes a one-line note like "Based on 3,200 rows in sales_q1.csv"

---

## Install and Run

### Prerequisites
- Python 3.11+
- Node.js 18+ and npm
- A Google Gemini API key 

### Backend

```bash
git clone <repo-url>
cd talk-to-data

# Install Python dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# Start the backend server
cd backend
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The UI will be available at `http://localhost:5173`.

---

## Tech Stack

| Layer     | Technology                          |
|-----------|-------------------------------------|
| Backend   | Python 3.11, FastAPI                |
| AI        | Google Gemini API (gemini-2.5-flash) |
| Data      | pandas (CSV), sqlite3 (DB)          |
| Frontend  | React (Vite), Recharts, Tailwind CSS|
| Env mgmt  | python-dotenv                       |

---

## Usage Examples

### Example 1: Understanding a change
**Question:** "Why did revenue drop last month?"

**Response:**
> Revenue decreased by 11% in February. The biggest contributor was a 22% drop in the South region, likely due to reduced ad spend. North and East regions remained relatively stable.
>
> Source: rows 1–3,200, column: revenue | Metric: revenue (sum)

### Example 2: Breakdown
**Question:** "What makes up total sales?"

**Response:**
> North region accounts for 40% of total sales ($22,000), followed by East at 28% ($20,500). South and West contribute the remaining 32%.
>
> Source: rows 1–48, column: revenue | Metric: revenue (sum)

---

## Architecture

The system follows a modular pipeline:

```
Upload → Sanitise → Intent Parser → Semantic Layer → Data Engine → Gemini API → Response
```

Key design decisions:
- **Gemini is used only for language** — all numerical computation happens locally in pandas
- **Raw data never enters the Gemini prompt** — only aggregated summaries are sent (privacy + token efficiency)
- **The semantic layer** normalises business terminology into canonical metrics with defined aggregations

For a detailed explanation, see [docs/architecture.md](docs/architecture.md).

---

## Limitations

- Maximum dataset size: 100,000 rows (configurable via `MAX_UPLOAD_ROWS`)
- No real-time database connections — upload-only model
- The semantic layer is manually configured (supports ~6 common metrics out of the box)
- No authentication or user management system
- In-memory session store — sessions are lost on server restart
- File size limit: 50 MB

---

## Future Improvements

- **Multi-modal charts** – allow users to choose chart types or auto-detect the best visualisation
- **Auto-detecting metric definitions** – infer aggregation types from column data patterns
- **Streaming responses** – use server-sent events for real-time answer generation
- **Persistent sessions** – Redis-backed session store with TTL for production deployments
- **Multi-table support** – allow joining multiple uploaded files or tables
- **Export to PDF/Excel** – one-click export of insights and charts
- **Natural-language follow-up** – conversation memory for iterative data exploration

---

## License

Licensed under the Apache License 2.0. See individual file headers for details.
