# 📧 MarMail

AI-Powered Multi-Agent Email Marketing Automation (Hackathon Demo)

MarMail is a multi-agent AI system that finds companies matching a target niche, enriches them with contact info, validates the data, and writes personalized outreach emails — all in one click.

Built free-tier friendly: no paid APIs required. Uses DuckDuckGo search for lead discovery and Hugging Face's free Inference Providers for the LLM.

---

## ✨ Features

### 🔍 LeadGen Agent
- Searches the web for companies matching your target niche
- Visits company sites and extracts a public contact email where possible

### 🗄️ DB Agent
- Deduplicates by domain
- Validates every lead (intent: usable record? output: well-formed email?) before storing
- Persists to a local SQLite database

### ✉️ Email Agent
- Writes a personalized subject + body per lead using your offer description
- Retries once if the draft fails validation (empty body, leftover placeholders, etc.)

### 🛡️ Rule-Based Critique
- No LLM tokens spent on validation — plain Python checks for intent + output correctness
- Rejects bad leads and malformed emails before they reach the next stage

### 🧭 Sequence Agent
- Orchestrates LeadGen → DB → Email in order and reports a final campaign summary

---

## 🏗️ Architecture

```
                Sequence Agent
                      │
      ┌───────────────┼───────────────┐
      │               │               │
      ▼               ▼               ▼
  LeadGen Agent    DB Agent      Email Agent
  (search +        (validate +   (personalize +
   enrich)          dedupe +      retry on
                     store)        critique fail)
      │               │               │
      └───────────────┴───────────────┘
                      │
                      ▼
              Campaign Report
```

---

## 🛠️ Tech Stack

- Python
- Streamlit
- LangGraph (`create_react_agent`)
- LangChain Core (tools)
- Hugging Face Inference Providers (free tier)
- DuckDuckGo Search (`duckduckgo-search`)
- SQLite

---

## 🚀 Installation

### Clone / copy the project

```bash
cd marmail-demo
```

### Create a virtual environment

```bash
python -m venv venv
```

### Activate it

Windows:
```bash
venv\Scripts\activate
```

Linux/Mac:
```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Add your Hugging Face token

Create `.streamlit/secrets.toml`:

```toml
HUGGINGFACE_API_KEY = "hf_your_token_here"
```

Get a free token at https://huggingface.co/settings/tokens — create a **Fine-grained** token and tick **Make calls to Inference Providers** under permissions.

### Run it

```bash
streamlit run app.py
```

---

## 📈 How to Use

1. Enter a **target niche** (e.g. "SaaS companies for project management")
2. Enter **what you're offering** (e.g. "AI-powered onboarding tool for new hires")
3. Click **Run MarMail Campaign**
4. Watch the agents work live in the status log
5. Review personalized emails per lead, expand any card to read the full draft
6. Download the campaign report as a `.txt` file
7. Browse stored leads in the SQLite viewer at the bottom

---

## ⚠️ Demo Limitations (by design)

This is a **hackathon-scale demo**, intentionally simplified:

- Emails found are best-effort (scraped from public pages) — not verified like Apollo/Hunter
- No emails are actually sent — this generates drafts only
- No multi-module microservice split, no MCP servers — one Streamlit script, three lightweight agents
- Critique is rule-based (regex/format checks), not a separate LLM call — keeps it fast and free

These trade-offs are exactly right for a demo. See the production notes below if you want to scale this up.

---

## 🔮 Scaling Beyond the Demo

If you want to take this from hackathon demo to production:

- Swap DuckDuckGo search for Apollo/Clearbit/LinkedIn APIs for verified lead data
- Add real email sending via Gmail/Outlook API (OAuth2)
- Move from SQLite to Postgres/MongoDB
- Split LeadGen/DB/Email into separate services if you're processing 10k+ leads/month
- Add an LLM-based Critique pass for intent validation once volume justifies the token cost

---

## 📜 License

MIT License

---

## 👨‍💻 Author

Built as a hackathon demo, inspired by the multi-agent orchestration pattern in [SignalIQ](https://github.com/kk590/SignalIq).
