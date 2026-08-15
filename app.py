import os
import re
import json
import sqlite3
import traceback
from datetime import datetime
from urllib.parse import urlparse
from typing import List, Tuple

import boto3
import streamlit as st
from langchain_aws import ChatBedrock
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# ── Config ─────────────────────────────────────────────────────
st.set_page_config(page_title="MarMail", page_icon="📧", layout="wide")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
DB_PATH = os.environ.get("SQLITE_PATH", "marmail_leads.db")

# ═══════════════════════════════════════════════════════════════
#  AWS CREDENTIAL CHECK
# ═══════════════════════════════════════════════════════════════

def check_aws_credentials() -> Tuple[bool, str]:
    """Check if AWS credentials are available for Bedrock."""
    try:
        session = boto3.Session(region_name=AWS_REGION)
        creds = session.get_credentials()
        if creds is None:
            return False, "No AWS credentials found. Run 'aws configure' or set env vars."
        # Try a lightweight Bedrock call
        client = session.client("bedrock-runtime", region_name=AWS_REGION)
        client.list_foundation_models(byInferenceType="ON_DEMAND")
        return True, "AWS Bedrock accessible"
    except Exception as e:
        return False, str(e)


# ═══════════════════════════════════════════════════════════════
#  LLM BUILDER
# ═══════════════════════════════════════════════════════════════

def build_llm(model_id: str, region: str):
    bedrock_client = boto3.client("bedrock-runtime", region_name=region)
    return ChatBedrock(
        model_id=model_id,
        client=bedrock_client,
        model_kwargs={"temperature": 0.4, "max_tokens": 800},
    )


# ═══════════════════════════════════════════════════════════════
#  LOCAL TOOL IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════

import requests
from duckduckgo_search import DDGS
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ── LeadGen ────────────────────────────────────────────────────
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GOOGLE_CSE_ID = os.environ.get("GOOGLE_CSE_ID", "")
BING_API_KEY = os.environ.get("BING_API_KEY", "")
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


def search_google(query: str, num_results: int = 5) -> str:
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        return "❌ Google API not configured."
    try:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {"key": GOOGLE_API_KEY, "cx": GOOGLE_CSE_ID, "q": query, "num": min(num_results, 10)}
        data = requests.get(url, params=params, timeout=12).json()
        items = data.get("items", [])
        if not items:
            return "No results."
        lines = [f"{i+1}. {it.get('title','')}\n   URL: {it.get('link','')}\n   {it.get('snippet','')[:180]}" for i, it in enumerate(items)]
        return "\n\n".join(lines)
    except Exception as e:
        return f"❌ Google error: {e}"


def search_bing(query: str, num_results: int = 5) -> str:
    if not BING_API_KEY:
        return "❌ Bing API not configured."
    try:
        endpoint = "https://api.bing.microsoft.com/v7.0/search"
        headers = {"Ocp-Apim-Subscription-Key": BING_API_KEY}
        params = {"q": query, "count": min(num_results, 10)}
        data = requests.get(endpoint, headers=headers, params=params, timeout=12).json()
        items = data.get("webPages", {}).get("value", [])
        if not items:
            return "No results."
        lines = [f"{i+1}. {it.get('name','')}\n   URL: {it.get('url','')}\n   {it.get('snippet','')[:180]}" for i, it in enumerate(items)]
        return "\n\n".join(lines)
    except Exception as e:
        return f"❌ Bing error: {e}"


def search_duckduckgo(query: str, num_results: int = 8) -> str:
    """Search DuckDuckGo with detailed error logging."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results))
        if not results:
            return "No results found."
        lines = []
        for i, r in enumerate(results, 1):
            title = r.get('title', '')
            href = r.get('href', '')
            body = r.get('body', '')[:180]
            lines.append(f"{i}. {title}\n   URL: {href}\n   Info: {body}")
        return "\n\n".join(lines)
    except Exception as e:
        return f"❌ DuckDuckGo error: {type(e).__name__}: {e}"


def extract_contact_email(website_url: str) -> str:
    try:
        text = requests.get(website_url, headers=HEADERS, timeout=10).text
        emails = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
        bad = ["png", "jpg", "jpeg", "gif", "example.com", "sentry.io", "w3.org", "github.com"]
        emails = [e for e in emails if not any(b in e.lower() for b in bad)]
        if emails:
            return f"✅ Found: {emails[0]}"
        domain = urlparse(website_url).netloc.replace("www.", "")
        return f"⚠️ No email found. Guess: contact@{domain}"
    except Exception as e:
        return f"❌ Error: {e}"


# ── DB ─────────────────────────────────────────────────────────
COCKROACH_DB_URL = os.environ.get("COCKROACH_DB_URL", "")
USE_PG = bool(COCKROACH_DB_URL)


def db_conn():
    if USE_PG:
        import psycopg2
        return psycopg2.connect(COCKROACH_DB_URL)
    return sqlite3.connect(DB_PATH)


def db_ph():
    return "%s" if USE_PG else "?"


def db_init_schema() -> str:
    conn = db_conn()
    try:
        cur = conn.cursor()
        if USE_PG:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS leads (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    company STRING NOT NULL, website STRING NOT NULL UNIQUE,
                    contact_email STRING NOT NULL, snippet STRING,
                    email_subject STRING, email_body STRING, status STRING,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """)
        else:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS leads (
                    id TEXT PRIMARY KEY, company TEXT NOT NULL, website TEXT NOT NULL UNIQUE,
                    contact_email TEXT NOT NULL, snippet TEXT,
                    email_subject TEXT, email_body TEXT, status TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
            """)
        conn.commit()
        return "✅ Schema ready"
    except Exception as e:
        return f"❌ Schema error: {e}"
    finally:
        conn.close()


def db_store_lead(company: str, website: str, contact_email: str, snippet: str = "", status: str = "found") -> str:
    conn = db_conn()
    ph = db_ph()
    try:
        cur = conn.cursor()
        if USE_PG:
            cur.execute(
                f"INSERT INTO leads (company, website, contact_email, snippet, status) VALUES ({ph},{ph},{ph},{ph},{ph}) ON CONFLICT (website) DO NOTHING RETURNING id",
                (company, website, contact_email, snippet, status),
            )
            ok = cur.fetchone() is not None
        else:
            try:
                cur.execute(f"INSERT INTO leads (company, website, contact_email, snippet, status) VALUES ({ph},{ph},{ph},{ph},{ph})",
                    (company, website, contact_email, snippet, status))
                ok = True
            except sqlite3.IntegrityError:
                ok = False
        conn.commit()
        return f"✅ Stored {company}" if ok else f"↩️ Duplicate: {website}"
    except Exception as e:
        return f"❌ DB error: {e}"
    finally:
        conn.close()


def db_update_lead_email(website: str, email_subject: str, email_body: str) -> str:
    conn = db_conn()
    ph = db_ph()
    try:
        cur = conn.cursor()
        cur.execute(f"UPDATE leads SET email_subject={ph}, email_body={ph} WHERE website={ph}", (email_subject, email_body, website))
        conn.commit()
        return f"✅ Updated {website}" if cur.rowcount else f"⚠️ Not found: {website}"
    except Exception as e:
        return f"❌ Update error: {e}"
    finally:
        conn.close()


# ── Email ──────────────────────────────────────────────────────
SMTP_HOST = os.environ.get("SMTP_HOST", "")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASS = os.environ.get("SMTP_PASS", "")
YAHOO_EMAIL = os.environ.get("YAHOO_EMAIL", SMTP_USER)
YAHOO_APP_PASSWORD = os.environ.get("YAHOO_APP_PASSWORD", SMTP_PASS)
GMAIL_TOKEN_FILE = os.environ.get("GMAIL_TOKEN_FILE", "")


def smtp_send(to_email, subject, body, from_email, host, port, user, password):
    msg = MIMEMultipart()
    msg["From"] = from_email or user
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    with smtplib.SMTP(host, port, timeout=15) as server:
        server.starttls()
        server.login(user, password)
        server.send_message(msg)


def email_smtp(to_email: str, subject: str, body: str, from_email: str = "", provider: str = "generic") -> str:
    presets = {"gmail": ("smtp.gmail.com", 587), "outlook": ("smtp.office365.com", 587),
               "yahoo": ("smtp.mail.yahoo.com", 587), "generic": (SMTP_HOST, SMTP_PORT)}
    host, port = presets.get(provider, (SMTP_HOST, SMTP_PORT))
    user, password = SMTP_USER, SMTP_PASS
    if provider == "yahoo":
        user, password = YAHOO_EMAIL or user, YAHOO_APP_PASSWORD or password
    if not user or not password:
        return f"❌ Credentials missing for {provider}"
    try:
        smtp_send(to_email, subject, body, from_email, host, port, user, password)
        return f"✅ Sent to {to_email} via {provider}"
    except Exception as e:
        return f"❌ Failed: {e}"


def email_gmail(to_email: str, subject: str, body: str, from_email: str = "") -> str:
    try:
        if not GMAIL_TOKEN_FILE or not os.path.exists(GMAIL_TOKEN_FILE):
            return email_smtp(to_email, subject, body, from_email, provider="gmail")
        import base64
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        creds = Credentials.from_authorized_user_file(GMAIL_TOKEN_FILE, ["https://www.googleapis.com/auth/gmail.send"])
        service = build("gmail", "v1", credentials=creds)
        message = MIMEText(body)
        message["to"] = to_email; message["subject"] = subject; message["from"] = from_email or SMTP_USER
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        service.users().messages().send(userId="me", body={"raw": raw}).execute()
        return f"✅ Sent to {to_email} via Gmail API"
    except Exception as e:
        return f"❌ Gmail API failed: {e}"


def email_outlook(to_email: str, subject: str, body: str, from_email: str = "") -> str:
    return email_smtp(to_email, subject, body, from_email, provider="outlook")


def email_yahoo(to_email: str, subject: str, body: str, from_email: str = "") -> str:
    return email_smtp(to_email, subject, body, from_email, provider="yahoo")


# ═══════════════════════════════════════════════════════════════
#  LANGCHAIN TOOLS (for the agent)
# ═══════════════════════════════════════════════════════════════

@tool
def tool_search_google(query: str, num_results: int = 5) -> str:
    """Search Google using Custom Search JSON API."""
    return search_google(query, num_results)

@tool
def tool_search_bing(query: str, num_results: int = 5) -> str:
    """Search Bing using Azure Web Search API."""
    return search_bing(query, num_results)

@tool
def tool_search_duckduckgo(query: str, num_results: int = 8) -> str:
    """Search DuckDuckGo — free, no API key required."""
    return search_duckduckgo(query, num_results)

@tool
def tool_extract_contact_email(website_url: str) -> str:
    """Scrape a public contact email from a company website."""
    return extract_contact_email(website_url)

@tool
def tool_store_lead(company: str, website: str, contact_email: str, snippet: str = "", status: str = "found") -> str:
    """Store a lead into the database. Skips duplicates by website."""
    return db_store_lead(company, website, contact_email, snippet, status)

@tool
def tool_update_lead_email(website: str, email_subject: str, email_body: str) -> str:
    """Attach a personalized email to an already-stored lead."""
    return db_update_lead_email(website, email_subject, email_body)

@tool
def tool_send_email_smtp(to_email: str, subject: str, body: str, from_email: str = "", provider: str = "generic") -> str:
    """Send email via SMTP. Provider can be generic, gmail, outlook, or yahoo."""
    return email_smtp(to_email, subject, body, from_email, provider)


AGENT_TOOLS = [
    tool_search_google, tool_search_bing, tool_search_duckduckgo, tool_extract_contact_email,
    tool_store_lead, tool_update_lead_email, tool_send_email_smtp,
]


# ═══════════════════════════════════════════════════════════════
#  CRITIQUE & LOOP
# ═══════════════════════════════════════════════════════════════

def critique_lead(lead: dict) -> Tuple[bool, str]:
    if not lead.get("company"):
        return False, "Missing company name"
    if not lead.get("website"):
        return False, "Missing website"
    email = (lead.get("contact_email") or "").strip()
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        return False, f"Invalid email: {email}"
    return True, "OK"


def critique_email(subject: str, body: str) -> Tuple[bool, str]:
    if not subject or len(subject) < 5:
        return False, "Subject too short"
    if not body or len(body) < 40:
        return False, "Body too short"
    if "{{" in body or "[your" in body.lower():
        return False, "Unfilled placeholders"
    return True, "OK"


# ═══════════════════════════════════════════════════════════════
#  MODULE AGENTS
# ═══════════════════════════════════════════════════════════════

class LeadGenModule:
    def __init__(self, llm, log, demo_mode: bool = False):
        self.llm = llm
        self.log = log
        self.demo_mode = demo_mode
        self.agent = None
        if llm and not demo_mode:
            try:
                self.agent = create_react_agent(
                    llm,
                    tools=[tool_search_google, tool_search_bing, tool_search_duckduckgo, tool_extract_contact_email],
                    prompt=(
                        "You are the LeadGen agent. Search for companies matching the niche, "
                        "extract contact emails from their websites, and return a structured list: "
                        "Company name | Website | Email. Use multiple search engines if needed."
                    ),
                )
                self.log("   🤖 LeadGen agent compiled successfully")
            except Exception as e:
                self.log(f"   ⚠️ Agent compilation failed: {e}")

    def run(self, niche: str, max_leads: int) -> List[dict]:
        self.log(f"🔍 LeadGen: searching for '{niche}'")

        # ── Demo mode: return mock data ────────────────────────
        if self.demo_mode:
            self.log("   🎮 DEMO MODE: returning mock leads")
            mock = [
                {"company": "Asana", "website": "https://asana.com", "contact_email": "sales@asana.com", "snippet": "Project management software", "status": "found"},
                {"company": "Monday", "website": "https://monday.com", "contact_email": "contact@monday.com", "snippet": "Work management platform", "status": "found"},
                {"company": "ClickUp", "website": "https://clickup.com", "contact_email": "hello@clickup.com", "snippet": "All-in-one productivity platform", "status": "found"},
                {"company": "Notion", "website": "https://notion.so", "contact_email": "team@makenotion.com", "snippet": "Connected workspace", "status": "found"},
                {"company": "Trello", "website": "https://trello.com", "contact_email": "support@trello.com", "snippet": "Visual collaboration tool", "status": "found"},
            ]
            return mock[:max_leads]

        # ── Try agent first ────────────────────────────────────
        raw_text = ""
        if self.agent:
            try:
                result = self.agent.invoke({
                    "messages": [(
                        "user",
                        f"Find {max_leads} companies matching: '{niche}'. "
                        f"For each, find the website and a contact email."
                    )]
                })
                raw_text = result["messages"][-1].content
                self.log(f"   ✅ Agent returned {len(raw_text)} chars")
            except Exception as e:
                self.log(f"   ⚠️ Agent invoke failed: {e}")

        # ── Fallback: manual search ────────────────────────────
        if not raw_text or len(raw_text) < 100:
            self.log("   🔄 Falling back to manual search...")
            # Try multiple queries
            queries = [
                f"{niche} official website",
                f"top {niche}",
                niche,
            ]
            for q in queries:
                self.log(f"      Trying query: '{q}'")
                raw_text = search_duckduckgo(q, num_results=max_leads * 2)
                self.log(f"      Result length: {len(raw_text)} chars")
                if len(raw_text) > 100:
                    break
                # Also try Google/Bing if configured
                if GOOGLE_API_KEY and GOOGLE_CSE_ID:
                    raw_text = search_google(q, num_results=max_leads)
                    if len(raw_text) > 100:
                        break
                if BING_API_KEY:
                    raw_text = search_bing(q, num_results=max_leads)
                    if len(raw_text) > 100:
                        break

        # Show first 500 chars of raw for debugging
        preview = raw_text[:500].replace("\n", " ")
        self.log(f"   📄 Raw preview: {preview}...")

        leads = self._parse(raw_text, max_leads)
        self.log(f"   📊 Parsed {len(leads)} candidates")

        # ── Enrich with emails ─────────────────────────────────
        enriched = []
        for lead in leads:
            if lead["contact_email"].startswith("contact@"):
                self.log(f"   🔍 Scraping email for {lead['website']}...")
                email_result = extract_contact_email(lead["website"])
                self.log(f"      {email_result}")
                if email_result.startswith("✅"):
                    lead["contact_email"] = email_result.replace("✅ Found: ", "").strip()
                    lead["status"] = "found"
            enriched.append(lead)

        # ── Critique ───────────────────────────────────────────
        valid = []
        for lead in enriched:
            ok, reason = critique_lead(lead)
            if ok:
                valid.append(lead)
            else:
                self.log(f"   ⛔ Rejected {lead.get('company','?')}: {reason}")
        self.log(f"✅ LeadGen: {len(valid)} leads passed critique")
        return valid

    def _parse(self, raw_text: str, max_leads: int) -> List[dict]:
        leads = []
        lines = [l.strip("-• ") for l in raw_text.split("\n") if l.strip()]
        seen = set()
        for line in lines:
            if len(leads) >= max_leads:
                break
            url_match = re.search(r"https?://[^\s\)\]]+", line)
            if not url_match:
                continue
            url = url_match.group(0).rstrip(".,)")
            domain = urlparse(url).netloc.replace("www.", "")
            if not domain or domain in seen or "." not in domain:
                continue
            seen.add(domain)
            email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", line)
            email = email_match.group(0) if email_match else f"contact@{domain}"
            # Better company name extraction
            parts = domain.split(".")
            company_guess = parts[0].capitalize() if parts else "Unknown"
            leads.append({
                "company": company_guess,
                "website": url,
                "contact_email": email,
                "snippet": line[:200],
                "status": "guess" if email.startswith("contact@") else "found",
            })
        return leads


class DBModule:
    def __init__(self, log):
        self.log = log
        result = db_init_schema()
        self.log(f"   🗄️ {result}")

    def run(self, leads: List[dict]) -> List[dict]:
        self.log("🗄️ DB: storing leads")
        stored = []
        for lead in leads:
            ok, reason = critique_lead(lead)
            if not ok:
                self.log(f"   ⛔ Rejected: {lead.get('company','?')} — {reason}")
                continue
            result = db_store_lead(
                lead["company"], lead["website"], lead["contact_email"],
                lead.get("snippet", ""), lead.get("status", "found")
            )
            self.log(f"   {result}")
            if "✅" in result or "↩️" in result:
                stored.append(lead)
        self.log(f"✅ DB: {len(stored)} leads stored")
        return stored

    def update_emails(self, leads: List[dict]):
        self.log("🗄️ DB: attaching personalized emails")
        for lead in leads:
            result = db_update_lead_email(
                lead["website"],
                lead.get("email_subject", ""),
                lead.get("email_body", "")
            )
            self.log(f"   {result}")


class EmailModule:
    def __init__(self, llm, log, demo_mode: bool = False):
        self.llm = llm
        self.log = log
        self.demo_mode = demo_mode

    def run(self, leads: List[dict], offer: str, email_provider: str = "smtp") -> List[dict]:
        self.log("✉️ Email: personalizing & sending")
        results = []
        for lead in leads:
            # ── Generate email ─────────────────────────────────
            if self.demo_mode:
                subject = f"Quick question about {lead['company']}"
                body = (
                    f"Hi {lead['company']} team,\n\n"
                    f"I came across {lead['website']} and was impressed by your work in the space. "
                    f"We're offering {offer} and I believe it could be a great fit. "
                    f"Would you be open to a brief 10-minute chat next week?\n\n"
                    f"Best regards"
                )
            else:
                prompt = (
                    "Write a short, personalized cold outreach email.\n"
                    f"Recipient company: {lead['company']}\n"
                    f"Website: {lead['website']}\n"
                    f"Context: {lead.get('snippet', '')}\n"
                    f"What we're offering: {offer}\n\n"
                    "Respond exactly as:\n"
                    "Subject: <subject line>\n"
                    "Body: <3-5 sentence body, no placeholders>"
                )
                subject, body = "", ""
                for attempt in range(2):
                    try:
                        resp = self.llm.invoke(prompt)
                        text = resp.content if hasattr(resp, "content") else str(resp)
                        subj_match = re.search(r"Subject:\s*(.+)", text)
                        body_match = re.search(r"Body:\s*(.+)", text, re.DOTALL)
                        subject = subj_match.group(1).strip() if subj_match else ""
                        body = body_match.group(1).strip() if body_match else text.strip()
                        ok, reason = critique_email(subject, body)
                        if ok:
                            break
                        self.log(f"   🔁 Retry for {lead['company']}: {reason}")
                    except Exception as e:
                        self.log(f"   ⚠️ LLM failed: {e}")
                        break

                subject = subject or f"Quick question for {lead['company']}"
                body = body or f"Hi {lead['company']} team,\n\nWe're reaching out about {offer}. Would love to connect.\n\nBest"

            lead["email_subject"] = subject
            lead["email_body"] = body

            # ── Send ─────────────────────────────────────────────
            if self.demo_mode:
                lead["delivery_status"] = "🎮 DEMO: not sent"
                self.log(f"   🎮 DEMO: skipped send to {lead['contact_email']}")
            else:
                sender_map = {
                    "gmail": email_gmail, "outlook": email_outlook,
                    "yahoo": email_yahoo, "smtp": email_smtp,
                }
                sender_fn = sender_map.get(email_provider, email_smtp)
                try:
                    send_result = sender_fn(lead["contact_email"], subject, body)
                    self.log(f"   {send_result}")
                    lead["delivery_status"] = send_result
                except Exception as e:
                    self.log(f"   ❌ Send failed: {e}")
                    lead["delivery_status"] = f"❌ Error: {e}"

            results.append(lead)
        self.log(f"✅ Email: processed {len(results)} leads")
        return results


# ═══════════════════════════════════════════════════════════════
#  SEQUENCE AGENT
# ═══════════════════════════════════════════════════════════════

class SequenceAgent:
    def __init__(self, llm, log, demo_mode: bool = False):
        self.log = log
        self.leadgen = LeadGenModule(llm, log, demo_mode)
        self.db = DBModule(log)
        self.email = EmailModule(llm, log, demo_mode)

    def run(self, niche: str, offer: str, max_leads: int, email_provider: str) -> List[dict]:
        self.log("🧭 Sequence: starting pipeline")
        leads = self.leadgen.run(niche, max_leads)
        if not leads:
            self.log("🧭 Sequence: no leads found — pipeline halted")
            return []
        leads = self.db.run(leads)
        if not leads:
            self.log("🧭 Sequence: no leads stored — pipeline halted")
            return []
        leads = self.email.run(leads, offer, email_provider)
        self.db.update_emails(leads)
        self.log("🎉 Sequence: pipeline complete")
        return leads


# ═══════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ═══════════════════════════════════════════════════════════════

def main():
    # ── Sidebar ──────────────────────────────────────────────
    st.sidebar.title("⚙️ System Controls")

    # Check AWS once
    aws_ok, aws_msg = check_aws_credentials()
    if aws_ok:
        st.sidebar.success("☁️ AWS Bedrock Ready")
    else:
        st.sidebar.warning("☁️ AWS Bedrock Unavailable")
        st.sidebar.caption(aws_msg)

    st.sidebar.caption(f"Region: `{AWS_REGION}`")
    st.sidebar.caption(f"DB: `{'CockroachDB' if USE_PG else 'SQLite'}`")

    # Demo mode toggle — DEFAULT ON for hackathon/demo
    demo_mode = st.sidebar.checkbox("🎮 Demo Mode (no AWS/SMTP needed)", value=True)
    if demo_mode:
        st.sidebar.info("✅ Demo mode active. Mock leads + generated emails. No real sends.")

    MODELS = {
        "Claude 3.5 Sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "Claude 3 Haiku": "anthropic.claude-3-haiku-20240307-v1:0",
        "Llama 3.1 70B": "meta.llama3-1-70b-instruct-v1:0",
        "Mistral Large": "mistral.mistral-large-2402-v1:0",
    }
    model_choice = st.sidebar.selectbox("Bedrock Model:", MODELS.keys())
    model_id = MODELS[model_choice]
    max_leads = st.sidebar.slider("Max leads:", 3, 15, 5)

    email_providers = ["smtp", "gmail", "outlook", "yahoo"]
    email_provider = st.sidebar.selectbox("Email Provider:", email_providers)

    with st.sidebar.expander("🏗️ Architecture", expanded=True):
        st.write("🧭 **Sequence Agent** — Orchestration")
        st.write("🔍 **LeadGen** → 🛡️ Critique → 🔁 Loop")
        st.write("🗄️ **DB** → 🛡️ Critique → 🔁 Loop")
        st.write("✉️ **Email** → 🛡️ Critique → 🔁 Loop")

    # ── Main UI ──────────────────────────────────────────────
    st.title("📧 MarMail — AI Multi-Agent Email Marketing")
    st.info(
        "**Architecture:** Sequence → LeadGen → DB → Email"  

        "**Each module:** Agent → Critique → Loop"  

        "**Demo Mode:** Mock leads, generated emails (no credentials needed)"
    )

    col1, col2 = st.columns(2)
    with col1:
        niche = st.text_input(
            "Target niche / criteria:",
            placeholder="e.g. SaaS companies for project management",
        )
    with col2:
        offer = st.text_input(
            "What are you offering?",
            placeholder="e.g. AI-powered onboarding tool for new hires",
        )

    # Button is disabled only when inputs are empty
    run_disabled = not (niche and offer)

    if st.button("🚀 Run MarMail Campaign", type="primary", disabled=run_disabled):
        st.markdown("---")
        st.subheader("🤖 Multi-Agent Pipeline Executing…")
        log_box = st.container()

        def log(msg):
            log_box.write(msg)

        with st.status("Agents working…", expanded=True) as status:
            st.write(f"🎯 Niche: {niche}")
            st.write(f"🤖 Model: {model_choice}")
            st.write(f"📧 Email via: {email_provider}")
            st.write(f"🎮 Demo mode: {'✅ ON' if demo_mode else '❌ OFF'}")

            leads = []
            try:
                llm = None
                if not demo_mode and aws_ok:
                    llm = build_llm(model_id, AWS_REGION)

                seq = SequenceAgent(llm, log, demo_mode)
                leads = seq.run(niche, offer, max_leads, email_provider)
                status.update(label="✅ Pipeline Complete!", state="complete", expanded=False)
            except Exception as e:
                error_detail = traceback.format_exc()
                status.update(label="❌ Pipeline Error", state="error")
                st.error(f"**Error:** {e}")
                with st.expander("🔍 Full traceback"):
                    st.code(error_detail, language="python")

        # ── Results ──────────────────────────────────────────
        if leads:
            st.success(f"✅ Campaign ready — {len(leads)} leads processed")
            st.markdown("---")
            st.markdown("## 📋 Campaign Results")

            for lead in leads:
                with st.expander(f"📨 {lead['company']} — {lead['contact_email']}"):
                    st.markdown(f"**Website:** {lead['website']}")
                    st.markdown(f"**Status:** `{lead.get('status','')}`")
                    st.markdown(f"**Delivery:** `{lead.get('delivery_status','')}`")
                    st.markdown(f"**Subject:** {lead.get('email_subject','')}")
                    st.markdown(f"**Body:** {lead.get('email_body','')}")

            # Download report
            report_lines = [
                f"MarMail Campaign Report — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                f"Niche: {niche}", f"Offer: {offer}", f"Leads: {len(leads)}", "",
            ]
            for lead in leads:
                report_lines += [
                    f"Company: {lead['company']}",
                    f"Website: {lead['website']}",
                    f"Email: {lead['contact_email']}",
                    f"Subject: {lead.get('email_subject','')}",
                    f"Body: {lead.get('email_body','')}",
                    f"Delivery: {lead.get('delivery_status','')}",
                    "-" * 40,
                ]
            report = ".join(report_lines)"
            st.download_button(
                "📥 Download Campaign Report",
                report,
                f"marmail_campaign_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                "text/plain",
            )

            # View DB
            with st.expander("🗄️ View stored leads"):
                conn = sqlite3.connect(DB_PATH)
                rows = conn.execute(
                    "SELECT company, website, contact_email, status, created_at "
                    "FROM leads ORDER BY id DESC LIMIT 50"
                ).fetchall()
                conn.close()
                st.table(rows)
        else:
            st.warning("No leads passed validation. Try a broader niche or check the logs above.")


if __name__ == "__main__":
    main()
