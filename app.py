import streamlit as st
import os
import re
import sqlite3
import requests
from datetime import datetime
from urllib.parse import urlparse

from duckduckgo_search import DDGS
from langchain_core.tools import tool
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.prebuilt import create_react_agent

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="MarMail Demo", page_icon="📧", layout="wide")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

DB_PATH = "marmail_leads.db"


def get_hf_key():
    try:
        key = st.secrets.get("HUGGINGFACE_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("HUGGINGFACE_API_KEY", "")


HF_KEY = get_hf_key()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS leads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company TEXT,
            website TEXT UNIQUE,
            contact_email TEXT,
            snippet TEXT,
            email_subject TEXT,
            email_body TEXT,
            status TEXT,
            created_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()


init_db()

# ============================================================
# TOOLS — used by the LeadGen agent (all free, no paid API keys)
# ============================================================


@tool
def web_search_leads(niche: str) -> str:
    """Search the web for businesses/companies matching a target niche or industry description."""
    try:
        results = DDGS().text(f"{niche} company official site", max_results=8)
        if not results:
            return "No results found."
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(
                f"{i}. {r.get('title', '')}\n"
                f"   URL: {r.get('href', '')}\n"
                f"   Info: {r.get('body', '')[:150]}"
            )
        return "\n\n".join(lines)
    except Exception as e:
        return f"❌ Search error: {e}"


@tool
def extract_contact_email(website_url: str) -> str:
    """Visit a company website and try to find a public contact email address."""
    try:
        resp = requests.get(website_url, headers=HEADERS, timeout=8)
        resp.raise_for_status()
        text = resp.text
        emails = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
        emails = [
            e for e in emails
            if not any(x in e.lower() for x in ["png", "jpg", "example.com", "sentry"])
        ]
        if emails:
            return f"✅ Found email on page: {emails[0]}"
        domain = urlparse(website_url).netloc.replace("www.", "")
        return f"⚠️ No public email found. Best guess: contact@{domain} (unverified)"
    except Exception as e:
        return f"❌ Could not reach site: {e}"


# ============================================================
# RULE-BASED CRITIQUE — intent + output validation, no LLM tokens spent
# ============================================================


def critique_lead(lead: dict) -> tuple:
    """Intent validation: is this a usable lead? Output validation: is it well-formed?"""
    if not lead.get("company"):
        return False, "Missing company name"
    if not lead.get("website"):
        return False, "Missing website"
    email = (lead.get("contact_email") or "").strip()
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        return False, "Email format looks invalid"
    return True, "OK"


def critique_email(subject: str, body: str) -> tuple:
    """Intent validation: does it read like a real email? Output validation: no broken placeholders."""
    if not subject or len(subject) < 5:
        return False, "Subject too short / missing"
    if not body or len(body) < 40:
        return False, "Body too short / missing"
    if "{{" in body or "[your" in body.lower():
        return False, "Unfilled template placeholders detected"
    return True, "OK"


# ============================================================
# AGENTS
# ============================================================


def build_llm(hf_key, model_id):
    endpoint = HuggingFaceEndpoint(
        repo_id=model_id,
        huggingfacehub_api_token=hf_key,
        max_new_tokens=800,
        temperature=0.4,
    )
    return ChatHuggingFace(llm=endpoint)


def run_leadgen_agent(llm, niche, max_leads, log):
    """LeadGen agent: searches the web and enriches results with contact emails."""
    agent = create_react_agent(
        llm,
        tools=[web_search_leads, extract_contact_email],
        prompt=(
            "You are the LeadGen agent. Use web_search_leads to find companies matching "
            "the niche, then use extract_contact_email on the most promising results. "
            "Return a clear list of: Company name | Website | Email."
        ),
    )
    log(f"🔍 LeadGen agent searching for: {niche}")
    result = agent.invoke(
        {
            "messages": [
                (
                    "user",
                    f"Find {max_leads} companies matching: '{niche}'. "
                    f"For each, find the website and a contact email.",
                )
            ]
        }
    )
    output = result["messages"][-1].content
    log("✅ LeadGen agent finished raw discovery")
    return output


def parse_leads_from_text(raw_text, max_leads):
    """Lightweight heuristic parser — turns the agent's free text into structured lead records."""
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
        if not domain or domain in seen:
            continue
        seen.add(domain)
        email_match = re.search(
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", line
        )
        email = email_match.group(0) if email_match else f"contact@{domain}"
        company_guess = domain.split(".")[0].capitalize()
        leads.append(
            {
                "company": company_guess,
                "website": url,
                "contact_email": email,
                "snippet": line[:200],
                "status": "guess" if email.startswith("contact@") else "found",
            }
        )
    return leads


def run_db_agent(leads, log):
    """DB agent: validates via critique_lead, dedupes, and writes to SQLite."""
    conn = sqlite3.connect(DB_PATH)
    stored, skipped = 0, 0
    for lead in leads:
        ok, reason = critique_lead(lead)
        if not ok:
            skipped += 1
            log(f"   ⛔ Rejected {lead.get('company', '?')}: {reason}")
            continue
        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO leads
                    (company, website, contact_email, snippet, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    lead["company"],
                    lead["website"],
                    lead["contact_email"],
                    lead["snippet"],
                    lead["status"],
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()
            stored += 1
        except Exception as e:
            log(f"   ⚠️ DB error for {lead['company']}: {e}")
    conn.close()
    log(f"✅ DB agent stored {stored} leads, rejected {skipped}")
    return stored, skipped


def run_email_agent(llm, leads, offer, log, max_retries=1):
    """Email agent: personalizes one email per lead, retries once if critique fails (Loop behavior)."""
    results = []
    for lead in leads:
        prompt = (
            "Write a short, personalized cold outreach email.\n"
            f"Recipient company: {lead['company']}\n"
            f"Website: {lead['website']}\n"
            f"Context: {lead.get('snippet', '')}\n"
            f"What we're offering: {offer}\n\n"
            "Respond in exactly this format:\n"
            "Subject: <subject line>\n"
            "Body: <3-5 sentence email body, no placeholders>"
        )
        attempt, subject, body = 0, "", ""
        while attempt <= max_retries:
            resp = llm.invoke(prompt)
            text = resp.content if hasattr(resp, "content") else str(resp)
            subj_match = re.search(r"Subject:\s*(.+)", text)
            body_match = re.search(r"Body:\s*(.+)", text, re.DOTALL)
            subject = subj_match.group(1).strip() if subj_match else ""
            body = body_match.group(1).strip() if body_match else text.strip()
            ok, reason = critique_email(subject, body)
            if ok:
                break
            log(f"   🔁 Retry email for {lead['company']}: {reason}")
            attempt += 1
        lead["email_subject"] = subject or f"Quick question for {lead['company']}"
        lead["email_body"] = body or "Personalization failed — needs manual review."
        results.append(lead)
    log(f"✅ Email agent personalized {len(results)} emails")
    return results


def update_db_with_emails(leads, log):
    conn = sqlite3.connect(DB_PATH)
    for lead in leads:
        conn.execute(
            "UPDATE leads SET email_subject=?, email_body=? WHERE website=?",
            (lead["email_subject"], lead["email_body"], lead["website"]),
        )
    conn.commit()
    conn.close()
    log("✅ DB agent updated records with personalized emails")


def run_sequence_agent(llm, niche, offer, max_leads, log):
    """Sequence agent: orchestrates LeadGen → DB → Email in order."""
    log("🧭 Sequence agent starting pipeline")
    raw = run_leadgen_agent(llm, niche, max_leads, log)
    leads = parse_leads_from_text(raw, max_leads)
    log(f"   Parsed {len(leads)} candidate leads")
    run_db_agent(leads, log)
    valid_leads = [l for l in leads if critique_lead(l)[0]]
    enriched = run_email_agent(llm, valid_leads, offer, log)
    update_db_with_emails(enriched, log)
    log("🎉 Sequence agent: pipeline complete")
    return enriched


# ============================================================
# STREAMLIT UI
# ============================================================


def main():
    with st.sidebar.expander("🤖 AI Status", expanded=True):
        if HF_KEY:
            st.success("✅ Hugging Face Connected")
            st.caption(f"Key: hf_…{HF_KEY[-6:]}")
        else:
            st.error("❌ HUGGINGFACE_API_KEY not found")
            st.code('HUGGINGFACE_API_KEY = "hf_your_token"', language="toml")

    if not HF_KEY:
        st.title("📧 MarMail Demo")
        st.error("🚫 HUGGINGFACE_API_KEY is missing!")
        st.markdown("### Setup — token needs **Inference Providers** permission:")
        st.markdown("1. Go to **https://huggingface.co/settings/tokens**")
        st.markdown("2. Click **New token** → choose **Fine-grained**")
        st.markdown("3. Under *User permissions* → tick ✅ **Make calls to Inference Providers**")
        st.markdown("4. Click **Generate** → **Copy** the token")
        st.markdown("5. Add to `.streamlit/secrets.toml`:")
        st.code('HUGGINGFACE_API_KEY = "hf_paste_here"', language="toml")
        st.stop()

    MODELS = {
        "Llama 3.3 70B (Best)": "meta-llama/Llama-3.3-70B-Instruct",
        "Qwen 2.5 72B (Smart)": "Qwen/Qwen2.5-72B-Instruct",
        "Mistral Small (Fast)": "mistralai/Mistral-Small-24B-Instruct-2501",
    }

    st.sidebar.title("⚙️ System Controls")
    model_choice = st.sidebar.selectbox("AI Model:", MODELS.keys())
    model_id = MODELS[model_choice]
    max_leads = st.sidebar.slider("Max leads to find:", 3, 15, 5)

    with st.sidebar.expander("👥 Active Agents", expanded=True):
        st.write("🧭 **Sequence Agent** — Orchestration")
        st.write("🔍 **LeadGen Agent** — Web Search + Enrichment")
        st.write("🗄️ **DB Agent** — Store + Dedupe (SQLite)")
        st.write("✉️ **Email Agent** — Personalization")
        st.write("🛡️ **Critique** — Rule-based validation (free, no LLM)")

    st.title("📧 MarMail — AI Multi-Agent Email Marketing Demo")
    st.info(
        "Agents: **Sequence → LeadGen → DB → Email**, each validated before moving to the next stage."
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

    if st.button(
        "🚀 Run MarMail Campaign", type="primary", disabled=not (niche and offer)
    ):
        st.markdown("---")
        st.subheader("🤖 Multi-Agent Pipeline Executing…")

        log_box = st.container()

        def log(msg):
            log_box.write(msg)

        with st.status("Agents working…", expanded=True) as status:
            st.write(f"🎯 Niche: {niche}")
            st.write(f"🤖 Model: {model_choice}")
            st.write("👥 Agents: Sequence → LeadGen → DB → Email")
            try:
                llm = build_llm(HF_KEY, model_id)
                leads = run_sequence_agent(llm, niche, offer, max_leads, log)
                status.update(
                    label="✅ MarMail Pipeline Complete!", state="complete", expanded=False
                )
            except Exception as e:
                import traceback

                status.update(label="❌ Error", state="error")
                st.error(str(e))
                st.code(traceback.format_exc())
                leads = []

        if leads:
            st.success(f"✅ Campaign ready — {len(leads)} personalized emails generated")
            st.markdown("---")
            st.markdown("## 📋 Campaign Results")

            for lead in leads:
                with st.expander(f"📨 {lead['company']} — {lead['contact_email']}"):
                    st.markdown(f"**Website:** {lead['website']}")
                    st.markdown(f"**Status:** `{lead['status']}`")
                    st.markdown(f"**Subject:** {lead['email_subject']}")
                    st.markdown(f"**Body:**\n\n{lead['email_body']}")

            report_lines = [
                f"MarMail Campaign Report — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                f"Niche: {niche}",
                f"Offer: {offer}",
                f"Leads: {len(leads)}",
                "",
            ]
            for lead in leads:
                report_lines += [
                    f"Company: {lead['company']}",
                    f"Website: {lead['website']}",
                    f"Email: {lead['contact_email']}",
                    f"Subject: {lead['email_subject']}",
                    f"Body: {lead['email_body']}",
                    "-" * 40,
                ]
            report = "\n".join(report_lines)
            st.download_button(
                "📥 Download Campaign Report",
                report,
                f"marmail_campaign_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                "text/plain",
            )

            with st.expander("🗄️ View stored leads (SQLite)"):
                conn = sqlite3.connect(DB_PATH)
                rows = conn.execute(
                    "SELECT company, website, contact_email, status, created_at "
                    "FROM leads ORDER BY id DESC LIMIT 50"
                ).fetchall()
                conn.close()
                st.table(rows)
        else:
            st.warning("No leads passed validation. Try a broader niche.")


if __name__ == "__main__":
    main()
