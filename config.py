import os

# Paths
TEXT_PATH = "HCAHPS_RAG_Facts.csv"
SUMM_PATH = "HCAHPS_RAG_Summaries.csv"

OUT_DIR = "hcahps_indexes"
FACTS_INDEX_PATH = os.path.join(OUT_DIR, "facts.faiss")
SUMM_INDEX_PATH  = os.path.join(OUT_DIR, "summaries.faiss")
FACTS_IDS_PATH   = os.path.join(OUT_DIR, "facts_ids.npy")
SUMM_IDS_PATH    = os.path.join(OUT_DIR, "summaries_ids.npy")

# Ollama Settings
OLLAMA_URL   = "http://localhost:11434"
DEFAULT_LLM  = "qwen3:8b"

# Keywords
FACTY = {
    "percent", "%", "score", "rate", "count", "how many",
    "what percentage", "increase", "decrease", "trend",
    "change", "median", "average", "mean", "compare", "facts"
}

INTERP = {
    "why", "how", "improve", "recommend", "insights",
    "summary", "drivers", "root cause", "opportunities",
    "strengths", "weaknesses", "because"
}

# Prompts
SYS = (
    "You are a precise hospital analytics assistant. "
    "Answer in complete sentences. Quote short phrases in \"double quotes\" "
    "and keep numbers as stated. "
    "Give recommendations for improvement when scores are not good. "
    "Compare performance between hospitals only when the user asks you to compare. "
    "You can speak freely when needed."
)

AGENT_SYS = f"""You are an intelligent HCAHPS assistant.
You have access to the following tools:

1. `rag`: Retrieve information from the HCAHPS database.
   - Use this for specific questions about hospital scores, comparisons, or regulations.
   - Args: {{"question": "..."}}

2. `chitchat`: Small talk or general greeting.
   - Use this if the user says "hello", "thanks", or asks something unrelated to hospitals.
   - Args: None

3. `dashboard`: Control the App UI to show data visualizations.
   - Use this if the user asks to "show", "plot", "graph", or "visualize" data.
   - Args: {{"mode": "Dashboard", "filters": {{"state": "TX", "hospitals": ["Hospital A", "Hospital B"], "measure": "..."}}}}

4. `code_interpreter`: Execute Python code for data analysis.
   - Use this for math, statistics, correlations, aggregations, or specific data queries not covered by RAG.
   - The dataframe is available as `df`.
   - Args: {{"code": "df['Score'].mean()"}}

Output ONLY a JSON object with the "tool" and "args" keys.
Examples:
User: "Hi" -> {{"tool": "chitchat"}}
User: "What is the cleanliness score for Texas?" -> {{"tool": "rag", "args": {{"question": "What is the cleanliness score for Texas?"}}}}
User: "Show me a plot of Texas hospitals" -> {{"tool": "dashboard", "args": {{"mode": "Dashboard", "filters": {{"state": "TX"}}}}}}
User: "Calculate the average rating for California" -> {{"tool": "code_interpreter", "args": {{"code": "df[df['State']=='CA']['Score'].mean()"}}}}
User: "What is the correlation between hygiene and quietness?" -> {{"tool": "code_interpreter", "args": {{"code": "df['Hygiene'].corr(df['Quietness'])"}}}}

Do NOT include any extra text outside the JSON object.
"""

# Regulations
CFR_RULES = [
    {
        "section": "CFR §482.41",
        "short_name": "Physical environment, cleanliness & quietness",
        "keywords": [
            "CLEAN", "CLEANLINESS", "ROOM CLEAN", "BATHROOM",
            "QUIET", "NOISE", "ENVIRONMENT", "PHYSICAL ENVIRONMENT",
            "H_CLEAN", "H_CLEAN_STAR", "H_QT"
        ],
        "summary": (
            "Hospitals must provide a safe, functional, and sanitary environment "
            "for patients, staff, and visitors. This includes maintaining clean "
            "rooms and bathrooms, controlling noise, and ensuring surfaces and "
            "equipment are properly maintained."
        ),
        "why_template": (
            "Your question and retrieved HCAHPS measures mention cleanliness and/or quietness. "
            "These topics correspond to the federal requirement for maintaining a sanitary, safe "
            "physical environment under §482.41."
        ),
    },
    {
        "section": "CFR §482.13",
        "short_name": "Patient rights & grievance handling",
        "keywords": [
            "RIGHTS", "RESPECT", "DIGNITY", "PRIVACY", "GRIEVANCE",
            "COMPLAINT", "COMPLAINTS", "PATIENT RIGHTS", "H_RECMND",
            "H_NURSE", "H_DOCTOR", "COMMUNICATION", "TREATED"
        ],
        "summary": (
            "Hospitals must protect and promote patient rights, including respect, privacy, "
            "informed decision-making, and a fair grievance process."
        ),
        "why_template": (
            "Your question and the retrieved HCAHPS measures focus on patient experience, respect, "
            "and communication. These are core patient rights protected under §482.13."
        ),
    },
]
