import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from io import BytesIO
from huggingface_hub import hf_hub_download

# ===================== DOWNLOAD LARGE FILES FROM HUGGING FACE =====================
def download_large_files():
    files_in_root = ["HCAHPS_RAG_Facts.csv", "HCAHPS_RAG_Summaries.csv"]
    files_in_indexes = ["facts.faiss", "summaries.faiss", "facts_ids.npy", "summaries_ids.npy"]

    os.makedirs("hcahps_indexes", exist_ok=True)

    for f in files_in_root:
        if not os.path.exists(f):
            hf_hub_download(
                repo_id="AliRasoulRifai/hcahps-data",
                filename=f,
                repo_type="dataset",
                local_dir="."
            )

    for f in files_in_indexes:
        local_path = f"hcahps_indexes/{f}"
        if not os.path.exists(local_path):
            hf_hub_download(
                repo_id="AliRasoulRifai/hcahps-data",
                filename=f,
                repo_type="dataset",
                local_dir="hcahps_indexes"
            )

download_large_files()

from config import DEFAULT_LLM, TEXT_PATH
from rag_engine import (
    load_data_and_indexes,
    agent_answer,
    rag_answer,
    chat_answer,
    research_answer,
    detect_applicable_cfr_sections
)
from utils import create_pdf_report, get_agent_df
from dashboard import render_dashboard

# ===================== STREAMLIT UI =====================
st.set_page_config(
    page_title="HCAHPS AI Agent",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
<style>
.small-chart {
    max-width: 350px;
    margin: 0 auto;
}
</style>
""", unsafe_allow_html=True)

# Sidebar Settings
with st.sidebar:
    st.header("Settings")
    st.sidebar.title("HCAHPS RAG Agent 🏥")
    
    if "requested_app_mode" in st.session_state:
        st.session_state["app_mode_state"] = st.session_state.pop("requested_app_mode")

    if "app_mode_state" not in st.session_state:
        st.session_state["app_mode_state"] = "Chat Agent"

    app_mode = st.sidebar.radio("App Mode", ["Chat Agent", "Dashboard"], key="app_mode_state")

    st.divider()
    
    st.subheader("Data Source")
    uploaded_file = st.file_uploader("Upload HCAHPS CSV (Optional)", type=["csv"])
    
    st.divider()

    if app_mode == "Chat Agent":
        model_name = st.text_input("Ollama model", value=DEFAULT_LLM)
        llm_mode = st.radio("Agent Strategy", ["Auto (Agent)", "Force RAG", "Force Chat", "Deep Research"], index=0)
        smart_s = st.toggle("Enable Smart Search (Decomposition)", value=False, help="Breaks complex questions into sub-queries.")
        k_each = st.slider("Retrieval Depth (k)", 1, 10, 6)
        rag_temp = st.slider("Temperature", 0.0, 1.0, 0.1)
        
        st.divider()
        if st.button("Clear Chat History"):
            st.session_state["messages"] = [st.session_state["messages"][0]]
            st.rerun()

# Load Data
try:
    if uploaded_file:
        df_display = pd.read_csv(uploaded_file)
        df_display = df_display.apply(pd.to_numeric, errors="ignore")
        st.toast(f"Loaded custom data: {len(df_display)} rows")
        # For agent analysis, we still rely on the 'text_df' structure for consistency unless mapped
        text_df, _, _, _, _, _ = load_data_and_indexes()
    else:
        text_df, _, _, _, _, _ = load_data_and_indexes()
        df_display = text_df
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello! I am your HCAHPS expert. Ask me about hospital ratings, compare facilities, or ask for analysis."}]

# ===================== MAIN APP LOGIC =====================

if app_mode == "Dashboard":
    render_dashboard(df_display)

else:
    # --- CHAT INTERFACE ---
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("has_report_data"):
                 st.info("📊 (Data analysis available in previous turn)")

    # Chat Input
    if prompt := st.chat_input("Ask a question about hospital data..."):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate Response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    history_for_llm = [m for m in st.session_state["messages"] if not m.get("has_report_data")]

                    if llm_mode == "Deep Research":
                        with st.status("🕵️‍♂️ Deep Research in progress...", expanded=True) as status:
                            st.write("Generating research plan...")
                            result = research_answer(prompt, k_each=k_each, model=model_name)
                            st.write(f"✅ Plan: {', '.join(result['plan'])}")
                            st.write(f"🔎 Executing {len(result['plan'])} search steps...")
                            st.write(f"📚 Found {len(result['hits'])} relevant snippets.")
                            status.update(label="Research Complete! Writing Report...", state="complete", expanded=False)
                            
                    elif llm_mode == "Force RAG":
                        result = rag_answer(prompt, k_each=k_each, model=model_name, temperature=rag_temp, history=history_for_llm, smart=smart_s)
                    elif llm_mode == "Force Chat":
                        result = chat_answer(prompt, model=model_name, temperature=0.5)
                    else:
                        result = agent_answer(prompt, k_each=k_each, model=model_name, rag_temperature=rag_temp, history=history_for_llm, smart=smart_s)
                    
                    # --- HANDLE TOOLS ---
                    
                    # 1. Dashboard Control
                    if result.get("mode") == "dashboard_control":
                        filters = result.get("args", {}).get("filters", {})
                        st.session_state["requested_app_mode"] = "Dashboard" 
                        if "state" in filters: st.session_state["dashboard_filter_state"] = filters["state"]
                        if "hospitals" in filters: st.session_state["dashboard_filter_hospitals"] = filters["hospitals"]
                        st.toast(f"🎛️ Agent switching to Dashboard with filters: {filters}")
                        st.rerun()

                    # 2. Code Interpreter
                    elif result.get("mode") == "code_interpreter":
                        code_snippet = result.get("args", {}).get("code", "")
                        st.caption("🐍 Executing Python Code:")
                        st.code(code_snippet, language="python")

                        with st.spinner("Preparing data for analysis..."):
                             agent_df = get_agent_df(text_df)

                        try:
                            local_env = {"df": agent_df, "pd": pd, "np": np}
                            try:
                                exec_result = eval(code_snippet, local_env)
                                st.write("**Result:**", exec_result)
                                response_text = f"I executed the code. Result: {exec_result}"
                            except SyntaxError:
                                exec(code_snippet, local_env)
                                if "result" in local_env:
                                    st.write("**Result:**", local_env["result"])
                                    response_text = f"Executed code. Result: {local_env['result']}"
                                else:
                                    st.success("Code executed successfully.")
                                    response_text = "Executed the code successfully."
                        except Exception as e:
                            st.error(f"Error executing code: {e}")
                            response_text = f"Error executing code: {e}"

                    # 3. Stream Response
                    elif "stream" in result:
                         stream_generator = result["stream"]
                         response_text = st.write_stream(stream_generator)
                    
                    else:
                         # Fallback if no stream and not tool (e.g. static content)
                         response_text = result.get("content", "I processed your request.")
                         st.write(response_text)

                    # Store assistant response (ONCE)
                    message_data = {"role": "assistant", "content": response_text}
                    
                    # --- EXTRA CONTENT (Hit-based) ---
                    hits = result.get("hits", [])
                    matched_rules = []
                    
                    if hits:
                         # Regulations
                         matched_rules = detect_applicable_cfr_sections(prompt, hits)
                         if matched_rules:
                             with st.expander("⚖️ Federal Regulations Alert"):
                                 for rule in matched_rules:
                                     st.markdown(f"**{rule['section']}**: {rule['summary']}")
                                     st.caption(rule['why_template'])
                        
                         # Evidence
                         with st.expander("🔍 Retrieved Evidence"):
                             df_hits = pd.DataFrame(hits)[["Facility Name", "State", "score", "measure_title", "snippet"]]
                             st.dataframe(df_hits, use_container_width=True)
                             
                         # Charts (Radar)
                         facts_hits = [h for h in hits if h.get("source") == "facts"]
                         radar_buf = None
                         if facts_hits:
                             col1, col2 = st.columns([1,1])
                             with col1:
                                 st.caption("Facility Comparison (Radar)")
                                 top_facility = facts_hits[0]["Facility Name"]
                                 
                                 # Radar logic using text_df
                                 fac_df = text_df[text_df["Facility Name"] == top_facility].copy()
                                 fac_df = fac_df.apply(pd.to_numeric, errors="ignore")
                                 num_cols = fac_df.select_dtypes(include="number").columns.tolist()
                                 valid_cols = [c for c in num_cols if "score" in c.lower() or "percent" in c.lower()]
                                 target_col = valid_cols[0] if valid_cols else (num_cols[0] if num_cols else None)
                                 
                                 if target_col:
                                     radar_df = fac_df[["measure_title", target_col]].dropna().head(6)
                                     if not radar_df.empty:
                                         categories = radar_df["measure_title"].tolist()
                                         values = radar_df[target_col].tolist()
                                         N = len(categories)
                                         angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
                                         angles += [angles[0]]
                                         values += [values[0]]
                                         fig, ax = plt.subplots(figsize=(3,3), subplot_kw={"projection": "polar"})
                                         ax.plot(angles, values, marker='o')
                                         ax.fill(angles, values, alpha=0.25)
                                         ax.set_xticks(angles[:-1])
                                         ax.set_xticklabels(['' for _ in categories])
                                         ax.set_title(f"{top_facility}", pad=10, fontsize=9)
                                         st.pyplot(fig, use_container_width=False)
                                         radar_buf = BytesIO()
                                         fig.savefig(radar_buf, format='png', bbox_inches='tight')
                                         radar_buf.seek(0)
                                         
                             with col2:
                                 st.write("") 

                         # PDF Report
                         if hits or matched_rules:
                             reg_text = "\n".join([f"{r['section']}: {r['summary']}" for r in matched_rules])
                             pdf_data = create_pdf_report(
                                 question=prompt,
                                 answer=response_text,
                                 radar_img_bytes=radar_buf,
                                 regulation_text=reg_text
                             )
                             st.download_button("📄 Download Report", pdf_data, "report.pdf", "application/pdf")
                             message_data["has_report_data"] = True
                    
                    # Final Save
                    st.session_state["messages"].append(message_data)

                except Exception as e:
                    st.error(f"Error: {e}")