import streamlit as st
import plotly.express as px
import pandas as pd

def render_dashboard(df: pd.DataFrame):
    st.header("📊 Comparative Analytics Dashboard")
    
    # 1. Hospital Comparison
    st.subheader("Hospital Comparison")
    
    # Get list of hospitals
    if "Facility Name" not in df.columns:
        st.error("Dataframe missing 'Facility Name' column.")
        return

    # Filter by State
    if "State" not in df.columns:
        st.error("Dataframe missing 'State' column for state filtering.")
        return

    states = sorted(df["State"].unique().tolist())
    
    # Check for agent-injected state
    default_state_index = 0
    if "dashboard_filter_state" in st.session_state:
        target_state = st.session_state.pop("dashboard_filter_state").upper() # Consume the event
        if target_state in states:
            default_state_index = states.index(target_state)

    selected_state = st.selectbox("Select State", states, index=default_state_index)
    
    # Filter Hospitals based on selected state
    state_df = df[df["State"] == selected_state]
    all_hospitals = sorted(state_df["Facility Name"].unique().tolist())
    
    # Check for agent-injected hospitals
    default_hospitals_selection = all_hospitals[:3] if len(all_hospitals) > 2 else all_hospitals # Original default logic
    if "dashboard_filter_hospitals" in st.session_state:
        injected = st.session_state.pop("dashboard_filter_hospitals")
        # Simple string matching
        matched_hospitals = [h for h in all_hospitals if any(i.lower() in h.lower() for i in injected)]
        if matched_hospitals:
            default_hospitals_selection = matched_hospitals
    
    # Selector
    selected_hospitals = st.multiselect(
        "Select Hospitals to Compare",
        options=all_hospitals,
        default=default_hospitals_selection,
        max_selections=5
    )
    
    if not selected_hospitals:
        st.info("Please select at least one hospital.")
        return

    # Filter data
    subset = df[df["Facility Name"].isin(selected_hospitals)].copy()
    
    # Metric Selector
    # Try to find numeric columns that look like HCAHPS scores
    # In HCAHPS data, 'measure_id' often defines the metric, and 'score' (or similar) is the value.
    # We need to pivot or filter.
    
    if "measure_id" in df.columns and "score" in df.columns:
        # HCAHPS Standard Format
        all_measures = sorted(subset["measure_id"].unique().tolist())
        selected_measures = st.multiselect(
            "Select Measures",
            options=all_measures,
            default=all_measures[:3] if len(all_measures) > 2 else all_measures
        )
        
        if selected_measures:
             # Chart
            chart_data = subset[subset["measure_id"].isin(selected_measures)]
            
            # Ensure score is numeric
            chart_data["score"] = pd.to_numeric(chart_data["score"], errors="coerce")
            
            fig = px.bar(
                chart_data,
                x="measure_id",
                y="score",
                color="Facility Name",
                barmode="group",
                title="Scores by Measure",
                hover_data=["measure_title"]
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Data Table
            st.dataframe(chart_data[["Facility Name", "measure_id", "score", "measure_title"]], use_container_width=True)

    else:
        # Generic CSV Fallback (if they upload something else)
        st.warning("Data format not recognized as standard HCAHPS vertical format. Trying generic numeric comparison.")
        num_cols = subset.select_dtypes(include="number").columns.tolist()
        
        if num_cols:
            target_col = st.selectbox("Select Metric", num_cols)
            fig = px.bar(
                subset,
                x="Facility Name",
                y=target_col,
                color="Facility Name",
                title=f"Comparison: {target_col}"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No numeric columns found to plot.")

    # 2. State/Aggregate View (Optional)
    st.markdown("---")
    st.subheader("Dataset Overview")
    st.write(f"Total Facilities: {df['Facility Name'].nunique()}")
    if "State" in df.columns:
        st.write(f"States represented: {df['State'].nunique()}")
