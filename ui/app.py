import os
import streamlit as st
import pandas as pd
import io
import json

from ui.client import ApiClient, ApiError

# Streamlit Page Configuration
st.set_page_config(
    page_title="Synthetic / Dirty Dataset Generator",
    page_icon="🧪",
    layout="wide",
)

# Backend URL configuration
DEFAULT_API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Initialize API client in session state
if "api_client" not in st.session_state:
    st.session_state.api_client = ApiClient(base_url=DEFAULT_API_URL)

client: ApiClient = st.session_state.api_client

# App Title & Description
st.title("🧪 Synthetic / Dirty Dataset Generator")
st.markdown(
    """
    Generate synthetic clean datasets and realistic dirty datasets for 
    **EDA, Data Cleaning, Preprocessing, and Machine Learning** experiments.
    """
)

# Sidebar: Backend Connection & Server Info
with st.sidebar:
    st.header("⚙️ Backend Connection")
    backend_url = st.text_input("FastAPI Base URL", value=DEFAULT_API_URL)
    if backend_url != client.base_url:
        client = ApiClient(base_url=backend_url)
        st.session_state.api_client = client

    is_healthy = client.check_health()
    if is_healthy:
        st.success("🟢 Backend Connected")
    else:
        st.error("🔴 Backend Offline")
        st.caption("Start the backend with: `uvicorn api.main:app --port 8000`")

st.markdown("---")

available_presets = ["customer_churn", "ecommerce", "house_price"]

# Verify Backend Availability
if not is_healthy:
    st.warning("⚠️ Cannot connect to the FastAPI backend. Please start the backend service to continue.")
    st.code("uvicorn api.main:app --host 127.0.0.1 --port 8000", language="bash")
    st.stop()
else:
    # Fetch presets dynamically from FastAPI
    try:
        available_presets = client.get_presets()
    except ApiError as e:
        st.error(f"Error fetching presets: {e.message}")
        st.stop()

preset_labels = {
    "customer_churn": "Customer Churn (Classification)",
    "ecommerce": "E-Commerce Transactions (Classification)",
    "house_price": "House Prices (Regression)",
}

# User Configuration Form
col_config, col_info = st.columns([2, 1])

with col_config:
    st.subheader("📋 Dataset Configuration")

    selected_preset = st.selectbox(
        "Preset Dataset",
        options=available_presets,
        format_func=lambda x: preset_labels.get(x, x),
    )

    rows = st.number_input(
        "Number of Rows",
        min_value=10,
        max_value=100000,
        value=1000,
        step=500,
        help="Total rows to generate for the dataset.",
    )

    dirtiness = st.slider(
        "Dirtiness Percentage (%)",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        step=1.0,
        help="Percentage of total rows selected as dirty rows.",
    )

    use_seed = st.checkbox("Specify Random Seed (for reproducibility)", value=False)
    seed = None
    if use_seed:
        seed = st.number_input(
            "Seed Value",
            min_value=0,
            max_value=2147483647,
            value=12345,
            step=1,
        )

    generate_clicked = st.button("🚀 Generate Dataset", type="primary", use_container_width=True)

with col_info:
    st.subheader("ℹ️ Rules & Safeguards")
    st.markdown(
        """
        - **Row-Based Dirtiness:** Dirtiness `X%` means `X%` of rows are selected for corruption.
        - **Protected Columns:** IDs and Target labels are strictly protected from corruption.
        - **Ground Truth Report:** A detailed JSON report records all applied corruptions.
        - **Deterministic:** Supplying a seed guarantees 100% reproducible output.
        """
    )

# Execution and Result Handling
if generate_clicked:
    with st.spinner("Generating clean and dirty datasets via FastAPI backend..."):
        try:
            result = client.generate_dataset(
                preset=selected_preset,
                rows=int(rows),
                dirtiness=float(dirtiness),
                seed=int(seed) if seed is not None else None,
            )
            st.session_state["last_result"] = result
        except ApiError as err:
            st.error(f"Generation failed: {err.message}")

if "last_result" in st.session_state:
    res = st.session_state["last_result"]

    st.success("✅ Dataset Generation Completed!")

    # Summary Metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Preset", res["preset"])
    m2.metric("Total Rows", f"{res['rows']:,}")
    m3.metric("Dirty Rows", f"{res['dirty_rows']:,} ({res['dirtiness']}%)")
    m4.metric("Total Corruptions", f"{res['total_corruptions']:,}")
    m5.metric("Seed", res["seed"])

    st.markdown("### 📥 Download Output Files")
    dl1, dl2, dl3 = st.columns(3)

    try:
        clean_bytes = client.download_file(res["clean_download_url"])
        dirty_bytes = client.download_file(res["dirty_download_url"])
        report_bytes = client.download_file(res["report_download_url"])

        with dl1:
            st.download_button(
                label=f"📄 Download Clean Dataset ({res['preset']}_clean.csv)",
                data=clean_bytes,
                file_name=f"{res['preset']}_clean.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with dl2:
            st.download_button(
                label=f"⚠️ Download Dirty Dataset ({res['preset']}_dirty.csv)",
                data=dirty_bytes,
                file_name=f"{res['preset']}_dirty.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with dl3:
            st.download_button(
                label="📊 Download Corruption Report (JSON)",
                data=report_bytes,
                file_name="corruption_report.json",
                mime="application/json",
                use_container_width=True,
            )

        # Quick Preview tabs
        st.markdown("### 🔍 Dataset Previews")
        clean_df = pd.read_csv(io.BytesIO(clean_bytes))
        dirty_df = pd.read_csv(io.BytesIO(dirty_bytes))
        report_data = json.loads(report_bytes.decode("utf-8"))

        tab1, tab2, tab3 = st.tabs(["Clean Dataset Preview", "Dirty Dataset Preview", "Corruption Report Ground Truth"])
        with tab1:
            st.dataframe(clean_df.head(20), use_container_width=True)
        with tab2:
            st.dataframe(dirty_df.head(20), use_container_width=True)
        with tab3:
            st.json(report_data)

    except ApiError as e:
        st.error(f"Error downloading generated files: {e.message}")
