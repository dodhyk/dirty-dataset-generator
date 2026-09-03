# Synthetic / Dirty Dataset Generator

A modular Python framework for generating realistic clean tabular synthetic datasets and corresponding dirty datasets for learning and benchmarking **Exploratory Data Analysis (EDA), Data Cleaning, Preprocessing, and Machine Learning**.

---

## 1. Architectural Overview

The project is strictly architected into three decoupled layers:

```
Streamlit (UI Client)
       │ HTTP
       ▼
FastAPI (REST Backend)
       │ Python Call
       ▼
Core Generator (Standalone Engine)
       │
       ▼
Generated Files (isolated under generated/<generation_id>/)
```

- **Core Generator (`core/`)**: Pure Python engine containing domain models, mathematical sampling distributions, preset schemas, clean dataset generators, row-based corruption engine, and ground truth report generation. Independent of FastAPI, HTTP servers, or Streamlit.
- **FastAPI Layer (`api/`)**: REST API exposing health checks, preset listings, dataset generation, and secure file downloads with path traversal defenses and UUID isolation.
- **Streamlit Layer (`ui/`)**: User interface client communicating with FastAPI solely via HTTP requests through `DatasetApiClient`. Never imports or accesses the Core Generator directly.

---

## 2. Presets Available

| Preset | Task Type | Key Features | Target Column | ID Column |
| :--- | :--- | :--- | :--- | :--- |
| **`customer_churn`** | Binary Classification | Demographics, tenure, monthly charges, contracts | `churn` (Protected) | `customer_id` (Protected) |
| **`ecommerce`** | Binary Classification | Orders, spend, products, devices, timestamps | `purchased_again` (Protected) | `customer_id` (Protected) |
| **`house_price`** | Continuous Regression | Area, beds, baths, location, condition, amenities | `price` (Protected) | `house_id` (Protected) |

---

## 3. Dirtiness & Protection Model

- **Row-Based Dirtiness:** A dirtiness parameter of `X%` means exactly `X%` of dataset rows are selected as dirty rows ($N = \text{round}(\text{rows} \times X / 100)$).
- **Untouched Rows:** Rows not selected remain 100% identical to the clean dataset.
- **Protected Columns:** IDs (`customer_id`, `house_id`) and Targets (`churn`, `purchased_again`, `price`) are **strictly protected** and never corrupted.
- **No Duplicate IDs:** Unique identifiers are maintained.
- **Ground Truth Report:** Generates `corruption_report.json` cataloging every corruption with row index, column name, corruption type, original value, and corrupted value.

---

## 4. Quickstart Guide

### Installation

```bash
pip install -r requirements.txt
```

### 1. Using Core Generator Standalone (No Server Needed)

```python
from generator import generate

# Generate 5,000 rows with 20% dirty rows and reproducible seed
result = generate(
    preset="customer_churn",
    rows=5000,
    dirtiness=20.0,
    seed=12345,
)

print("Clean dataset shape:", result.clean_df.shape)
print("Dirty dataset shape:", result.dirty_df.shape)
print("Dirty rows count:", result.report.dirty_rows)
print("Total corruptions:", result.report.total_corruptions)

# Access DataFrames directly
clean_df = result.clean_df
dirty_df = result.dirty_df
report_dict = result.report.to_dict()
```

### 2. Running the FastAPI Backend

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

Interactive Swagger API documentation is available at:
`http://127.0.0.1:8000/docs`

Available Endpoints:
- `GET  /health`
- `GET  /presets`
- `POST /generate`
- `GET  /download/{generation_id}/clean`
- `GET  /download/{generation_id}/dirty`
- `GET  /download/{generation_id}/report`

### 3. Running the Streamlit UI Client

In a separate terminal:

```bash
streamlit run ui/app.py
```

The UI will automatically connect to `http://127.0.0.1:8000` and provide interactive controls for presets, rows, dirtiness, and one-click downloads for clean CSV, dirty CSV, and JSON corruption reports.

---

## 5. Running Automated Tests

Run the full pytest suite:

```bash
python -m pytest -v
```

Test coverage includes:
- Schema, column types, and protection flags
- Clean dataset distributions, domains, and non-trivial feature-target relationships
- Row-based dirtiness precision, untouched row bitwise equality, and corruption report truthfulness
- Exact reproducibility with identical seeds vs divergence with different seeds
- REST API endpoint behavior, validation rules, HTTP status codes, and path traversal prevention
- Live HTTP integration between `ApiClient` and running FastAPI backend
