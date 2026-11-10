# Nassau Candy Factory Optimization Dashboard

A data-driven **factory allocation and shipping optimization system** built using Python and Streamlit. The project analyzes product demand, factory assignments, and shipping distances to identify opportunities to reduce transportation distance and improve allocation efficiency.

## Project Overview

The system helps analyze how products are distributed across factories and destinations. It provides factory allocation recommendations, what-if scenario analysis, shipping comparisons, and analytical insights through an interactive dashboard.

The goal is to support better factory allocation decisions using data analysis, optimization techniques, and machine learning.

## Key Features

- **Factory Optimization Simulator:** Explore factory allocation recommendations based on shipping distances.
- **What-If Scenario Analysis:** Test different allocation preferences and compare possible outcomes.
- **Recommendation Dashboard:** Review optimized factory allocations and product distribution.
- **Risk & Impact Analysis:** Understand allocation changes and their potential impact.
- **Model & Analytics:** Explore model performance, clustering results, and analytical insights.
- **Interactive Visualizations:** View factory-wise allocation, shipping distance, product demand, and performance metrics.

## Technologies Used

- **Programming:** Python
- **Dashboard:** Streamlit
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Plotly
- **Optimization:** PuLP
- **Machine Learning:** Scikit-learn
- **Database / Data Storage:** CSV files

## Project Structure

```text
nassau_candy_dashboard/
│
├── app.py
├── Nassau Candy Distributor.csv
├── distance_matrix.csv
├── optimized_shipping_plan.csv
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd nassau_candy_dashboard
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

### 3. Activate the environment

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit application

```bash
streamlit run app.py
```

The dashboard will open in your browser.

## Required Input Files

The dashboard uses the following CSV files:

| File | Description |
|---|---|
| `Nassau Candy Distributor.csv` | Original sales and distribution dataset |
| `distance_matrix.csv` | Factory-to-destination distance information |
| `optimized_shipping_plan.csv` | Optimized product-to-factory allocation plan |

If the additional CSV files are not included in the repository, upload them using the dashboard's sidebar file upload options.

### Distance Matrix Format

The distance matrix should contain:

- `Factory`
- `City`
- `State` or `State/Province`
- `Country` or `Country/Region`
- `Distance_Miles` or `Distance`

## Analytical Approach

The project combines several analytical techniques:

1. Data cleaning and preprocessing.
2. Factory-to-destination distance calculation.
3. Factory allocation optimization using PuLP.
4. What-if scenario simulation.
5. Regression model comparison.
6. Customer/destination clustering analysis.
7. Shipping distance and hypothetical cost sensitivity analysis.

## Important Notes and Limitations

- Shipping distances are calculated using straight-line geographical distances, not actual road distances.
- What-if recommendations use a distance-based nearest-factory approach, while the optimized allocation dashboard uses the supplied optimization plan.
- The speed/profit preference is a scenario setting, not a calculation of actual factory profitability.
- Shipping cost estimates are hypothetical and depend on the selected cost-per-mile assumption.
- Lead-time improvements are represented using distance as a proxy, not validated delivery-time data.
- Factory capacity, manufacturing costs, and product eligibility should be validated before using recommendations for real operational decisions.
- Machine learning models evaluate patterns in the optimization-generated data and do not independently establish real-world shipping performance.

## Project Objective

To demonstrate how **data analytics, machine learning, and mathematical optimization** can be combined to support factory allocation and shipping decisions through an interactive business intelligence dashboard.

---

**Developed as a Data Science project.**