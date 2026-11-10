
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Nassau Candy Factory Optimization",
    page_icon="🏭",
    layout="wide"
)

st.title("🏭 Factory Reallocation & Shipping Optimization")
st.markdown(
    "A decision-support dashboard for factory allocation "
    "and distance-based shipping optimization."
)

ROOT = Path(__file__).parent

DATA_PATH = ROOT / "Nassau Candy Distributor.csv"
DISTANCE_PATH = ROOT / "distance_matrix.csv"
OPT_PATH = ROOT / "optimized_factory_allocation.csv"

FACTORIES = [
    "Lot's O' Nuts",
    "Wicked Choccy's",
    "Sugar Shack",
    "Secret Factory",
    "The Other Factory"
]

CURRENT_FACTORY = {
    "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows": "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious": "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate": "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",
    "Laffy Taffy": "Sugar Shack",
    "SweeTARTS": "Sugar Shack",
    "Nerds": "Sugar Shack",
    "Fun Dip": "Sugar Shack",
    "Fizzy Lifting Drinks": "Sugar Shack",
    "Everlasting Gobstopper": "Secret Factory",
    "Hair Toffee": "The Other Factory",
    "Lickable Wallpaper": "Secret Factory",
    "Wonka Gum": "Secret Factory",
    "Kazookles": "The Other Factory"
}

# --------------------------------------------------
# 2. HELPER FUNCTIONS
# --------------------------------------------------

@st.cache_data
def read_csv_file(file):
    return pd.read_csv(file)


def clean_text(series):
    return series.astype(str).str.strip()


def normalize_distance_columns(df):
    df = df.copy()
    df.columns = df.columns.str.strip()

    rename_map = {}

    for col in df.columns:
        low = col.lower().strip()

        if low == "factory":
            rename_map[col] = "Factory"
        elif low == "city":
            rename_map[col] = "City"
        elif low in ["state", "state/province"]:
            rename_map[col] = "State"
        elif low in ["country", "country/region"]:
            rename_map[col] = "Country"
        elif low in ["distance_miles", "distance"]:
            rename_map[col] = "Distance_Miles"

    df = df.rename(columns=rename_map)

    required = [
        "Factory", "City", "State",
        "Country", "Distance_Miles"
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Distance matrix is missing columns: {missing}"
        )

    for col in ["Factory", "City", "State", "Country"]:
        df[col] = clean_text(df[col])

    df["Distance_Miles"] = pd.to_numeric(
        df["Distance_Miles"], errors="coerce"
    )

    return df.dropna(subset=["Distance_Miles"])


def normalize_optimized_columns(df):
    df = df.copy()

    # Remove extra spaces from column names
    df.columns = df.columns.str.strip()

    # Rename columns to match the dashboard
    df = df.rename(columns={
        "Recommended Factory": "Optimized_Factory",
        "Current Factory": "Current_Factory",
        "State": "State/Province",
        "Country": "Country/Region",
        "Optimized Units": "Allocated_Units"
    })

    # Required columns
    required = [
        "Product Name",
        "City",
        "State/Province",
        "Country/Region",
        "Optimized_Factory",
        "Allocated_Units"
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Optimized plan is missing columns: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Clean text columns
    text_columns = [
        "Product Name",
        "City",
        "State/Province",
        "Country/Region",
        "Optimized_Factory",
        "Current_Factory"
    ]

    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Convert allocated units to numbers
    df["Allocated_Units"] = pd.to_numeric(
        df["Allocated_Units"],
        errors="coerce"
    ).fillna(0)

    return df


def add_keys(df):
    df = df.copy()

    for col in [
        "Product Name", "City",
        "State/Province", "Country/Region"
    ]:
        df[col] = clean_text(df[col])

    return df


def get_distance(distance_df, factory, city, state, country):
    match = distance_df[
        (distance_df["Factory"] == str(factory).strip()) &
        (distance_df["City"] == str(city).strip()) &
        (distance_df["State"] == str(state).strip()) &
        (distance_df["Country"] == str(country).strip())
    ]

    if match.empty:
        return np.nan

    return float(match.iloc[0]["Distance_Miles"])


def calculate_factory_summary(df, factory_col, units_col, distance_col):
    temp = df.copy()

    temp[units_col] = pd.to_numeric(
        temp[units_col], errors="coerce"
    ).fillna(0)

    temp[distance_col] = pd.to_numeric(
        temp[distance_col], errors="coerce"
    )

    temp["Distance_Weighted_Miles"] = (
        temp[units_col] * temp[distance_col]
    )

    summary = temp.groupby(factory_col).agg(
        Total_Units=(units_col, "sum"),
        Total_Distance_Weighted_Miles=(
            "Distance_Weighted_Miles", "sum"
        )
    ).reset_index()

    summary = summary.rename(
        columns={factory_col: "Factory"}
    )

    return summary


# --------------------------------------------------
# 3. SIDEBAR INPUTS
# --------------------------------------------------

with st.sidebar:
    st.header("📂 Inputs")

    data_upload = st.file_uploader(
        "Source dataset (CSV) - Required",
        type=["csv"]
    )

    dist_upload = st.file_uploader(
        "Distance matrix (CSV) - Required",
        type=["csv"]
    )

    opt_upload = st.file_uploader(
        "Optimized shipping plan (CSV) - Optional",
        type=["csv"]
    )

    st.divider()

    st.info(
        "Upload your source dataset and distance matrix. "
        "Upload the PuLP optimized plan to view its results."
    )


# --------------------------------------------------
# 4. LOAD DATA
# --------------------------------------------------

try:
    if data_upload is not None:
        df = read_csv_file(data_upload)
    elif DATA_PATH.exists():
        df = read_csv_file(DATA_PATH)
    else:
        st.error("Please upload the source dataset CSV.")
        st.stop()

    if dist_upload is not None:
        distance_df = read_csv_file(dist_upload)
    elif DISTANCE_PATH.exists():
        distance_df = read_csv_file(DISTANCE_PATH)
    else:
        st.error("Please upload the distance matrix CSV.")
        st.stop()

    distance_df = normalize_distance_columns(distance_df)

    if opt_upload is not None:
        optimized_plan = normalize_optimized_columns(
            read_csv_file(opt_upload)
        )
    elif OPT_PATH.exists():
        optimized_plan = normalize_optimized_columns(
            read_csv_file(OPT_PATH)
        )
    else:
        optimized_plan = None

except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()


# --------------------------------------------------
# 5. VALIDATE SOURCE DATA
# --------------------------------------------------

df.columns = df.columns.str.strip()

required_source_columns = [
    "Product Name",
    "City",
    "State/Province",
    "Country/Region",
    "Units"
]

missing = [
    c for c in required_source_columns
    if c not in df.columns
]

if missing:
    st.error(
        f"Source dataset is missing columns: {missing}"
    )
    st.stop()

df = add_keys(df)

df["Units"] = pd.to_numeric(
    df["Units"], errors="coerce"
).fillna(0)

if "Sales" in df.columns:
    df["Sales"] = pd.to_numeric(
        df["Sales"], errors="coerce"
    ).fillna(0)

if "Gross Profit" in df.columns:
    df["Gross Profit"] = pd.to_numeric(
        df["Gross Profit"], errors="coerce"
    ).fillna(0)

# Create product-destination demand
demand = df.groupby(
    [
        "Product Name",
        "City",
        "State/Province",
        "Country/Region"
    ],
    as_index=False
).agg(
    Demand_Units=("Units", "sum")
)

demand["Current_Factory"] = demand[
    "Product Name"
].map(CURRENT_FACTORY)

demand = demand.dropna(
    subset=["Current_Factory"]
)


# --------------------------------------------------
# 6. CREATE BASELINE DISTANCE DATA
# --------------------------------------------------

baseline = demand.copy()

baseline["Current_Distance_Miles"] = baseline.apply(
    lambda row: get_distance(
        distance_df,
        row["Current_Factory"],
        row["City"],
        row["State/Province"],
        row["Country/Region"]
    ),
    axis=1
)
# Check for missing baseline distances
missing_baseline = baseline[
    "Current_Distance_Miles"
].isna().sum()

if missing_baseline > 0:
    st.error(
        f"Found {missing_baseline} destinations "
        "without matching baseline distances. "
        "Please check the distance matrix."
    )
    st.stop()

baseline["Current_Distance_Weighted_Miles"] = (
    baseline["Demand_Units"] *
    baseline["Current_Distance_Miles"]
)


# --------------------------------------------------
# 7. CREATE DISTANCE-BASED SCENARIO
# --------------------------------------------------

scenario_rows = []

for _, row in demand.iterrows():
    destination_distances = distance_df[
        (distance_df["City"] == row["City"]) &
        (distance_df["State"] == row["State/Province"]) &
        (distance_df["Country"] == row["Country/Region"])
]

    nearest = destination_distances.sort_values(
        "Distance_Miles"
    ).iloc[0]

    current_distance = get_distance(
        distance_df,
        row["Current_Factory"],
        row["City"],
        row["State/Province"],
        row["Country/Region"]
    )

    scenario_rows.append({
        "Product Name": row["Product Name"],
        "City": row["City"],
        "State/Province": row["State/Province"],
        "Country/Region": row["Country/Region"],
        "Demand_Units": row["Demand_Units"],
        "Current_Factory": row["Current_Factory"],
        "Recommended_Factory": nearest["Factory"],
        "Current_Distance_Miles": current_distance,
        "Recommended_Distance_Miles": nearest["Distance_Miles"]
    })

scenario = pd.DataFrame(scenario_rows)

if scenario.empty:
    st.error(
        "No matching destinations were found in the distance matrix."
    )
    st.stop()

scenario["Current_Weighted_Miles"] = (
    scenario["Demand_Units"] *
    scenario["Current_Distance_Miles"]
)

scenario["Recommended_Weighted_Miles"] = (
    scenario["Demand_Units"] *
    scenario["Recommended_Distance_Miles"]
)


# --------------------------------------------------
# 8. APPLY OPTIMIZED PLAN IF UPLOADED
# --------------------------------------------------

optimized_available = False
optimized_summary = None
optimized_total_units = 0
optimized_weighted_miles = 0

if optimized_plan is not None:

    optimized_available = True

    # Match the optimized plan to the distance matrix
    optimized_plan["Distance_Miles"] = optimized_plan.apply(
        lambda row: get_distance(
            distance_df,
            row["Optimized_Factory"],
            row["City"],
            row["State/Province"],
            row["Country/Region"]
        ),
        axis=1
    )

# Check for missing optimized distances

    # Check for missing optimized distances
    missing_optimized = optimized_plan["Distance_Miles"].isna().sum()

    if missing_optimized > 0:
        st.error(
            f"Found {missing_optimized} optimized allocations "
            "without matching distances. Please check the distance matrix."
        )
        st.stop()

    # Calculate weighted miles
    optimized_plan["Weighted_Miles"] = (
        optimized_plan["Allocated_Units"] *
        optimized_plan["Distance_Miles"]
    )

    optimized_total_units = optimized_plan[
        "Allocated_Units"
    ].sum()

    optimized_weighted_miles = optimized_plan[
        "Weighted_Miles"
    ].sum()

    optimized_summary = optimized_plan.groupby(
        "Optimized_Factory",
        as_index=False
    ).agg(
        Total_Units=("Allocated_Units", "sum"),
        Distance_Weighted_Miles=("Weighted_Miles", "sum")
    ).rename(
        columns={"Optimized_Factory": "Factory"}
    )
    # Aggregate the optimized plan by product and destination
    optimized_by_destination = optimized_plan.groupby(
        [
            "Product Name",
            "City",
            "State/Province",
            "Country/Region"
        ],
        as_index=False
    ).agg(
        Optimized_Units=("Allocated_Units", "sum")
    )

    scenario = scenario.merge(
        optimized_by_destination,
        on=[
            "Product Name",
            "City",
            "State/Province",
            "Country/Region"
        ],
        how="left"
    )

    scenario["Optimized_Units"] = scenario[
        "Optimized_Units"
    ].fillna(0)

    # A single recommended factory is shown only when
    # all allocated units for that destination go to one factory.
    factory_counts = optimized_plan.groupby(
        [
            "Product Name",
            "City",
            "State/Province",
            "Country/Region"
        ]
    )["Optimized_Factory"].nunique()

    single_factory_keys = factory_counts[
        factory_counts == 1
    ].index

    single_factory_plan = optimized_plan.set_index(
        [
            "Product Name",
            "City",
            "State/Province",
            "Country/Region"
        ]
    )

    def get_single_factory(row):
        key = (
            row["Product Name"],
            row["City"],
            row["State/Province"],
            row["Country/Region"]
        )

        if key not in single_factory_keys:
            return "Multiple factories"

        return single_factory_plan.loc[
            key, "Optimized_Factory"
        ].iloc[0] if isinstance(
            single_factory_plan.loc[key], pd.DataFrame
        ) else single_factory_plan.loc[key, "Optimized_Factory"]

    scenario["Optimized_Factory"] = scenario.apply(
        get_single_factory,
        axis=1
    )

    # The optimized allocation may split a destination
    # across factories. Use the full plan for its metrics.
     


# --------------------------------------------------
# 9. CALCULATE COMPARISON METRICS
# --------------------------------------------------

current_total_units = baseline["Demand_Units"].sum()

current_total_weighted = baseline[
    "Current_Distance_Weighted_Miles"
].sum()

scenario_total_weighted = scenario[
    "Recommended_Weighted_Miles"
].sum()

if optimized_available:
    comparison_weighted = optimized_weighted_miles
    comparison_units = optimized_total_units
    comparison_label = "PuLP Optimized Plan"
else:
    comparison_weighted = scenario_total_weighted
    comparison_units = scenario["Demand_Units"].sum()
    comparison_label = "Distance-Based Scenario"

if current_total_weighted > 0:
    reduction = (
        (current_total_weighted - comparison_weighted)
        / current_total_weighted
    ) * 100
else:
    reduction = 0


# --------------------------------------------------
# 10. TOP LEVEL METRICS
# --------------------------------------------------

st.subheader("📊 Project Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Total Demand Units",
    f"{int(current_total_units):,}"
)

c2.metric(
    "Product-Destination Records",
    f"{len(demand):,}"
)

c3.metric(
    "Factories",
    len(FACTORIES)
)

c4.metric(
    "Optimization Status",
    "Loaded" if optimized_available else "Not uploaded"
)

st.divider()


# --------------------------------------------------
# 11. DASHBOARD TABS
# --------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏭 Factory Optimization Simulator",
    "🔄 What-If Scenario Analysis",
    "💡 Recommendation Dashboard",
    "⚠️ Risk & Impact Panel",
    "📊 Model & Analytics"
])


# ==================================================
# TAB 1: FACTORY OPTIMIZATION SIMULATOR
# ==================================================

with tab1:

    st.header("Factory Optimization Simulator")

    st.write(
        "Compare the current factory allocation with the "
        "distance-based scenario or the uploaded PuLP plan."
    )

    st.subheader("Baseline / Historical Factory Allocation")

    current_summary = baseline.groupby(
        "Current_Factory",
        as_index=False
    ).agg(
        Total_Units=("Demand_Units", "sum"),
        Distance_Weighted_Miles=(
            "Current_Distance_Weighted_Miles", "sum"
        )
    ).rename(
        columns={"Current_Factory": "Factory"}
    )

    st.dataframe(
        current_summary,
        use_container_width=True
    )

    st.subheader("Recommended Factory Allocation")

    if optimized_available:
        st.info(
            "The uploaded PuLP plan is being used for optimized "
            "factory allocation and performance calculations."
        )

        st.dataframe(
            optimized_summary,
            use_container_width=True
        )
         
        allocation_chart = px.bar(
            optimized_summary,
            x="Factory",
            y="Total_Units",
            title="PuLP Optimized Units by Factory",
            text_auto=True
        )

    else:
        st.info(
            "No PuLP plan uploaded. Showing the nearest-factory "
            "distance-based scenario instead."
        )

        recommendation_summary = scenario.groupby(
            "Recommended_Factory",
            as_index=False
        ).agg(
            Total_Units=("Demand_Units", "sum"),
            Distance_Weighted_Miles=(
                "Recommended_Weighted_Miles", "sum"
            )
        ).rename(
            columns={"Recommended_Factory": "Factory"}
        )

        st.dataframe(
            recommendation_summary,
            use_container_width=True
        )

        allocation_chart = px.bar(
            recommendation_summary,
            x="Factory",
            y="Total_Units",
            title="Distance-Based Recommended Units by Factory",
            text_auto=True
        )

    st.plotly_chart(
        allocation_chart,
        use_container_width=True
    )

    st.subheader("Current vs Recommended Distance")

    comparison_data = pd.DataFrame({
        "Allocation": [
            "Current Allocation",
            comparison_label
        ],
        "Distance-Weighted Miles": [
            current_total_weighted,
            comparison_weighted
        ]
    })

    fig = px.bar(
        comparison_data,
        x="Allocation",
        y="Distance-Weighted Miles",
        color="Allocation",
        text_auto=".2s",
        title="Distance-Weighted Comparison"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Distance-weighted miles = allocated units × "
        "straight-line distance in miles. This is a distance "
        "proxy, not actual freight cost or delivery time."
    )


# ==================================================
# TAB 2: WHAT-IF SCENARIO ANALYSIS
# ==================================================

with tab2:

    st.header("What-If Scenario Analysis")

    st.write(
        "Explore how changing the distance preference affects "
        "factory recommendations."
    )

    st.info(
        "Note: What-If simulation uses a distance-based "
        "nearest-factory heuristic. The Recommendation Dashboard "
        "and PuLP allocation charts use the uploaded optimized "
        "solver plan. These are separate allocation methods."
)

    preference = st.slider(
        "Preference for reducing distance (%)",
        min_value=0,
        max_value=100,
        value=100,
        step=10
)

    st.caption(
        "At 100%, the scenario selects the nearest factory. "
        "At 0%, it keeps the current factory assignment."
    )
    st.write("Scenario columns:", scenario.columns.tolist())
    # Create What-If scenario
    what_if = scenario.copy()

    what_if["Scenario_Factory"] = np.where(
        preference == 100,
        what_if["Recommended_Factory"],
        what_if["Current_Factory"]
    )

    what_if["Scenario_Distance_Miles"] = np.where(
        preference == 100,
        what_if["Recommended_Distance_Miles"],
        what_if["Current_Distance_Miles"]
    )

    what_if["Current_Weighted_Miles"] = (
        what_if["Demand_Units"] * what_if["Current_Distance_Miles"]
    )

    what_if["Scenario_Weighted_Miles"] = (
        what_if["Demand_Units"] * what_if["Scenario_Distance_Miles"]
    )

    what_if_current = what_if["Current_Weighted_Miles"].sum()
    what_if_new = what_if["Scenario_Weighted_Miles"].sum()

    what_if_reduction = (
        (what_if_current - what_if_new) / what_if_current * 100
        if what_if_current > 0 else 0
    )

    a, b, c = st.columns(3)

    a.metric(
        "Current Distance-Weighted Miles",
        f"{what_if_current:,.0f}"
    )

    b.metric(
        "Scenario Distance-Weighted Miles",
        f"{what_if_new:,.0f}"
    )

    c.metric(
        "Reduction",
        f"{what_if_reduction:.2f}%"
    )

    st.subheader("Scenario Factory Allocation")

    what_if_summary = what_if.groupby(
        "Scenario_Factory",
        as_index=False
    ).agg(
        Total_Units=("Demand_Units", "sum")
    ).rename(
        columns={"Scenario_Factory": "Factory"}
    )

    fig = px.bar(
        what_if_summary,
        x="Factory",
        y="Total_Units",
        color="Factory",
        text_auto=True,
        title="Units by Factory in What-If Scenario"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Scenario Details")

    st.dataframe(
        what_if,
        use_container_width=True
    )


# ==================================================
# TAB 3: RECOMMENDATION DASHBOARD
# ==================================================

with tab3:

    st.header("Recommendation Dashboard")

    st.write(
        "View product and destination allocation recommendations."
    )

    if optimized_available:

        st.success(
            "PuLP optimized allocation is loaded."
        )

        recommendation_view = optimized_plan.copy()

        st.subheader("Optimized Factory Recommendations")

        st.dataframe(
            recommendation_view[
                [
                    "Product Name",
                    "City",
                    "State/Province",
                    "Country/Region",
                    "Optimized_Factory",
                    "Allocated_Units",
                    "Distance_Miles"
                ]
            ],
            use_container_width=True
        )

        st.download_button(
            "⬇️ Download Optimized Recommendations",
            data=optimized_plan.to_csv(index=False).encode("utf-8"),
            file_name="optimized_factory_recommendations.csv",
            mime="text/csv"
        )

        # Use Current_Factory already present in the optimized plan
        reassigned = optimized_plan.copy()

        # Fallback: get Current_Factory from demand if it is missing
        if "Current_Factory" not in reassigned.columns:
            reassigned = reassigned.merge(
                demand[
                    [
                        "Product Name",
                        "City",
                        "State/Province",
                        "Country/Region",
                        "Current_Factory"
                    ]
                ],
                on=[
                    "Product Name",
                    "City",
                    "State/Province",
                    "Country/Region"
                ],
                how="left"
            )

        # Compare optimized factory with original factory
        reassigned["Is_Reassigned"] = (
            reassigned["Optimized_Factory"] !=
            reassigned["Current_Factory"]
        )

        reassigned_units = reassigned.loc[
            reassigned["Is_Reassigned"],
            "Allocated_Units"
        ].sum()

        unchanged_units = reassigned.loc[
            ~reassigned["Is_Reassigned"],
            "Allocated_Units"
        ].sum()

        x, y = st.columns(2)

        x.metric(
            "Units Reassigned",
            f"{int(reassigned_units):,}"
        )

        y.metric(
            "Units Remaining at Current Factory",
            f"{int(unchanged_units):,}"
        )

        factory_chart = px.bar(
            optimized_summary,
            x="Factory",
            y="Total_Units",
            color="Factory",
            text_auto=True,
            title="Optimized Factory Workload"
        )

        st.plotly_chart(
            factory_chart,
            use_container_width=True
        )

    else:

        st.warning(
            "Upload optimized_factory_allocation.csv to see "
            "the PuLP recommendations here."
        )

        st.subheader("Nearest Factory Recommendations")

        st.dataframe(
            scenario[
                [
                    "Product Name",
                    "City",
                    "State/Province",
                    "Country/Region",
                    "Current_Factory",
                    "Recommended_Factory",
                    "Demand_Units",
                    "Recommended_Distance_Miles"
                ]
            ],
            use_container_width=True
        )

    st.subheader("Product-Level Demand")

    product_summary = demand.groupby(
        "Product Name",
        as_index=False
    ).agg(
        Total_Demand=("Demand_Units", "sum")
    ).sort_values(
        "Total_Demand",
        ascending=False
    )

    fig = px.bar(
        product_summary,
        x="Product Name",
        y="Total_Demand",
        title="Demand by Product"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ==================================================
# TAB 4: RISK & IMPACT PANEL
# ==================================================

with tab4:

    st.header("Risk & Impact Panel")

    st.write(
        "Review data quality, allocation risks, and the "
        "estimated impact of factory reallocation."
    )

    st.subheader("Dataset Information")

    r1, r2, r3 = st.columns(3)

    r1.metric(
        "Source Dataset Rows",
        f"{len(df):,}"
    )

    r2.metric(
        "Demand Records",
        f"{len(demand):,}"
    )

    r3.metric(
        "Distance Matrix Records",
        f"{len(distance_df):,}"
    )

    st.subheader("Data Quality Checks")

    quality = pd.DataFrame({
        "Check": [
            "Missing values in source dataset",
            "Duplicate rows in source dataset",
            "Negative demand units",
            "Missing distance values",
            "Unmapped products"
        ],
        "Result": [
            int(df.isna().sum().sum()),
            int(df.duplicated().sum()),
            int((df["Units"] < 0).sum()),
            int(distance_df["Distance_Miles"].isna().sum()),
            int(df["Product Name"].map(CURRENT_FACTORY).isna().sum())
        ]
    })

    st.dataframe(
        quality,
        use_container_width=True
    )

    st.subheader("Optimization Impact")

    impact_value = comparison_weighted

    i1, i2, i3 = st.columns(3)

    i1.metric(
        "Current Distance-Weighted Miles",
        f"{current_total_weighted:,.0f}"
    )

    i2.metric(
        "Selected Plan Distance-Weighted Miles",
        f"{impact_value:,.0f}"
    )

    i3.metric(
        "Estimated Distance Reduction",
        f"{reduction:.2f}%"
    )

    if optimized_available:
        st.info(
            "The impact figures use the uploaded PuLP optimized "
            "plan and its allocated units."
        )
    else:
        st.info(
            "The impact figures use the nearest-factory "
            "distance-based scenario."
        )

    st.subheader("Important Model Limitations")

    st.warning(
        """
        1. Factory-to-destination distances are straight-line
           Haversine distances, not actual road distances.

        2. Distance-weighted miles are a proxy. They are not
           actual shipping cost savings.

        3. Factory capacities in the optimization model are
           assumed values and need business verification.

        4. Product-to-factory compatibility must be confirmed
           before implementing recommendations.

        5. The dashboard does not predict actual delivery time.
        """
    )

    st.subheader("Download Data")

    st.download_button(
        "⬇️ Download Demand Summary",
        data=demand.to_csv(index=False).encode("utf-8"),
        file_name="demand_summary.csv",
        mime="text/csv"
    )

    st.download_button(
        "⬇️ Download Distance Matrix",
        data=distance_df.to_csv(index=False).encode("utf-8"),
        file_name="distance_matrix.csv",
        mime="text/csv"
    )

    if optimized_available:
        st.download_button(
            "⬇️ Download Optimized Plan",
            data=optimized_plan.to_csv(index=False).encode("utf-8"),
            file_name="optimized_factory_allocation.csv",
            mime="text/csv"
        )


# ==================================================
# TAB 5: MODEL & ANALYTICS
# ==================================================

with tab5:

    st.header("📊 Machine Learning Model Evaluation")

    st.write(
        "Comparison of regression models used to predict "
        "the optimized average shipping distance."
    )

    # Model evaluation results from Google Colab
    model_results = pd.DataFrame({
        "Model": [
            "Random Forest",
            "Gradient Boosting",
            "Linear Regression"
        ],
        "MAE": [
            68.596317,
            83.766681,
            138.732259
        ],
        "RMSE": [
            114.663226,
            123.205636,
            186.555183
        ],
        "R2 Score": [
            0.787027,
            0.754112,
            0.436244
        ]
    })

    st.subheader("Regression Model Comparison")

    st.dataframe(
        model_results.round(3),
        use_container_width=True,
        hide_index=True
    )

    # R2 comparison chart
    fig = px.bar(
        model_results,
        x="Model",
        y="R2 Score",
        color="Model",
        text_auto=".3f",
        title="R² Score Comparison"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.success(
        "Random Forest performed best among the three tested "
        "models, with an R² score of 0.787 on unseen destinations."
    )

    st.caption(
        "MAE and RMSE measure prediction error in miles. "
        "R² measures how well the model explains variation "
        "in the target values. Lower MAE/RMSE and higher R² "
        "are preferred."
    )

    # ------------------------------------------
    # RANDOM FOREST FEATURE IMPORTANCE
    # ------------------------------------------

    st.divider()

    st.subheader("🌳 Random Forest Feature Importance")

    importance_df = pd.DataFrame({
        "Feature": [
            "Current Distance",
            "Region",
            "Current Factory",
            "Units",
            "Product Name",
            "Gross Profit",
            "Sales",
            "Cost"
        ],
        "Importance": [
            149.639489,
            135.980888,
            68.288826,
            0.196969,
            0.055705,
            -0.138793,
            -0.340604,
            -1.039752
        ]
    })

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=True
    )

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        title="Random Forest Feature Importance (Permutation)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.write("**Key Findings:**")

    st.write(
        "- Current Distance is the most influential feature."
    )

    st.write(
        "- Region is the second most influential feature."
    )

    st.write(
        "- Current Factory also contributes significantly."
    )

    st.write(
        "- Sales, Cost and Gross Profit showed little "
        "predictive contribution in this evaluation."
    )

    st.caption(
        "Permutation importance measures the change in "
        "MAE when a feature is shuffled. Negative values "
        "mean shuffling slightly improved the model score."
    )

    # ------------------------------------------
    # DESTINATION CLUSTERING ANALYSIS
    # ------------------------------------------

    st.divider()

    st.subheader("📍 K-Means Destination Clustering")

    st.write(
        "Destinations were grouped based on their distance "
        "from the five factories and their total demand."
    )

    cluster_results = pd.DataFrame({
        "Cluster": ["Cluster 0", "Cluster 1"],
        "Destinations": [155, 461],
        "Average Units": [76.43, 58.15],
        "Total Units": [11847, 26807]
    })

    col1, col2 = st.columns(2)

    col1.metric(
        "Best Number of Clusters",
        "2"
    )

    col2.metric(
        "Silhouette Score",
        "0.5229"
    )

    st.subheader("Cluster Summary")

    st.dataframe(
        cluster_results,
        use_container_width=True,
        hide_index=True
    )

    fig = px.bar(
        cluster_results,
        x="Cluster",
        y="Destinations",
        color="Cluster",
        text_auto=True,
        title="Number of Destinations in Each Cluster"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.write("**Cluster Insights:**")

    st.write(
        "- Cluster 0 contains 155 destinations and is "
        "mostly closest to Lot's O' Nuts."
    )

    st.write(
        "- Cluster 1 contains 461 destinations and has "
        "a more geographically distributed factory pattern."
    )

    st.caption(
        "Cluster numbers are labels, not rankings. "
        "The silhouette score measures how well-separated "
        "the clusters are."
    )

    # ------------------------------------------
    # SHIPPING COST SENSITIVITY ANALYSIS
    # ------------------------------------------

    st.divider()

    st.subheader("💰 Shipping Cost Sensitivity Analysis")

    st.write(
        "Estimate potential shipping savings using different "
        "hypothetical costs per unit-mile."
    )

    sensitivity_df = pd.DataFrame({
        "Cost per Unit-Mile ($)": [
            0.001, 0.005, 0.010, 0.020
        ],
        "Baseline Shipping Cost ($)": [
            48383.83, 241919.17, 483838.33, 967676.67
        ],
        "Optimized Shipping Cost ($)": [
            19574.42, 97872.11, 195744.22, 391488.45
        ],
        "Potential Savings ($)": [
            28809.41, 144047.05, 288094.11, 576188.22
        ],
        "Reduction (%)": [
            59.54, 59.54, 59.54, 59.54
        ]
    })

    st.subheader("Shipping Cost Comparison")

    st.dataframe(
        sensitivity_df.round(3),
        use_container_width=True,
        hide_index=True
    )

    fig = px.line(
        sensitivity_df,
        x="Cost per Unit-Mile ($)",
        y=[
            "Baseline Shipping Cost ($)",
            "Optimized Shipping Cost ($)"
        ],
        markers=True,
        title="Baseline vs Optimized Shipping Cost"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.metric(
        "Distance-Weighted Reduction",
        "59.54%"
    )

    st.caption(
        "These are hypothetical cost estimates, not actual "
        "shipping expenses or verified company savings. "
        "The calculation uses straight-line distance multiplied "
        "by allocated units."
    )

    # ------------------------------------------
    # FINAL PROJECT KPIs
    # ------------------------------------------

    st.divider()

    st.subheader("🏆 Final Project Performance KPIs")

    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        "Total Products",
        "15"
    )

    k2.metric(
        "Total Destinations",
        "616"
    )

    k3.metric(
        "Total Units",
        "38,654"
    )

    k4.metric(
        "Total Allocations",
        "2,267"
    )

    k5, k6, k7, k8 = st.columns(4)

    k5.metric(
        "Reassigned Units",
        "26,557"
    )

    k6.metric(
        "Reassigned Allocations",
        "1,654"
    )

    k7.metric(
        "Baseline Unit-Miles",
        "48.38M"
    )

    k8.metric(
        "Optimized Unit-Miles",
        "19.57M"
    )

    st.divider()

    st.subheader("Overall Optimization Result")

    st.metric(
        "Distance-Weighted Reduction",
        "59.54%",
        delta="Improvement"
    )

    st.success(
        "Random Forest was the best-performing regression model "
        "with an R² score of 0.787 on unseen destinations."
    )

    st.write(
        "**Project Conclusion:** The proposed factory "
        "reallocation plan reduced distance-weighted miles "
        "by 59.54% in the evaluated scenario. The analysis "
        "also identified destination clusters, evaluated "
        "regression models and estimated shipping-cost "
        "sensitivity under hypothetical transportation rates."
    )

    st.caption(
        "All distance improvements are based on straight-line "
        "Haversine distances multiplied by allocated units. "
        "They do not represent verified road-distance or "
        "actual freight-cost savings."
    )

# --------------------------------------------------
# 12. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Nassau Candy Distributor | Factory Reallocation & "
    "Shipping Optimization Recommendation System"
)
