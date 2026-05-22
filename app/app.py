import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="Retail Customer Intelligence Dashboard",
    layout="wide"
)

# -----------------------------------
# LOAD DATA
# -----------------------------------

clean_df = pd.read_csv(
    "data/cleaned/clean_retail.zip",
    compression="zip"
)

rules = pd.read_csv(
    "data/cleaned/association_rules.csv"
)

# -----------------------------------
# RULES PREPROCESSING
# -----------------------------------

rules["antecedents"] = (
    rules["antecedents"]
    .astype(str)
)

rules["consequents"] = (
    rules["consequents"]
    .astype(str)
)

# -----------------------------------
# DATA PREPROCESSING
# -----------------------------------

clean_df["InvoiceDate"] = pd.to_datetime(
    clean_df["InvoiceDate"]
)

clean_df["Month"] = (
    clean_df["InvoiceDate"]
    .dt.month
)

clean_df["Hour"] = (
    clean_df["InvoiceDate"]
    .dt.hour
)

# -----------------------------------
# SIDEBAR FILTERS
# -----------------------------------

st.sidebar.header(
    "Dashboard Filters"
)

country = st.sidebar.selectbox(
    "Select Country",
    sorted(clean_df["Country"].dropna().unique())
)

# -----------------------------------
# FILTER DATA
# -----------------------------------

filtered_df = clean_df[
    clean_df["Country"] == country
]

# -----------------------------------
# KPI CALCULATIONS
# -----------------------------------

total_revenue = (
    filtered_df["TotalPrice"]
    .sum()
)

total_customers = (
    filtered_df["CustomerID"]
    .nunique()
)

total_orders = (
    filtered_df["InvoiceNo"]
    .nunique()
)

average_order_value = (
    total_revenue / total_orders
    if total_orders != 0 else 0
)

# -----------------------------------
# TITLE
# -----------------------------------

st.title(
    "Retail Customer Intelligence Dashboard"
)

st.write(
    "End-to-End Retail Analytics Platform"
)

# -----------------------------------
# DASHBOARD TABS
# -----------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Customer Analytics",
    "Recommendations",
    "Forecasting"
])

# ===================================
# TAB 1 — OVERVIEW
# ===================================

with tab1:

    # KPI CARDS

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Revenue",
        f"${total_revenue:,.0f}"
    )

    col2.metric(
        "Customers",
        total_customers
    )

    col3.metric(
        "Orders",
        total_orders
    )

    col4.metric(
        "Avg Order Value",
        f"${average_order_value:,.2f}"
    )

    # MONTHLY REVENUE TREND

    st.subheader(
        "Monthly Revenue Trend"
    )

    monthly_sales = (
        filtered_df
        .groupby("Month")["TotalPrice"]
        .sum()
        .reset_index()
    )

    fig_monthly = px.line(
        monthly_sales,
        x="Month",
        y="TotalPrice",
        title="Monthly Revenue Trend",
        markers=True
    )

    st.plotly_chart(
        fig_monthly,
        use_container_width=True
    )

    # TOP PRODUCTS

    st.subheader(
        "Top Selling Products"
    )

    top_products = (
        filtered_df
        .groupby("Description")["Quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig_products = px.bar(
        top_products,
        x="Quantity",
        y="Description",
        orientation="h",
        title="Top 10 Products"
    )

    st.plotly_chart(
        fig_products,
        use_container_width=True
    )

    # HOURLY SALES

    st.subheader(
        "Purchase Trend by Hour"
    )

    hourly_sales = (
        filtered_df
        .groupby("Hour")["InvoiceNo"]
        .count()
        .reset_index()
    )

    fig_hourly = px.line(
        hourly_sales,
        x="Hour",
        y="InvoiceNo",
        title="Orders by Hour",
        markers=True
    )

    st.plotly_chart(
        fig_hourly,
        use_container_width=True
    )

# ===================================
# TAB 2 — CUSTOMER ANALYTICS
# ===================================

with tab2:

    # CUSTOMER SPENDING

    st.subheader(
        "Customer Spending Distribution"
    )

    customer_spending = (
        filtered_df
        .groupby("CustomerID")["TotalPrice"]
        .sum()
        .reset_index()
    )

    fig_spending = px.histogram(
        customer_spending,
        x="TotalPrice",
        nbins=40,
        title="Customer Spending Distribution"
    )

    st.plotly_chart(
        fig_spending,
        use_container_width=True
    )

    # RFM ANALYSIS

    st.subheader(
        "Customer Segmentation (RFM)"
    )

    snapshot_date = (
        filtered_df["InvoiceDate"]
        .max()
    )

    rfm = (
        filtered_df
        .groupby("CustomerID")
        .agg({
            "InvoiceDate": lambda x:
                (snapshot_date - x.max()).days,

            "InvoiceNo": "nunique",

            "TotalPrice": "sum"
        })
    )

    rfm.columns = [
        "Recency",
        "Frequency",
        "Monetary"
    ]

    rfm["R_Score"] = pd.qcut(
        rfm["Recency"],
        4,
        labels=[4,3,2,1]
    )

    rfm["F_Score"] = pd.qcut(
        rfm["Frequency"].rank(method="first"),
        4,
        labels=[1,2,3,4]
    )

    rfm["M_Score"] = pd.qcut(
        rfm["Monetary"],
        4,
        labels=[1,2,3,4]
    )

    rfm["R_Score"] = rfm["R_Score"].astype(int)
    rfm["F_Score"] = rfm["F_Score"].astype(int)
    rfm["M_Score"] = rfm["M_Score"].astype(int)

    # SEGMENTATION

    def segment_customer(row):

        if (
            row["R_Score"] >= 3 and
            row["F_Score"] >= 3 and
            row["M_Score"] >= 3
        ):
            return "VIP"

        elif row["F_Score"] >= 3:
            return "Loyal Customer"

        elif row["M_Score"] >= 3:
            return "Big Spender"

        elif row["R_Score"] >= 3:
            return "Recent Customer"

        else:
            return "Regular"

    rfm["Segment"] = (
        rfm
        .apply(segment_customer, axis=1)
    )

    segment_counts = (
        rfm["Segment"]
        .value_counts()
        .reset_index()
    )

    segment_counts.columns = [
        "Segment",
        "Count"
    ]

    fig_segments = px.bar(
        segment_counts,
        x="Segment",
        y="Count",
        color="Segment",
        title="Customer Segments"
    )

    st.plotly_chart(
        fig_segments,
        use_container_width=True
    )

    st.dataframe(
        rfm.head(20)
    )

# ===================================
# TAB 3 — RECOMMENDATIONS
# ===================================

with tab3:

    st.subheader(
        "Product Recommendation Engine"
    )

    product_list = (
        rules["antecedents"]
        .dropna()
        .unique()
    )

    selected_product = st.selectbox(
        "Select a Product",
        sorted(product_list)
    )

    recommendations = rules[
        rules["antecedents"]
        .str.contains(
            selected_product,
            case=False,
            na=False
        )
    ]

    if len(recommendations) > 0:

        recommendations = (
            recommendations[
                [
                    "consequents",
                    "confidence",
                    "lift"
                ]
            ]
            .sort_values(
                by="lift",
                ascending=False
            )
            .head(10)
        )

        st.dataframe(
            recommendations
        )

    else:

        st.write(
            "No recommendations found."
        )

# ===================================
# TAB 4 — FORECASTING
# ===================================

with tab4:

    st.subheader(
        "Sales Forecasting"
    )

    st.info(
        "Forecasting dashboard will be integrated in next phase."
    )

# -----------------------------------
# DATASET PREVIEW
# -----------------------------------

st.subheader(
    "Filtered Dataset Preview"
)

st.dataframe(
    filtered_df.head(10)
)

# -----------------------------------
# FOOTER
# -----------------------------------

st.markdown("---")

st.write(
    "Built using Python, Streamlit, Plotly, Pandas, and Machine Learning concepts."
)
