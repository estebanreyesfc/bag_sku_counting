import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(page_title="Bag Counting + SKU Dashboard", layout="wide")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("results_videos_sku_with_url.csv")

    df["camera"] = df["name"].str.extract(r'(Camera\d+)')[0]

    def extract_start_time(name):
        try:
            return pd.to_datetime(name.split("_")[2], format="%Y%m%d%H%M%S")
        except:
            return pd.NaT

    df["start_time"] = df["name"].apply(extract_start_time)

    # ✅ FIX: Unique video identifier
    df["folder"] = df["folder"].fillna("root")
    df["video_id"] = df["folder"] + "||" + df["name"]

    # Fix video URL
    df["video_url"] = df["video_url"].str.extract(r'id=([^&]+)')[0] \
        .apply(lambda x: f"https://drive.google.com/file/d/{x}/preview" if pd.notna(x) else np.nan)

    return df


df = load_data()

# -----------------------------
# TREE UI
# -----------------------------
def build_tree(df):
    tree = {}
    for _, row in df.iterrows():
        folder_path = row["folder"]
        parts = folder_path.split("/")

        current = tree
        for part in parts:
            current = current.setdefault(part, {})

        current.setdefault("_videos", []).append({
            "name": row["name"],
            "video_id": row["video_id"]
        })
    return tree


def render_tree(tree, path=""):
    for key, value in tree.items():
        if key == "_videos":
            for i, video in enumerate(value):
                vid = video["video_id"]
                label = video["name"]

                unique_key = f"{path}_{vid}_{i}"

                if st.button(f"🎥 {label}", key=unique_key, use_container_width=True):
                    st.session_state["selected_video"] = vid
            continue

        new_path = f"{path}/{key}"
        with st.expander(f"📁 {key}", expanded=False):
            render_tree(value, new_path)

# -----------------------------
# FILTERS
# -----------------------------
st.sidebar.title("⚙️ Filters")

folders = st.sidebar.multiselect(
    "Folder",
    df["folder"].unique(),
    default=df["folder"].unique()
)

cameras = st.sidebar.multiselect(
    "Camera",
    df["camera"].dropna().unique(),
    default=df["camera"].dropna().unique()
)

df_filtered = df[
    (df["folder"].isin(folders)) &
    (df["camera"].isin(cameras))
].copy()

# -----------------------------
# PRECOMPUTE
# -----------------------------
df_filtered["error"] = df_filtered["count"] - df_filtered["manual_count"]
df_filtered["abs_error"] = df_filtered["error"].abs()

df_valid = df_filtered.dropna(subset=["count", "manual_count"])


# -----------------------------
# GLOBAL METRICS
# -----------------------------
total_error = df_valid["abs_error"].sum()
total_gt = df_valid["manual_count"].sum()

volume_accuracy = 1 - total_error / total_gt if total_gt > 0 else np.nan

weighted_acc_1 = (
    (df_valid["abs_error"] <= 1) * df_valid["manual_count"]
).sum() / total_gt if total_gt > 0 else np.nan

weighted_acc_2 = (
    (df_valid["abs_error"] <= 2) * df_valid["manual_count"]
).sum() / total_gt if total_gt > 0 else np.nan

mae = df_valid["abs_error"].mean()
bias = df_valid["error"].mean()

correlation = df_valid[["manual_count", "count"]].corr().iloc[0, 1]
correlation = correlation if pd.notna(correlation) else 0

# -----------------------------
# VIDEO METRICS (FIXED)
# -----------------------------
df_video = df_valid.groupby("video_id").agg(
    total_gt=("manual_count", "sum"),
    total_error=("abs_error", "sum"),
    mae=("abs_error", "mean"),
    video_url=("video_url", "first"),
    start_time=("start_time", "first"),
    name=("name", "first"),
    folder=("folder", "first")
).reset_index()

df_video["accuracy"] = np.where(
    df_video["total_gt"] > 0,
    1 - df_video["total_error"] / df_video["total_gt"],
    np.nan
)

perfect_pct = (df_video["total_error"] == 0).mean()
near_perfect = (df_video["accuracy"] >= 0.98).mean()

all_videos = set(df["video_id"].unique())
valid_videos = set(df_valid["video_id"].unique())

missing_videos = all_videos - valid_videos

# -----------------------------
# KPI DISPLAY
# -----------------------------
st.title("📦 Bag Counting + SKU Dashboard")

c1, c2, c3, c4, c5, c6 = st.columns(6)

c1.metric("Volume Accuracy", f"{volume_accuracy:.1%}")
c2.metric("±1 (Vol)", f"{weighted_acc_1:.1%}")
c3.metric("±2 (Vol)", f"{weighted_acc_2:.1%}")
c4.metric("MAE", f"{mae:.2f}")
c5.metric("Bias", f"{bias:+.2f}")
c6.metric("Corr", f"{correlation:.3f}")

c7, c8 = st.columns(2)
c7.metric("Perfect Videos (100%)", f"{perfect_pct:.1%}")
c8.metric("Near Perfect (≥98%)", f"{near_perfect:.1%}")

c9, c10 = st.columns(2)
c9.metric("Total Loads", df_video.shape[0])
c10.metric("Total Bags", f"{total_gt:,.0f}")

# -----------------------------
# CORRELATION
# -----------------------------
st.subheader("🔗 Prediction vs Ground Truth")

fig = px.scatter(
    df_valid,
    x="manual_count",
    y="count",
    color="sku",
    trendline="ols"
)
st.plotly_chart(fig, width="stretch")

# -----------------------------
# ERROR VS LOAD SIZE
# -----------------------------
st.subheader("📏 Error vs Load Size")

fig = px.scatter(
    df_video,
    x="total_gt",
    y="total_error",
    trendline="ols",
    hover_data=["name", "folder"]
)
st.plotly_chart(fig, width="stretch")

# -----------------------------
# ERROR DISTRIBUTION
# -----------------------------
st.subheader("📊 Error Distribution")

fig = px.histogram(df_valid, x="error", nbins=30)
st.plotly_chart(fig, width="stretch")

# -----------------------------
# SKU ANALYSIS
# -----------------------------
st.subheader("🎯 Accuracy per SKU")

sku_acc = df_valid.groupby("sku").agg(
    total_gt=("manual_count", "sum"),
    total_error=("abs_error", "sum"),
    mae=("abs_error", "mean"),
    samples=("sku", "count")
).reset_index()

sku_acc["accuracy"] = np.where(
    sku_acc["total_gt"] > 0,
    1 - sku_acc["total_error"] / sku_acc["total_gt"],
    np.nan
)

sku_acc = sku_acc.sort_values("accuracy")

fig = px.bar(sku_acc, x="sku", y="accuracy", text="samples")
st.plotly_chart(fig, width="stretch")

st.dataframe(sku_acc, width="stretch")

# -----------------------------
# WORST ROWS
# -----------------------------
st.subheader("🚨 Worst Belt-Level Errors")

worst_rows = df_valid.sort_values("abs_error", ascending=False).head(20)

st.dataframe(
    worst_rows[["folder", "name", "belt", "sku", "count", "manual_count", "error"]],
    width="stretch"
)

# -----------------------------
# BEST / WORST VIDEOS
# -----------------------------
st.subheader("🏆 Best Videos (Perfect)")
st.dataframe(df_video[df_video["total_error"] == 0].head(10), width="stretch")

st.subheader("🚨 Worst Videos")
st.dataframe(df_video.sort_values("total_error", ascending=False).head(10), width="stretch")

# -----------------------------
# VIDEO INSPECTOR
# -----------------------------
col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("📂 Video Explorer")

    tree = build_tree(df_video)
    render_tree(tree)

    if "selected_video" not in st.session_state and len(df_video) > 0:
        st.session_state["selected_video"] = df_video["video_id"].iloc[0]

selected_video = st.session_state.get("selected_video")

with col_right:
    st.subheader("🎥 Video Inspector")

    video_rows = df_valid[df_valid["video_id"] == selected_video]

    if len(video_rows) > 0:
        video_rows = video_rows.sort_values("abs_error", ascending=False)

        video_url = video_rows["video_url"].iloc[0]

        col1, col2 = st.columns([2, 1])

        with col1:
            if pd.notna(video_url):
                st.iframe(video_url, height=500)

        with col2:
            st.markdown("### 📊 Belts")

            for _, row in video_rows.iterrows():
                gt = row["manual_count"]
                pred = row["count"]

                color = "🟢" if pred == gt else "🔴" if pred > gt else "🟡"
                st.write(f"{color} Belt {row['belt']} | {row['sku']}: {pred} / {gt}")

# -----------------------------
# BIAS
# -----------------------------
st.subheader("📉 Bias Analysis")

fig = px.histogram(df_valid, x="error", nbins=50)
st.plotly_chart(fig, width="stretch")
