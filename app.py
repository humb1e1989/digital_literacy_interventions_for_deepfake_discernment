import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import re
from pathlib import Path
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Deepfake Discernment Study",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).parent

# ── Helpers ───────────────────────────────────────────────────────────────────
def clean_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(text))
    return re.sub(r"\s+", " ", text).strip()

@st.cache_data
def load_raw_data():
    g1 = pd.read_csv(BASE / "Data/group1_data_raw.csv")
    g2 = pd.read_csv(BASE / "Data/group2_data_raw.csv")
    g1["data_group"] = "Group 1"
    g2["data_group"] = "Group 2"
    return pd.concat([g1, g2], ignore_index=True)

@st.cache_data
def load_metrics():
    df  = pd.read_csv(BASE / "Data/all_participant_metrics.zip", compression="zip")
    dfu = pd.read_csv(BASE / "Data/all_participant_metrics_followup.zip", compression="zip")
    dft = pd.read_csv(BASE / "Data/processed/pilot2/metrics_results_over_time.csv")
    return df, dfu, dft

@st.cache_data
def load_qsf(filename: str):
    with open(BASE / "Survey" / filename, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_qsf(qsf):
    elements = qsf["SurveyElements"]
    q_map = {}
    for e in elements:
        if e.get("Element") != "SQ":
            continue
        p = e.get("Payload", {})
        qid = p.get("QuestionID", "")
        choices = []
        raw_choices = p.get("Choices", {})
        if isinstance(raw_choices, dict):
            for _, cv in raw_choices.items():
                if isinstance(cv, dict):
                    choices.append(clean_html(cv.get("Display", "")))
        q_map[qid] = {
            "id": qid,
            "type": p.get("QuestionType", ""),
            "text": clean_html(p.get("QuestionText", "")),
            "choices": choices,
        }
    bl_el = next((e for e in elements if e.get("Element") == "BL"), None)
    blocks = {}
    if bl_el:
        payload = bl_el.get("Payload", {})
        items = payload.values() if isinstance(payload, dict) else payload
        for blk in items:
            if not isinstance(blk, dict):
                continue
            name = blk.get("Description", "Unnamed")
            if "Trash" in name:
                continue
            qs = [q_map[be["QuestionID"]]
                  for be in blk.get("BlockElements", [])
                  if be.get("Type") == "Question" and be.get("QuestionID") in q_map]
            blocks[name] = qs
    return blocks

GROUP_COLORS = {
    "control":        "#B22222",
    "intervention_1": "#4D8E8A",
    "intervention_2": "#5C2D91",
    "intervention_3": "#D16DB3",
    "intervention_4": "#72AEE6",
    "intervention_5": "#1F4B82",
}
GROUP_LABELS = {
    "control":        "Control",
    "intervention_1": "Knowledge",
    "intervention_2": "Textual",
    "intervention_3": "Visual",
    "intervention_4": "Gamified",
    "intervention_5": "Feedback",
}
ORDER = ["control", "intervention_2", "intervention_3", "intervention_4", "intervention_5", "intervention_1"]

QTYPE_LABELS = {
    "MC": "Multiple Choice", "Matrix": "Matrix", "TE": "Text Entry",
    "DB": "Display Image", "Slider": "Slider", "CS": "Constant Sum",
    "RO": "Rank Order", "SBS": "Side by Side", "": "Other",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🔍 Deepfake Discernment")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["📋 Project Overview", "📊 Raw Data Analysis", "📝 Survey Structure (QSF)",
     "📈 Statistical Results", "🎨 Visualizations"],
    label_visibility="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PROJECT OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📋 Project Overview":
    st.title("🔍 Designing Effective Digital Literacy Interventions for Boosting Deepfake Discernment")
    st.markdown("*Designing Effective Digital Literacy Interventions for Boosting Deepfake Discernment*")
    st.markdown("---")

    st.header("Abstract")
    st.info(
        "Deepfake images can erode trust in institutions and compromise election outcomes, "
        "as people often struggle to discern real images from deepfake images. Improving digital "
        "literacy can help address these challenges. This study compares the efficacy of five digital "
        "literacy interventions to boost deepfake discernment with **N=1,200** participants from the "
        "United States.\n\n"
        "Results show that lightweight, easy-to-understand interventions can boost deepfake image "
        "discernment by up to **13 percentage points** while maintaining trust in real images."
    )

    st.markdown("---")
    st.header("Study Design")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Participants", "1,200")
        st.metric("Intervention Groups", "6 (incl. 1 control)")
    with c2:
        st.metric("Measurement Time Points", "2 (T1 main + T2 follow-up)")
        st.metric("Images per Task", "15 (5 real + 10 deepfake)")
    with c3:
        st.metric("Follow-up Participants", "~811 (~67%)")
        st.metric("Country", "United States (Prolific)")

    st.markdown("---")
    st.header("Six Intervention Conditions")
    interventions = [
        ("🔴 Control", "No intervention. Participants complete the image detection task directly, providing a baseline."),
        ("🟣 Textual Tips", "Participants read five text-based tips describing common visual artifacts in deepfakes (unnatural skin, eye anomalies, blurred edges, etc.)."),
        ("🟠 Visual Demonstrations", "Participants view annotated example images highlighting deepfake artifacts, providing intuitive visual recognition cues."),
        ("🔵 Gamified Exercise", "Participants play an interactive game: judge each image as real or fake and receive immediate score feedback."),
        ("🔷 Implicit Learning (Feedback)", "Participants rate an image, then immediately receive the correct label — learning through repeated exposure and feedback."),
        ("🟢 AI Knowledge", "Participants read an explanation of how deepfakes are generated (GANs, diffusion models, etc.)."),
    ]
    for title, desc in interventions:
        with st.expander(title):
            st.write(desc)

    st.markdown("---")
    st.header("Repository Structure")
    st.code(
        "digital_literacy_interventions_for_deepfake_discernment/\n"
        "├── Data/              # Raw CSV data (group1/group2 main + follow-up)\n"
        "├── Code/              # Analysis scripts\n"
        "│   ├── main.py        # Data processing + statistical tests\n"
        "│   ├── regressions.py # Robustness regression checks\n"
        "│   └── visualizations.py\n"
        "├── Survey/            # Four Qualtrics QSF survey files\n"
        "└── Results/pilot2/    # Output figures (PDF + PNG)",
        language="text",
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — RAW DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Raw Data Analysis":
    st.title("Raw Data Analysis")
    with st.spinner("Loading data..."):
        raw = load_raw_data()
    metrics, _, _ = load_metrics()

    g1_n = (raw["data_group"] == "Group 1").sum()
    g2_n = (raw["data_group"] == "Group 2").sum()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Group 1 Participants", g1_n)
    c2.metric("Group 2 Participants", g2_n)
    c3.metric("Total (raw)", g1_n + g2_n)
    c4.metric("After Cleaning (main)", len(metrics))

    st.markdown("---")
    st.markdown("### Participant Demographics")
    tab_age, tab_gen, tab_edu, tab_eth, tab_grp, tab_time = st.tabs(
        ["Age", "Gender", "Education", "Ethnicity", "Intervention Groups", "Completion Time"]
    )

    with tab_age:
        age = raw["age"].dropna()
        age = age[age >= 18]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(age, bins=30, color="#4682B4", edgecolor="white", alpha=0.85)
        ax.axvline(age.mean(),   color="red",    linestyle="--", label=f"Mean {age.mean():.1f}")
        ax.axvline(age.median(), color="green",  linestyle=":",  label=f"Median {age.median():.0f}")
        ax.set_xlabel("Age"); ax.set_ylabel("Count"); ax.legend()
        ax.set_title("Age Distribution")
        plt.tight_layout()
        st.pyplot(fig); plt.close()
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean Age",   f"{age.mean():.1f}")
        c2.metric("Median Age", f"{age.median():.0f}")
        c3.metric("Range",      f"{int(age.min())}–{int(age.max())}")

    with tab_gen:
        gcnt = raw["gender"].value_counts()
        fig = go.Figure(go.Pie(
            labels=gcnt.index.tolist(), values=gcnt.values.tolist(),
            textinfo="percent+label",
            marker_colors=["#4682B4", "#FF8C69", "#90EE90", "#DDA0DD"],
        ))
        fig.update_layout(title="Gender Distribution", height=380)
        st.plotly_chart(fig, use_container_width=True)

    with tab_edu:
        edu_map = {
            "0-6 (up to Primary School)":  "Primary",
            "7-12 (up to High School)":    "High School",
            "13-16 (college/undergraduate university/certificate training)": "College/Bachelor",
            "More than 17 years (doctorate degree, medical degree, etc.)":   "Graduate",
        }
        edu = raw["education"].map(edu_map).fillna("Other").value_counts()
        edu_order = ["Primary", "High School", "College/Bachelor", "Graduate"]
        edu = edu.reindex([x for x in edu_order if x in edu.index])
        fig = go.Figure(go.Bar(
            x=edu.index.tolist(), y=edu.values.tolist(),
            marker_color=["#FFD700", "#FFA500", "#4682B4", "#2E86AB"],
            text=edu.values.tolist(), textposition="outside",
        ))
        fig.update_layout(title="Education Distribution", xaxis_title="", yaxis_title="Count", height=380)
        st.plotly_chart(fig, use_container_width=True)

    with tab_eth:
        def simplify_eth(x):
            return "Multiracial" if "," in str(x) else str(x)
        eth = raw["ethnicity"].apply(simplify_eth).value_counts()
        fig = go.Figure(go.Bar(
            y=eth.index.tolist(), x=eth.values.tolist(), orientation="h",
            marker=dict(color=eth.values.tolist(), colorscale="Blues", showscale=False),
            text=eth.values.tolist(), textposition="outside",
        ))
        fig.update_layout(title="Ethnicity Distribution", xaxis_title="Count", height=430,
                          yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    with tab_grp:
        grp_cnt = raw["intervention_group"].value_counts().reset_index()
        grp_cnt.columns = ["group", "count"]
        labels  = [GROUP_LABELS.get(g, g) for g in grp_cnt["group"]]
        colors  = [GROUP_COLORS.get(g, "#888") for g in grp_cnt["group"]]
        fig = go.Figure(go.Bar(
            x=labels, y=grp_cnt["count"].tolist(),
            marker_color=colors,
            text=grp_cnt["count"].tolist(), textposition="outside",
        ))
        fig.update_layout(title="Participants per Intervention Group",
                          xaxis_title="", yaxis_title="Count", height=380)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Group 1 vs Group 2 breakdown (different image orderings):**")
        g1r = raw[raw["data_group"] == "Group 1"]["intervention_group"].value_counts()
        g2r = raw[raw["data_group"] == "Group 2"]["intervention_group"].value_counts()
        cross = pd.DataFrame({"Group 1": g1r, "Group 2": g2r}).fillna(0).astype(int)
        cross.index = [GROUP_LABELS.get(i, i) for i in cross.index]
        st.dataframe(cross, use_container_width=True)

    with tab_time:
        dur = raw["Duration (in seconds)"].dropna() / 60
        dur_filt = dur[dur < 60]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(dur_filt, bins=40, color="#2C7BB6", edgecolor="white", alpha=0.85)
        ax.axvline(dur.median(), color="orange", linestyle="--",
                   label=f"Median {dur.median():.1f} min")
        ax.set_xlabel("Completion Time (minutes)"); ax.set_ylabel("Count")
        ax.set_title("Survey Completion Time Distribution (>60 min excluded)")
        ax.legend(); plt.tight_layout()
        st.pyplot(fig); plt.close()
        c1, c2, c3 = st.columns(3)
        c1.metric("Median",  f"{dur.median():.1f} min")
        c2.metric("Mean",    f"{dur.mean():.1f} min")
        c3.metric("Min/Max", f"{dur.min():.1f} / {dur.max():.0f} min")

    st.markdown("---")
    st.markdown("### Data Cleaning Pipeline")
    raw_n = len(raw); final_n = len(metrics)
    c1, c2, c3 = st.columns(3)
    c1.metric("Raw Participants", raw_n)
    c2.metric("After Cleaning (main study)", final_n)
    c3.metric("Exclusion Rate", f"{(raw_n - final_n) / raw_n * 100:.1f}%")
    st.markdown("""
**Exclusion criteria (applied in order):**
1. Remove duplicates (keep first record per ID)
2. Remove top 5% duration outliers per group
3. Remove participants who failed both attention checks
4. Remove participants who failed honesty checks
5. Remove participants giving identical responses to all 15 images
6. Remove participants who had seen too many images before (validation check)
""")

    st.markdown("---")
    st.markdown("### Raw Data Preview (first 20 rows)")
    preview = ["intervention_group", "gender", "age", "education", "ethnicity",
               "Duration (in seconds)", "slider_confidence_1"]
    st.dataframe(raw[[c for c in preview if c in raw.columns]].head(20), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SURVEY STRUCTURE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📝 Survey Structure (QSF)":
    st.title("Survey Structure — Qualtrics QSF Files")

    QSF_FILES = {
        "Main Study — Group 1":    "media_literacy_training_group1.qsf",
        "Main Study — Group 2":    "media_literacy_training_group2.qsf",
        "Follow-up — Group 1":     "media_literacy_training_group1_followup.qsf",
        "Follow-up — Group 2":     "media_literacy_training_group2_followup.qsf",
    }

    selected = st.selectbox("Select survey file", list(QSF_FILES.keys()))
    qsf    = load_qsf(QSF_FILES[selected])
    blocks = parse_qsf(qsf)
    active = {k: v for k, v in blocks.items() if v}
    total_q = sum(len(v) for v in active.values())

    st.markdown(f"**Survey name:** `{qsf['SurveyEntry'].get('SurveyName','')}`")
    c1, c2 = st.columns(2)
    c1.metric("Active Blocks", len(active))
    c2.metric("Total Questions", total_q)
    st.markdown("---")

    # Block bar chart
    st.subheader("Questions per Block")
    bnames  = list(active.keys())
    bcounts = [len(v) for v in active.values()]
    fig = go.Figure(go.Bar(
        x=bnames, y=bcounts,
        marker=dict(color=bcounts, colorscale="Blues", showscale=False),
        text=bcounts, textposition="outside",
    ))
    fig.update_layout(xaxis_tickangle=-35, height=420,
                      xaxis_title="", yaxis_title="# Questions", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Question type pie
    st.subheader("Question Type Distribution")
    all_types = [QTYPE_LABELS.get(q["type"], q["type"] or "Other")
                 for qs in active.values() for q in qs]
    type_counts = pd.Series(all_types).value_counts()
    fig = go.Figure(go.Pie(
        labels=type_counts.index.tolist(),
        values=type_counts.values.tolist(),
        textinfo="percent+label",
        marker_colors=["#4682B4","#FF8C69","#90EE90","#DDA0DD",
                       "#FFD700","#CD853F","#87CEEB","#F08080"],
    ))
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Survey flow horizontal bar
    st.subheader("Survey Flow Timeline")
    BLOCK_KEYWORD_COLORS = {
        "attention": "#FF6B6B", "honesty": "#FF6B6B",
        "welcome": "#95E1D3",   "prolific": "#95E1D3",
        "detection": "#F7DC6F",
        "intervention": "#AED6F1", "feedback": "#AED6F1",
        "demographic": "#F0B27A",
        "social": "#C39BD3",    "media": "#C39BD3", "literacy": "#C39BD3",
        "crt": "#82E0AA",
        "debrief": "#D5DBDB",
    }
    def block_color(name):
        nl = name.lower()
        for kw, col in BLOCK_KEYWORD_COLORS.items():
            if kw in nl:
                return col
        return "#BDC3C7"

    fig = go.Figure()
    for bname, qs in active.items():
        n = len(qs)
        fig.add_trace(go.Bar(
            x=[n], y=[bname], orientation="h",
            marker_color=block_color(bname),
            text=f"{n} q", textposition="inside" if n > 1 else "outside",
            name=bname, showlegend=False,
            hovertemplate=f"<b>{bname}</b><br>Questions: {n}<extra></extra>",
        ))
    fig.update_layout(
        barmode="overlay",
        height=max(300, len(active) * 28),
        xaxis_title="Number of Questions",
        yaxis={"categoryorder": "array",
               "categoryarray": list(reversed(list(active.keys())))},
        margin={"l": 10, "r": 20, "t": 10, "b": 40},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Per-block question browser
    st.subheader("Browse Questions by Block")
    sel_block = st.selectbox("Select block", list(active.keys()))
    for q in active[sel_block]:
        type_label = QTYPE_LABELS.get(q["type"], q["type"])
        preview = q["text"][:100] + ("..." if len(q["text"]) > 100 else "")
        with st.expander(f"[{type_label}] {preview}"):
            st.markdown(f"**ID:** `{q['id']}` | **Type:** {type_label}")
            st.markdown(f"**Question text:** {q['text']}")
            if q["choices"]:
                st.markdown("**Answer choices:**")
                for c in q["choices"]:
                    if c.strip():
                        st.markdown(f"  - {c}")

    st.markdown("---")

    # 4-survey comparison
    st.subheader("Comparison Across All Four Surveys")
    comp_rows = []
    for name, fname in QSF_FILES.items():
        b = parse_qsf(load_qsf(fname))
        act = {k: v for k, v in b.items() if v}
        comp_rows.append({
            "Survey": name,
            "Active Blocks": len(act),
            "Total Questions": sum(len(v) for v in act.values()),
        })
    comp_df = pd.DataFrame(comp_rows)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Active Blocks",
        x=comp_df["Survey"].tolist(), y=comp_df["Active Blocks"].tolist(),
        marker_color="#4682B4",
        text=comp_df["Active Blocks"].tolist(), textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Total Questions",
        x=comp_df["Survey"].tolist(), y=comp_df["Total Questions"].tolist(),
        marker_color="#FF8C00",
        text=comp_df["Total Questions"].tolist(), textposition="outside",
    ))
    fig.update_layout(barmode="group", height=420, xaxis_tickangle=-10)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — STATISTICAL RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Statistical Results":
    st.title("Statistical Analysis Results")
    with st.spinner("Loading data..."):
        metrics, metrics_fu, metrics_time = load_metrics()

    metrics["group_label"]    = metrics["intervention_group"].map(GROUP_LABELS)
    metrics_fu["group_label"] = metrics_fu["intervention_group"].map(GROUP_LABELS)

    # Descriptive stats
    st.markdown("### Descriptive Statistics by Group")
    metric_sel = st.selectbox("Select metric", [
        "fake_accuracy", "real_accuracy", "fake_sharing_rate",
        "real_sharing_rate", "overall_accuracy", "balanced_accuracy",
    ], format_func=lambda x: {
        "fake_accuracy":     "Deepfake Detection Accuracy",
        "real_accuracy":     "Real Image Accuracy",
        "fake_sharing_rate": "Deepfake Sharing Rate",
        "real_sharing_rate": "Real Image Sharing Rate",
        "overall_accuracy":  "Overall Accuracy",
        "balanced_accuracy": "Balanced Accuracy",
    }.get(x, x))

    desc = (metrics.groupby("intervention_group")[metric_sel]
            .agg(["mean", "std", "median", "count"])
            .round(2).reset_index())
    desc["Group"] = desc["intervention_group"].map(GROUP_LABELS)
    ctrl_mean = desc.loc[desc["intervention_group"] == "control", "mean"].values[0]
    desc["vs Control"] = (desc["mean"] - ctrl_mean).round(2)
    desc.columns = ["_key", "Mean (%)", "SD", "Median (%)", "N", "Group", "vs Control (pp)"]
    desc = desc[["Group", "Mean (%)", "SD", "Median (%)", "N", "vs Control (pp)"]]

    def highlight(val):
        if isinstance(val, (int, float)):
            if val > 2:  return "background-color:#c8f7c5"
            if val < -2: return "background-color:#f7c5c5"
        return ""
    st.dataframe(desc.style.applymap(highlight, subset=["vs Control (pp)"]),
                 use_container_width=True, hide_index=True)

    st.markdown("---")

    # Mann-Whitney U tests
    st.markdown("### Mann-Whitney U Tests vs. Control (Bonferroni ×5 correction)")
    for mvar, mlabel in [
        ("fake_accuracy",     "Deepfake Detection Accuracy"),
        ("real_accuracy",     "Real Image Accuracy"),
        ("fake_sharing_rate", "Deepfake Sharing Rate"),
    ]:
        rows = []
        ctrl = metrics[metrics["intervention_group"] == "control"][mvar]
        for grp in ORDER[1:]:
            inter = metrics[metrics["intervention_group"] == grp][mvar]
            U, p  = mannwhitneyu(ctrl, inter, method="asymptotic")
            p_adj = min(p * 5, 1.0)
            diff  = inter.mean() - ctrl.mean()
            stars = ("***" if p_adj < 0.001 else "**" if p_adj < 0.01
                     else "*" if p_adj < 0.05 else "ns")
            rows.append({
                "Group":         GROUP_LABELS[grp],
                "Control Mean":  f"{ctrl.mean():.1f}%",
                "Group Mean":    f"{inter.mean():.1f}%",
                "Difference":    f"{diff:+.1f} pp",
                "U Statistic":   f"{U:.0f}",
                "p (adjusted)":  f"{p_adj:.4f}",
                "Significance":  stars,
            })
        st.markdown(f"**{mlabel} — T1 (main study)**")
        rdf = pd.DataFrame(rows)
        def color_sig(val):
            if val in ("*", "**", "***"):
                return "background-color:#c8f7c5;font-weight:bold"
            return ""
        st.dataframe(rdf.style.applymap(color_sig, subset=["Significance"]),
                     use_container_width=True, hide_index=True)
        st.markdown("")

    st.markdown("---")

    # Longitudinal T1 vs T2
    st.markdown("### Longitudinal Comparison — T1 (main) vs T2 (follow-up, ~4 weeks later)")
    fig = go.Figure()
    long_rows = []
    for grp in ORDER:
        t1    = metrics[metrics["intervention_group"] == grp]["fake_accuracy"].mean()
        t2    = metrics_fu[metrics_fu["intervention_group"] == grp]["fake_accuracy"].mean()
        color = GROUP_COLORS[grp]
        label = GROUP_LABELS[grp]
        fig.add_trace(go.Scatter(
            x=["T1 (Main Study)", "T2 (Follow-up)"],
            y=[round(t1, 1), round(t2, 1)],
            mode="lines+markers+text",
            name=label,
            line={"color": color, "width": 2.5},
            marker={"size": 11, "color": color},
            text=[f"{t1:.1f}%", f"{t2:.1f}%"],
            textposition=["middle left", "middle right"],
        ))
        long_rows.append({"Group": label, "T1 (%)": round(t1, 1),
                          "T2 (%)": round(t2, 1), "Change (pp)": round(t2 - t1, 1)})
    fig.update_layout(
        title="Deepfake Detection Accuracy: T1 vs T2",
        yaxis_title="Accuracy (%)", xaxis_title="",
        yaxis_range=[55, 85], height=450, legend_title="Group",
    )
    st.plotly_chart(fig, use_container_width=True)

    long_df = pd.DataFrame(long_rows)
    def color_change(val):
        if isinstance(val, (int, float)):
            if val > 1:  return "background-color:#c8f7c5"
            if val < -1: return "background-color:#f7c5c5"
        return ""
    st.dataframe(long_df.style.applymap(color_change, subset=["Change (pp)"]),
                 use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Full Metrics Summary — Main Study")
    cols_show = ["fake_accuracy", "real_accuracy", "fake_sharing_rate",
                 "real_sharing_rate", "overall_accuracy", "balanced_accuracy"]
    summary = (metrics.groupby("intervention_group")[cols_show]
               .mean().round(2).reset_index())
    summary["Group"] = summary["intervention_group"].map(GROUP_LABELS)
    summary = summary.drop("intervention_group", axis=1)
    summary.columns = ["Deepfake Acc.", "Real Acc.", "Fake Share Rate",
                       "Real Share Rate", "Overall Acc.", "Balanced Acc.", "Group"]
    summary = summary[["Group"] + [c for c in summary.columns if c != "Group"]]
    st.dataframe(summary, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎨 Visualizations":
    st.title("Visualizations")
    with st.spinner("Loading data..."):
        metrics, metrics_fu, _ = load_metrics()

    def make_violin(df, metric, title, y_label):
        fig = go.Figure()
        for grp in ORDER:
            data  = df[df["intervention_group"] == grp][metric].dropna().tolist()
            label = GROUP_LABELS[grp]
            color = GROUP_COLORS[grp]
            fig.add_trace(go.Violin(
                y=data, x=[label] * len(data),
                name=label, side="negative",
                line_color=color, fillcolor=color, opacity=0.65,
                meanline_visible=True, width=0.8,
                box_visible=True, box_width=0.1,
                points=False, showlegend=False,
            ))
            mean_v = float(np.mean(data))
            se_v   = float(np.std(data) / np.sqrt(len(data)))
            fig.add_trace(go.Scatter(
                x=[label], y=[mean_v], mode="markers",
                marker={"color": color, "size": 12, "symbol": "circle",
                        "line": {"width": 2, "color": "white"}},
                showlegend=False,
                error_y={"type": "data", "array": [se_v], "visible": True,
                         "color": color, "thickness": 2.5, "width": 6},
            ))
        ctrl_mean = float(df[df["intervention_group"] == "control"][metric].mean())
        fig.add_hline(y=ctrl_mean, line_dash="dash", line_color="gray",
                      opacity=0.5, line_width=1.5,
                      annotation_text=f"Control mean {ctrl_mean:.1f}%",
                      annotation_position="right")
        fig.update_layout(title=title, yaxis_title=y_label, xaxis_title="Intervention Group",
                          violingap=0.05, height=470, showlegend=False)
        return fig

    def add_sig_annotations(fig, df, metric):
        ctrl  = df[df["intervention_group"] == "control"][metric]
        y_max = float(df[metric].quantile(0.98))
        for grp in ORDER[1:]:
            inter = df[df["intervention_group"] == grp][metric]
            _, p  = mannwhitneyu(ctrl, inter, method="asymptotic")
            p_adj = min(p * 5, 1.0)
            stars = ("***" if p_adj < 0.001 else "**" if p_adj < 0.01
                     else "*" if p_adj < 0.05 else None)
            if stars:
                fig.add_annotation(
                    x=GROUP_LABELS[grp], y=y_max + 3,
                    text=stars, showarrow=False,
                    font={"size": 18, "color": "#333"},
                )
        return fig

    # Deepfake accuracy
    st.subheader("Deepfake Image Detection Accuracy")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**T1 — Main Study**")
        fig = make_violin(metrics,    "fake_accuracy", "Deepfake Accuracy (T1)", "Accuracy (%)")
        fig = add_sig_annotations(fig, metrics, "fake_accuracy")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**T2 — Follow-up (~4 weeks later)**")
        fig = make_violin(metrics_fu, "fake_accuracy", "Deepfake Accuracy (T2)", "Accuracy (%)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Real image accuracy
    st.subheader("Real Image Detection Accuracy")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**T1 — Main Study**")
        fig = make_violin(metrics,    "real_accuracy", "Real Image Accuracy (T1)", "Accuracy (%)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**T2 — Follow-up**")
        fig = make_violin(metrics_fu, "real_accuracy", "Real Image Accuracy (T2)", "Accuracy (%)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Sharing rates
    st.subheader("Image Sharing Intention Rates (T1)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Deepfake Image Sharing Rate**")
        fig = make_violin(metrics, "fake_sharing_rate", "Deepfake Sharing Rate", "Rate (%)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**Real Image Sharing Rate**")
        fig = make_violin(metrics, "real_sharing_rate", "Real Image Sharing Rate", "Rate (%)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Summary grouped bar
    st.subheader("All Metrics Comparison Across Groups")
    metric_cols = [
        ("fake_accuracy",     "Deepfake Accuracy"),
        ("real_accuracy",     "Real Accuracy"),
        ("fake_sharing_rate", "Fake Share Rate"),
        ("real_sharing_rate", "Real Share Rate"),
    ]
    bar_colors = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0"]
    fig = go.Figure()
    for (col, lbl), color in zip(metric_cols, bar_colors):
        ys = [metrics[metrics["intervention_group"] == g][col].mean() for g in ORDER]
        fig.add_trace(go.Bar(
            name=lbl,
            x=[GROUP_LABELS[g] for g in ORDER],
            y=[round(v, 1) for v in ys],
            marker_color=color,
            text=[f"{v:.1f}" for v in ys], textposition="outside",
        ))
    fig.update_layout(barmode="group", height=470,
                      xaxis_title="", yaxis_title="Mean (%)",
                      title="Mean Values per Metric and Group",
                      legend_title="Metric")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Pre-generated half-violin plots
    st.subheader("Paper-Style Half-Violin Plots (generated by visualizations.py)")
    result_dir = BASE / "Results/pilot2"
    png_map = [
        ("Deepfake Accuracy T1",    "acc_fake_t1.png"),
        ("Deepfake Accuracy T2",    "acc_fake_t2.png"),
        ("Real Image Accuracy T1",  "acc_real_t1.png"),
        ("Real Image Accuracy T2",  "acc_real_t2.png"),
        ("Fake Sharing Rate T1",    "sharing_fake_t1.png"),
        ("Real Sharing Rate T1",    "sharing_real_t1.png"),
    ]
    cols = st.columns(2)
    for i, (label, fname) in enumerate(png_map):
        fpath = result_dir / fname
        if fpath.exists():
            with cols[i % 2]:
                st.markdown(f"**{label}**")
                st.image(str(fpath), use_column_width=True)
    st.caption("Figures saved to `Results/pilot2/` as both PDF and PNG.")
