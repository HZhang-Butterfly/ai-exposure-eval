"""
AI Automation Exposure Dashboard
Visualises LLM evaluation results for 114 digital occupations.
Run:  streamlit run dashboard.py
"""

import json
import glob
import os
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Automation Exposure Dashboard",
    page_icon="🤖",
    layout="wide",
)

DIMENSIONS = ["Correctness", "Completeness", "Best Practices", "Domain Accuracy", "Clarity"]
RESULTS_DIR = "results"

# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load_all_results(results_dir: str) -> dict:
    data = {}
    for fpath in glob.glob(os.path.join(results_dir, "*_results_auto.json")):
        try:
            d = json.load(open(fpath, encoding="utf-8"))
            m = d.get("meta", {})
            score = m.get("exposure_score", {})
            overall = score.get("overall", 0) if isinstance(score, dict) else 0
            if overall and overall > 0:
                data[m["job_title"]] = d
        except Exception:
            continue
    return data


def build_summary_df(data: dict) -> pd.DataFrame:
    rows = []
    for title, d in data.items():
        m = d["meta"]
        score = m.get("exposure_score", {})
        dims = score.get("dimensions", {}) if isinstance(score, dict) else {}
        row = {
            "Job Title": title,
            "Overall Score": score.get("overall", 0),
            "Evaluated Tasks": m.get("evaluated_tasks", 0),
            "Total O*NET Tasks": m.get("total_onet_tasks", 0),
        }
        for dim in DIMENSIONS:
            row[dim] = dims.get(dim, None)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("Overall Score", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df


all_data = load_all_results(RESULTS_DIR)
summary_df = build_summary_df(all_data)

# ── Sidebar navigation ─────────────────────────────────────────────────────────
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Job Comparison", "Job Detail", "Dimension Analysis"],
    index=0,
)
st.sidebar.markdown("---")
st.sidebar.metric("Jobs Evaluated", len(all_data))
st.sidebar.metric("Avg Exposure Score", f"{summary_df['Overall Score'].mean():.2f} / 5")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("🤖 AI Automation Exposure Dashboard")
    st.markdown(
        "Quantifying how well a large language model can perform the tasks of **114 digital occupations**, "
        "based on O\\*NET task data and a Teacher–Student–Judge evaluation framework."
    )

    st.markdown("---")

    # ── Score distribution ─────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("Jobs Evaluated", len(all_data))
    col2.metric("Avg Exposure Score", f"{summary_df['Overall Score'].mean():.2f} / 5")
    col3.metric("Score Range", f"{summary_df['Overall Score'].min():.2f} – {summary_df['Overall Score'].max():.2f}")

    st.markdown("---")

    # ── Ranked bar chart ───────────────────────────────────────────────────────
    st.subheader("Occupation Exposure Score Ranking")
    st.caption("Higher score = LLM performs better on this occupation's tasks (scale 1–5)")

    top_n = st.slider("Show top / bottom N jobs", min_value=10, max_value=len(summary_df), value=30, step=5)
    view = st.radio("View", ["Top (most exposed)", "Bottom (least exposed)", "All"], horizontal=True)

    if view == "Top (most exposed)":
        plot_df = summary_df.head(top_n).copy()
    elif view == "Bottom (least exposed)":
        plot_df = summary_df.tail(top_n).sort_values("Overall Score").copy()
    else:
        plot_df = summary_df.copy()

    color_scale = px.colors.diverging.RdYlGn
    fig_bar = px.bar(
        plot_df,
        x="Overall Score",
        y="Job Title",
        orientation="h",
        color="Overall Score",
        color_continuous_scale=color_scale,
        range_color=[2.5, 5.0],
        text="Overall Score",
        labels={"Overall Score": "Exposure Score (1–5)"},
        height=max(500, len(plot_df) * 22),
    )
    fig_bar.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_bar.update_layout(
        yaxis={"categoryorder": "total ascending"},
        coloraxis_showscale=False,
        margin=dict(l=10, r=60, t=20, b=20),
        xaxis_range=[0, 5.5],
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Score distribution histogram ───────────────────────────────────────────
    st.markdown("---")
    st.subheader("Score Distribution")
    fig_hist = px.histogram(
        summary_df,
        x="Overall Score",
        nbins=20,
        color_discrete_sequence=["#4C78A8"],
        labels={"Overall Score": "Exposure Score"},
    )
    fig_hist.update_layout(bargap=0.1, height=320)
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── Dimension heatmap ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Dimension Score Heatmap (Top 30)")
    heat_df = summary_df.head(30).set_index("Job Title")[DIMENSIONS].fillna(0)
    fig_heat = px.imshow(
        heat_df,
        color_continuous_scale="RdYlGn",
        zmin=1, zmax=5,
        aspect="auto",
        height=650,
        labels={"color": "Score"},
    )
    fig_heat.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_heat, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — JOB COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Job Comparison":
    st.title("Job Comparison — Radar Chart")
    st.markdown("Select 2–5 occupations to compare their dimension score profiles.")

    job_list = summary_df["Job Title"].tolist()
    defaults = job_list[:3] if len(job_list) >= 3 else job_list

    selected_jobs = st.multiselect(
        "Choose occupations to compare",
        options=job_list,
        default=defaults,
        max_selections=5,
    )

    if len(selected_jobs) < 2:
        st.info("Please select at least 2 occupations.")
    else:
        fig_radar = go.Figure()
        theta = DIMENSIONS + [DIMENSIONS[0]]

        for job in selected_jobs:
            row = summary_df[summary_df["Job Title"] == job].iloc[0]
            values = [row.get(d, 0) or 0 for d in DIMENSIONS]
            values_closed = values + [values[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=values_closed,
                theta=theta,
                fill="toself",
                name=job,
                opacity=0.7,
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
            showlegend=True,
            height=550,
            margin=dict(l=40, r=40, t=60, b=40),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # ── Parallel coordinates ───────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Parallel Coordinates — Dimension Profile")
        st.caption("Each line is one occupation; traces that score consistently high stay in the green band.")
        pc_df = summary_df[summary_df["Job Title"].isin(selected_jobs)].copy()
        pc_df = pc_df[["Job Title", "Overall Score"] + DIMENSIONS].fillna(0)
        color_vals = pc_df["Overall Score"].tolist()
        fig_pc = go.Figure(go.Parcoords(
            line=dict(
                color=color_vals,
                colorscale="RdYlGn",
                cmin=1, cmax=5,
                showscale=True,
                colorbar=dict(title="Overall"),
            ),
            dimensions=[
                dict(range=[1, 5], label=dim, values=pc_df[dim].tolist())
                for dim in DIMENSIONS
            ],
        ))
        fig_pc.update_layout(height=380, margin=dict(l=60, r=60, t=40, b=20))
        st.plotly_chart(fig_pc, use_container_width=True)

        # ── Side-by-side score table ───────────────────────────────────────────
        st.markdown("---")
        st.subheader("Score Breakdown")
        compare_df = summary_df[summary_df["Job Title"].isin(selected_jobs)][
            ["Job Title", "Overall Score"] + DIMENSIONS
        ].set_index("Job Title").T
        st.dataframe(compare_df.style.format("{:.2f}").background_gradient(
            cmap="RdYlGn", vmin=1, vmax=5, axis=None
        ), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — JOB DETAIL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Job Detail":
    st.title("Job Detail — Task-Level Scores")

    job_list = summary_df["Job Title"].tolist()
    selected_job = st.selectbox("Select an occupation", options=job_list)

    if selected_job:
        d = all_data[selected_job]
        m = d["meta"]
        score = m.get("exposure_score", {})

        # ── Job summary header ─────────────────────────────────────────────────
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Overall Exposure Score", f"{score.get('overall', 0):.2f} / 5")
        col2.metric("Evaluated Tasks", m.get("evaluated_tasks", "—"))
        col3.metric("Total O*NET Tasks", m.get("total_onet_tasks", "—"))
        col4.metric("O*NET Code", m.get("onet_code", "—"))

        # ── Dimension bar + radar ──────────────────────────────────────────────
        dims = score.get("dimensions", {}) if isinstance(score, dict) else {}
        if dims:
            st.markdown("---")
            col_bar, col_radar = st.columns([1, 1])

            with col_bar:
                st.subheader("Dimension Averages")
                dim_df = pd.DataFrame({
                    "Dimension": list(dims.keys()),
                    "Score": list(dims.values()),
                })
                fig_dim = px.bar(
                    dim_df, x="Dimension", y="Score",
                    color="Score",
                    color_continuous_scale="RdYlGn",
                    range_color=[1, 5],
                    range_y=[0, 5.5],
                    text="Score",
                    height=360,
                )
                fig_dim.update_traces(texttemplate="%{text:.2f}", textposition="outside")
                fig_dim.update_layout(coloraxis_showscale=False, margin=dict(t=20, b=20))
                st.plotly_chart(fig_dim, use_container_width=True)

            with col_radar:
                st.subheader("Radar Profile")
                dim_keys = [d for d in DIMENSIONS if d in dims]
                values = [dims[d] for d in dim_keys]
                theta_closed = dim_keys + [dim_keys[0]]
                values_closed = values + [values[0]]
                fig_r = go.Figure(go.Scatterpolar(
                    r=values_closed,
                    theta=theta_closed,
                    fill="toself",
                    fillcolor="rgba(76,120,168,0.3)",
                    line=dict(color="#4C78A8", width=2),
                    name=selected_job,
                ))
                fig_r.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
                    showlegend=False,
                    height=360,
                    margin=dict(l=40, r=40, t=40, b=40),
                )
                st.plotly_chart(fig_r, use_container_width=True)

        # ── Task-level detail ──────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Task-Level Scores")

        tasks = d.get("tasks", [])
        scored_tasks = [t for t in tasks if t.get("dimension_scores")]

        if not scored_tasks:
            st.warning("No scored tasks found in this result file.")
        else:
            task_rows = []
            for t in scored_tasks:
                labels = t.get("dimension_labels", DIMENSIONS)
                scores = t.get("dimension_scores", [])
                row = {
                    "Task ID": t.get("task_id", "—"),
                    "Task Type": t.get("task_type", "—"),
                    "Avg": round(sum(scores) / len(scores), 2) if scores else 0,
                }
                for lbl, s in zip(labels, scores):
                    row[lbl] = s
                task_rows.append(row)

            task_df = pd.DataFrame(task_rows).sort_values("Avg", ascending=False)

            st.dataframe(
                task_df.style.background_gradient(
                    subset=["Avg"] + [c for c in task_df.columns if c in DIMENSIONS],
                    cmap="RdYlGn", vmin=1, vmax=5
                ).format({c: "{:.0f}" for c in task_df.columns if c in DIMENSIONS + ["Avg"]}),
                use_container_width=True,
                height=400,
            )

            # ── Judge reason detail ────────────────────────────────────────────
            st.markdown("---")
            st.subheader("Judge Reasoning Detail")
            st.caption("Click a task to see the student answer and judge's evaluation rationale.")

            for t in scored_tasks:
                scores = t.get("dimension_scores", [])
                avg = round(sum(scores) / len(scores), 2) if scores else 0
                task_label = f"[{avg:.1f}/5]  {t.get('task_type', t.get('task_id', '?'))}"

                with st.expander(task_label):
                    col_a, col_b = st.columns([1, 1])

                    with col_a:
                        st.markdown("**Task Prompt**")
                        st.markdown(t.get("user_prompt", "—"))

                        st.markdown("**Student Answer**")
                        answer = t.get("student_answer", "(no response)")
                        st.markdown(answer[:1500] + ("…" if len(answer) > 1500 else ""))

                    with col_b:
                        st.markdown("**Dimension Scores**")
                        labels = t.get("dimension_labels", DIMENSIONS)
                        for lbl, s in zip(labels, scores):
                            color = "🟢" if s >= 4 else ("🟡" if s == 3 else "🔴")
                            st.markdown(f"{color} **{lbl}**: {s}/5")
                        st.markdown(f"**Average: {avg}/5**")

                        st.markdown("**Judge Reasoning**")
                        st.info(t.get("reason", "No reason provided."))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — DIMENSION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Dimension Analysis":
    st.title("Dimension Analysis")
    st.markdown(
        "How do the five evaluation dimensions distribute across all occupations? "
        "Use these charts to spot structural patterns — e.g., which dimension is hardest for AI."
    )
    st.markdown("---")

    dim_data = summary_df[DIMENSIONS].fillna(0)

    # ── Box plot per dimension ─────────────────────────────────────────────────
    st.subheader("Score Distribution per Dimension (Box Plot)")
    st.caption("Shows median, spread, and outliers for each evaluation dimension across all jobs.")
    box_rows = []
    for dim in DIMENSIONS:
        for val in summary_df[dim].dropna():
            box_rows.append({"Dimension": dim, "Score": val})
    box_df = pd.DataFrame(box_rows)
    fig_box = px.box(
        box_df, x="Dimension", y="Score",
        color="Dimension",
        points="all",
        range_y=[0, 5.5],
        height=420,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_box.update_layout(showlegend=False, margin=dict(t=20, b=20))
    st.plotly_chart(fig_box, use_container_width=True)

    # ── Bubble chart: task count vs overall score ──────────────────────────────
    st.markdown("---")
    st.subheader("Task Coverage vs. Exposure Score (Bubble Chart)")
    st.caption("Bubble size = number of evaluated tasks. Larger + higher = more complete & higher-scoring job.")
    bubble_df = summary_df[summary_df["Overall Score"] > 0].copy()
    bubble_df["Coverage %"] = (bubble_df["Evaluated Tasks"] / bubble_df["Total O*NET Tasks"] * 100).round(1)
    fig_bubble = px.scatter(
        bubble_df,
        x="Coverage %",
        y="Overall Score",
        size="Evaluated Tasks",
        color="Overall Score",
        color_continuous_scale="RdYlGn",
        range_color=[2.5, 5],
        hover_name="Job Title",
        hover_data={"Evaluated Tasks": True, "Coverage %": True, "Overall Score": ":.2f"},
        labels={"Coverage %": "Task Coverage (%)", "Overall Score": "Exposure Score"},
        height=480,
        size_max=35,
    )
    fig_bubble.update_layout(margin=dict(t=20, b=20))
    st.plotly_chart(fig_bubble, use_container_width=True)

    # ── Scatter: Correctness vs Domain Accuracy ────────────────────────────────
    st.markdown("---")
    st.subheader("Correctness vs. Domain Accuracy")
    st.caption(
        "Jobs in the top-right corner score high on both — AI gets the right answer AND uses correct domain knowledge. "
        "Jobs in the top-left are fluent but factually shaky."
    )
    scat_df = summary_df[summary_df["Correctness"].notna() & summary_df["Domain Accuracy"].notna()].copy()
    fig_scat = px.scatter(
        scat_df,
        x="Domain Accuracy",
        y="Correctness",
        color="Overall Score",
        color_continuous_scale="RdYlGn",
        range_color=[2.5, 5],
        hover_name="Job Title",
        text="Job Title",
        height=520,
        labels={"Domain Accuracy": "Domain Accuracy Score", "Correctness": "Correctness Score"},
    )
    fig_scat.update_traces(textposition="top center", textfont_size=9)
    fig_scat.update_layout(margin=dict(t=20, b=20))
    fig_scat.add_shape(type="line", x0=1, y0=1, x1=5, y1=5,
                       line=dict(dash="dot", color="gray", width=1))
    st.plotly_chart(fig_scat, use_container_width=True)

    # ── Correlation heatmap ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Dimension Correlation Matrix")
    st.caption("How correlated are the five dimensions? High correlation means they tend to move together.")
    corr = summary_df[DIMENSIONS].corr().round(2)
    fig_corr = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        height=420,
    )
    fig_corr.update_layout(margin=dict(t=20, b=20))
    st.plotly_chart(fig_corr, use_container_width=True)
