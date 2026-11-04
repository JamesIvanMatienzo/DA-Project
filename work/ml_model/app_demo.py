import os
import sys
import numpy as np
import pandas as pd
import streamlit as st

# Ensure local imports from work/ml_model work regardless of working directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from live_predictor import (
    DEMO_PRESETS,
    train_and_save_bundle,
    load_unseen_test_dataframe,
    row_to_preset_dict,
    predict_from_raw_inputs,
    simulate_rank_curve,
)

st.set_page_config(
    page_title="FlyRank SEO Visibility System",
    page_icon="🎯",
    layout="centered",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner="Loading Champion Stacking Ensemble...")
def load_cached_bundle():
    return train_and_save_bundle(force_retrain=False)


@st.cache_data(show_spinner="Loading 6,163 Unseen Test Rows...")
def get_cached_test_df():
    return load_unseen_test_dataframe()


bundle = load_cached_bundle()
test_df = get_cached_test_df()

DEFAULT_PRESET = DEMO_PRESETS[list(DEMO_PRESETS.keys())[0]]


def _set_state_from_dict(preset_dict: dict):
    """Populate widget session_state keys safely BEFORE widget instantiation."""
    st.session_state["in_word_count"] = int(preset_dict["word_count"])
    st.session_state["in_avg_position"] = float(preset_dict["avg_position"])
    st.session_state["in_search_volume"] = int(preset_dict["search_volume"])
    st.session_state["in_sessions_90d"] = int(preset_dict["sessions_90d"])
    st.session_state["in_days_update"] = int(preset_dict["days_since_last_update"])
    st.session_state["in_actual_impr"] = int(preset_dict["actual_impressions_90d"])
    ct = preset_dict.get("content_type", "keyword article")
    mi = preset_dict.get("main_intent", "informational")
    st.session_state["in_content_type"] = ct if ct in ["keyword article", "comparison article", "feedly article"] else "keyword article"
    st.session_state["in_main_intent"] = mi if mi in ["informational", "commercial", "transactional", "navigational"] else "informational"
    st.session_state["base_preset"] = dict(preset_dict)


def cb_load_test_row():
    idx = int(st.session_state.get("picked_test_idx", 0))
    row_dict = row_to_preset_dict(test_df.iloc[idx])
    _set_state_from_dict(row_dict)
    st.session_state["loaded_row_notice"] = f"Loaded Row #{idx} (`{test_df.iloc[idx]['content_id']}`) into the Live Calculator (Tab 1)!"


# Initialize default state on first load (runs BEFORE any widgets are created)
if "in_word_count" not in st.session_state:
    _set_state_from_dict(DEFAULT_PRESET)

# ==============================================================================
# CLEAN HEADER
# ==============================================================================
st.markdown("## 🎯 SEO Search Visibility & Content Refresh System")
st.caption(
    "**Live System Demonstration (`Champion Stacking Ensemble: 50% Poly Ridge + 50% HGBR`)** "
    "• Enter real-world article metrics ($X$) below to predict 90-day Google Search Impressions ($\\hat{y}$) and generate an SEO action plan."
)

tab_calc, tab_sim, tab_data = st.tabs([
    "🎯 1. Live System Calculator",
    "📈 2. What-If Rank Simulator & Model Breakdown",
    "📂 3. Unseen Test Dataset (6,163 Rows)",
])

# ==============================================================================
# TAB 1: DIRECT 5-INPUT LIVE SYSTEM CALCULATOR
# ==============================================================================
with tab_calc:
    st.markdown("#### 1. Enter the 5 Pure Model Inputs ($X$)")
    with st.container(border=True):
        c_left, c_right = st.columns(2)
        with c_left:
            word_count = st.number_input(
                "1. Article Word Count (`word_count`)",
                min_value=0,
                max_value=25000,
                step=100,
                key="in_word_count",
                help="Total words on the page (Typical range: 300 to 7,500 words).",
            )
            avg_position = st.number_input(
                "2. Google Rank Position (`avg_position` | 1 = Top)",
                min_value=0.0,
                max_value=250.0,
                step=0.5,
                format="%.1f",
                key="in_avg_position",
                help="Average Google rank position (1.0–10.0 = Page 1; 11.0–20.0 = Page 2; 0 = Unranked).",
            )
            search_volume = st.number_input(
                "3. Keyword Search Volume (`search_volume`)",
                min_value=0,
                max_value=200000,
                step=100,
                key="in_search_volume",
                help="Monthly Google search demand for the target keyword.",
            )
        with c_right:
            sessions_90d = st.number_input(
                "4. 90-Day Website Sessions (`sessions_90d`)",
                min_value=0,
                max_value=50000,
                step=10,
                key="in_sessions_90d",
                help="Total user visits to the page over the past 90 days.",
            )
            days_since_last_update = st.number_input(
                "5. Days Since Last Update (`days_since_last_update`)",
                min_value=0,
                max_value=3650,
                step=10,
                key="in_days_update",
                help="Days since the article was last updated (>180 days indicates stale content).",
            )
            st.caption(
                "💡 **Automatic Feature Engineering:** From these **5 core inputs ($X$)**, the system automatically "
                "computes all **48 model features** (`1/rank` power law, `log(1+x)` transforms, demand synergies, and category tiers)."
            )

    with st.expander("🔍 Optional Comparison: Enter Current Impressions (to calculate Refresh Upside Gap)", expanded=False):
        st.caption(
            "**Defense Note:** `Current Impressions` is **never** passed into the ML model ($X$). "
            "It is only subtracted *after* the model predicts ($\\hat{y} - y$) to measure how much extra traffic an existing page can gain."
        )
        oc1, oc2, oc3 = st.columns(3)
        with oc1:
            actual_impressions_90d = st.number_input(
                "Current 90d Impressions (Optional — for Upside Gap)",
                min_value=0,
                max_value=2000000,
                step=50,
                key="in_actual_impr",
            )
        with oc2:
            content_type = st.selectbox(
                "Content Format (`content_type`)",
                options=["keyword article", "comparison article", "feedly article"],
                key="in_content_type",
            )
        with oc3:
            main_intent = st.selectbox(
                "Keyword Intent (`main_intent`)",
                options=["informational", "commercial", "transactional", "navigational"],
                key="in_main_intent",
            )

    # Auto-derive the remaining engineered features
    base_p = st.session_state.get("base_preset", {})
    core_unchanged = (
        bool(base_p)
        and word_count == int(base_p.get("word_count", -1))
        and abs(avg_position - float(base_p.get("avg_position", -1))) < 1e-4
        and search_volume == int(base_p.get("search_volume", -1))
        and sessions_90d == int(base_p.get("sessions_90d", -1))
        and days_since_last_update == int(base_p.get("days_since_last_update", -1))
    )

    if core_unchanged:
        live_input = dict(base_p)
        live_input["content_type"] = content_type
        live_input["main_intent"] = main_intent
        live_input["actual_impressions_90d"] = actual_impressions_90d
    else:
        pageviews_90d = int(round(sessions_90d * 1.15))
        users_90d = int(round(sessions_90d * 0.96))
        engaged_sessions_90d = int(min(15, round(sessions_90d * 0.02)))
        days_with_sessions = int(min(90, max(0 if sessions_90d == 0 else 1, round(np.sqrt(sessions_90d) * 4.0))))
        engagement_rate = round((engaged_sessions_90d / max(1, sessions_90d)) * 100.0, 2) if sessions_90d > 0 else 0.0
        scroll_rate = 6.0 if sessions_90d > 0 else 0.0
        scroll_events_90d = int(round(pageviews_90d * (scroll_rate / 100.0)))
        live_input = {
            "content_type": content_type,
            "main_intent": main_intent,
            "word_count": word_count,
            "char_count": int(round(word_count * 6.2)),
            "avg_position": avg_position,
            "search_volume": search_volume,
            "competition": float(base_p.get("competition", 0.25)),
            "cpc": float(base_p.get("cpc", 1.20)),
            "content_age_days": max(int(base_p.get("content_age_days", 300)), days_since_last_update + 30),
            "days_since_last_update": days_since_last_update,
            "sessions_90d": sessions_90d,
            "pageviews_90d": pageviews_90d,
            "users_90d": users_90d,
            "engaged_sessions_90d": engaged_sessions_90d,
            "days_with_sessions": days_with_sessions,
            "engagement_rate": engagement_rate,
            "scroll_rate": scroll_rate,
            "scroll_events_90d": scroll_events_90d,
            "ai_sessions_90d": 0,
            "ai_traffic_pct": 0.0,
            "actual_impressions_90d": actual_impressions_90d,
        }

    # Execute model pipeline
    pred_result = predict_from_raw_inputs(live_input, bundle)

    # ==============================================================================
    # 2. ONE CLEAN SYSTEM OUTPUT BANNER
    # ==============================================================================
    st.markdown("#### 2. System Prediction ($\\hat{y}$) & Recommended SEO Action")
    with st.container(border=True):
        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            st.metric(
                label="🏆 System Predicted 90d Impressions",
                value=f"{pred_result['pred_impressions_ensemble']:,}",
                delta=f"Pred Log(1+y) = {pred_result['pred_log_ensemble']:.2f}",
            )
        with mcol2:
            st.metric(
                label="📏 Naive Word-Count Rule (Baseline)",
                value=f"{pred_result['pred_impressions_univariate']:,}",
                delta=f"Log(1+y) = {pred_result['pred_log_univariate']:.2f} (Words Only)",
                delta_color="off",
            )
        with mcol3:
            gap_val = pred_result["opportunity_gap_log"]
            upside = pred_result["impression_upside_delta"]
            st.metric(
                label="🚀 Refresh Upside Gap (ŷ - y)",
                value=f"{upside:+,}",
                delta=f"vs. Current {pred_result['actual_impressions']:,} ({gap_val:+.2f} log)",
                delta_color="normal" if gap_val >= 0.5 else "off",
            )

        rc = pred_result["reason_code"]
        if rc in ("striking_distance_opportunity", "thin_content_expansion", "stale_authority_decay", "under_indexed_refresh"):
            st.success(
                f"**SYSTEM RECOMMENDATION — {pred_result['badge_title']}**\n\n"
                f"{pred_result['playbook_text']}"
            )
        elif rc == "performing_at_capacity":
            st.info(
                f"**SYSTEM RECOMMENDATION — {pred_result['badge_title']}**\n\n"
                f"{pred_result['playbook_text']}"
            )
        else:
            st.warning(
                f"**SYSTEM RECOMMENDATION — {pred_result['badge_title']}**\n\n"
                f"{pred_result['playbook_text']}"
            )

# ==============================================================================
# TAB 2: WHAT-IF RANK SIMULATOR & MODEL BREAKDOWN
# ==============================================================================
with tab_sim:
    st.markdown("#### 🔬 What-If Simulation: Moving Google Rank from #30 to #1")
    st.caption(
        "How does the system respond if you keep this article's exact attributes and improve its Google Rank from Page 3 (#30) to Page 1 (#1)?"
    )
    sim_df = simulate_rank_curve(live_input, bundle)
    plot_df = sim_df.set_index("avg_position")[
        ["Champion Ensemble (Pred Impressions)", "Baseline 2: Word Count Rule (Pred Impressions)"]
    ]
    st.line_chart(plot_df, height=300)
    st.dataframe(
        sim_df[
            [
                "Google Rank Position",
                "SERP Power (1/pos)",
                "Champion Ensemble (Pred Impressions)",
                "Baseline 2: Word Count Rule (Pred Impressions)",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.markdown("#### ⚖️ All 5 Competing Architectures on Current Input")
    st.dataframe(pd.DataFrame(pred_result["models_comparison"]), use_container_width=True, hide_index=True)

    with st.expander("🧬 Inspect Auto-Computed 48-Feature Pipeline Signals"):
        sig_df = pd.DataFrame([
            {"Engineered Signal": k, "Value": str(v)}
            for k, v in {**pred_result["key_engineered_signals"], **pred_result["derived_tiers"]}.items()
        ])
        st.dataframe(sig_df, use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 3: UNSEEN TEST DATASET BROWSER (6,163 HELD-OUT ROWS)
# ==============================================================================
with tab_data:
    st.markdown("#### 📂 Load Any Row from the 6,163 Unseen Validation Articles")
    if "loaded_row_notice" in st.session_state:
        st.success(st.session_state.pop("loaded_row_notice"))

    dcol1, dcol2 = st.columns([2, 1])
    with dcol1:
        st.number_input(
            f"Select Validation Row Index (0 to {len(test_df) - 1}):",
            min_value=0,
            max_value=len(test_df) - 1,
            value=0,
            step=1,
            key="picked_test_idx",
        )
    with dcol2:
        st.write("")
        st.write("")
        st.button(
            "📥 Load into Calculator",
            type="primary",
            use_container_width=True,
            on_click=cb_load_test_row,
        )

    preview_cols = [
        "content_id",
        "word_count",
        "avg_position",
        "search_volume",
        "sessions_90d",
        "days_since_last_update",
        "impressions_90d",
        "pred_impressions_90d",
        "opportunity_gap_log",
    ]
    st.dataframe(test_df[preview_cols].head(100), use_container_width=True)
