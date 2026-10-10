"""
Step 5: Interactive app.

Run from the project folder (after simulation.py):
    streamlit run app.py

Shows the model's risk scores for Chicago restaurant inspections from the test
period (2024 onward). Every score comes from a model trained only on earlier
years, so the app shows how the model would have done in real life.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parent
SCORED_PATH = ROOT / "data" / "processed" / "scored_inspections.csv"
SIM_FIG = ROOT / "reports" / "figures" / "11_simulation.png"

TIERS = ["High", "Medium", "Low"]
# Colors from the Chicago city flag (red stars, light blue stripes), plus amber in between
NAVY, FLAG_BLUE, FLAG_RED, AMBER = "#10263B", "#41B6E6", "#E4002B", "#F28C28"
TIER_HEX = {"High": FLAG_RED, "Medium": AMBER, "Low": FLAG_BLUE}
TIER_COLORS = {t: [int(h[i:i + 2], 16) for i in (1, 3, 5)] for t, h in TIER_HEX.items()}
RESULTS = ["Pass", "Pass w/ Conditions", "Fail"]


# ---------- Data ----------

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["inspection_date", "sim_date"], dtype={"zip": str})
    # The model's raw scores aren't calibrated probabilities (it was trained with
    # balanced class weights), so show each score as a percentile instead.
    df["risk_percentile"] = df["risk_score"].rank(pct=True)
    df["risk_tier"] = np.select([df["risk_percentile"] > 0.8, df["risk_percentile"] > 0.5],
                                ["High", "Medium"], default="Low")
    fallback = pd.Series(np.where(df["failed"] == 1, "Fail", "Pass"), index=df.index)
    df["results"] = df["results"].fillna(fallback)
    df["dba_name"] = df["dba_name"].fillna("Unknown").str.title()
    df["address"] = df["address"].fillna("").str.title()
    return df


def latest_per_restaurant(df: pd.DataFrame) -> pd.DataFrame:
    return (df.sort_values(["inspection_date", "inspection_id"])
              .groupby("license_number", as_index=False).tail(1))


def headline_numbers(df: pd.DataFrame) -> dict:
    fails = df[df["failed"] == 1]
    base_flag = df["last_routine_failed"] == 1
    k = int(base_flag.sum())
    return {
        "avg_days": fails["days_earlier"].mean(),
        "mid_month_model": (fails["sim_position"] <= 0.5).mean(),
        "mid_month_actual": (fails["actual_position"] <= 0.5).mean(),
        "extra_found": int(df.nlargest(k, "risk_score")["failed"].sum() - df.loc[base_flag, "failed"].sum()),
        "n_inspections": len(df),
        "n_fails": len(fails),
    }


def risk_reasons(row: pd.Series) -> list[str]:
    """Plain-language facts behind a restaurant's score (from its history before that inspection)."""
    reasons = []
    if row.get("prior_inspections", 0) == 0 or pd.isna(row.get("prior_inspections")):
        return ["No earlier inspections on record, so the model has little history to go on."]
    if pd.notna(row.get("prior_fail_rate")):
        reasons.append(f"Failed {row['prior_fail_rate']:.0%} of its {int(row['prior_inspections'])} earlier inspections.")
    if pd.notna(row.get("prior_avg_violations")):
        reasons.append(f"Averaged {row['prior_avg_violations']:.1f} violations per earlier inspection.")
    if row.get("last_routine_failed") == 1:
        reasons.append("Failed its last routine inspection.")
    elif row.get("last_routine_cond") == 1:
        reasons.append("Passed its last routine inspection only with conditions.")
    if pd.notna(row.get("days_since_last_routine")):
        reasons.append(f"{int(row['days_since_last_routine'])} days since its last routine inspection.")
    if row.get("prior_complaints", 0) > 0:
        reasons.append(f"{int(row['prior_complaints'])} earlier complaint-based inspection(s).")
    return reasons


# ---------- Look and feel ----------

def star_svg(size: int = 26, color: str = FLAG_RED) -> str:
    """A six-pointed star like the ones on the Chicago flag."""
    pts = []
    for i in range(12):
        r = 10 if i % 2 == 0 else 5.4
        a = math.pi / 2 + i * math.pi / 6
        pts.append(f"{12 + r * math.cos(a):.2f},{12 - r * math.sin(a):.2f}")
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" aria-hidden="true">'
            f'<polygon points="{" ".join(pts)}" fill="{color}"/></svg>')


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Barlow:wght@400;500;600&display=swap');

.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea,
.stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stCaptionContainer"] {{
    font-family: 'Barlow', sans-serif;
}}
.stApp h1, .stApp h2, .stApp h3 {{
    font-family: 'Barlow Condensed', sans-serif;
    color: {NAVY};
    letter-spacing: 0.01em;
}}
.stApp h3 {{ font-size: 1.65rem; font-weight: 600; margin-top: 0.6rem; }}

/* Header styled after the Chicago flag */
.flag {{ background: #FFFFFF; margin: 0 0 1.75rem 0; }}
.flag .stripe {{ height: 14px; background: {FLAG_BLUE}; }}
.flag .body {{
    display: flex; align-items: center; justify-content: space-between; gap: 2rem;
    padding: 1.6rem 1.75rem;
}}
.flag h1 {{
    font-family: 'Barlow Condensed', sans-serif; color: {NAVY};
    font-size: 3.1rem; font-weight: 700; line-height: 1.0; margin: 0; padding: 0;
}}
.flag p {{
    font-size: 1.08rem; color: #3A5068; max-width: 62ch; margin: 0.6rem 0 0 0; line-height: 1.5;
}}
.flag .stars {{ display: flex; gap: 0.9rem; flex-shrink: 0; }}
@media (max-width: 760px) {{
    .flag .body {{ flex-direction: column-reverse; align-items: flex-start; gap: 1rem; }}
    .flag h1 {{ font-size: 2.3rem; }}
}}

/* Headline numbers */
[data-testid="stMetric"] {{
    background: #FFFFFF; border-left: 5px solid {FLAG_BLUE}; padding: 0.9rem 1.1rem;
}}
[data-testid="stMetricValue"] {{
    font-family: 'Barlow Condensed', sans-serif; font-weight: 600; color: {NAVY};
}}

/* Tier bars and map legend */
.tier-row {{ display: grid; grid-template-columns: 7.5rem 1fr 3.5rem; align-items: center;
             gap: 0.8rem; margin: 0.7rem 0; }}
.tier-name {{ font-weight: 600; color: {NAVY}; }}
.tier-track {{ background: #E3EFF6; height: 26px; }}
.tier-fill {{ height: 100%; }}
.tier-val {{ font-family: 'Barlow Condensed', sans-serif; font-size: 1.35rem; font-weight: 600;
             color: {NAVY}; text-align: right; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 1.4rem; margin: 0.4rem 0 0.2rem 0; color: #3A5068; }}
.legend span {{ display: inline-flex; align-items: center; gap: 0.45rem; }}
.legend i {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; }}
</style>
"""

HEADER = f"""
<div class="flag">
  <div class="stripe"></div>
  <div class="body">
    <div>
      <h1>Which Chicago kitchens should inspectors visit first?</h1>
      <p>A model trained on restaurant inspections before 2024 ranks every restaurant by its risk of
         failing. Here's how that ranking would have played out on real inspections from 2024 onward.</p>
    </div>
    <div class="stars">{star_svg() * 4}</div>
  </div>
  <div class="stripe"></div>
</div>
"""


def tier_bars(rates: pd.Series) -> str:
    top = max(rates.max(), 0.01)
    rows = []
    for tier, rate in rates.items():
        rows.append(
            f'<div class="tier-row"><span class="tier-name">{tier} risk</span>'
            f'<div class="tier-track"><div class="tier-fill" style="width:{rate / top * 100:.1f}%;'
            f'background:{TIER_HEX[tier]}"></div></div><span class="tier-val">{rate:.0%}</span></div>')
    return "".join(rows)


LEGEND = ('<div class="legend">'
          + "".join(f'<span><i style="background:{TIER_HEX[t]}"></i>{t} risk</span>' for t in TIERS)
          + "</div>")


# ---------- Page ----------

st.set_page_config(page_title="Chicago Inspection Risk", page_icon="✶", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(HEADER, unsafe_allow_html=True)

if not SCORED_PATH.exists():
    st.error("No scored data yet. In the terminal, run `python src/simulation.py` first, then reload.")
    st.stop()

data = load_data(str(SCORED_PATH))
nums = headline_numbers(data)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Failures found sooner", f"{nums['avg_days']:.1f} days",
          help="Average days earlier a failing restaurant would have been inspected in the model's risk order.")
c2.metric("Failures found by mid-month", f"{nums['mid_month_model']:.0%}",
          f"{(nums['mid_month_model'] - nums['mid_month_actual']) * 100:+.0f} pts vs. actual order")
c3.metric("Extra failures caught", f"{nums['extra_found']:+,}",
          help="Versus flagging restaurants that failed their last routine inspection, with the same number of flags.")
c4.metric("Inspections scored", f"{nums['n_inspections']:,}", f"{nums['n_fails']:,} failed", delta_color="off")

# Sidebar filters
st.sidebar.header("Filters")
tiers = st.sidebar.multiselect("Risk tier", TIERS, default=TIERS,
                               help="High = top 20% of scores, Medium = next 30%, Low = bottom 50%.")
results = st.sidebar.multiselect("Actual result", RESULTS, default=RESULTS)
search = st.sidebar.text_input("Restaurant name contains")

latest = latest_per_restaurant(data)
view = latest[latest["risk_tier"].isin(tiers) & latest["results"].isin(results)]
if search:
    view = view[view["dba_name"].str.contains(search, case=False, regex=False)]

# Map
st.subheader("Every restaurant at its most recent routine inspection")
map_df = view.dropna(subset=["latitude", "longitude"]).copy()
if map_df.empty:
    st.info("No restaurants match these filters.")
else:
    map_df[["r", "g", "b"]] = pd.DataFrame(map_df["risk_tier"].map(TIER_COLORS).tolist(), index=map_df.index)
    map_df["risk_label"] = map_df["risk_percentile"].map(lambda p: f"riskier than {p:.0%} of inspections")
    map_df["date_label"] = map_df["inspection_date"].dt.strftime("%b %d, %Y")
    layer = pdk.Layer("ScatterplotLayer", data=map_df, get_position="[longitude, latitude]",
                      get_fill_color="[r, g, b, 170]", get_radius=70, radius_min_pixels=2,
                      radius_max_pixels=10, pickable=True)
    st.pydeck_chart(pdk.Deck(
        layers=[layer], map_provider="carto", map_style="light",
        initial_view_state=pdk.ViewState(latitude=41.85, longitude=-87.68, zoom=9.8),
        tooltip={"html": "<b>{dba_name}</b><br/>{address}<br/>{risk_tier} risk ({risk_label})"
                         "<br/>Actual result: <b>{results}</b> on {date_label}"},
    ))
    st.markdown(LEGEND, unsafe_allow_html=True)
    st.caption(f"Showing {len(map_df):,} restaurants. Hover over a dot to see the restaurant and its result.")

# Did the tiers work?
left, right = st.columns(2)
with left:
    st.subheader("How often each risk tier actually failed")
    tier_rates = data.groupby("risk_tier")["failed"].mean().reindex(TIERS).fillna(0)
    st.markdown(tier_bars(tier_rates), unsafe_allow_html=True)
    st.caption(f"Overall failure rate: {data['failed'].mean():.1%}. High = top 20% of scores, "
               "medium = next 30%, low = bottom 50%.")
with right:
    st.subheader("Same inspectors, riskiest first")
    if SIM_FIG.exists():
        st.image(str(SIM_FIG))
    st.caption("Each month keeps its real inspection dates. Only the order changes, and failing "
               "kitchens get found sooner.")

# Riskiest restaurants
st.subheader("The 25 highest-risk restaurants right now")
top = view.nlargest(25, "risk_score")[["dba_name", "address", "risk_tier", "risk_percentile",
                                       "inspection_date", "results"]]
st.dataframe(
    top, hide_index=True,
    column_config={
        "dba_name": "Restaurant", "address": "Address", "risk_tier": "Risk tier",
        "risk_percentile": st.column_config.ProgressColumn("Risk percentile", min_value=0, max_value=1,
                                                           format="%.2f"),
        "inspection_date": st.column_config.DateColumn("Inspection date", format="MMM D, YYYY"),
        "results": "Actual result",
    },
)

# Restaurant lookup
st.subheader("Look up a restaurant")
query = st.text_input("Type part of a restaurant's name", key="lookup")
if query:
    matches = latest[latest["dba_name"].str.contains(query, case=False, regex=False)]
    if matches.empty:
        st.info("No restaurants found with that name.")
    else:
        matches = matches.sort_values("dba_name")
        options = {f"{r.dba_name} ({r.address})": r.license_number for r in matches.itertuples()}
        choice = st.selectbox("Choose a restaurant", list(options))
        history = data[data["license_number"] == options[choice]].sort_values("inspection_date")
        last = history.iloc[-1]
        st.markdown(f"**{last['risk_tier']} risk** at its most recent inspection "
                    f"(riskier than {last['risk_percentile']:.0%} of inspections). "
                    f"Actual result: **{last['results']}**.")
        st.markdown("\n".join(f"- {reason}" for reason in risk_reasons(last)))
        st.dataframe(
            history[["inspection_date", "risk_tier", "risk_percentile", "results"]],
            hide_index=True,
            column_config={
                "inspection_date": st.column_config.DateColumn("Inspection date", format="MMM D, YYYY"),
                "risk_tier": "Risk tier",
                "risk_percentile": st.column_config.ProgressColumn("Risk percentile", min_value=0,
                                                                   max_value=1, format="%.2f"),
                "results": "Actual result",
            },
        )

st.caption("Data: City of Chicago Food Inspections (Chicago Data Portal).")
