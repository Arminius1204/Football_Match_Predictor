"""EPL goal-difference predictor: Streamlit front end.

    python -m streamlit run app.py

UI only. All ML lives in src/ (Predictor reads artifacts/ written by train_app_model.py).
Every number shown is read from artifacts/meta.json, so the page can't drift from
the evaluated model.
"""

from datetime import date
from html import escape

import pandas as pd
import streamlit as st

from src import charts
from src.config import ARTIFACT_DIR
from src.predict import Predictor

st.set_page_config(page_title="Who's got the edge? EPL goal-difference model",
                   page_icon="⚽", layout="centered")

# ---------------------------------------------------------------------------
# Styling: Barlow / Barlow Condensed over the turf theme from config.toml
# ---------------------------------------------------------------------------
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&family=Barlow+Condensed:wght@500;600;700&display=swap');
/* Set the family on the app root and let it inherit. A blanket [class*="st-"] rule would
   also override Streamlit's Material icon font and print icon names as text. */
html, body, .stApp, .stApp p, .stApp li, .stApp label, .stApp button, .stApp input,
.stApp summary, .stApp td, .stApp th {{ font-family: 'Barlow', sans-serif; }}
.block-container {{ max-width: 860px; padding-top: 2.2rem; padding-bottom: 4rem; }}
h1, h2, h3, h4 {{ font-family: 'Barlow Condensed', sans-serif !important; letter-spacing: 0.01em; }}
h1 {{ font-size: 3.1rem !important; font-weight: 700 !important; line-height: 1.0 !important; margin-bottom: 0.2rem; }}
h3 {{ font-size: 1.55rem !important; font-weight: 600 !important; margin-top: 0.6rem; }}
h4 {{ font-size: 1.3rem !important; font-weight: 600 !important; }}
.lede {{ color: {charts.CHALK_DIM}; font-size: 1.02rem; margin: 0 0 1.4rem 0; max-width: 62ch; }}

/* Scoreboard: the one big moment on the page */
.board {{ display: grid; grid-template-columns: 1fr auto 1fr; align-items: end; gap: 1rem;
          margin: 1.6rem 0 0.2rem 0; }}
.board .team {{ font-family: 'Barlow Condensed', sans-serif; font-weight: 600; font-size: 2.2rem;
                line-height: 1.05; }}
.board .home {{ color: {charts.BRASS}; text-align: left; }}
.board .away {{ color: {charts.SKY}; text-align: right; }}
.board .side {{ display: block; font-family: 'Barlow', sans-serif; font-weight: 500; font-size: 0.85rem;
                color: {charts.CHALK_DIM}; letter-spacing: 0.02em; }}
.board .gd {{ font-family: 'Barlow Condensed', sans-serif; font-weight: 700; font-size: 5.4rem;
              line-height: 0.9; text-align: center; font-variant-numeric: tabular-nums; }}
.verdict {{ text-align: center; font-size: 1.25rem; font-weight: 600; margin: 0.6rem 0 0.15rem 0; }}
.subverdict {{ text-align: center; color: {charts.CHALK_DIM}; font-size: 0.98rem; margin-bottom: 0.4rem; }}

/* Recent results */
.recent {{ list-style: none; padding: 0 !important; margin: 0.2rem 0 1.6rem 0 !important; }}
.recent li {{ margin: 0 !important; display: grid; grid-template-columns: 1.7rem 2.9rem 1fr auto; gap: 0.5rem; align-items: center;
              padding: 0.32rem 0; border-bottom: 1px solid {charts.CHALK_FAINT}; font-size: 0.96rem; }}
.res {{ display: inline-flex; justify-content: center; align-items: center; width: 1.45rem; height: 1.45rem;
        border-radius: 3px; font-weight: 600; font-size: 0.8rem; }}
.res.W {{ background: {charts.CHALK}; color: {charts.TURF}; }}
.res.D {{ border: 1.5px solid {charts.CHALK_DIM}; color: {charts.CHALK}; }}
.res.L {{ background: rgba(238,241,234,0.12); color: {charts.CHALK_DIM}; }}
.score {{ font-variant-numeric: tabular-nums; font-weight: 600; }}
.when {{ color: {charts.CHALK_DIM}; font-size: 0.85rem; }}
.teamhead {{ font-family: 'Barlow Condensed', sans-serif; font-size: 1.45rem; font-weight: 600; margin: 0; }}
.elo {{ color: {charts.CHALK_DIM}; font-size: 0.92rem; margin: 0 0 0.3rem 0; }}

.stButton > button {{ font-family: 'Barlow Condensed', sans-serif; font-size: 1.05rem; }}
:focus-visible {{ outline: 2px solid {charts.BRASS} !important; outline-offset: 2px; }}
@media (max-width: 640px) {{
  h1 {{ font-size: 2.4rem !important; }}
  .board .team {{ font-size: 1.5rem; }}
  .board .gd {{ font-size: 3.6rem; }}
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@st.cache_resource
def get_predictor():
    """Load the ensemble, scaler and team snapshots once per server process."""
    return Predictor(ARTIFACT_DIR)


def nice_date(iso):
    """'2026-05-24' -> '24 May 2026'."""
    d = date.fromisoformat(iso)
    return f"{d.day} {d.strftime('%B %Y')}"


def season_label(code):
    """'2526' -> '2025-26'."""
    return f"20{code[:2]}-{code[2:]}"


try:
    P = get_predictor()
except FileNotFoundError:
    st.error("Model files are missing. Run `python train_app_model.py`, then reload this page.")
    st.stop()

META = P.meta
LATEST = season_label(P.latest_season)
AS_OF = nice_date(META["state"]["ratings_as_of"])


# ---------------------------------------------------------------------------
# Header and team pickers
# ---------------------------------------------------------------------------
def render_header():
    st.title("Who's got the edge?")
    st.markdown(
        f'<p class="lede">Pick a Premier League fixture and a neural network predicts the final '
        f'goal difference from each club\'s Elo rating and its last five matches. '
        f'Ratings as of {AS_OF.replace(" ", "&nbsp;")}.</p>', unsafe_allow_html=True)


def _ensure_valid_selection(options):
    """Keep the current picks if they are still offered, otherwise fall back to defaults."""
    defaults = [t for t in ("Arsenal", "Liverpool") if t in options] or options[:2]
    if st.session_state.get("home") not in options:
        st.session_state.home = defaults[0]
    if st.session_state.get("away") not in options or st.session_state.away == st.session_state.home:
        preferred = [t for t in defaults[1:] + options if t != st.session_state.home]
        st.session_state.away = preferred[0]


def _swap():
    st.session_state.home, st.session_state.away = st.session_state.away, st.session_state.home


def team_label(team):
    """Dropdown label. Clubs outside the latest season are tagged."""
    if P.is_current(team):
        return team
    return f"{team} (last in the league {season_label(P.snapshots[team]['last_season'])})"


def render_pickers():
    include_all = st.session_state.get("include_all", False)
    options = P.teams(include_all=include_all)
    _ensure_valid_selection(options)

    c1, c2, c3 = st.columns([5, 1.3, 5], vertical_alignment="bottom")
    with c1:
        st.selectbox("Home team", options, key="home", format_func=team_label)
    with c2:
        st.button("Swap", on_click=_swap, width="stretch",
                  help="Swap the home and away teams")
    with c3:
        st.selectbox("Away team", [t for t in options if t != st.session_state.home],
                     key="away", format_func=team_label)
    st.toggle(f"Include clubs outside the {LATEST} Premier League", key="include_all")
    st.caption(f"The default list is the 20 clubs of {LATEST}. League membership for 2026-27 "
               f"is not reflected: clubs promoted this summer are missing, and clubs relegated "
               f"in May are still listed.")
    return st.session_state.home, st.session_state.away


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def verdict_text(pred):
    """Headline wording from the validated band, or a neutral line if bands are off."""
    leader = pred.home if pred.lean == "H" else pred.away
    margin = abs(pred.gd)
    if pred.band is None:
        return f"{leader} by {margin:.2f} goals", ""
    if pred.band == "Too close to call":
        return "Too close to call", f"The model leans slightly to {leader}, by {margin:.2f} goals."
    return f"{pred.band} to {leader}", f"The model expects {leader} to win by about {margin:.1f} goals."


def band_caption():
    """Explain the gauge shading using the validation cutoffs."""
    if not P.show_bands:
        return ""
    c1, c2 = P.band_cutoffs
    return (f"Shading, from the halfway line outwards: too close to call (under {c1:.2f} goals), "
            f"slight edge ({c1:.2f} to {c2:.2f}), clear edge ({c2:.2f} or more).")


def render_prediction(home, away):
    pred = P.predict(home, away)
    head, sub = verdict_text(pred)
    lo, hi = min(pred.gd_members), max(pred.gd_members)
    st.markdown(f"""
<div class="board" role="group" aria-label="Predicted goal difference">
  <div class="team home"><span class="side">Home</span>{escape(home)}</div>
  <div class="gd">{pred.gd:+.2f}</div>
  <div class="team away"><span class="side">Away</span>{escape(away)}</div>
</div>
<p class="verdict">{escape(head)}</p>
<p class="subverdict">{escape(sub)} Positive numbers favour {escape(home)}.</p>
""", unsafe_allow_html=True)

    st.plotly_chart(charts.gd_gauge(pred.gd, pred.gd_members, P.band_cutoffs if P.show_bands else [],
                                    home, away),
                    width="stretch", config=charts.PLOT_CONFIG)
    outcome_name = {"H": f"{home} win", "A": f"{away} win", "D": "draw"}[pred.outcome]
    st.caption(f"The five networks (seeds 0-4) range from {lo:+.2f} to {hi:+.2f} goals; the large marker "
               f"is their average. {band_caption()} The assignment's win/draw/loss "
               f"rule (draw if the margin is within {P.threshold:.2f}) gives: {outcome_name}.")
    for w in pred.warnings:
        st.warning(w, icon="⏳")
    return pred


# ---------------------------------------------------------------------------
# Team comparison
# ---------------------------------------------------------------------------
def recent_list(team):
    """Last five results as an accessible list (letter + score, not colour alone)."""
    items = []
    for m in reversed(P.snapshots[team]["recent"]):
        where = "home" if m["venue"] == "H" else "away"
        items.append(
            f'<li><span class="res {m["result"]}" aria-label="{m["result"]}">{m["result"]}</span>'
            f'<span class="score">{m["gf"]}-{m["ga"]}</span>'
            f'<span>{escape(m["opponent"])} <span class="when">({where})</span></span>'
            f'<span class="when">{nice_date(m["date"])}</span></li>')
    return f'<ul class="recent">{"".join(items)}</ul>'


def render_comparison(home, away):
    st.subheader("How they compare")
    sh, sa = P.snapshots[home], P.snapshots[away]
    c1, c2 = st.columns(2, gap="large")
    for col, team, snap, colour in ((c1, home, sh, charts.BRASS), (c2, away, sa, charts.SKY)):
        with col:
            st.markdown(f'<p class="teamhead" style="color:{colour}">{escape(team)}</p>'
                        f'<p class="elo">Elo {snap["elo"]:.0f}, last played {nice_date(snap["last_match_date"])}</p>'
                        + recent_list(team), unsafe_allow_html=True)

    st.markdown("#### Last five matches, per-game averages")
    st.plotly_chart(charts.form_mirror(home, away, sh["form"], sa["form"]),
                    width="stretch", config=charts.PLOT_CONFIG)
    st.caption(f"{home} on the left, {away} on the right. Within each row the bars compare the two "
               f"clubs and the labels give the real averages. These eight numbers per club, plus Elo, "
               f"are exactly what the network sees.")

    st.markdown("#### Elo rating since 2014")
    st.plotly_chart(charts.elo_history(home, away, sh["elo_history"], sa["elo_history"]),
                    width="stretch", config=charts.PLOT_CONFIG)
    st.caption(f"Elo starts at 1500 (dotted line) and moves after every league match, more for "
               f"bigger margins. Home advantage is worth {P.home_adv:.0f} points. Ratings only change "
               f"while a club is in the Premier League.")


# ---------------------------------------------------------------------------
# About: method, held-out results and honest caveats
# ---------------------------------------------------------------------------
def pct(x):
    return f"{100 * x:.1f}%"


def render_about():
    m, b = META["ensemble_test_metrics"], META["baseline_test_metrics"]
    split, pe, bands = META["split"], META["probability_eval"], META["bands"]
    n_test = split["n_test"]
    test_span = f"{season_label(split['test_seasons'][0])} and {season_label(split['test_seasons'][-1])}"

    with st.expander("About this model", expanded=False):
        st.markdown(f"""
**What it is.** A multilayer perceptron with hidden layers of
{", ".join(map(str, META["model"]["hidden"]))} neurons and {META["model"]["activation"]} activations.
It is the configuration that won on validation in the Deep Learning assignment. The app averages five
copies trained with seeds 0-4. Inputs are 19 numbers known before kick-off: both clubs' Elo ratings
and their per-game averages over the last five league matches. Club names are never inputs.

**How it was tested.** Trained on {season_label(split['train_seasons'][0])} to
{season_label(split['train_seasons'][-1])}, tuned on {season_label(split['val_seasons'][0])},
and scored once on {test_span} ({n_test} matches), with no shuffling across time.
""")
        st.dataframe(pd.DataFrame({
            "": ["This model (5-seed average)", "Always predict the average margin"],
            "Mean absolute error (goals)": [f"{m['MAE']:.3f}", f"{b['MAE']:.3f}"],
            "RMSE": [f"{m['RMSE']:.3f}", f"{b['RMSE']:.3f}"],
            "R²": [f"{m['R2']:.3f}", f"{b['R2']:.3f}"],
            "Win/draw/loss accuracy": [pct(m["WDL_acc"]), f"{pct(b['WDL_acc_always_home'])} (always home)"],
        }), hide_index=True, width="stretch")
        st.markdown(f"""
An R² of {m['R2']:.2f} means most of a single match's outcome is noise the model can't see.
It is better at telling strong favourites from coin flips than at calling exact scores.

**What the wording means.** The bands were fixed using only the {season_label(split['val_seasons'][0])}
season: each covers a third of that season's predicted margins (cut-offs at
{bands['cutoffs'][0]:.2f} and {bands['cutoffs'][1]:.2f} goals). On the {n_test} test matches,
the side the model leaned toward did this:
""")
        st.dataframe(pd.DataFrame([{
            "Band": r["band"],
            "Predicted margin": (f"under {r['abs_gd_to']:.2f}" if r["abs_gd_from"] == 0 else
                                 f"{r['abs_gd_from']:.2f} or more" if r["abs_gd_to"] is None else
                                 f"{r['abs_gd_from']:.2f} to {r['abs_gd_to']:.2f}"),
            "Matches": r["n"], "Lean won": pct(r["lean_won"]),
            "Draw": pct(r["draw"]), "Lean lost": pct(r["lean_lost"])} for r in bands["rows"]]),
            hide_index=True, width="stretch")

        h_rows = [r for r in pe["calibration"]["H"] if r["n"] >= 50]
        over = sorted(-100 * r["gap"] for r in h_rows if r["gap"] < 0)
        st.markdown(f"""
**Why there are no percentages.** The predicted margin was also turned into win/draw/loss
probabilities. They scored better than simply using historical base rates (log loss
{pe['log_loss_model']:.3f} vs {pe['log_loss_baseline']:.3f}), but they did not clear a calibration bar
set in advance: the home-win calibration error was {pe['ece']['H']:.3f} against a limit of
{pe['max_ece_allowed']:.2f}. Home wins were overstated, predicted {pct(pe['mean_pred_probs']['H'])}
of the time on average against {pct(pe['observed_rates']['H'])} observed, and by about
{over[0]:.0f}-{over[-1]:.0f} points in most probability ranges. That is likely because home advantage
weakened in {test_span}. Draws and away wins were within the limit. Rather than adjust the numbers
after seeing the test results, the app shows the margin only.

**Keep in mind.**
- Ratings are as of {AS_OF}, the last match in the data. Summer transfers, injuries, managers and
  promoted clubs are not reflected.
- The {season_label(META['state']['latest_season'])} season refreshes team ratings only. The network,
  its scaling and its test results are frozen at the assignment version.
- A club outside the Premier League keeps the rating it had when it went down. Ratings for clubs
  tagged with an earlier season are likely stale.
- The model predicts an average margin. Even in the "Clear edge" band, the favourite failed to
  win {pct(1 - bands['rows'][-1]['lean_won'])} of test matches.
""")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
render_header()
home, away = render_pickers()
render_prediction(home, away)
st.divider()
render_comparison(home, away)
st.divider()
render_about()
st.caption(f"Data: football-data.co.uk via github.com/datasets/football-datasets, "
           f"{season_label(META['state']['seasons'][0])} to {LATEST}. Deep Learning (CSC403-8), CIA 1.")
