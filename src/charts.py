"""Plotly figures for the app. Pure presentation: inputs are plain numbers and dicts.

Visual language: a floodlit pitch. Turf-green ground, chalk lines, brass for the home
side and pale sky for the away side. Every chart uses the same two team colours.
"""

import plotly.graph_objects as go

# Palette (mirrors .streamlit/config.toml and the CSS in app.py)
TURF = "#0E3B2C"
TURF_LIGHT = "#16503B"
CHALK = "#EEF1EA"
CHALK_DIM = "rgba(238, 241, 234, 0.55)"
CHALK_FAINT = "rgba(238, 241, 234, 0.10)"
BRASS = "#D9B44A"      # home
SKY = "#9CC3E6"        # away
FONT = "Barlow, sans-serif"
FONT_DISPLAY = "Barlow Condensed, sans-serif"

PLOT_CONFIG = {"displayModeBar": False, "responsive": True}


def _base_layout(fig, height):
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=8, b=8),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family=FONT, color=CHALK, size=13), showlegend=False,
                      hoverlabel=dict(bgcolor=TURF_LIGHT, font=dict(family=FONT, color=CHALK)))
    return fig


def gd_gauge(gd, members, cutoffs, home, away, span=3.0):
    """Predicted goal difference on a 'pitch': home half on the left, away half on the right.

    Plot x = -GD, so a home edge points into the home half. Shaded zones are the wording
    bands (validation cutoffs). The bar is the range of the five seeds, and the marker is
    the ensemble mean.
    """
    x, xs = -gd, [-m for m in members]
    fig = go.Figure()

    # Wording-band zones: chalk shading grows with distance from the halfway line.
    edges = [0.0] + list(cutoffs) + [span]
    for i, alpha in enumerate([0.0, 0.05, 0.10]):
        for side in (-1, 1):
            if alpha:
                fig.add_shape(type="rect", x0=side * edges[i], x1=side * edges[i + 1], y0=-1, y1=1,
                              fillcolor=f"rgba(238,241,234,{alpha})", line_width=0, layer="below")
    # Pitch outline and halfway line (zero goal difference)
    fig.add_shape(type="rect", x0=-span, x1=span, y0=-1, y1=1, line=dict(color=CHALK_DIM, width=1.5))
    fig.add_shape(type="line", x0=0, x1=0, y0=-1, y1=1, line=dict(color=CHALK_DIM, width=1.5))
    fig.add_shape(type="circle", x0=-0.35, x1=0.35, y0=-0.55, y1=0.55, line=dict(color=CHALK_FAINT, width=1.5))

    colour = BRASS if gd > 0 else SKY
    # Seed spread (model disagreement), drawn above the marker row so it stays visible
    y_seed = 0.38
    fig.add_trace(go.Scatter(x=[min(xs), max(xs)], y=[y_seed, y_seed], mode="lines",
                             line=dict(color=colour, width=8), opacity=0.55,
                             hovertemplate="Seed range<extra></extra>"))
    fig.add_trace(go.Scatter(x=xs, y=[y_seed] * len(xs), mode="markers",
                             marker=dict(color=colour, size=9, symbol="line-ns",
                                         line=dict(color=CHALK, width=2)),
                             customdata=[[i, m] for i, m in enumerate(members)],
                             hovertemplate="Seed %{customdata[0]}: %{customdata[1]:+.2f}<extra></extra>"))
    fig.add_annotation(x=(min(xs) + max(xs)) / 2, y=y_seed + 0.22, text="five seeds", showarrow=False,
                       font=dict(size=11, color=CHALK_DIM))
    # Ensemble mean, with a drop line to the margin axis
    fig.add_shape(type="line", x0=x, x1=x, y0=-1, y1=y_seed, line=dict(color=colour, width=1, dash="dot"))
    fig.add_trace(go.Scatter(x=[x], y=[0], mode="markers",
                             marker=dict(color=colour, size=18, line=dict(color=TURF, width=3)),
                             hovertemplate=f"Ensemble: {gd:+.2f}<extra></extra>"))

    for side, name, col in ((-1, home, BRASS), (1, away, SKY)):
        fig.add_annotation(x=side * span * 0.97, y=0.72, text=f"{name} half", showarrow=False,
                           xanchor="left" if side < 0 else "right",
                           font=dict(family=FONT_DISPLAY, size=15, color=col))
    ticks = [t for t in range(-int(span), int(span) + 1)]
    fig.update_xaxes(range=[-span - 0.05, span + 0.05], tickvals=ticks,
                     ticktext=[str(abs(t)) for t in ticks], showgrid=False, zeroline=False,
                     tickfont=dict(color=CHALK_DIM), fixedrange=True,
                     title=dict(text="Predicted winning margin (goals)", font=dict(color=CHALK_DIM, size=12)))
    fig.update_yaxes(range=[-1.05, 1.05], visible=False, fixedrange=True)
    return _base_layout(fig, 190)


def elo_history(home, away, home_hist, away_hist, since="2014-08-01"):
    """Both teams' Elo over time (pre-match ratings, ending at the current rating)."""
    fig = go.Figure()
    for name, hist, col in ((home, home_hist, BRASS), (away, away_hist, SKY)):
        pts = [h for h in hist if h["date"] >= since]
        fig.add_trace(go.Scatter(x=[p["date"] for p in pts], y=[p["elo"] for p in pts],
                                 mode="lines", name=name, line=dict(color=col, width=2),
                                 connectgaps=False,
                                 hovertemplate=f"{name}<br>%{{x|%d %b %Y}}: %{{y:.0f}}<extra></extra>"))
    fig.add_hline(y=1500, line=dict(color=CHALK_FAINT, width=1, dash="dot"))
    fig.update_xaxes(showgrid=False, tickfont=dict(color=CHALK_DIM), fixedrange=True)
    fig.update_yaxes(gridcolor=CHALK_FAINT, tickfont=dict(color=CHALK_DIM), fixedrange=True,
                     title=dict(text="Elo", font=dict(color=CHALK_DIM, size=12)))
    fig = _base_layout(fig, 280)
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.08, x=0, font=dict(size=13)))
    return fig


FORM_LABELS = {"gf": "Goals scored", "ga": "Goals conceded", "sf": "Shots", "sa": "Shots faced",
               "stf": "Shots on target", "sta": "On target faced", "cf": "Corners", "pts": "Points"}


def form_mirror(home, away, home_form, away_form):
    """Last-5 averages as mirrored bars: home to the left, away to the right, per statistic."""
    keys = [k for k in FORM_LABELS if k in home_form][::-1]   # first stat at the top
    labels = [FORM_LABELS[k] for k in keys]
    hv, av = [home_form[k] for k in keys], [away_form[k] for k in keys]
    # Shots (~15) would dwarf goals (~1.5) on one scale, so each row is scaled so the
    # larger team reaches 1. Bar lengths compare the two teams within a row, and the
    # labels show the real values.
    scale = [max(h, a) or 1.0 for h, a in zip(hv, av)]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=labels, x=[-h / s for h, s in zip(hv, scale)], orientation="h",
                         marker_color=BRASS, text=[f"{v:.1f}" for v in hv], textposition="outside",
                         cliponaxis=False, customdata=hv,
                         hovertemplate=f"{home}: %{{customdata:.2f}}<extra></extra>"))
    fig.add_trace(go.Bar(y=labels, x=[a / s for a, s in zip(av, scale)], orientation="h",
                         marker_color=SKY, text=[f"{v:.1f}" for v in av], textposition="outside",
                         cliponaxis=False, customdata=av,
                         hovertemplate=f"{away}: %{{customdata:.2f}}<extra></extra>"))
    fig.update_traces(width=0.6, textfont=dict(color=CHALK, size=12))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(visible=False, range=[-1.35, 1.35], fixedrange=True)
    fig.update_yaxes(tickfont=dict(color=CHALK, size=13), fixedrange=True)
    return _base_layout(fig, 330)
