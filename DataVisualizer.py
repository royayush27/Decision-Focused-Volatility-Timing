"""
Visualiseur interactif du panel GARCH-LSTM, action par action.

Lancement :  python viewer_dashboard.py   -> s'ouvre dans le navigateur (http://127.0.0.1:8050)
Prérequis :  pip install dash plotly pyarrow

- Filtre par taille de capitalisation (micro / small / mid / large), liste avec recherche,
  boutons précédent / suivant / au hasard (dans le filtre)
- Séries empilées sur un axe temporel commun : prix, performance vs benchmark, rendements,
  volatilité, volume, spread, capitalisation
- Benchmark S&P 500 ou Russell 2000 : performance base 100, vol réalisée de l'indice,
  bêta / corrélation / surperformance dans la fiche
- VIX superposé à la volatilité de l'action
- Zoom synchronisé (molette ou sélection), double-clic pour revenir, survol unifié
- Anomalies marquées sur le prix, fiche de synthèse de l'action à droite
"""
import glob
import os
import socket
import threading
import webbrowser

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyarrow.parquet as pq
from dash import Dash, Input, Output, State, ctx, dcc, html
from plotly.subplots import make_subplots

# ============================================================================
# Paramètres
# ============================================================================
BASE_DIR = r"C:\Users\Royal\OneDrive - De Vinci\Financial Engineering - Hanyang"
PANEL_PATH = os.path.join(BASE_DIR, "datasets", "panel_garch_lstm.parquet")
VOL_WINDOW = 21   # jours pour les vols glissantes
PORT = 8050          # si déjà occupé (ancien visualiseur ouvert), le port suivant libre est utilisé
VERSION = "v4"       # affiché dans l'en-tête : permet de vérifier que c'est bien la dernière version

# Classes de capitalisation (seuils usuels, en dollars nominaux)
CAP_BINS = [0, 250e6, 2e9, 10e9, np.inf]
CAP_LABELS = ["Micro", "Small", "Mid", "Large"]
CAP_DESC = {"Micro": "< 250 M$", "Small": "250 M$ – 2 Md$", "Mid": "2 – 10 Md$", "Large": "> 10 Md$",
            "Inconnue": "capi manquante"}

BENCHMARKS = {"SPX": "S&P 500", "RUT": "Russell 2000"}
TR_OF = {"SPX": "SPXTR", "RUT": "RUTTR"}   # version total return (dividendes réinvestis) de chaque indice
BASIS_LABEL = {"tr": "total return", "prix": "prix seul"}

STOCK_COLS = ["Instrument", "Date", "Open", "High", "Low", "Close", "Volume", "Bid", "Ask",
              "TotalReturnPct", "MarketCap", "LogRet", "RelSpread", "VarParkinson",
              "VarGarmanKlass", "InUniverse"]
MARKET_COLS = ["SPX", "VIX", "RUT", "SPXTR", "RUTTR"]

# Palette catégorielle (ordre fixe, validée daltonisme). Chaque entité garde sa couleur partout :
# action = bleu (et ses mesures de vol : aqua, orange), benchmark = vert, VIX = violet.
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "green": "#008300", "violet": "#4a3aa7"}
CRITICAL = "#d03b3b"
SURFACE, TEXT, TEXT2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983", "#e4e3de"
FONT = "Inter, Segoe UI, system-ui, sans-serif"

PANELS = {
    "prix": "Prix",
    "perf": "Performance",
    "rendements": "Rendements",
    "vol": "Volatilité",
    "volume": "Volume",
    "spread": "Spread",
    "capi": "Capitalisation",
}
TITLES = {
    "prix": "Prix ($)",
    "perf": "Performance cumulée, base 100",
    "rendements": "Log-rendement journalier (%)",
    "vol": f"Volatilité annualisée ({VOL_WINDOW} j glissants)",
    "volume": "Volume échangé (titres)",
    "spread": "Spread bid-ask relatif (points de base)",
    "capi": "Capitalisation boursière (Md$, échelle log)",
}


# ============================================================================
# Chargement (une seule fois, puis tout se fait en mémoire)
# ============================================================================
def load_panel():
    print(f"Chargement de {PANEL_PATH} ...")
    available = set(pq.read_schema(PANEL_PATH).names)
    cols = [c for c in STOCK_COLS + MARKET_COLS if c in available]
    df = pd.read_parquet(PANEL_PATH, columns=cols)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Instrument", "Date"], kind="stable").reset_index(drop=True)

    # séries de marché : une ligne par date, retirées du panel pour économiser la mémoire
    mcols = [c for c in MARKET_COLS if c in df]
    market = pd.DataFrame()
    if mcols:
        market = (df[["Date"] + mcols].drop_duplicates("Date").set_index("Date")
                  .sort_index().astype("float64"))
    df = df.drop(columns=mcols)
    market = add_market_fallback(market)
    market = market.dropna(axis=1, how="all")

    # position de chaque action dans le tableau trié -> accès instantané
    codes, uniques = pd.factorize(df["Instrument"], sort=False)
    bounds = np.flatnonzero(np.diff(codes)) + 1
    starts, ends = np.r_[0, bounds], np.r_[bounds, len(df)]
    slices = {uniques[i]: (starts[i], ends[i]) for i in range(len(uniques))}

    # classe de capi de chaque action = selon sa DERNIÈRE capitalisation connue
    if "MarketCap" in df:
        last_cap = df.groupby("Instrument", sort=False)["MarketCap"].last()
        cls = pd.cut(last_cap, CAP_BINS, labels=CAP_LABELS, right=False).astype(object).fillna("Inconnue")
        classes = cls.to_dict()
    else:
        classes = {r: "Inconnue" for r in slices}

    print(f"{len(df):,} lignes, {len(slices):,} actions chargées.")
    return df, (market if len(market.columns) else None), slices, classes


def add_market_fallback(market):
    """Si le panel n'a pas une série de marché, la chercher dans les fichiers de marché
    écrits par l'extraction (market.parquet ou MARKET_<nom>.parquet)."""
    missing = [c for c in MARKET_COLS if c not in market or market[c].notna().sum() == 0]
    if not missing:
        return market
    folder = os.path.dirname(PANEL_PATH)
    ric_to_name = {".SPX": "SPX", ".VIX": "VIX", ".RUT": "RUT", ".SPXTR": "SPXTR", ".RUTTR": "RUTTR"}
    found = {}
    path = os.path.join(folder, "market.parquet")
    if os.path.exists(path):
        m = pd.read_parquet(path)
        if {"Instrument", "Date", "Close"} <= set(m.columns):
            for ric, name in ric_to_name.items():
                s = m[m["Instrument"] == ric].set_index(pd.to_datetime(m.loc[m["Instrument"] == ric, "Date"]))["Close"]
                if name in missing and s.notna().any():
                    found[name] = s
    for path in glob.glob(os.path.join(folder, "MARKET_*.parquet")):
        name = os.path.basename(path)[7:-8]
        if name in missing and name not in found:
            m = pd.read_parquet(path)
            date_col = next((c for c in m.columns if c.lower() == "date"), None)
            vals = [c for c in m.columns if c not in ("Instrument", date_col)]
            if date_col and vals:
                found[name] = m.set_index(pd.to_datetime(m[date_col]))[vals[0]]
    for name, s in found.items():
        s = pd.to_numeric(s, errors="coerce").groupby(level=0).last().astype("float64")
        s.index.name = "Date"
        market = market.join(s.rename(name), how="outer") if len(market.columns) else s.rename(name).to_frame()
        print(f"  {name} absent du panel : repris depuis les fichiers de marché.")
    return market


DF, MARKET, SLICES, CLASSES = load_panel()
RICS = sorted(SLICES)
DEFAULT_RIC = max(SLICES, key=lambda r: SLICES[r][1] - SLICES[r][0])  # historique le plus long
CLASS_COUNTS = pd.Series(CLASSES).value_counts()
CLASS_ORDER = [c for c in CAP_LABELS + ["Inconnue"] if c in CLASS_COUNTS]

# séries dérivées des indices (calculées une fois)
ANN = np.sqrt(252) * 100
BENCH_RET, BENCH_VOL = {}, {}      # clé = SPX, RUT (prix) ou SPXTR, RUTTR (total return)
if MARKET is not None:
    for code in list(BENCHMARKS) + list(TR_OF.values()):
        if code in MARKET and MARKET[code].notna().sum() > 250:
            lvl = MARKET[code].dropna()
            r = np.log(lvl / lvl.shift(1))
            BENCH_RET[code] = r
            BENCH_VOL[code] = r.rolling(VOL_WINDOW, min_periods=VOL_WINDOW // 2).std() * ANN
AVAILABLE_BENCH = [c for c in BENCHMARKS if c in BENCH_RET or TR_OF[c] in BENCH_RET]


def load_sources():
    """Source réelle de chaque série de marché (indice LSEG, ETF de repli, FRED), si connue."""
    path = os.path.join(os.path.dirname(PANEL_PATH), "market.parquet")
    try:
        m = pd.read_parquet(path, columns=["Instrument", "Source"])
    except Exception:
        return {}
    names = {".SPX": "SPX", ".VIX": "VIX", ".RUT": "RUT", ".SPXTR": "SPXTR", ".RUTTR": "RUTTR"}
    return {names[i]: str(src) for i, src in m.groupby("Instrument")["Source"].first().items() if i in names}


SOURCES = load_sources()


def via(code):
    """' (via ETF SPY)' quand la série vient d'un ETF de repli."""
    src = SOURCES.get(code, "")
    for etf in ("SPY", "IWM"):
        if etf in src:
            return f" (via ETF {etf})"
    return ""


def comparison(bench, basis):
    """Choisit des séries de même nature pour l'action et l'indice.
    Retourne (base effective, code de la série indice, libellé indice, note)."""
    name = BENCHMARKS[bench]
    tr_code = TR_OF[bench]
    if basis == "tr" and tr_code in BENCH_RET:
        note = "Total return des deux côtés : dividendes réinvestis pour l'action et pour l'indice."
        if via(tr_code):
            note += f" L'indice est reconstruit depuis l'ETF{via(tr_code)[9:-1]}, frais de gestion inclus (~0,1-0,2 %/an)."
        return "tr", tr_code, f"{name} total return{via(tr_code)}", note
    if bench in BENCH_RET:
        note = "Prix seul des deux côtés : dividendes exclus pour l'action et pour l'indice."
        if basis == "tr":
            note = ("Pas de série total return pour cet indice : comparaison faite en prix des deux côtés "
                    "(dividendes exclus), pour rester cohérente. Relance l'extraction pour récupérer "
                    f"{tr_code}.")
        return "prix", bench, f"{name} prix{via(bench)}", note
    # seule la version TR existe : on compare en total return
    return "tr", tr_code, f"{name} total return{via(tr_code)}", \
        "Seule la version total return de l'indice est disponible : comparaison en total return."
HAS_VIX = MARKET is not None and "VIX" in MARKET and MARKET["VIX"].notna().any()
HAS_CAP = "MarketCap" in DF and DF["MarketCap"].notna().any()

# Diagnostic : ce qui manque dans les données, affiché en haut de la page et dans la console
DIAG = []
if not HAS_CAP:
    DIAG.append("Pas de colonne MarketCap exploitable dans le panel : le filtre par taille est désactivé.")
if not HAS_VIX:
    DIAG.append("Pas de série VIX dans le panel ni dans les fichiers de marché : l'option VIX est désactivée.")
for code, name in BENCHMARKS.items():
    if code not in AVAILABLE_BENCH:
        DIAG.append(f"Pas de série {name} ({code}) : ce benchmark n'est pas proposé.")
    elif TR_OF[code] not in BENCH_RET:
        DIAG.append(f"Pas de version total return du {name} ({TR_OF[code]}) : la comparaison se fera en prix. "
                    f"Relance l'extraction pour la récupérer.")
if MARKET is not None:
    print("Séries de marché disponibles :", ", ".join(f"{c} ({MARKET[c].notna().sum():,} jours)"
                                                        for c in MARKET.columns))
for msg in DIAG:
    print("  !", msg)


# ============================================================================
# Préparation des séries d'une action
# ============================================================================
def stock_frame(ric):
    a, b = SLICES[ric]
    d = DF.iloc[a:b].copy()
    for c in d.columns:
        if d[c].dtype == "float32":
            d[c] = d[c].astype("float64")
    w, mp = VOL_WINDOW, max(5, VOL_WINDOW // 2)
    if "VarGarmanKlass" in d:
        d["VolGK"] = np.sqrt(d["VarGarmanKlass"].clip(lower=0).rolling(w, min_periods=mp).mean()) * ANN
    if "VarParkinson" in d:
        d["VolPK"] = np.sqrt(d["VarParkinson"].rolling(w, min_periods=mp).mean()) * ANN
    if "LogRet" in d:
        d["VolCC"] = d["LogRet"].rolling(w, min_periods=mp).std() * ANN
    if "Close" in d:
        d["LogRetPrix"] = np.log(d["Close"] / d["Close"].shift(1))
    if "MarketCap" in d:
        d["CapClass"] = pd.cut(d["MarketCap"], CAP_BINS, labels=CAP_LABELS, right=False)

    # anomalies (mêmes règles que check_dataset.py)
    reasons = pd.Series("", index=d.index)

    def flag(mask, text):
        nonlocal reasons
        mask = mask.fillna(False)
        reasons = reasons.where(~mask, reasons + text + " ; ")

    if {"Open", "High", "Low", "Close"} <= set(d):
        flag(d["High"] < d["Low"], "High < Low")
        flag((d["Close"] > d["High"] * 1.001) | (d["Close"] < d["Low"] * 0.999), "Close hors [Low, High]")
        flag((d[["Open", "High", "Low", "Close"]] <= 0).any(axis=1), "prix <= 0")
    if {"Close", "LogRet"} <= set(d):
        diff = (np.log(d["Close"] / d["Close"].shift(1)) - d["LogRet"]).abs()
        flag(diff > 0.05, "rendement prix ≠ total return (split ?)")
    if "LogRet" in d:
        flag(d["LogRet"].abs() > np.log(1.5), "|rendement| > 50 %")
    if {"Bid", "Ask"} <= set(d):
        flag(d["Bid"] > d["Ask"], "Bid > Ask")
    d["Anomalie"] = reasons.str.rstrip(" ;")
    return d


def benchmark_stats(d, code, basis):
    """Bêta, corrélation, rendements sur la fenêtre affichée, séries de même nature."""
    col = "LogRet" if basis == "tr" else "LogRetPrix"
    if code not in BENCH_RET or col not in d:
        return None
    rs = d.set_index("Date")[col]
    rb = BENCH_RET[code].reindex(rs.index)
    ok = rs.notna() & rb.notna()
    rs, rb = rs[ok], rb[ok]
    if len(rs) < 60:
        return None
    beta = np.cov(rs, rb)[0, 1] / rb.var()
    return {
        "beta": beta,
        "corr": rs.corr(rb),
        "ret_s": np.expm1(rs.mean() * 252),
        "ret_b": np.expm1(rb.mean() * 252),
        "vol_s": rs.std() * np.sqrt(252),
        "vol_b": rb.std() * np.sqrt(252),
        "n": len(rs),
    }


# ============================================================================
# Figure
# ============================================================================
def build_figure(ric, panels, options, period="tout", bench="aucun", basis="tr", since=None):
    full = stock_frame(ric)   # vols calculées sur tout l'historique, puis fenêtre d'affichage
    d = full
    if period == "depuis" and since:
        d = d[d["Date"] >= pd.Timestamp(since)]
    elif period not in ("tout", "depuis"):
        start = d["Date"].max() - pd.DateOffset(years=int(period))
        d = d[d["Date"] >= start]
    x = d["Date"]
    bench = bench if bench in AVAILABLE_BENCH else None
    cmp = comparison(bench, basis) if bench else ("tr" if basis == "tr" else "prix", None, "", "")
    eff_basis, bcode, blabel, _note = cmp
    stock_col = "LogRet" if eff_basis == "tr" else "LogRetPrix"

    panels = [p for p in PANELS if p in panels] or ["prix"]
    weights = [2.4 if p == "prix" else 1.6 if p == "perf" else 1.3 for p in panels]
    titles = dict(TITLES)
    if len(x):
        titles["perf"] = (f"Performance cumulée, base 100 au {x.iloc[0]:%d/%m/%Y} "
                          f"({BASIS_LABEL[eff_basis]})")
    fig = make_subplots(rows=len(panels), cols=1, shared_xaxes=True, vertical_spacing=0.035,
                        row_heights=weights, subplot_titles=[titles[p] for p in panels])

    def line(row, xs, y, name, color, fmt, dash=None, width=1.6, legend=True):
        fig.add_trace(go.Scattergl(
            x=xs, y=y, name=name, mode="lines", line=dict(color=color, width=width, dash=dash),
            hovertemplate=f"{name} : %{{y:{fmt}}}<extra></extra>", showlegend=legend,
            legendgroup=name), row=row, col=1)

    for i, p in enumerate(panels, start=1):
        if p == "prix" and "Close" in d:
            if "chandeliers" in options and {"Open", "High", "Low"} <= set(d):
                fig.add_trace(go.Candlestick(
                    x=x, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
                    name="OHLC", showlegend=False,
                    increasing=dict(line=dict(color=C["aqua"], width=1), fillcolor=C["aqua"]),
                    decreasing=dict(line=dict(color=C["orange"], width=1), fillcolor=C["orange"])),
                    row=i, col=1)
            else:
                line(i, x, d["Close"], "Clôture", C["blue"], ",.2f", legend=False)
            bad = d[d["Anomalie"] != ""]
            if len(bad):
                fig.add_trace(go.Scatter(
                    x=bad["Date"], y=bad["Close"], mode="markers", name=f"Anomalies ({len(bad)})",
                    marker=dict(symbol="x", size=9, color=CRITICAL, line=dict(width=2, color=CRITICAL)),
                    customdata=bad["Anomalie"], hovertemplate="⚠ %{customdata}<extra></extra>"),
                    row=i, col=1)
            if "log" in options:
                fig.update_yaxes(type="log", row=i, col=1)

        elif p == "perf" and stock_col in d and len(d):
            r = d[stock_col].fillna(0)
            perf = 100 * np.exp(r.cumsum() - r.iloc[0])   # 100 le premier jour de la fenêtre
            line(i, x, perf, f"{ric} ({BASIS_LABEL[eff_basis]})", C["blue"], ".1f")
            if bcode:
                lvl = MARKET[bcode].reindex(pd.DatetimeIndex(x)).ffill()
                first = lvl.dropna()
                if len(first):
                    line(i, x, 100 * lvl / first.iloc[0], blabel, C["green"], ".1f")
            fig.add_hline(y=100, line=dict(color=MUTED, width=1), row=i, col=1)

        elif p == "rendements" and "LogRet" in d:
            line(i, x, d["LogRet"] * 100, "Log-rendement (%)", C["blue"], ".2f", width=1, legend=False)

        elif p == "vol":
            if "VolGK" in d:
                line(i, x, d["VolGK"], f"{ric} · vol Garman-Klass", C["blue"], ".1f")
            if "VolPK" in d:
                line(i, x, d["VolPK"], f"{ric} · vol Parkinson", C["aqua"], ".1f", width=1.2)
            if "VolCC" in d:
                line(i, x, d["VolCC"], f"{ric} · vol close-to-close", C["orange"], ".1f", width=1.2)
            if bcode:
                bv = BENCH_VOL[bcode].reindex(pd.DatetimeIndex(x))
                line(i, x, bv, f"{BENCHMARKS[bench]} · vol réalisée", C["green"], ".1f", width=2)
            if "vix" in options and HAS_VIX and len(x):
                v = MARKET["VIX"].reindex(pd.DatetimeIndex(x)).ffill()
                line(i, x, v, "VIX (vol implicite S&P 500)", C["violet"], ".1f", width=2.2)
            fig.update_yaxes(ticksuffix=" %", rangemode="tozero", row=i, col=1)

        elif p == "volume" and "Volume" in d:
            fig.add_trace(go.Bar(x=x, y=d["Volume"], name="Volume", showlegend=False,
                                 marker=dict(color=C["blue"], line_width=0),
                                 hovertemplate="Volume : %{y:,.0f}<extra></extra>"), row=i, col=1)

        elif p == "spread" and "RelSpread" in d:
            line(i, x, d["RelSpread"] * 1e4, "Spread relatif (pb)", C["blue"], ".1f", width=1.2, legend=False)

        elif p == "capi" and "MarketCap" in d:
            fig.add_trace(go.Scattergl(
                x=x, y=d["MarketCap"] / 1e9, mode="lines", line=dict(color=C["blue"], width=1.6),
                customdata=d["CapClass"].astype(str), showlegend=False,
                hovertemplate="Capi : %{y:,.2f} Md$ (%{customdata})<extra></extra>"), row=i, col=1)
            fig.update_yaxes(type="log", row=i, col=1)
            caps = d["MarketCap"].dropna() / 1e9
            if len(caps):
                lo, hi = caps.min(), caps.max()
                for thr, label in [(0.25, "Small ≥ 250 M$"), (2, "Mid ≥ 2 Md$"), (10, "Large ≥ 10 Md$")]:
                    if lo / 3 <= thr <= hi * 3:  # seuils proches de la série seulement
                        fig.add_hline(y=thr, line=dict(color=MUTED, width=1, dash="dash"), row=i, col=1)
                        # sur un axe log, Plotly place les annotations en log10(valeur)
                        yref = "y" if i == 1 else f"y{i}"
                        fig.add_annotation(x=0, xref="x domain" if i == 1 else f"x{i} domain",
                                           y=np.log10(thr), yref=yref, text=label, showarrow=False,
                                           xanchor="left", yanchor="bottom",
                                           font=dict(size=10, color=TEXT2))

    fig.update_layout(
        height=150 + sum(weights) * 150, margin=dict(l=60, r=24, t=40, b=30),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font=dict(family=FONT, size=12, color=TEXT2),
        hovermode="x unified", hoverlabel=dict(bgcolor="white", font_size=12, font_color=TEXT),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, font=dict(size=11)),
        bargap=0, uirevision=f"{ric}-{period}",  # garde ton zoom quand tu coches un panneau
    )
    fig.update_annotations(selector=dict(xref="paper"), font=dict(size=12, color=TEXT), xanchor="left", x=0)
    fig.update_xaxes(showgrid=False, linecolor=GRID, hoverformat="%d/%m/%Y", showspikes=True,
                     spikemode="across", spikethickness=1, spikecolor=MUTED, spikedash="solid",
                     rangeslider=dict(visible=False))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    stats = benchmark_stats(d, bcode, eff_basis) if bcode else None
    info = {"bench": bench, "label": blabel, "note": _note, "basis": eff_basis,
            "start": x.iloc[0] if len(x) else None}
    return fig, full, stats, info


def empty_figure(message):
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font=dict(size=15, color=TEXT2))
    fig.update_layout(height=400, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


# ============================================================================
# Fiche de synthèse
# ============================================================================
SECTION = {"fontSize": 12, "fontWeight": 600, "color": TEXT2, "textTransform": "uppercase",
           "letterSpacing": "0.04em", "margin": "16px 0 6px"}


def summary_card(ric, d, stats, info, period):
    def row(k, v, warn=False):
        return html.Tr([html.Td(k, style={"color": TEXT2, "padding": "3px 12px 3px 0"}),
                        html.Td(v, style={"textAlign": "right", "fontVariantNumeric": "tabular-nums",
                                          "color": CRITICAL if warn else TEXT,
                                          "fontWeight": 600 if warn else 400})])

    def table(rows):
        return html.Table(rows, style={"fontSize": 13, "borderCollapse": "collapse", "width": "100%"})

    cls = CLASSES.get(ric, "Inconnue")
    n_anom = int((d["Anomalie"] != "").sum())
    rows = [row("Période", f"{d['Date'].min():%d/%m/%Y} → {d['Date'].max():%d/%m/%Y}"),
            row("Jours de données", f"{len(d):,}".replace(",", " "))]
    if "InUniverse" in d:
        rows.append(row("Dans l'univers", f"{d['InUniverse'].fillna(False).mean():.0%}"))
    if "MarketCap" in d and d["MarketCap"].notna().any():
        rows.append(row("Dernière capi", f"{d['MarketCap'].dropna().iloc[-1] / 1e9:,.2f} Md$"))
    if "VolGK" in d and d["VolGK"].notna().any():
        rows.append(row("Vol GK médiane", f"{d['VolGK'].median():.1f} %"))
    rows.append(row("Anomalies", f"⚠ {n_anom}" if n_anom else "✓ aucune", warn=n_anom > 0))

    blocks = [
        html.Div([html.Span(ric, style={"fontSize": 20, "fontWeight": 700, "color": TEXT}),
                  html.Span(f"{cls} cap" if cls != "Inconnue" else "taille inconnue", style={"marginLeft": 10, "fontSize": 12, "padding": "2px 8px",
                                                 "border": f"1px solid {GRID}", "borderRadius": 12,
                                                 "color": TEXT2, "verticalAlign": "middle"})],
                 style={"marginBottom": 8}),
        table(rows),
    ]

    if "CapClass" in d and d["CapClass"].notna().any():
        share = d["CapClass"].value_counts(normalize=True)
        blocks += [html.Div("Temps passé par classe", style=SECTION),
                   table([row(f"{c} ({CAP_DESC[c]})", f"{share.get(c, 0):.0%}")
                          for c in CAP_LABELS if share.get(c, 0) > 0])]

    if stats:
        bench = info["bench"]
        start = info["start"]
        per = f"depuis le {start:%d/%m/%Y}" if start is not None else ""
        excess = stats["ret_s"] - stats["ret_b"]
        blocks += [html.Div(f"Vs {BENCHMARKS[bench]}, {BASIS_LABEL[info['basis']]}", style=SECTION),
                   html.Div(per, style={"fontSize": 12, "color": TEXT2, "marginBottom": 4}),
                   table([row("Bêta", f"{stats['beta']:.2f}"),
                          row("Corrélation", f"{stats['corr']:.2f}"),
                          row("Rendement annualisé", f"{stats['ret_s']:+.1%}"),
                          row(f"{BENCHMARKS[bench]} annualisé", f"{stats['ret_b']:+.1%}"),
                          row("Surperformance", f"{excess:+.1%}"),
                          row("Vol action / indice", f"{stats['vol_s']:.0%} / {stats['vol_b']:.0%}")]),
                   html.Div(info["note"],
                            style={"fontSize": 11, "color": TEXT2, "marginTop": 6, "lineHeight": 1.35})]

    miss_cols = [c for c in ["Close", "Open", "High", "Low", "Volume", "Bid", "Ask",
                             "TotalReturnPct", "MarketCap"] if c in d]
    blocks += [html.Div("Valeurs manquantes", style=SECTION),
               table([row(c, f"{d[c].isna().mean():.1%}", warn=d[c].isna().mean() > 0.05) for c in miss_cols])]

    anomalies = d.loc[d["Anomalie"] != "", ["Date", "Anomalie"]].head(8)
    if len(anomalies):
        blocks += [html.Div("Premières anomalies", style=SECTION),
                   html.Ul([html.Li(f"{r.Date:%d/%m/%Y} : {r.Anomalie}", style={"marginBottom": 4})
                            for r in anomalies.itertuples()],
                           style={"fontSize": 12, "color": TEXT, "paddingLeft": 16, "margin": 0})]
    return blocks


# ============================================================================
# Application
# ============================================================================
BUTTON = {"border": f"1px solid {GRID}", "background": "white", "borderRadius": 6, "padding": "6px 12px",
          "cursor": "pointer", "fontSize": 13, "color": TEXT, "fontFamily": FONT}
CARD = {"background": "white", "border": f"1px solid {GRID}", "borderRadius": 10, "padding": 16}
CHOICE = dict(inline=True, inputStyle={"marginRight": 4},
              labelStyle={"marginRight": 12, "fontSize": 13, "color": TEXT})
ROW = {"display": "flex", "gap": 14, "alignItems": "center", "flexWrap": "wrap", "marginBottom": 10}


def group(label, *children):
    return html.Div([html.Span(label, style={"fontSize": 12, "color": TEXT2, "marginRight": 8,
                                             "fontWeight": 600})] + list(children),
                    style={"display": "flex", "alignItems": "center"})


def sep():
    return html.Div(style={"width": 1, "height": 24, "background": GRID})


MIN_DATE = DF["Date"].min().date()
MAX_DATE = DF["Date"].max().date()
DEFAULT_SINCE = max(MIN_DATE, pd.Timestamp("2015-01-01").date())

app = Dash(__name__, title="Panel GARCH-LSTM")
app.layout = html.Div(style={"fontFamily": FONT, "background": SURFACE, "minHeight": "100vh",
                             "padding": "16px 20px", "color": TEXT}, children=[
    html.Div([
        html.Span("Panel GARCH-LSTM", style={"fontSize": 18, "fontWeight": 700}),
        html.Span(f"  ·  {len(RICS):,} actions".replace(",", " "), style={"color": TEXT2, "fontSize": 14}),
        html.Span(f"  ·  visualiseur {VERSION}", style={"color": MUTED, "fontSize": 12}),
    ], style={"marginBottom": 12}),
    *([html.Div([html.Div("⚠ " + m) for m in DIAG],
                style={"background": "#fff4e5", "border": "1px solid #f0c98a", "borderRadius": 8,
                       "padding": "8px 12px", "fontSize": 13, "color": TEXT, "marginBottom": 12})]
      if DIAG else []),

    # ligne 1 : filtre de capi + choix de l'action
    html.Div(style=ROW, children=[
        group("Taille", dcc.Checklist(
            id="caps", value=CLASS_ORDER, **CHOICE,
            options=[{"label": f"{c} ({CLASS_COUNTS[c]:,})".replace(",", " "), "value": c,
                      "disabled": not HAS_CAP}
                     for c in CLASS_ORDER])),
        sep(),
        dcc.Dropdown(id="ric", options=[{"label": f"{r}  ·  {CLASSES.get(r, 'Inconnue')}", "value": r}
                                        for r in RICS], value=DEFAULT_RIC, clearable=False, searchable=True,
                     placeholder="Rechercher un RIC…", style={"width": 280, "fontSize": 14}),
        html.Button("◀ Précédent", id="prev", n_clicks=0, style=BUTTON),
        html.Button("Suivant ▶", id="next", n_clicks=0, style=BUTTON),
        html.Button("Au hasard", id="rand", n_clicks=0, style=BUTTON),
        html.Span(id="count", style={"fontSize": 13, "color": TEXT2}),
    ]),

    # ligne 2 : panneaux affichés
    html.Div(style=ROW, children=[
        group("Graphiques", dcc.Checklist(id="panels", value=list(PANELS), **CHOICE,
                                          options=[{"label": v, "value": k} for k, v in PANELS.items()])),
    ]),

    # ligne 3 : période, benchmark, options
    html.Div(style=ROW, children=[
        group("Période", dcc.RadioItems(id="period", value="tout", **CHOICE, options=[
            {"label": "1 an", "value": "1"}, {"label": "5 ans", "value": "5"},
            {"label": "10 ans", "value": "10"}, {"label": "Tout", "value": "tout"},
            {"label": "Depuis le", "value": "depuis"}]),
              dcc.DatePickerSingle(id="since", date=DEFAULT_SINCE, display_format="DD/MM/YYYY",
                                   first_day_of_week=1, min_date_allowed=MIN_DATE, max_date_allowed=MAX_DATE,
                                   initial_visible_month=DEFAULT_SINCE, placeholder="jj/mm/aaaa",
                                   style={"fontSize": 13})),
        sep(),
        group("Comparer à", dcc.RadioItems(
            id="bench", value=AVAILABLE_BENCH[0] if AVAILABLE_BENCH else "aucun", **CHOICE,
            options=[{"label": "Aucun", "value": "aucun"}]
                    + [{"label": BENCHMARKS[c], "value": c} for c in AVAILABLE_BENCH])),
        sep(),
        group("Base", dcc.RadioItems(id="basis", value="tr", **CHOICE, options=[
            {"label": "Total return", "value": "tr"}, {"label": "Prix seul", "value": "prix"}])),
        sep(),
        group("Options", dcc.Checklist(id="options", value=["log"], **CHOICE, options=[
            {"label": "Échelle log", "value": "log"},
            {"label": "Chandeliers", "value": "chandeliers"},
            {"label": "VIX sur la volatilité" if HAS_VIX else "VIX (absent des données)",
             "value": "vix", "disabled": not HAS_VIX}])),
    ]),

    html.Div(style={"display": "flex", "gap": 16, "alignItems": "flex-start"}, children=[
        html.Div(dcc.Loading(dcc.Graph(id="chart", config={"displaylogo": False, "scrollZoom": True}),
                             type="dot", color=C["blue"]),
                 style={**CARD, "flex": 1, "minWidth": 0, "padding": 8}),
        html.Div(id="summary", style={**CARD, "width": 290, "flexShrink": 0}),
    ]),
])


@app.callback(Output("ric", "options"), Output("ric", "value"), Output("count", "children"),
              Input("caps", "value"), Input("prev", "n_clicks"), Input("next", "n_clicks"),
              Input("rand", "n_clicks"), State("ric", "value"))
def navigate(caps, _p, _n, _r, current):
    caps = set(caps or [])
    rics = [r for r in RICS if CLASSES.get(r, "Inconnue") in caps]
    options = [{"label": f"{r}  ·  {CLASSES.get(r, 'Inconnue')}", "value": r} for r in rics]
    count = f"{len(rics):,} action(s) dans le filtre".replace(",", " ")
    if not rics:
        return [], None, count
    i = rics.index(current) if current in rics else None
    trig = ctx.triggered_id
    if trig == "prev":
        value = rics[((i if i is not None else 0) - 1) % len(rics)]
    elif trig == "next":
        value = rics[((i if i is not None else -1) + 1) % len(rics)]
    elif trig == "rand":
        value = rics[np.random.randint(len(rics))]
    else:  # changement de filtre ou démarrage : on garde l'action si elle est encore dans le filtre
        value = current if i is not None else (DEFAULT_RIC if DEFAULT_RIC in rics else rics[0])
    return options, value, count


@app.callback(Output("period", "value"), Input("since", "date"), prevent_initial_call=True)
def pick_date(_date):
    return "depuis"   # choisir une date bascule sur "Depuis le"


@app.callback(Output("chart", "figure"), Output("summary", "children"),
              Input("ric", "value"), Input("panels", "value"), Input("options", "value"),
              Input("period", "value"), Input("bench", "value"), Input("basis", "value"),
              Input("since", "date"))
def update(ric, panels, options, period="tout", bench="aucun", basis="tr", since=None):
    if ric not in SLICES:
        return empty_figure("Aucune action ne correspond au filtre de taille."), []
    period = period or "tout"
    fig, d, stats, info = build_figure(ric, panels or [], options or [], period, bench or "aucun",
                                       basis or "tr", since)
    return fig, summary_card(ric, d, stats, info, period)


def free_port(start):
    for port in range(start, start + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("Aucun port libre entre 8050 et 8070.")


if __name__ == "__main__":
    port = free_port(PORT)
    if port != PORT:
        print(f"  ! Le port {PORT} est déjà pris (un ancien visualiseur tourne encore ?). "
              f"Utilisation du port {port}. Pense à fermer l'ancien terminal.")
    url = f"http://127.0.0.1:{port}"
    print(f"Ouvre {url} dans ton navigateur (Ctrl+C pour arrêter).")
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    app.run(debug=False, port=port)