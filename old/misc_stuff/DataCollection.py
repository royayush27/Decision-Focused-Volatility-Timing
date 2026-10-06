"""
Extraction LSEG optimisée pour un univers large (~3 300 actions US) -> panel GARCH-LSTM.

- Prix (OHLC, volume, bid, ask) : ld.get_history, 1 appel par action pour 20 ans.
- Fondamentaux (total return, vol implicite, market cap) : 1 seul get_data groupé
  par lot d'actions et par tranche de 2 ans.
- 4 requêtes en parallèle, limiteur de débit, retries + découpage adaptatif.
- Checkpoint par morceau : on peut arrêter / relancer à tout moment.
- Arrêt propre si le quota journalier semble atteint -> relancer le lendemain.
"""
import hashlib
import os
import re
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import lseg.data as ld

warnings.simplefilter(action="ignore", category=FutureWarning)

# ============================================================================
# Paramètres
# ============================================================================
EXCEL_PATH = r"C:\Users\Royal\Downloads\GridExport_September_27_2026_10_14_17.xlsx"
BASE_DIR = r"C:\Users\Royal\OneDrive - De Vinci\Financial Engineering - Hanyang"
OUT_DIR = os.path.join(BASE_DIR, "datasets")
# Checkpoints hors OneDrive : des milliers de petits fichiers réécrits en continu
# ralentissent la synchro et OneDrive peut les verrouiller pendant l'écriture.
TEMP_DIR = os.path.join(os.path.expanduser("~"), "lseg_temp_batches")
START_DATE = "2006-01-01"
END_DATE = "2026-09-27"
MIN_MARKET_CAP = 250e6  # filtre de ton screen, réappliqué date par date dans le panel

# Données de pricing historique (get_history) -> nom de colonne dans le panel
PRICING_FIELDS = {
    "OPEN_PRC": "Open",
    "HIGH_1": "High",
    "LOW_1": "Low",
    "TRDPRC_1": "Close",
    "ACVOL_UNS": "Volume",
    "BID": "Bid",
    "ASK": "Ask",
}

# Champs fondamentaux (get_data, demandés ensemble)
FUND_FIELDS = {
    "TR.TotalReturn": "TotalReturnPct",
    "TR.CompanyMarketCapitalization": "MarketCap",
}

# Séries de marché US (clé = RIC canonique utilisé dans le panel)
MARKET_RICS = {
    ".SPX": "SPX",  # S&P 500
    ".VIX": "VIX",  # vol implicite S&P 500
    ".RUT": "RUT",  # Russell 2000 (small caps, proche de ton univers)
    ".SPXTR": "SPXTR",  # S&P 500 total return (dividendes réinvestis)
    ".RUTTR": "RUTTR",  # Russell 2000 total return
}
MARKET_FIELDS = {"TRDPRC_1": "Close"}

# Sources essayées dans l'ordre pour chaque série. Les indices S&P et CBOE (VIX) demandent
# des droits LSEG spécifiques, souvent absents des accès universitaires : on bascule alors
# sur l'ETF qui réplique l'indice (même dynamique, prix d'ETF) ou sur FRED pour le VIX.
#   ("history", ric)  -> ld.get_history, champ TRDPRC_1
#   ("datagrid", ric) -> ld.get_data, champ TR.PriceClose
#   ("fred", série)   -> téléchargement public FRED (hors LSEG)
MARKET_SOURCES = {
    ".SPX": [("history", ".SPX"), ("datagrid", ".SPX"), ("history", "SPY.P"), ("datagrid", "SPY.P")],
    ".VIX": [("history", ".VIX"), ("datagrid", ".VIX"), ("fred", "VIXCLS")],
    ".RUT": [("history", ".RUT"), ("datagrid", ".RUT"), ("history", "IWM.P"), ("datagrid", "IWM.P")],
    # Total return : l'indice TR, sinon l'ETF avec TR.TotalReturn (dividendes réinvestis, frais de l'ETF inclus)
    ".SPXTR": [("history", ".SPXTR"), ("datagrid", ".SPXTR"), ("tr_etf", "SPY.P")],
    ".RUTTR": [("history", ".RUTTR"), ("datagrid", ".RUTTR"), ("tr_etf", "IWM.P")],
}

# Biais de survivance : ton screen ne contient que des sociétés actives aujourd'hui.
# ".RUA" (Russell 3000) ajoute les constituants historiques, délistés compris.
# Attention : ça fait grossir l'univers (et la durée) d'environ 2x.
HISTORICAL_INDEX = None  # ex. ".RUA"

# Nettoyage de l'univers (appliqué à la construction du panel : rien n'est retéléchargé).
# La liste des exclusions et leur raison est écrite dans datasets/universe.csv.
EXCLUDE_OTC = True                       # marchés OTC : cotations peu liquides, capis peu fiables
OTC_SUFFIXES = (".PK", ".PQ", ".OB")
EXCLUDE_SPAC = True                      # SPAC : prix quasi figé sur la valeur du trust -> vol ~ 0
SPAC_NAME_PATTERN = (r"\bAcquisition (?:Corp|Corporation|Co|Company|Holdings|Ltd)\b|\bBlank Check|\bSPAC\b"
                     r"|\b(?:Corp|Corporation|Inc)\.? (?:I|II|III|IV|V|VI|VII|VIII|IX|X)\b")

# Capitalisation de repli pour les actions où TR.CompanyMarketCapitalization est vide
# (introductions récentes, entités restructurées) : même champ que le screener LSEG.
CAP_FALLBACK_FIELD = "TR.CompanyMarketCap"

# Performance / robustesse
MAX_WORKERS = 4                # requêtes simultanées
MAX_REQUESTS_PER_SEC = 4       # sous la limite API (~5/s)
FUND_BATCH_SIZE = 25           # actions par requête fondamentaux
FUND_YEARS_PER_CHUNK = 2       # années par requête fondamentaux
HISTORY_ROW_LIMIT = 10_000     # plafond de lignes d'un get_history
REQUEST_TIMEOUT = 180
MAX_RETRIES = 3
BASE_BACKOFF = 5
MIN_PERIOD_DAYS = 60
MAX_CONSECUTIVE_FAILURES = 25  # au-delà : quota probablement atteint -> arrêt propre

NO_DATA_MARKERS = ("no data", "not found", "unable to resolve", "does not exist", "invalid ric")
SESSION_MARKERS = ("session", "unauthorized", "401")
THROTTLE_MARKERS = ("429", "too many", "quota", "limit exceeded", "rate limit")


# ============================================================================
# Session, débit et garde-fou quota (partagés entre threads)
# ============================================================================
class Stopped(Exception):
    pass


_session_lock = threading.Lock()
_last_reopen = [0.0]
STOP = threading.Event()
_fail_lock = threading.Lock()
_consecutive_failures = [0]


def open_session():
    ld.get_config().set_param("http.request-timeout", REQUEST_TIMEOUT)
    ld.open_session()


def reopen_session():
    with _session_lock:
        if time.time() - _last_reopen[0] < 60:  # un autre thread vient de le faire
            return
        try:
            ld.close_session()
        except Exception:
            pass
        time.sleep(5)
        open_session()
        _last_reopen[0] = time.time()


class RateLimiter:
    def __init__(self, rps):
        self.interval = 1.0 / rps
        self.lock = threading.Lock()
        self.next_slot = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            slot = max(now, self.next_slot)
            self.next_slot = slot + self.interval
        time.sleep(max(0.0, slot - now))


LIMITER = RateLimiter(MAX_REQUESTS_PER_SEC)


def record(success):
    with _fail_lock:
        if success:
            _consecutive_failures[0] = 0
        else:
            _consecutive_failures[0] += 1
            if _consecutive_failures[0] >= MAX_CONSECUTIVE_FAILURES and not STOP.is_set():
                print(f"\n!!! {MAX_CONSECUTIVE_FAILURES} échecs d'affilée : quota probablement atteint. "
                      f"Arrêt propre, relance le script plus tard (il reprendra où il en est).\n")
                STOP.set()


def call_with_retry(fn, desc):
    """Appelle fn() avec limiteur de débit et retries. Renvoie None si 'pas de données'."""
    last_err = None
    for attempt in range(MAX_RETRIES):
        if STOP.is_set():
            raise Stopped()
        LIMITER.wait()
        try:
            res = fn()
            record(True)
            return res
        except Exception as err:
            msg = str(err).lower()
            if any(m in msg for m in NO_DATA_MARKERS):
                record(True)
                return None
            last_err = err
            print(f"    ! {desc} : tentative {attempt + 1}/{MAX_RETRIES} échouée : {err}")
            if any(m in msg for m in SESSION_MARKERS):
                reopen_session()
            wait = BASE_BACKOFF * 2 ** attempt
            if any(m in msg for m in THROTTLE_MARKERS):
                wait *= 6
            time.sleep(wait)
    record(False)
    raise last_err


# ============================================================================
# Utilitaires
# ============================================================================
def periods(start, end, years):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    out, y = [], start.year
    while y <= end.year:
        s = max(start, pd.Timestamp(year=y, month=1, day=1))
        e = min(end, pd.Timestamp(year=y + years - 1, month=12, day=31))
        out.append((s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")))
        y += years
    return out


def fname(text):
    """Nom de fichier sûr (les RICs délistés contiennent '^', etc.)."""
    short = re.sub(r"[^A-Za-z0-9]+", "_", text)[:40]
    return f"{short}_{hashlib.md5(text.encode()).hexdigest()[:8]}"


def to_naive_dates(s):
    s = pd.to_datetime(s, errors="coerce")
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_convert(None)
    return s.dt.normalize()


def save(df, path):
    tmp = path + ".tmp"
    df.to_parquet(tmp)
    os.replace(tmp, path)  # écriture atomique : pas de checkpoint corrompu si on coupe


def write_output(write, path, retries=5, wait=3):
    """Écrit un fichier de sortie (dossier OneDrive). Si le fichier est verrouillé
    (ouvert dans Excel, synchro OneDrive en cours), réessaie puis écrit sous un autre nom."""
    for attempt in range(retries):
        try:
            write(path)
            return path
        except PermissionError:
            if attempt == 0:
                print(f"  ! {os.path.basename(path)} est verrouillé (ouvert dans Excel ? synchro OneDrive ?), "
                      f"nouvel essai dans {wait} s…")
            time.sleep(wait)
    root, ext = os.path.splitext(path)
    alt = f"{root}_{time.strftime('%Y%m%d_%H%M%S')}{ext}"
    write(alt)
    print(f"  ! {os.path.basename(path)} toujours verrouillé : écrit sous {os.path.basename(alt)}. "
          f"Ferme le fichier puis relance pour retrouver le nom normal.")
    return alt


# ============================================================================
# Prix : get_history (1 appel par action)
# ============================================================================
def normalize_history(df, ric, mapping):
    cols = ["Instrument", "Date"] + list(mapping.values())
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=cols)
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    df.columns = [str(c) for c in df.columns]
    df.index.name = "Date"
    df = df.reset_index().rename(columns=mapping)
    df["Date"] = to_naive_dates(df["Date"])
    df["Instrument"] = ric
    for c in mapping.values():
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df else np.nan
    return df[cols].dropna(subset=["Date"])


def fetch_history(ric, mapping):
    """20 ans de quotidien ; si le plafond de lignes est atteint, on remonte par morceaux."""
    frames, end = [], END_DATE
    while True:
        raw = call_with_retry(
            lambda e=end: ld.get_history(
                universe=ric, fields=list(mapping), interval="daily", start=START_DATE, end=e
            ),
            f"prix {ric}",
        )
        df = normalize_history(raw, ric, mapping)
        if df.empty:
            break
        frames.append(df)
        first = df["Date"].min()
        near_start = first <= pd.Timestamp(START_DATE) + pd.Timedelta(days=10)
        if len(df) < HISTORY_ROW_LIMIT * 0.98 or near_start:
            break
        end = (first - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    if not frames:
        return normalize_history(None, ric, mapping)
    return pd.concat(frames, ignore_index=True).drop_duplicates(["Instrument", "Date"])


def pricing_task(ric, mapping, folder):
    path = os.path.join(folder, f"{fname(ric)}.parquet")

    def run():
        try:
            save(fetch_history(ric, mapping), path)
            return []
        except Stopped:
            return []
        except Exception as err:
            return [{"type": "prix", "ric": ric, "start": START_DATE, "end": END_DATE, "error": str(err)}]

    return path, run


# ============================================================================
# Fondamentaux : get_data groupé
# ============================================================================
def reshape_fund(raw):
    """get_data renvoie Instrument, date1, val1, date2, val2... -> format long fusionné."""
    cols = ["Instrument", "Date"] + list(FUND_FIELDS.values())
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=cols)
    expected = 1 + 2 * len(FUND_FIELDS)
    if raw.shape[1] != expected:
        raise ValueError(f"{raw.shape[1]} colonnes reçues au lieu de {expected} : {list(raw.columns)}")
    merged = None
    for i, col in enumerate(FUND_FIELDS.values()):
        part = pd.DataFrame({
            "Instrument": raw.iloc[:, 0].astype(str),
            "Date": to_naive_dates(raw.iloc[:, 1 + 2 * i]),
            col: pd.to_numeric(raw.iloc[:, 2 + 2 * i], errors="coerce"),
        }).dropna(subset=["Date"])
        part = part.groupby(["Instrument", "Date"], as_index=False)[col].last()
        merged = part if merged is None else merged.merge(part, on=["Instrument", "Date"], how="outer")
    return merged[cols]


def fetch_fund(rics, s, e):
    fields = []
    for f in FUND_FIELDS:
        fields += [f"{f}.date", f]
    raw = call_with_retry(
        lambda: ld.get_data(universe=rics, fields=fields, parameters={"SDate": s, "EDate": e, "Frq": "D"}),
        f"fondamentaux {len(rics)} actions {s}->{e}",
    )
    return reshape_fund(raw)


def fetch_fund_adaptive(rics, s, e, failures):
    """En cas d'échec : coupe le lot en deux, puis la période en deux."""
    try:
        return fetch_fund(rics, s, e)
    except Stopped:
        raise
    except Exception as err:
        if len(rics) > 1:
            mid = len(rics) // 2
            parts = [fetch_fund_adaptive(rics[:mid], s, e, failures),
                     fetch_fund_adaptive(rics[mid:], s, e, failures)]
        else:
            sd, ed = pd.Timestamp(s), pd.Timestamp(e)
            if (ed - sd).days <= MIN_PERIOD_DAYS:
                failures.append({"type": "fondamentaux", "ric": rics[0], "start": s, "end": e, "error": str(err)})
                return reshape_fund(None)
            mid = sd + (ed - sd) / 2
            parts = [fetch_fund_adaptive(rics, s, mid.strftime("%Y-%m-%d"), failures),
                     fetch_fund_adaptive(rics, (mid + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), e, failures)]
        parts = [p for p in parts if not p.empty]
        return pd.concat(parts, ignore_index=True) if parts else reshape_fund(None)


def fund_task(batch, s, e, folder):
    # la liste des champs fait partie de la clé : si on change FUND_FIELDS,
    # les anciens checkpoints ne sont pas réutilisés par erreur
    key = hashlib.md5((",".join(batch) + "|" + ",".join(FUND_FIELDS)).encode()).hexdigest()[:10]
    path = os.path.join(folder, f"{key}__{s}__{e}.parquet")

    def run():
        failures = []
        try:
            df = fetch_fund_adaptive(batch, s, e, failures)
        except Stopped:
            return []
        if not failures:  # sinon on ne sauvegarde pas : retenté au prochain lancement
            save(df, path)
        return failures

    return path, run


# ============================================================================
# Exécution parallèle avec progression
# ============================================================================
def run_parallel(name, tasks):
    todo = [(p, fn) for p, fn in tasks if not os.path.exists(p)]
    print(f"\n{'=' * 60}\n{name} : {len(tasks) - len(todo)}/{len(tasks)} déjà faits, {len(todo)} à faire\n{'=' * 60}")
    failures, done, t0 = [], 0, time.time()
    if not todo or STOP.is_set():
        return failures
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(fn) for _, fn in todo]
        for fut in as_completed(futures):
            done += 1
            try:
                failures.extend(fut.result())
            except Exception as err:
                failures.append({"type": name, "error": str(err)})
            if done % 25 == 0 or done == len(todo):
                elapsed = time.time() - t0
                eta = elapsed / done * (len(todo) - done)
                print(f"  {name} : {done}/{len(todo)} | écoulé {elapsed / 60:.1f} min | "
                      f"reste ~{eta / 60:.0f} min | échecs {len(failures)}")
    return failures


def consolidate(folder, out_path):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".parquet")]
    dfs = [pd.read_parquet(f) for f in files]
    dfs = [d for d in dfs if not d.empty]
    if not dfs:
        print(f">>> Rien dans {folder}")
        return None
    df = pd.concat(dfs, ignore_index=True).drop_duplicates(["Instrument", "Date"], keep="last")
    num = df.select_dtypes("number").columns
    df[num] = df[num].astype("float32")  # divise la mémoire par 2 (~17 M lignes)
    write_output(df.to_parquet, out_path)
    print(f">>> {out_path} : {len(df):,} lignes, {df['Instrument'].nunique()} instruments")
    return df


# ============================================================================
# Séries de marché (avec sources de repli)
# ============================================================================
def market_from_history(ric):
    return fetch_history(ric, MARKET_FIELDS)


def market_from_datagrid(ric):
    raw = ld.get_data(universe=[ric], fields=["TR.PriceClose.date", "TR.PriceClose"],
                      parameters={"SDate": START_DATE, "EDate": END_DATE, "Frq": "D"})
    if raw is None or len(raw) == 0 or raw.shape[1] < 3:
        return pd.DataFrame(columns=["Instrument", "Date", "Close"])
    return pd.DataFrame({"Instrument": ric, "Date": to_naive_dates(raw.iloc[:, 1]),
                         "Close": pd.to_numeric(raw.iloc[:, 2], errors="coerce")}).dropna()


def market_from_tr_etf(ric):
    """Indice de total return reconstruit à partir des rendements quotidiens TR.TotalReturn d'un ETF."""
    raw = ld.get_data(universe=[ric], fields=["TR.TotalReturn.date", "TR.TotalReturn"],
                      parameters={"SDate": START_DATE, "EDate": END_DATE, "Frq": "D"})
    if raw is None or len(raw) == 0 or raw.shape[1] < 3:
        return pd.DataFrame(columns=["Instrument", "Date", "Close"])
    df = pd.DataFrame({"Date": to_naive_dates(raw.iloc[:, 1]),
                       "r": pd.to_numeric(raw.iloc[:, 2], errors="coerce")}).dropna().sort_values("Date")
    df = df.drop_duplicates("Date")
    df["Close"] = 100 * (1 + df["r"] / 100).cumprod()
    df["Instrument"] = ric
    return df[["Instrument", "Date", "Close"]]


def market_from_fred(series):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
    raw = pd.read_csv(url)
    raw.columns = ["Date", "Close"]
    df = pd.DataFrame({"Instrument": series, "Date": pd.to_datetime(raw["Date"], errors="coerce"),
                       "Close": pd.to_numeric(raw["Close"], errors="coerce")}).dropna()
    return df[(df["Date"] >= START_DATE) & (df["Date"] <= END_DATE)]


def fetch_market_series(canonical, folder):
    """Essaie chaque source jusqu'à obtenir des données. Ne sauvegarde jamais un résultat vide."""
    path = os.path.join(folder, f"{fname(canonical)}.parquet")
    if os.path.exists(path):
        src = pd.read_parquet(path)["Source"].iloc[0]
        print(f"  {canonical} : déjà téléchargé (source {src})")
        return None
    tried = []
    for kind, code in MARKET_SOURCES[canonical]:
        try:
            if kind == "history":
                df = market_from_history(code)
            elif kind == "datagrid":
                df = call_with_retry(lambda c=code: market_from_datagrid(c), f"marché {code}")
            elif kind == "tr_etf":
                df = call_with_retry(lambda c=code: market_from_tr_etf(c), f"marché TR {code}")
            else:
                df = market_from_fred(code)
            n = 0 if df is None else int(df["Close"].notna().sum())
        except Stopped:
            raise
        except Exception as err:
            tried.append(f"{kind}:{code} -> erreur ({str(err)[:80]})")
            continue
        if n < 250:
            tried.append(f"{kind}:{code} -> {n} valeurs")
            continue
        df = df.copy()
        df["Source"] = f"{kind}:{code}"
        df["Instrument"] = canonical          # nom canonique pour le panel
        save(df[["Instrument", "Date", "Close", "Source"]], path)
        note = "" if code == canonical else "  (repli : l'indice lui-même n'est pas accessible)"
        print(f"  {canonical} : {n:,} jours via {kind}:{code}{note}")
        return None
    print(f"  ! {canonical} introuvable. Essais : " + " | ".join(tried))
    return {"type": "marché", "ric": canonical, "start": START_DATE, "end": END_DATE, "error": " | ".join(tried)}


# ============================================================================
# Univers
# ============================================================================
def historical_constituents(index_ric):
    rics = set()
    for s, _ in periods(START_DATE, END_DATE, 1):
        try:
            df = call_with_retry(
                lambda d=s: ld.get_data(universe=[index_ric], fields=["TR.IndexConstituentRIC"],
                                        parameters={"SDate": d}),
                f"constituants {index_ric} {s}",
            )
            if df is None or df.empty:
                continue
            col = [c for c in df.columns if c != "Instrument"][0]
            found = df[col].dropna().astype(str).str.strip()
            rics.update(r for r in found if r)
            print(f"  {index_ric} au {s} : {len(found)} constituants")
        except Exception as err:
            print(f"  ! constituants {index_ric} au {s} indisponibles : {err}")
    return rics


def build_universe():
    df_excel = pd.read_excel(EXCEL_PATH)
    rics = {r.strip() for r in df_excel.iloc[:, 0].dropna().astype(str) if r.strip()}
    print(f"Screen Excel : {len(rics)} RICs")
    if HISTORICAL_INDEX:
        hist = historical_constituents(HISTORICAL_INDEX)
        print(f"Constituants historiques ajoutés : {len(hist - rics)}")
        rics |= hist
    rics = sorted(rics)  # ordre stable -> checkpoints réutilisables
    write_output(lambda p: pd.Series(rics, name="RIC").to_csv(p, index=False),
                 os.path.join(OUT_DIR, "universe.csv"))
    return rics


def universe_info(rics, folder):
    """Nom et secteur TRBC de chaque RIC (une requête par 500), puis règles d'exclusion."""
    path = os.path.join(folder, "infos_univers.parquet")
    if os.path.exists(path):
        info = pd.read_parquet(path)
    else:
        parts = []
        for i in range(0, len(rics), 500):
            batch = rics[i:i + 500]
            try:
                raw = call_with_retry(lambda b=batch: ld.get_data(universe=b, fields=["TR.CommonName",
                                                                                     "TR.TRBCIndustry"]),
                                      f"infos univers {i}-{i + len(batch)}")
            except Stopped:
                raise
            except Exception as err:
                print(f"  ! noms/secteurs indisponibles pour {len(batch)} RICs : {err}")
                raw = None
            if raw is not None and len(raw) and raw.shape[1] >= 2:
                # le secteur peut manquer si LSEG ne reconnaît pas le champ : on garde le nom
                parts.append(pd.DataFrame({
                    "RIC": raw.iloc[:, 0].astype(str),
                    "Nom": raw.iloc[:, 1].fillna("").astype(str),
                    "Secteur": raw.iloc[:, 2].fillna("").astype(str) if raw.shape[1] >= 3 else "",
                }))
        info = pd.concat(parts, ignore_index=True).drop_duplicates("RIC") if parts else \
            pd.DataFrame(columns=["RIC", "Nom", "Secteur"])
        if len(info) == len(rics):
            save(info, path)
    info = pd.DataFrame({"RIC": rics}).merge(info, on="RIC", how="left").fillna({"Nom": "", "Secteur": ""})

    reasons = pd.Series("", index=info.index)
    if EXCLUDE_OTC:
        reasons[info["RIC"].str.upper().str.endswith(OTC_SUFFIXES)] = "OTC"
    if EXCLUDE_SPAC:
        spac = info["Nom"].str.contains(SPAC_NAME_PATTERN, case=False, regex=True) | \
            info["Secteur"].str.contains("Blank Check|Special Purpose", case=False, regex=True)
        reasons[spac & (reasons == "")] = "SPAC"
    info["Exclu"] = reasons
    n_otc, n_spac = (reasons == "OTC").sum(), (reasons == "SPAC").sum()
    print(f"Univers : {len(info)} RICs, dont {n_otc} OTC et {n_spac} SPAC exclus du panel "
          f"(liste dans universe.csv).")
    write_output(lambda p: info.to_csv(p, index=False, encoding="utf-8-sig"),
                 os.path.join(OUT_DIR, "universe.csv"))
    return set(info.loc[info["Exclu"] != "", "RIC"])


def missing_cap_rics(fund_folder, rics):
    """RICs sans aucune capitalisation dans les checkpoints de fondamentaux."""
    have = set()
    for f in os.listdir(fund_folder):
        if f.endswith(".parquet"):
            d = pd.read_parquet(os.path.join(fund_folder, f), columns=["Instrument", "MarketCap"])
            have.update(d.loc[d["MarketCap"].notna(), "Instrument"].unique())
    return [r for r in rics if r not in have]


def cap_fallback_task(ric, folder):
    path = os.path.join(folder, f"{fname(ric)}.parquet")

    def run():
        try:
            raw = call_with_retry(
                lambda: ld.get_data(universe=[ric],
                                    fields=[f"{CAP_FALLBACK_FIELD}.date", CAP_FALLBACK_FIELD],
                                    parameters={"SDate": START_DATE, "EDate": END_DATE, "Frq": "D"}),
                f"capi de repli {ric}")
        except Stopped:
            return []
        except Exception as err:
            return [{"type": "capi repli", "ric": ric, "start": START_DATE, "end": END_DATE, "error": str(err)}]
        df = pd.DataFrame(columns=["Instrument", "Date", "MarketCap"])
        if raw is not None and len(raw) and raw.shape[1] >= 3:
            df = pd.DataFrame({"Instrument": ric, "Date": to_naive_dates(raw.iloc[:, 1]),
                               "MarketCap": pd.to_numeric(raw.iloc[:, 2], errors="coerce")}).dropna()
        save(df, path)  # même vide : on ne redemande pas à chaque lancement
        return []

    return path, run


def apply_cap_fallback(fund, folder):
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".parquet")]
    rep = [d for d in (pd.read_parquet(f) for f in files) if not d.empty]
    if fund is None or not rep:
        return fund
    rep = pd.concat(rep, ignore_index=True).rename(columns={"MarketCap": "MarketCapRepli"})
    rep["MarketCapRepli"] = rep["MarketCapRepli"].astype("float32")
    fund = fund.merge(rep, on=["Instrument", "Date"], how="outer")
    fund["MarketCap"] = fund["MarketCap"].fillna(fund["MarketCapRepli"])
    print(f"Capi de repli ({CAP_FALLBACK_FIELD}) utilisée pour {rep['Instrument'].nunique()} action(s).")
    return fund.drop(columns="MarketCapRepli")


# ============================================================================
# Panel GARCH-LSTM
# ============================================================================
def build_panel(prices, fund, market, excluded=frozenset()):
    print(f"\n{'=' * 60}\nCONSTRUCTION DU PANEL\n{'=' * 60}")
    frames = [d[~d["Instrument"].isin(excluded)] for d in (prices, fund) if d is not None]
    if not frames:
        print(">>> Rien à assembler")
        return
    panel = frames[0]
    for d in frames[1:]:
        panel = panel.merge(d, on=["Instrument", "Date"], how="outer")
    panel = panel.sort_values(["Instrument", "Date"]).reset_index(drop=True)
    cols = set(panel.columns)

    if "TotalReturnPct" in cols:  # en % -> log-rendement décimal
        r = panel["TotalReturnPct"] / 100
        panel["LogRet"] = np.log1p(r.where(r > -1))

    if {"Bid", "Ask"} <= cols:
        ok = (panel["Bid"] > 0) & (panel["Ask"] >= panel["Bid"])
        mid = (panel["Bid"] + panel["Ask"]) / 2
        panel["RelSpread"] = ((panel["Ask"] - panel["Bid"]) / mid).where(ok)

    if {"Open", "High", "Low", "Close"} <= cols:
        ohlc = panel[["Open", "High", "Low", "Close"]]
        valid = (ohlc > 0).all(axis=1) & (panel["High"] >= panel["Low"])
        hl = np.log(panel["High"] / panel["Low"]).where(valid)
        co = np.log(panel["Close"] / panel["Open"]).where(valid)
        panel["VarParkinson"] = hl ** 2 / (4 * np.log(2))
        panel["VarGarmanKlass"] = 0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2

    if "Volume" in cols:
        panel["LogVolume"] = np.log1p(panel["Volume"].clip(lower=0))

    if "MarketCap" in cols:
        panel["LogMarketCap"] = np.log(panel["MarketCap"].where(panel["MarketCap"] > 0))
        # Filtre du screen réappliqué avec l'info de la veille (pas de look-ahead)
        prev_cap = panel.groupby("Instrument")["MarketCap"].transform(lambda s: s.ffill().shift(1))
        panel["InUniverse"] = prev_cap >= MIN_MARKET_CAP
        # Classe de taille à chaque date, avec la capi de la veille (pas de look-ahead)
        panel["CapClass"] = pd.cut(prev_cap, [0, 250e6, 2e9, 10e9, np.inf],
                                   labels=["Micro", "Small", "Mid", "Large"], right=False)

    if market is not None:
        for name in MARKET_RICS.values():
            ric = next(r for r, n in MARKET_RICS.items() if n == name)
            m = market[market["Instrument"] == ric][["Date", "Close"]].sort_values("Date")
            if m.empty:
                continue
            m = m.rename(columns={"Close": name})
            m[f"{name}_LogRet"] = np.log(m[name] / m[name].shift(1))
            panel = panel.merge(m, on="Date", how="left")

    num = panel.select_dtypes("number").columns
    panel[num] = panel[num].astype("float32")
    out = os.path.join(OUT_DIR, "panel_garch_lstm.parquet")
    out = write_output(panel.to_parquet, out)
    print(f">>> {out} : {len(panel):,} lignes, {panel['Instrument'].nunique()} actions")

    cov = panel.drop(columns=["Instrument", "Date"]).notna().mean().sort_values()
    write_output(lambda p: cov.rename("taux_non_manquant").to_csv(p), os.path.join(OUT_DIR, "coverage.csv"))
    print("\nCouverture par colonne (part de valeurs non manquantes) :")
    print(cov.map("{:.1%}".format).to_string())


# ============================================================================
# Main
# ============================================================================
def main():
    # "marche_v2" : les anciens checkpoints de marché (parfois vides) sont ignorés
    dirs = {k: os.path.join(TEMP_DIR, k) for k in ("prix", "fondamentaux", "marche_v2", "capi_repli")}
    dirs["marche"] = dirs.pop("marche_v2")
    for d in [OUT_DIR, *dirs.values()]:
        os.makedirs(d, exist_ok=True)

    t0 = time.time()
    all_failures = []
    excluded = set()
    open_session()
    try:
        rics = build_universe()
        try:
            excluded = universe_info(rics, TEMP_DIR)
        except Stopped:
            raise
        except Exception as err:
            # repli : seules les exclusions OTC (qui ne demandent aucune requête)
            excluded = {r for r in rics if EXCLUDE_OTC and r.upper().endswith(OTC_SUFFIXES)}
            print(f"  ! Noms/secteurs non récupérés ({err}) : seuls les {len(excluded)} RICs OTC sont exclus, "
                  f"la détection des SPAC est sautée pour ce lancement.")

        print(f"\n{'=' * 60}\nMARCHÉ\n{'=' * 60}")
        for canonical in MARKET_RICS:
            fail = fetch_market_series(canonical, dirs["marche"])
            if fail:
                all_failures.append(fail)

        tasks = [pricing_task(r, PRICING_FIELDS, dirs["prix"]) for r in rics]
        all_failures += run_parallel("PRIX", tasks)

        batches = [rics[i:i + FUND_BATCH_SIZE] for i in range(0, len(rics), FUND_BATCH_SIZE)]
        tasks = [fund_task(b, s, e, dirs["fondamentaux"])
                 for b in batches for s, e in periods(START_DATE, END_DATE, FUND_YEARS_PER_CHUNK)]
        all_failures += run_parallel("FONDAMENTAUX", tasks)

        if not STOP.is_set():
            missing = [r for r in missing_cap_rics(dirs["fondamentaux"], rics) if r not in excluded]
            if missing:
                print(f"\n{len(missing)} action(s) sans capitalisation : essai avec {CAP_FALLBACK_FIELD}")
                tasks = [cap_fallback_task(r, dirs["capi_repli"]) for r in missing]
                all_failures += run_parallel("CAPI DE REPLI", tasks)
    finally:
        ld.close_session()
        fail_path = os.path.join(OUT_DIR, "failures.csv")
        if all_failures:
            fail_path = write_output(lambda p: pd.DataFrame(all_failures).to_csv(p, index=False), fail_path)
            print(f"\n{len(all_failures)} échecs listés dans {fail_path} -> relance le script pour les retenter.")
        elif os.path.exists(fail_path):
            try:
                os.remove(fail_path)
            except PermissionError:
                pass
        print(f"Durée de ce lancement : {(time.time() - t0) / 3600:.2f} h")

    if STOP.is_set():
        print("Extraction incomplète (arrêt quota) : panel non construit. Relance le script plus tard.")
        return

    market = consolidate(dirs["marche"], os.path.join(OUT_DIR, "market.parquet"))
    prices = consolidate(dirs["prix"], os.path.join(OUT_DIR, "prices.parquet"))
    fund = consolidate(dirs["fondamentaux"], os.path.join(OUT_DIR, "fundamentals.parquet"))
    fund = apply_cap_fallback(fund, dirs["capi_repli"])
    build_panel(prices, fund, market, excluded)


if __name__ == "__main__":
    main()