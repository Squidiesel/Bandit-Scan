"""Zone Ticket core: zone detection, play building, filters, sizing, trade journal, limits, alerts.

No Streamlit in here, so everything can be tested on its own (see tests/test_core.py).
"""
import base64
import csv
import io
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

NY = ZoneInfo("America/New_York")
SYD = ZoneInfo("Australia/Sydney")


# ================= scanner (zone detection, volume, VWAP, structure) =================
RV_STRONG = 2.0                      # relative volume at a zone touch that counts as strong interest
IDX_BARS, IDX_NET, IDX_BAR = 4, 3.0, 2.0   # SPY/QQQ: net move over 4 bars, or one bar, in ATR


def atr_series(df, n=14):
    pc = df.Close.shift()
    tr = pd.concat([df.High - df.Low, (df.High - pc).abs(), (df.Low - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def completed(d15, minutes=15):
    """Drop the last bar when it is still forming, so its partial volume is not misread."""
    idx = d15.index
    if isinstance(idx, pd.DatetimeIndex) and len(d15) > 1:
        now = pd.Timestamp.now(tz=idx.tz) if idx.tz is not None else pd.Timestamp.now()
        if now - idx[-1] < pd.Timedelta(minutes=minutes):
            return d15.iloc[:-1]
    return d15


def session_vwap(d15):
    """Volume weighted average price of the latest session. None when there is no volume feed."""
    idx = d15.index
    if not isinstance(idx, pd.DatetimeIndex) or float(d15.Volume.sum()) <= 0:
        return None
    day = np.array(idx.date)
    last = d15[day == day[-1]]
    v = last.Volume.values.astype(float)
    if v.sum() <= 0:
        return None
    tp = ((last.High + last.Low + last.Close) / 3).values
    return float((tp * v).sum() / v.sum())


def bar_rvol_at(pv, day, slot):
    """One bar's volume divided by the average for the same time of day on other sessions."""
    if slot not in pv.columns:
        return None
    col = pv[slot]
    cur = col.get(day)
    others = col.drop(day, errors="ignore").dropna()
    if cur is None or pd.isna(cur) or len(others) < 3 or others.mean() <= 0:
        return None
    return float(cur / others.mean())


def intraday_rvol(d15c):
    """Time-of-day adjusted relative volume.
    Returns ({bar timestamp: rvol} for the last 8 bars, cumulative rvol for the session so far).
    Comparing with the same clock time on earlier sessions removes the open and close surges,
    so no fixed time-of-day multiplier is needed."""
    idx = d15c.index
    if not isinstance(idx, pd.DatetimeIndex) or len(d15c) < 30 or float(d15c.Volume.sum()) <= 0:
        return {}, None
    d = pd.DataFrame({"v": d15c.Volume.values.astype(float), "day": idx.date,
                      "slot": idx.hour * 60 + idx.minute})
    pv = d.pivot_table(index="day", columns="slot", values="v", aggfunc="sum")
    if len(pv) < 4:
        return {}, None
    today = pv.index[-1]
    bars = {ts: bar_rvol_at(pv, ts.date(), int(ts.hour * 60 + ts.minute)) for ts in idx[-8:]}
    last_slot = int(idx[-1].hour * 60 + idx[-1].minute)
    cum = pv[[c for c in pv.columns if c <= last_slot]].sum(axis=1, min_count=1)
    others = cum.drop(today).dropna()
    cum_rv = None
    if len(others) >= 3 and others.mean() > 0 and not pd.isna(cum.get(today)):
        cum_rv = float(cum[today] / others.mean())
    return bars, cum_rv


def structure(d1h, n=2):
    """1 hour market structure from swing points: 'up' (HH and HL), 'down' (LH and LL) or 'range'."""
    hi = (d1h["High"] if "High" in d1h else d1h["Close"]).values
    lo = (d1h["Low"] if "Low" in d1h else d1h["Close"]).values
    ph = [hi[i] for i in range(n, len(hi) - n) if hi[i] == max(hi[i - n:i + n + 1])]
    pl = [lo[i] for i in range(n, len(lo) - n) if lo[i] == min(lo[i - n:i + n + 1])]
    if len(ph) >= 2 and len(pl) >= 2:
        if ph[-1] < ph[-2] and pl[-1] < pl[-2]:
            return "down"
        if ph[-1] > ph[-2] and pl[-1] > pl[-2]:
            return "up"
    return "range"


def index_flags(d):
    """Is this index selling off or spiking right now? Uses 15 minute bars."""
    a = float(atr_series(d).iloc[-1])
    dc = completed(d)
    if not a or np.isnan(a) or len(dc) <= IDX_BARS:
        return None
    net = (dc.Close.iloc[-1] - dc.Close.iloc[-1 - IDX_BARS]) / a
    bar = (dc.Close.iloc[-1] - dc.Open.iloc[-1]) / a
    now = pd.Timestamp.now(tz=d.index.tz) if d.index.tz is not None else pd.Timestamp.now()
    return dict(open=bool(now - d.index[-1] < pd.Timedelta(minutes=40)),
                down=bool(net <= -IDX_NET or bar <= -IDX_BAR),
                up=bool(net >= IDX_NET or bar >= IDX_BAR))


# ---------- zone detection ----------
def find_zones(df, a, impulse=2.0):
    """A zone is 1-3 small candles (the base) followed by a big candle that
    leaves the base and travels at least `impulse` x ATR within 3 candles."""
    o, h, l, c, v = (df[k].values for k in ("Open", "High", "Low", "Close", "Volume"))
    n = len(df)
    body = np.abs(c - o)
    rng = h - l
    small = (body <= 0.5 * a) & (rng <= 1.2 * a)
    avgv = float(np.mean(v)) if n else 0.0
    zones = []
    for j in range(1, n - 1):
        k = 0
        while k < 3 and j - 1 - k >= 0 and small[j - 1 - k]:
            k += 1
        if k == 0 or body[j] < 0.8 * a:
            continue
        base = slice(j - k, j)
        end = min(j + 3, n)
        if c[j] > o[j]:
            low = l[base].min()
            move = h[j:end].max() - low
            if move < impulse * a:
                continue
            typ, prox, dist = "demand", max(o[base].max(), c[base].max()), low
        elif c[j] < o[j]:
            high = h[base].max()
            move = high - l[j:end].min()
            if move < impulse * a:
                continue
            typ, prox, dist = "supply", min(o[base].min(), c[base].min()), high
        else:
            continue
        height = abs(prox - dist)
        if height < 0.05 * a:
            continue
        after = slice(min(j + 3, n), n)
        if typ == "demand":
            broken = bool((c[after] < dist).any())
            hits = np.nonzero(l[after] <= prox)[0]
        else:
            broken = bool((c[after] > dist).any())
            hits = np.nonzero(h[after] >= prox)[0]
        # touches inside the last 5 bars are the current reaction, not a prior test of the zone
        first = after.start + int(hits[0]) if len(hits) else None
        touched = first is not None and first < n - 5
        zones.append(dict(type=typ, prox=float(prox), dist=float(dist), height=float(height),
                          idx=j, depart=float(move / a), volx=float(v[j] / avgv) if avgv else 0.0,
                          fresh=not touched, broken=broken))
    return zones


# ---------- play building ----------
ATR_RANGE = {"stock": (1.5, 10), "forex": (0.25, 2.5), "commodity": (0.8, 5),
             "index": (0.6, 6), "crypto": (2, 10)}
SCREEN_RVOL = 1.2   # stocks: relative volume needed to pass the screen


def reaction(d15c, prox, dist, long, look=4):
    """Did price reach the zone in the last `look` completed bars, and did it reject?
    Returns (state, time): 'rejected', 'touched' (no rejection yet) or 'waiting' (not there yet).
    Rejection = the bar tagged the zone and closed back out of it with a long wick
    or an engulfing candle in the trade direction."""
    t = d15c.tail(look + 1)
    if len(t) < 2:
        return "waiting", None
    o, h, l, c = (t[k].values for k in ("Open", "High", "Low", "Close"))
    for i in range(len(t) - 1, 0, -1):
        hit = (l[i] <= prox) if long else (h[i] >= prox)
        if not hit:
            continue
        rng = h[i] - l[i]
        if rng <= 0:
            return "touched", t.index[i]
        if long:
            out = c[i] > prox
            wick = (min(o[i], c[i]) - l[i]) >= 0.5 * rng
            engulf = c[i] > o[i] and c[i] >= o[i - 1] and o[i] <= c[i - 1] and c[i - 1] < o[i - 1]
        else:
            out = c[i] < prox
            wick = (h[i] - max(o[i], c[i])) >= 0.5 * rng
            engulf = c[i] < o[i] and c[i] <= o[i - 1] and o[i] >= c[i - 1] and c[i - 1] > o[i - 1]
        return ("rejected" if out and (wick or engulf) else "touched"), t.index[i]
    return "waiting", None


def decimals(price, sym=""):
    if sym.endswith("=X") and "JPY" in sym:
        return 3
    return 2 if price >= 100 else 3 if price >= 10 else 5 if price >= 1 else 6


def build_plays(tk, d15, d1h, dd, spread=None, max_dist=8.0, buf_pct=10.0, kind="stock", bar_min=15):
    """`d15` is the intraday frame at `bar_min` minute bars (15 by default, 5 for scalping)."""
    a = float(atr_series(d15).iloc[-1])
    price = float(d15.Close.iloc[-1])
    nd = decimals(price, tk)
    bar_ts = to_utc(d15.index[-1])
    adv = float(dd.Volume.iloc[-21:-1].mean())
    rvol_daily = float(dd.Volume.iloc[-1] / adv) if adv else 0.0
    atr_pct = float(atr_series(dd).iloc[-1] / dd.Close.iloc[-1] * 100)

    d15c = completed(d15, bar_min)
    bars, cum_rv = intraday_rvol(d15c)
    rvol = cum_rv if cum_rv is not None else rvol_daily
    vwap = session_vwap(d15)
    htf = structure(d1h.tail(80))

    ema1h = d1h.Close.ewm(span=20, adjust=False).mean()
    up1h = d1h.Close.iloc[-1] > ema1h.iloc[-1] and ema1h.iloc[-1] > ema1h.iloc[-5]
    dn1h = d1h.Close.iloc[-1] < ema1h.iloc[-1] and ema1h.iloc[-1] < ema1h.iloc[-5]
    emad = dd.Close.ewm(span=20, adjust=False).mean()
    upd = dd.Close.iloc[-1] > emad.iloc[-1]
    dnd = dd.Close.iloc[-1] < emad.iloc[-1]

    lo, hi = ATR_RANGE.get(kind, ATR_RANGE["stock"])
    if kind == "stock":
        # the spread is judged later against your own setting
        screen = [price > 5, adv >= 1e6, rvol >= SCREEN_RVOL, lo <= atr_pct <= hi]
    else:
        # forex, commodities, indices and crypto have no reliable volume feed here,
        # so the screen only asks for enough daily movement to be worth trading.
        screen = [lo <= atr_pct <= hi]
    screen_ok = all(screen)

    zones = [z for z in find_zones(d15, a) if not z["broken"]]
    session = d15.tail(max(10, int(52 * 15 / bar_min)))  # roughly the last two sessions
    plays = []
    for z in zones:
        long = z["type"] == "demand"
        entry = z["prox"]
        if long and not price > entry:
            continue
        if (not long) and not price < entry:
            continue
        if abs(price - entry) > max_dist * a:
            continue
        buf = z["height"] * buf_pct / 100
        stop = z["dist"] - buf if long else z["dist"] + buf
        # target: nearest opposing zone ahead of price, else the recent swing
        opp = [x for x in zones if x["type"] == ("supply" if long else "demand")]
        if long:
            ahead = [x["prox"] for x in opp if x["prox"] > price]
            tgt = min(ahead) if ahead else float(session.High.max())
            if tgt <= entry:
                continue
        else:
            ahead = [x["prox"] for x in opp if x["prox"] < price]
            tgt = max(ahead) if ahead else float(session.Low.min())
            if tgt >= entry:
                continue
        risk = abs(entry - stop)
        if risk <= 0:
            continue
        rr = abs(tgt - entry) / risk

        # zone quality extras: age, and whether the entry sits inside a bigger opposing zone
        age_bars = len(d15) - 1 - z["idx"]
        nested = any(x["height"] > z["height"] and min(x["prox"], x["dist"]) <= entry <= max(x["prox"], x["dist"])
                     for x in opp)
        react, react_ts = reaction(d15c, entry, z["dist"], long, look=4)

        # volume at the zone: the bar that touched it, otherwise the last two bars as price approaches
        rv, basis = None, None
        if bars:
            for k in range(1, 5):
                r, ts = d15c.iloc[-k], d15c.index[-k]
                if (long and r.Low <= entry) or ((not long) and r.High >= entry):
                    rv, basis = bars.get(ts), "touch"
                    break
            if rv is None:
                vals = [bars[t] for t in d15c.index[-2:] if bars.get(t) is not None]
                if vals:
                    rv, basis = float(np.mean(vals)), "approach"

        checks = [
            z["depart"] >= 3.0,                                  # strong departure
            z["fresh"],                                          # untouched since forming
            z["height"] <= 1.0 * a,                              # tight base
            (up1h and upd) if long else (dn1h and dnd),          # higher timeframe agrees
            ((z["volx"] >= 1.5 or rvol >= 2.0 or (rv is not None and rv >= RV_STRONG))
             if kind == "stock" else z["depart"] >= 4.0),        # volume or momentum
        ]
        score = int(sum(checks))
        take = screen_ok and score >= FILTERS["min_score"] and rr >= FILTERS["min_rr_take"]
        plays.append(dict(
            tkr=tk, side="long" if long else "short", kind=kind, nd=nd,
            bar_ts=bar_ts, px=round(price, nd), adv=int(round(adv)), rvol=round(rvol, 2), atr=round(atr_pct, 2),
            spr=None if spread is None else round(spread, 3),
            prox=round(entry, nd), dist=round(z["dist"], nd), tgt=round(tgt, nd),
            rr=round(rr, 1), score=score, sc=[bool(x) for x in checks],
            screen=[bool(x) for x in screen], take=bool(take), tf=f"{int(bar_min)}m", bar_min=int(bar_min),
            bar_lo=float(d15.Low.iloc[-1]), bar_hi=float(d15.High.iloc[-1]),
            atr15=a, vwap=None if vwap is None else round(vwap, nd),
            rv=None if rv is None else round(rv, 2), rv_basis=basis,
            htf=htf, depart=round(z["depart"], 1), age_bars=int(age_bars), nested=bool(nested),
            react=react, react_ts=react_ts,
        ))
    plays.sort(key=lambda p: (p["take"], p["score"], p["rr"]), reverse=True)
    return plays


# ---------- data ----------
def to_utc(ts):
    t = pd.Timestamp(ts)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return t.to_pydatetime()


def _retry(fn, tries=3, pause=0.7):
    """Yahoo drops requests now and then. Retry on errors only, with a growing pause."""
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(pause * (i + 1))
    raise last


def fetch_spread(tk):
    """Bid-ask spread in percent, or None. Slow (one extra request), so only call it for real candidates."""
    import yfinance as yf
    try:
        info = _retry(lambda: yf.Ticker(tk).info, tries=2)
        bid, ask = info.get("bid") or 0, info.get("ask") or 0
        if bid > 0 and ask > 0:
            return (ask - bid) / ((ask + bid) / 2) * 100
    except Exception:  # noqa: BLE001
        pass
    return None


def fetch_intraday(tk, bar_min=15, period=None):
    """Intraday bars at `bar_min` minutes. The frame is still called d15 elsewhere.
    `period` overrides the default window, for example "58d" for a backtest (Yahoo keeps about 60 days)."""
    import yfinance as yf
    bar_min = int(bar_min)
    period = period or ("20d" if bar_min >= 15 else "10d")
    d = _retry(lambda: yf.Ticker(tk).history(period=period, interval=f"{bar_min}m"))
    return d.dropna()


def fetch_slow(tk, hist=False):
    """1 hour and daily bars. They change slowly, so a scan can reuse them for much longer than the intraday bars.
    `hist` loads a longer stretch of both, so a backtest has context for its earliest steps."""
    import yfinance as yf
    t = yf.Ticker(tk)
    d1h = _retry(lambda: t.history(period="3mo" if hist else "1mo", interval="1h"))
    dd = _retry(lambda: t.history(period="1y" if hist else "3mo", interval="1d"))
    return d1h.dropna(), dd.dropna()


def fetch(tk, with_info=True, bar_min=15):
    d15 = fetch_intraday(tk, bar_min)
    d1h, dd = fetch_slow(tk)
    spread = fetch_spread(tk) if with_info else None
    return d15, d1h, dd, spread


def cached(store, key, ttl, fn):
    """Tiny thread-safe TTL cache. `store` is a dict with a lock, shared by all scans."""
    now = time.time()
    with store["lock"]:
        hit = store["d"].get(key)
        if hit and now - hit[0] < ttl:
            return hit[1]
    val = fn()
    with store["lock"]:
        store["d"][key] = (now, val)
        if len(store["d"]) > 400:
            for k, _ in sorted(store["d"].items(), key=lambda kv: kv[1][0])[:100]:
                store["d"].pop(k, None)
    return val



# ================= news, events and earnings =================
NEG_RE = re.compile(
    r"\b(halt(ed|s)?|investigat\w+|probe[ds]?|subpoena\w*|lawsuit|sues?|sued|fraud\w*|bankrupt\w*|chapter 11|"
    r"delist\w*|recall\w*|downgrad\w+|plunge\w*|tumbl\w+|crash\w*|slump\w*|miss(es|ed)?|"
    r"(cuts?|lowers?|slashes?) (guidance|forecast|outlook)|warning|resigns?|steps? down|"
    r"(public |stock |share )?offering|dilut\w+|short[- ]seller|SEC charges|data breach|cyberattack|"
    r"complete response letter|rejects?|tariffs?)\b", re.I)
POS_RE = re.compile(
    r"\b(upgrad\w+|surg\w+|soar\w*|jump\w*|rall(y|ies|ied)|beats?|"
    r"raises? (guidance|forecast|outlook)|record (high|revenue|earnings)|approv\w+|acquir\w+|"
    r"acquisition|buyout|takeover|partnership|wins?|breakthrough)\b", re.I)
MAJOR_RE = re.compile(
    r"(\bcpi\b|consumer price|\bppi\b|producer price|non-?farm|payrolls|fomc|fed funds|"
    r"interest rate (decision|announcement)|rate decision|\bgdp\b|retail sales|\bpce\b|"
    r"unemployment rate|jobless claims)", re.I)
# Fed decision days, announced 2:00 pm New York time (federalreserve.gov calendar).
FOMC_DECISIONS = ["2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
                  "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09"]
COUNTRY_CCY = {"US": "USD", "EU": "EUR", "EMU": "EUR", "DE": "EUR", "FR": "EUR", "IT": "EUR", "ES": "EUR",
               "GB": "GBP", "UK": "GBP", "JP": "JPY", "AU": "AUD", "NZ": "NZD", "CA": "CAD",
               "CH": "CHF", "CN": "CNY", "HK": "HKD"}


def _get_json(url, timeout=10):
    req = urllib.request.Request(url, headers={"User-Agent": "zone-ticket"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def parse_manual(text):
    out = []
    for line in text.splitlines():
        m = re.match(r"\s*(\d{4}-\d{2}-\d{2})[ T](\d{1,2}):(\d{2})\s*(.*)", line)
        if not m:
            continue
        try:
            t = datetime.strptime(f"{m[1]} {int(m[2]):02d}:{m[3]}", "%Y-%m-%d %H:%M").replace(tzinfo=SYD)
        except ValueError:
            continue
        out.append(dict(t=t.astimezone(timezone.utc), name=(m[4].strip() or "Your blackout"), ccy="ALL"))
    return out



def fmp_events_raw(key):
    """High impact economic releases from Financial Modeling Prep. Raises if nothing could be read."""
    today = datetime.now(timezone.utc).date()
    a, b = today - timedelta(days=1), today + timedelta(days=3)
    problem = "no response"
    for base in ("https://financialmodelingprep.com/stable/economic-calendar",
                 "https://financialmodelingprep.com/api/v3/economic_calendar"):
        try:
            data = _get_json(f"{base}?from={a}&to={b}&apikey={urllib.parse.quote(key)}")
        except urllib.error.HTTPError as e:
            problem = f"HTTP {e.code}"
            continue
        except Exception:
            problem = "network error"
            continue
        if not isinstance(data, list):
            problem = "the key was refused or the plan does not include the calendar"
            continue
        out = []
        for e in data:
            name = str(e.get("event", ""))
            if str(e.get("impact", "")).lower() != "high" and not MAJOR_RE.search(name):
                continue
            try:
                t = pd.Timestamp(e["date"])
                t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
            except Exception:
                continue
            ccy = (e.get("currency") or "").upper() or COUNTRY_CCY.get(str(e.get("country", "")).upper(), "")
            if ccy:
                out.append(dict(t=t.to_pydatetime(), name=name, ccy=ccy))
        return out
    raise RuntimeError(problem)



def build_events(manual_text, key="", fmp=None):
    """Event blackouts: built-in FOMC dates, your own times, and (with a key) FMP releases.
    `fmp` is the function that loads FMP events, so the caller can cache it."""
    fmp = fmp or fmp_events_raw
    ev = []
    for d in FOMC_DECISIONS:
        y, m, dd_ = (int(x) for x in d.split("-"))
        ev.append(dict(t=datetime(y, m, dd_, 14, 0, tzinfo=NY).astimezone(timezone.utc),
                       name="FOMC rate decision", ccy="USD"))
    ev += parse_manual(manual_text)
    status = "Event blackouts: built-in FOMC dates"
    if manual_text.strip():
        status += " and your own times"
    if key:
        try:
            more = fmp(key)
            ev += more
            status += f", plus {len(more)} high-impact releases from FMP"
        except Exception as e:  # noqa: BLE001
            status += f". FMP calendar not loaded ({e})"
    else:
        status += ". Add an FMP key in Filters to include CPI, PPI and jobs data"
    now = datetime.now(timezone.utc)
    return [e for e in ev if abs((e["t"] - now).total_seconds()) < 2 * 86400], status


def index_state_raw():
    """SPY and QQQ 15 minute state. None when neither could be read."""
    import yfinance as yf
    res = {}
    for sym in ("SPY", "QQQ"):
        try:
            f = index_flags(_retry(lambda: yf.Ticker(sym).history(period="5d", interval="15m"), tries=2).dropna())
            if f:
                res[sym] = f
        except Exception:  # noqa: BLE001
            pass
    if not res:
        return None
    return dict(open=any(v["open"] for v in res.values()),
                down=[k for k, v in res.items() if v["open"] and v["down"]],
                up=[k for k, v in res.items() if v["open"] and v["up"]])



def fetch_news(tk):
    import yfinance as yf
    out = []
    for it in yf.Ticker(tk).news or []:
        c = it.get("content") if isinstance(it.get("content"), dict) else it
        title = c.get("title") or it.get("title")
        ts = c.get("pubDate") or c.get("displayTime") or it.get("providerPublishTime")
        if not title or ts is None:
            continue
        try:
            t = pd.Timestamp(ts, unit="s", tz="UTC") if isinstance(ts, (int, float)) else pd.Timestamp(ts)
            t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
        except Exception:
            continue
        out.append((t.to_pydatetime(), str(title)))
    return out


def fetch_earnings(tk):
    import yfinance as yf
    cal = yf.Ticker(tk).calendar
    vals = []
    if isinstance(cal, dict):
        vals = cal.get("Earnings Date") or []
    elif cal is not None and hasattr(cal, "loc"):
        try:
            vals = list(cal.loc["Earnings Date"])
        except Exception:
            vals = []
    out = []
    for v in vals:
        try:
            out.append(pd.Timestamp(v).date())
        except Exception:
            pass
    return out




# ================= markets and settings =================

DEFAULT_WATCH = ("AAPL MSFT NVDA AMD TSLA META AMZN GOOGL NFLX AVGO PLTR SMCI COIN MSTR HOOD SOFI "
                 "RKLB ASTS LUNR IONQ RGTI AVAV KTOS MRNA CRSP "
                 "MU INTC ARM TSM ORCL CRM ADBE UBER SHOP SNOW CRWD PANW ANET PYPL AFRM UPST RBLX U DKNG "
                 "ABNB DASH RIVN LCID NIO F BA DIS NKE JPM BAC GS XOM CVX OXY LLY UNH WMT COST "
                 "MARA RIOT CLSK SNAP ROKU")

# name -> (screen kind, default leverage limit, [(yahoo symbol, label)])
MARKETS = {
    "Forex": ("forex", 30.0, [
        ("EURUSD=X", "EUR/USD"), ("GBPUSD=X", "GBP/USD"), ("USDJPY=X", "USD/JPY"),
        ("AUDUSD=X", "AUD/USD"), ("NZDUSD=X", "NZD/USD"), ("USDCAD=X", "USD/CAD"),
        ("USDCHF=X", "USD/CHF"), ("EURJPY=X", "EUR/JPY"), ("GBPJPY=X", "GBP/JPY"),
        ("AUDJPY=X", "AUD/JPY"), ("EURGBP=X", "EUR/GBP"), ("EURAUD=X", "EUR/AUD"),
        ("AUDNZD=X", "AUD/NZD"), ("AUDCAD=X", "AUD/CAD"), ("AUDCHF=X", "AUD/CHF"),
        ("EURCAD=X", "EUR/CAD"), ("EURCHF=X", "EUR/CHF"), ("EURNZD=X", "EUR/NZD"),
        ("GBPAUD=X", "GBP/AUD"), ("GBPCAD=X", "GBP/CAD"), ("GBPCHF=X", "GBP/CHF"),
        ("GBPNZD=X", "GBP/NZD"), ("NZDJPY=X", "NZD/JPY"), ("CADJPY=X", "CAD/JPY"),
        ("CHFJPY=X", "CHF/JPY"), ("NZDCAD=X", "NZD/CAD"), ("CADCHF=X", "CAD/CHF")]),
    "Commodities": ("commodity", 10.0, [
        ("GC=F", "Gold"), ("SI=F", "Silver"), ("PL=F", "Platinum"), ("HG=F", "Copper"),
        ("CL=F", "WTI Oil"), ("BZ=F", "Brent Oil"), ("NG=F", "Natural Gas"),
        ("ZW=F", "Wheat"), ("ZC=F", "Corn"), ("KC=F", "Coffee"),
        ("PA=F", "Palladium"), ("HO=F", "Heating Oil"), ("RB=F", "Gasoline"),
        ("ZS=F", "Soybeans"), ("ZL=F", "Soybean Oil"), ("SB=F", "Sugar"),
        ("CC=F", "Cocoa"), ("CT=F", "Cotton"), ("LE=F", "Live Cattle"), ("HE=F", "Lean Hogs")]),
    "Indices": ("index", 20.0, [
        ("ES=F", "S&P 500"), ("NQ=F", "Nasdaq 100"), ("YM=F", "Dow Jones"), ("RTY=F", "Russell 2000"),
        ("^GDAXI", "Germany 40"), ("^FTSE", "UK 100"), ("^N225", "Japan 225"),
        ("^AXJO", "Australia 200"), ("^HSI", "Hong Kong 50"),
        ("^FCHI", "France 40"), ("^STOXX50E", "Euro Stoxx 50"), ("^IXIC", "Nasdaq Composite"),
        ("^RUT", "Russell 2000 Index")]),
    "Crypto": ("crypto", 2.0, [
        ("BTC-USD", "Bitcoin"), ("ETH-USD", "Ethereum"), ("SOL-USD", "Solana"),
        ("XRP-USD", "XRP"), ("DOGE-USD", "Dogecoin"), ("ADA-USD", "Cardano"),
        ("BNB-USD", "BNB"), ("AVAX-USD", "Avalanche"), ("LINK-USD", "Chainlink"),
        ("LTC-USD", "Litecoin"), ("DOT-USD", "Polkadot"), ("TRX-USD", "Tron"),
        ("BCH-USD", "Bitcoin Cash"), ("SHIB-USD", "Shiba Inu")]),
    "ETFs": ("index", 5.0, [
        ("SPY", "SPY"), ("QQQ", "QQQ"), ("IWM", "IWM"), ("DIA", "DIA"), ("TQQQ", "TQQQ"),
        ("SQQQ", "SQQQ"), ("SOXL", "SOXL"), ("GLD", "GLD"), ("SLV", "SLV"), ("TLT", "TLT"),
        ("SMH", "SMH"), ("XLK", "XLK"), ("XLF", "XLF"), ("XLE", "XLE"), ("XLV", "XLV"),
        ("XLY", "XLY"), ("XLI", "XLI"), ("ARKK", "ARKK"), ("XBI", "XBI"), ("EEM", "EEM"),
        ("FXI", "FXI"), ("USO", "USO"), ("UNG", "UNG"), ("TNA", "TNA"), ("SPXL", "SPXL"),
        ("LABU", "LABU"), ("IBIT", "IBIT"), ("UVXY", "UVXY")]),
    "Stocks": ("stock", 5.0, []),
}
US_MARKETS = ("Stocks", "ETFs")
LOOKUP = {sym: (label, MARKETS[m][0], m) for m in MARKETS for sym, label in MARKETS[m][2]}
INDEX_CCY = {"^GDAXI": "EUR", "^FTSE": "GBP", "^N225": "JPY", "^AXJO": "AUD", "^HSI": "HKD",
             "^FCHI": "EUR", "^STOXX50E": "EUR"}



# ---------------- settings: defaults, limits, validation ----------------
SORTS = ["Best setup", "Reward:risk", "Closest to entry", "Market"]
DIRS = ["All", "Long", "Short"]

FILTERS = {"min_score": 3.0, "min_rr_take": 2.5, "max_dist": 15.0, "per_inst": 3.0,
           "min_stop_atr": 0.5, "max_tgt_atr": 10.0,
           "f_events": True, "ev_min": 15.0, "manual_ev": "", "f_news": True, "news_hrs": 3.0,
           "f_earn": True, "f_rvol": True, "rv_weak": 0.8, "f_vwap": True, "vwap_tol": 1.0, "f_htf": True,
           "f_index": True, "max_spread": 0.05,
           "f_depart": True, "zq_depart": 3.0, "f_age": True, "zq_age": 96.0, "f_nest": True,
           "f_react": False,
           "f_stale": True, "stale_min": 60.0,
           "f_limits": True, "day_loss_pct": 6.0, "max_open": 3.0, "max_open_risk": 6.0}
DEFAULTS = {"acct": "AUD", "equity": 10000.0, "risk_pct": 2.0,
            "sel": ["Forex", "Commodities", "Indices", "Crypto", "ETFs", "Stocks"],
            "watch": DEFAULT_WATCH, "extras": "", "show_close": False, "auto": False,
            "sort_by": SORTS[0], "dir_f": DIRS[0], "min_rr": 0.0,
            "alerts_on": False, "ntfy_topic": "",
            "tf": "15m", "rescan_min": 60.0, "take_expire_min": 120.0}
DEFAULTS.update(FILTERS)
for _m in MARKETS:
    DEFAULTS[f"lev_{_m}"] = MARKETS[_m][1]

TIMEFRAMES = {"15m": 15, "5m": 5}
# One tap preset for quick scalps: 5 minute bars, zones close to price, smaller reward, fast rescans.
SCALP_PRESET = {"tf": "5m", "max_dist": 4.0, "min_rr_take": 1.5, "per_inst": 2.0, "zq_age": 8.0,
                "min_stop_atr": 1.0, "max_tgt_atr": 8.0,
                "stale_min": 15.0, "rescan_min": 5.0, "take_expire_min": 60.0, "sort_by": "Closest to entry",
                "auto": True}

CHOICES = {"acct": ["AUD", "USD"], "sort_by": SORTS, "dir_f": DIRS, "tf": list(TIMEFRAMES)}
BOUNDS = {"equity": (100.0, 1e9), "risk_pct": (0.1, 10.0), "min_rr": (0.0, 20.0),
          "ev_min": (0.0, 120.0), "news_hrs": (0.5, 24.0), "rv_weak": (0.1, 2.0), "max_spread": (0.0, 2.0),
          "vwap_tol": (0.0, 5.0), "zq_depart": (2.0, 10.0), "zq_age": (4.0, 120.0),
          "stale_min": (5.0, 2880.0), "day_loss_pct": (0.5, 50.0), "max_open": (1.0, 50.0),
          "max_open_risk": (0.5, 100.0), "min_score": (1.0, 5.0), "min_rr_take": (1.0, 10.0),
          "max_dist": (2.0, 40.0), "per_inst": (1.0, 6.0),
          "rescan_min": (2.0, 240.0), "take_expire_min": (10.0, 1440.0),
          "min_stop_atr": (0.0, 5.0), "max_tgt_atr": (1.0, 100.0)}
for _m in MARKETS:
    BOUNDS[f"lev_{_m}"] = (1.0, 500.0)


def clamp(key, v):
    lo, hi = BOUNDS.get(key, (None, None))
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v


def from_q(key, raw):
    """Read one setting from the page address. Out-of-range numbers are pulled back into range,
    because the number boxes refuse values outside their limits."""
    d = DEFAULTS[key]
    if isinstance(d, bool):
        return raw == "1"
    if isinstance(d, float):
        v = float(raw)
        if not math.isfinite(v):
            raise ValueError("not a number")
        return clamp(key, v)
    if isinstance(d, list):
        return [x for x in raw.split(",") if x in MARKETS]
    if key in CHOICES:
        return raw if raw in CHOICES[key] else d
    return raw


def to_q(v):
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, list):
        return ",".join(v)
    return str(v)


def clean_setting(key, v):
    """Validate one value that came from a settings code. Raises ValueError when it does not fit."""
    d = DEFAULTS[key]
    if isinstance(d, bool):
        if not isinstance(v, bool):
            raise ValueError(key)
        return v
    if isinstance(d, float):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(key)
        return clamp(key, float(v))
    if isinstance(d, list):
        if not isinstance(v, list):
            raise ValueError(key)
        return [x for x in v if x in MARKETS]
    if not isinstance(v, str):
        raise ValueError(key)
    if key in CHOICES and v not in CHOICES[key]:
        raise ValueError(key)
    return v[:4000]


CODE_PREFIX = "ZT1:"


def settings_to_code(settings):
    data = {k: settings[k] for k in DEFAULTS if k in settings}
    raw = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return CODE_PREFIX + base64.urlsafe_b64encode(raw).decode()


def code_to_settings(code):
    """Decode a settings code into {key: value}. Raises ValueError for anything that is not a valid code."""
    code = (code or "").strip()
    if not code.startswith(CODE_PREFIX):
        raise ValueError("That does not look like a settings code.")
    try:
        data = json.loads(base64.urlsafe_b64decode(code[len(CODE_PREFIX):].encode()).decode())
    except Exception as e:  # noqa: BLE001
        raise ValueError("The code is damaged.") from e
    if not isinstance(data, dict):
        raise ValueError("The code is damaged.")
    out = {}
    for k, v in data.items():
        if k in DEFAULTS:
            try:
                out[k] = clean_setting(k, v)
            except ValueError:
                continue
    if not out:
        raise ValueError("The code has no usable settings.")
    return out



# ================= helpers =================

def infer(sym):
    if sym in LOOKUP:
        return LOOKUP[sym]
    if sym.endswith("=X"):
        return sym, "forex", "Forex"
    if sym.endswith("=F"):
        return sym, "commodity", "Commodities"
    if sym.endswith("-USD"):
        return sym, "crypto", "Crypto"
    if sym.startswith("^"):
        return sym, "index", "Indices"
    return sym, "stock", "Stocks"


def fmt(v, nd):
    return f"{v:,.{nd}f}"



def quote_ccy(p):
    sym = p["tkr"]
    if p.get("kind") == "forex" and len(sym) >= 6:
        return sym[3:6]
    return INDEX_CCY.get(sym, "USD")



# What each check is for. Shown on the page and next to any check a play fails.
CHECK_WHY = {
    "Passes the screen": "Thin, cheap or flat instruments have wide spreads and do not travel far enough to pay for the risk.",
    "Zone score": "Five quality points: strong departure, untouched, tight base, higher timeframes agree, volume or momentum. More points means the zone is more likely to still hold unfilled orders.",
    "Reward : risk": "At 2.5 to 1 you can lose well over half your trades and still make money. Low reward to risk needs a very high win rate.",
    "Stop size": "A stop smaller than normal candle noise gets hit by random movement, and the spread eats a large share of it. A tiny stop also inflates reward to risk, so a 14 to 1 plan on paper is not a 14 to 1 trade.",
    "Target within reach": "A target many ATRs away needs a long, clean move that most scalps never get. The plan looks good on paper, but price rarely travels that far before something reverses it.",
    "Affordable size": "If your risk amount cannot buy a single unit, the stop is too wide for your account size.",
    "Leverage within limit": "The trade must fit the leverage your broker allows, or it cannot be placed.",
    "Margin under 50%": "Tying up more than half your equity in margin leaves no room for a second trade or a bad move.",
    "Fresh data": "If the last candle is old the market is closed or the feed is lagging, so the levels are not live.",
    "No news event nearby": "Rate decisions and big data releases blow out spreads and run straight through zones.",
    "Spread": "A wide bid-ask spread is a cost paid on entry and exit, and it eats a tight stop.",
    "No earnings risk": "Earnings cause overnight gaps that jump past your stop with no chance to exit.",
    "No bad headline": "Fresh bad news against a long (or good news against a short) can overrun a zone.",
    "Volume at the zone": "A zone with no volume when price arrives has no real buyers or sellers defending it.",
    "Index calm": "US stocks follow SPY and QQQ. A long into an index selloff usually gets dragged through its zone.",
    "VWAP side": "VWAP is the day's average price. Buying far below it or shorting far above it fights the day's balance.",
    "1h trend agrees": "Trading with the hourly structure means you are not fighting the bigger move.",
    "Strong departure": "A zone that price left slowly had little imbalance behind it, so it is less likely to hold.",
    "Young zone": "Old zones have been seen and traded by more people and tend to be used up.",
    "Not nested": "A bigger opposing zone around your entry can stop price before it reaches your target.",
    "Rejection candle": "Waiting for a wick or engulfing candle shows the zone is actually holding right now. This turns plays into watch-only until price arrives, so it is off by default.",
    "Within your trading limits": "Daily loss, open trade count and total open risk limits stop one bad day from getting out of control.",
}


def number_checks(p, s, lev_cap, S=None):
    """The play's own numbers as (label, state, detail) rows. state is True (pass), False (fail) or None (off)."""
    S = S or {}
    ms = float(S.get("min_score", FILTERS["min_score"]))
    mr = float(S.get("min_rr_take", FILTERS["min_rr_take"]))
    out = []
    if p.get("kind") == "stock":
        names = ["price above $5", "average volume 1M+", f"relative volume {SCREEN_RVOL:g}x+", "daily range"]
    else:
        names = ["daily range"]
    failed = [n for n, ok in zip(names, p["screen"]) if not ok]
    out.append(("Passes the screen", not failed,
                ("fails the screen: " + ", ".join(failed)) if failed else "enough volume and movement to be worth trading"))
    ok = p["score"] >= ms
    out.append(("Zone score", ok, f"zone score {p['score']}/5, needs {ms:.0f}" if not ok
                else f"{p['score']}/5, needs {ms:.0f}"))
    ok = p["rr"] >= mr
    out.append(("Reward : risk", ok, f"reward to risk {p['rr']:.1f}, needs {mr:g}" if not ok
                else f"{p['rr']:.1f} to 1, needs {mr:g}"))
    atr = p.get("atr15") or 0
    if atr > 0:
        mn = float(S.get("min_stop_atr", FILTERS["min_stop_atr"]))
        stop_atr = abs(p["prox"] - s["stop"]) / atr
        ok = stop_atr >= mn
        out.append(("Stop size", ok, f"stop is only {stop_atr:.2f} ATR from entry, needs {mn:g}, normal noise will hit it"
                    if not ok else f"{stop_atr:.2f} ATR from entry, needs {mn:g}"))
        mx = float(S.get("max_tgt_atr", FILTERS["max_tgt_atr"]))
        tgt_atr = abs(p["tgt"] - p["prox"]) / atr
        ok = tgt_atr <= mx
        out.append(("Target within reach", ok, f"target is {tgt_atr:.1f} ATR from entry, limit {mx:g}, too far to expect"
                    if not ok else f"{tgt_atr:.1f} ATR from entry, limit {mx:g}"))
    ok = s["units"] >= 1
    out.append(("Affordable size", ok, "risk amount too small for one unit" if not ok else f"{s['units']:,} units"))
    ok = s["lev"] <= lev_cap
    out.append(("Leverage within limit", ok, f"needs {s['lev']:.1f}x leverage, limit {lev_cap:.0f}x" if not ok
                else f"{s['lev']:.1f}x of {lev_cap:.0f}x allowed"))
    ok = s["margin_pct"] <= 50
    out.append(("Margin under 50%", ok, f"margin is {s['margin_pct']:.0f}% of equity, keep under 50%" if not ok
                else f"{s['margin_pct']:.0f}% of equity"))
    return out


def reasons(p, s, lev_cap, S=None):
    """Reasons a play fails on its own numbers."""
    return [d for _, st, d in number_checks(p, s, lev_cap, S) if st is False]


def zone_points(p):
    """The five zone quality points as (label, earned)."""
    last = "Volume or interest at the zone" if p.get("kind") == "stock" else "Strong momentum (4+ ATR departure)"
    labels = ["Strong departure (3+ ATR)", "Untouched since forming", "Tight base",
              "Higher timeframes agree", last]
    return list(zip(labels, p.get("sc") or []))


def ccy_applies(ccy, p):
    if ccy in ("ALL", "USD"):
        return True
    sym = p["tkr"]
    if p.get("kind") == "forex" and len(sym) >= 6:
        return ccy in (sym[:3], sym[3:6])
    return INDEX_CCY.get(sym) == ccy


def prev_bday(d):
    return (pd.Timestamp(d) - pd.offsets.BDay(1)).date()


def filter_checks(p, S, ctx, now=None):
    """Every context filter as (label, state, detail). state is True (pass), False (holds the play back)
    or None (switched off, or no data to judge with). US-only checks are left out for other markets."""
    now = now or datetime.now(timezone.utc)
    out = []
    long = p["side"] == "long"

    def add(label, state, detail):
        out.append((label, state, detail))

    # fresh data
    if not S.get("f_stale"):
        add("Fresh data", None, "filter off")
    elif p.get("bar_ts") is None:
        add("Fresh data", None, "no candle time")
    else:
        age = (now - p["bar_ts"]).total_seconds() / 60
        if age > S["stale_min"]:
            add("Fresh data", False, f"stale data: last candle {fmt_age(age)} old "
                f"({p['bar_ts'].astimezone(SYD):%a %H:%M} Sydney), market probably closed")
        else:
            add("Fresh data", True, f"last candle {fmt_age(age)} old")

    # news events
    if not S["f_events"]:
        add("No news event nearby", None, "filter off")
    else:
        hit = None
        for e in ctx.get("events", []):
            if not ccy_applies(e["ccy"], p):
                continue
            mins = (e["t"] - now).total_seconds() / 60
            if -S["ev_min"] <= mins <= S["ev_min"]:
                hit = (e, mins)
                break
        if hit:
            e, mins = hit
            add("No news event nearby", False,
                f"news blackout: {e['name']} {e['t'].astimezone(SYD):%H:%M} Sydney ({mins:+.0f} min)")
        else:
            add("No news event nearby", True, f"no major event within {S['ev_min']:.0f} minutes")

    if p.get("us"):
        spr = p.get("spr")
        if spr is None:
            add("Spread", None, "spread unknown")
        elif spr > S["max_spread"]:
            add("Spread", False, f"spread {spr:.3f}% is over your {S['max_spread']:.3f}% limit")
        else:
            add("Spread", True, f"{spr:.3f}%, limit {S['max_spread']:.3f}%")

        if not S["f_earn"]:
            add("No earnings risk", None, "filter off")
        elif p.get("market") != "Stocks":
            add("No earnings risk", None, "not a single stock")
        else:
            today = now.astimezone(NY).date()
            msg = None
            for d in p.get("earn") or []:
                if d == today:
                    msg = f"earnings today ({d:%d %b})"
                    break
                if d == prev_bday(today):
                    msg = f"earnings {d:%d %b}, gap risk the day after"
                    break
            add("No earnings risk", msg is None, msg or "no earnings today or yesterday")

        if not S["f_news"]:
            add("No bad headline", None, "filter off")
        elif p.get("news_ok") is False:
            add("No bad headline", None, "headlines could not be read")
        else:
            bad = NEG_RE if long else POS_RE
            msg = None
            for t, title in sorted(p.get("news") or [], reverse=True):
                age = (now - t).total_seconds() / 3600
                if 0 <= age <= S["news_hrs"] and bad.search(title):
                    msg = f"{'negative' if long else 'positive'} headline {age:.1f}h ago: \"{title[:70]}\""
                    break
            add("No bad headline", msg is None, msg or f"nothing against the trade in the last {S['news_hrs']:g}h")

        if not S["f_rvol"]:
            add("Volume at the zone", None, "filter off")
        elif p.get("rv") is None:
            add("Volume at the zone", None, "no volume reading")
        elif p["rv"] < S["rv_weak"]:
            add("Volume at the zone", False, f"weak volume at the zone ({p['rv']:.1f}x normal for this time)")
        else:
            add("Volume at the zone", True, f"{p['rv']:.1f}x normal for this time of day")

        idx = ctx.get("index")
        if not S["f_index"]:
            add("Index calm", None, "filter off")
        elif not idx or not idx["open"]:
            add("Index calm", None, "US market closed or index unreadable")
        elif long and idx["down"]:
            add("Index calm", False, f"{'/'.join(idx['down'])} selling off, wait for it to settle")
        elif (not long) and idx["up"]:
            add("Index calm", False, f"{'/'.join(idx['up'])} spiking, wait for it to settle")
        else:
            add("Index calm", True, "SPY and QQQ are not making a sharp move")

    if not S["f_vwap"]:
        add("VWAP side", None, "filter off")
    elif p.get("vwap") is None or not p.get("atr15"):
        add("VWAP side", None, "no volume feed, so no VWAP")
    else:
        gap = (p["prox"] - p["vwap"]) / p["atr15"]
        if long and gap < -S["vwap_tol"]:
            add("VWAP side", False, f"demand zone is {abs(gap):.1f} ATR below session VWAP")
        elif (not long) and gap > S["vwap_tol"]:
            add("VWAP side", False, f"supply zone is {gap:.1f} ATR above session VWAP")
        else:
            add("VWAP side", True, f"zone is {gap:+.1f} ATR from VWAP, limit {S['vwap_tol']:g}")

    if not S["f_htf"]:
        add("1h trend agrees", None, "filter off")
    else:
        against = p.get("htf") == ("down" if long else "up")
        add("1h trend agrees", not against,
            ("counter-trend: 1h makes " + ("lower highs and lower lows" if long else "higher highs and higher lows"))
            if against else f"1h structure is {p.get('htf') or 'unknown'}, not against the trade")

    if not S["f_depart"]:
        add("Strong departure", None, "filter off")
    elif p.get("depart") is None:
        add("Strong departure", None, "no reading")
    elif p["depart"] < S["zq_depart"]:
        add("Strong departure", False,
            f"weak departure: the zone only moved {p['depart']:.1f} ATR away (needs {S['zq_depart']:.1f})")
    else:
        add("Strong departure", True, f"moved {p['depart']:.1f} ATR away, needs {S['zq_depart']:.1f}")

    if not S["f_age"]:
        add("Young zone", None, "filter off")
    elif p.get("age_bars") is None:
        add("Young zone", None, "no reading")
    elif zone_age_min(p) > S["zq_age"] * 60:
        add("Young zone", False,
            f"old zone: formed about {fmt_age(zone_age_min(p))} of trading ago (limit {S['zq_age']:.0f}h)")
    else:
        add("Young zone", True, f"about {fmt_age(zone_age_min(p))} old, limit {S['zq_age']:.0f}h")

    if not S["f_nest"]:
        add("Not nested", None, "filter off")
    elif p.get("nested"):
        add("Not nested", False, "the zone sits inside a bigger " + ("supply" if long else "demand") + " zone")
    else:
        add("Not nested", True, "no bigger opposing zone around the entry")

    if not S["f_react"]:
        add("Rejection candle", None, "filter off, the reaction is still shown on the card")
    elif p.get("react") is None:
        add("Rejection candle", None, "no reading")
    elif p["react"] == "waiting":
        add("Rejection candle", False, "price has not reached the zone yet, wait for a rejection candle")
    elif p["react"] == "touched":
        add("Rejection candle", False, "price tagged the zone but has not printed a rejection candle")
    else:
        add("Rejection candle", True, "rejection candle printed at the zone")
    return out


def filter_reasons(p, S, ctx, now=None):
    """Reasons a play is held back by the context filters."""
    return [d for _, st, d in filter_checks(p, S, ctx, now) if st is False]


def context_lines(p, S):
    out = []
    if p.get("bar_ts") is not None:
        age = (datetime.now(timezone.utc) - p["bar_ts"]).total_seconds() / 60
        out.append(f"Last candle {fmt_age(age)} ago ({p['bar_ts'].astimezone(SYD):%a %H:%M} Sydney)")
    if p.get("vwap") is not None and p.get("atr15"):
        gap = (p["prox"] - p["vwap"]) / p["atr15"]
        out.append(f"Session VWAP {fmt(p['vwap'], p['nd'])}, zone is {gap:+.1f} ATR from it")
    if p.get("rv") is not None:
        out.append(f"Volume {'at the touch' if p.get('rv_basis') == 'touch' else 'approaching the zone'}: "
                   f"{p['rv']:.1f}x normal for this time of day")
    if p.get("depart") is not None:
        extra = " · inside a bigger opposing zone" if p.get("nested") else ""
        out.append(f"Zone left by {p['depart']:.1f} ATR, formed about {fmt_age(zone_age_min(p))} of trading ago{extra}")
    if p.get("react"):
        out.append({"rejected": "Rejection candle printed at the zone",
                    "touched": "Price tagged the zone, no rejection candle yet",
                    "waiting": "Price has not reached the zone yet"}[p["react"]])
    if p.get("htf"):
        out.append("1h structure: " + {"up": "higher highs and higher lows", "down": "lower highs and lower lows",
                                       "range": "sideways"}[p["htf"]])
    if p.get("us"):
        out.append("Spread: " + ("unknown" if p.get("spr") is None else f"{p['spr']:.3f}%"))
        if p.get("news_ok") is False:
            out.append("Headlines could not be read, so the headline check was skipped for this one")
        now = datetime.now(timezone.utc)
        recent = [(t, ti) for t, ti in (p.get("news") or []) if 0 <= (now - t).total_seconds() < 86400]
        if recent:
            t, ti = max(recent)
            out.append(f"Latest headline ({(now - t).total_seconds() / 3600:.1f}h ago): \"{ti[:100]}\"")
        soon = [d for d in (p.get("earn") or []) if 0 <= (d - now.astimezone(NY).date()).days <= 3]
        if soon:
            out.append(f"Earnings {soon[0]:%d %b}")
    return out





def zone_age_min(p):
    """How old the zone is in minutes of trading, from its age in bars and the scan bar size."""
    return (p.get("age_bars") or 0) * p.get("bar_min", 15)


def fmt_age(mins):
    if mins < 90:
        return f"{max(mins, 0):.0f} min"
    if mins < 48 * 60:
        return f"{mins / 60:.1f} h"
    return f"{mins / 1440:.1f} days"


def size(p, equity, risk_pct, lev_cap, acct="AUD", f=None, buf_pct=10.0):
    """Position size from the risk amount. `f` is the value of one unit of the quote currency in the
    account currency; None means it could not be read and the numbers are approximate."""
    long = p["side"] == "long"
    height = abs(p["prox"] - p["dist"])
    stop = p["dist"] - height * buf_pct / 100 if long else p["dist"] + height * buf_pct / 100
    ccy = quote_ccy(p)
    approx = f is None
    f = 1.0 if approx else f
    per_unit = abs(p["prox"] - stop) * f
    units = math.floor(equity * risk_pct / 100 / per_unit) if per_unit > 0 else 0
    notional = units * p["prox"] * f
    margin = notional / lev_cap if lev_cap > 0 else float("inf")
    return dict(stop=stop, units=units, notional=notional, margin=margin, approx=approx, ccy=ccy,
                acct=acct, cur="A$" if acct == "AUD" else "$", fx=f,
                margin_pct=margin / equity * 100, lev=notional / equity,
                loss=units * per_unit, gain=units * abs(p["tgt"] - p["prox"]) * f)


# ---------------- results: evaluate, sort, filter ----------------
def sort_key(r, sort_by):
    p = r[0]
    if sort_by == "Reward:risk":
        return (-p["rr"], -p["score"])
    if sort_by == "Closest to entry":
        return (abs(p["px"] - p["prox"]) / p["px"], -p["score"])
    if sort_by == "Market":
        return (p["market"], p["label"], -p["score"])
    return (-p["score"], -p["rr"])


def apply_view(rows, sort_by="Best setup", dir_f="All", min_rr=0.0):
    out = [r for r in rows if r[0]["rr"] >= min_rr and (dir_f == "All" or r[0]["side"] == dir_f.lower())]
    return sorted(out, key=lambda r: sort_key(r, sort_by))


def evaluate(plays, equity, risk_pct, levs, acct, S, ctx, rate_fn, lim=None):
    """One row per play: (play, size, reasons it fails on its numbers, reasons the filters hold it).
    Each play also gets p["checks"], the full pass/fail list shown on its card."""
    rows = []
    for p in plays:
        lev = levs.get(p["market"], 5.0)
        sz = size(p, equity, risk_pct, lev, acct, rate_fn(quote_ccy(p), acct))
        nchecks = number_checks(p, sz, lev, S)
        fchecks = filter_checks(p, S, ctx)
        flt = [d for _, st, d in fchecks if st is False]
        checks = nchecks + fchecks
        if lim:
            halt = list(lim["halt"])
            extra = limit_play_reasons(lim, sz, S)
            flt = halt + flt + extra
            if S.get("f_limits"):
                bad = halt + extra
                checks.append(("Within your trading limits", not bad, "; ".join(bad) or "under every limit"))
        p["checks"] = checks
        rows.append((p, sz, [d for _, st, d in nchecks if st is False], flt))
    return rows


# ---------------- trade journal ----------------
TRADE_FIELDS = ["id", "logged", "status", "tkr", "label", "market", "side", "entry", "stop", "target",
                "units", "risk", "fx", "acct", "rr", "score", "nd", "exit", "closed", "r", "pnl", "notes"]
_FLOATS = {"entry", "stop", "target", "units", "risk", "fx", "rr", "exit", "r", "pnl"}
_INTS = {"score", "nd"}


def new_trade(p, sz, acct, now=None):
    now = now or datetime.now(timezone.utc)
    return dict(id=uuid.uuid4().hex[:8], logged=now.isoformat(), status="open", tkr=p["tkr"],
                label=p.get("label", p["tkr"]), market=p.get("market", ""), side=p["side"],
                entry=float(p["prox"]), stop=float(sz["stop"]), target=float(p["tgt"]),
                units=float(sz["units"]), risk=float(sz["loss"]), fx=float(sz["fx"]), acct=acct,
                rr=float(p["rr"]), score=int(p["score"]), nd=int(p.get("nd", 2)),
                exit=None, closed=None, r=None, pnl=None, notes="")


def trade_r(t, exit_price):
    """Result in R (1R = the distance from entry to stop)."""
    risk_per_unit = abs(t["entry"] - t["stop"])
    if risk_per_unit <= 0:
        return 0.0
    d = 1 if t["side"] == "long" else -1
    return d * (exit_price - t["entry"]) / risk_per_unit


def close_trade(t, exit_price, now=None):
    now = now or datetime.now(timezone.utc)
    d = 1 if t["side"] == "long" else -1
    t.update(status="closed", exit=float(exit_price), closed=now.isoformat(),
             r=round(trade_r(t, exit_price), 3),
             pnl=round(d * (exit_price - t["entry"]) * t["units"] * t["fx"], 2))
    return t


def cancel_trade(t, now=None):
    now = now or datetime.now(timezone.utc)
    t.update(status="cancelled", closed=now.isoformat(), r=None, pnl=None, exit=None)
    return t


def duplicate_open(trades, p):
    return any(t["status"] == "open" and t["tkr"] == p["tkr"] and t["side"] == p["side"]
               and abs(t["entry"] - p["prox"]) <= 1e-9 for t in trades)


def journal_stats(trades):
    """Summary of the closed trades. All zeros when there are none."""
    cl = sorted((t for t in trades if t["status"] == "closed" and t.get("pnl") is not None),
                key=lambda t: t["closed"])
    n = len(cl)
    if not n:
        return dict(n=0, wins=0, losses=0, win_rate=0.0, avg_r=0.0, total_r=0.0, total_pnl=0.0,
                    profit_factor=None, max_dd_r=0.0, avg_win_r=0.0, avg_loss_r=0.0, curve=[])
    wins = [t for t in cl if t["pnl"] > 0]
    losses = [t for t in cl if t["pnl"] < 0]
    gross_w = sum(t["pnl"] for t in wins)
    gross_l = -sum(t["pnl"] for t in losses)
    curve, run, peak, dd = [], 0.0, 0.0, 0.0
    for t in cl:
        run += t["r"]
        peak = max(peak, run)
        dd = max(dd, peak - run)
        curve.append(round(run, 3))
    return dict(n=n, wins=len(wins), losses=len(losses), win_rate=len(wins) / n * 100,
                avg_r=sum(t["r"] for t in cl) / n, total_r=run, total_pnl=sum(t["pnl"] for t in cl),
                profit_factor=(gross_w / gross_l) if gross_l > 0 else None, max_dd_r=dd,
                avg_win_r=(sum(t["r"] for t in wins) / len(wins)) if wins else 0.0,
                avg_loss_r=(sum(t["r"] for t in losses) / len(losses)) if losses else 0.0, curve=curve)


def trades_to_csv(trades):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=TRADE_FIELDS)
    w.writeheader()
    for t in trades:
        w.writerow({k: ("" if t.get(k) is None else t.get(k)) for k in TRADE_FIELDS})
    return buf.getvalue()


def trades_from_csv(text):
    """Parse a journal backup. Rows that are not valid trades are skipped."""
    out = []
    for row in csv.DictReader(io.StringIO(text)):
        try:
            t = {}
            for k in TRADE_FIELDS:
                v = (row.get(k) or "").strip()
                if k in _FLOATS:
                    t[k] = float(v) if v != "" else None
                elif k in _INTS:
                    t[k] = int(float(v)) if v != "" else 0
                else:
                    t[k] = v if v != "" else None
            if t["status"] not in ("open", "closed", "cancelled") or not t["id"] or t["side"] not in ("long", "short"):
                continue
            if None in (t["entry"], t["stop"], t["units"], t["fx"]):
                continue
            # dates must parse, and a closed trade needs its result, or the Journal tab would crash on it
            datetime.fromisoformat(t["logged"])
            if t["status"] != "open":
                datetime.fromisoformat(t["closed"])
            if t["status"] == "closed" and None in (t["r"], t["pnl"], t["exit"]):
                continue
            t["notes"] = t["notes"] or ""
            out.append(t)
        except (ValueError, TypeError):
            continue
    return out


def merge_trades(existing, incoming):
    """Add incoming trades whose id is not already present. Returns (merged list, number added)."""
    have = {t["id"] for t in existing}
    new = [t for t in incoming if t["id"] not in have]
    return existing + new, len(new)


def money(cur, v):
    return f"{'-' if v < 0 else ''}{cur}{abs(v):,.0f}"


def limit_state(trades, S, equity, cur="$", now=None):
    """Trading limits from the journal: daily loss, open trade count, open risk."""
    now = now or datetime.now(timezone.utc)
    today = now.astimezone(SYD).date()
    open_t = [t for t in trades if t["status"] == "open"]
    day_pnl = sum(t["pnl"] for t in trades if t["status"] == "closed" and t.get("pnl") is not None and t.get("closed")
                  and datetime.fromisoformat(t["closed"]).astimezone(SYD).date() == today)
    lim = dict(halt=[], open_n=len(open_t), open_risk=sum(t["risk"] for t in open_t), day_pnl=day_pnl,
               loss_cap=equity * S.get("day_loss_pct", 6.0) / 100,
               risk_cap=equity * S.get("max_open_risk", 6.0) / 100, max_open=int(S.get("max_open", 3)), cur=cur)
    if not S.get("f_limits"):
        return lim
    if day_pnl <= -lim["loss_cap"]:
        lim["halt"].append(f"daily loss limit reached: {money(cur, day_pnl)} today (limit {money(cur, -lim['loss_cap'])}), stop for the day")
    if lim["open_n"] >= lim["max_open"]:
        lim["halt"].append(f"{lim['open_n']} trades already open (limit {lim['max_open']})")
    elif lim["open_risk"] >= lim["risk_cap"]:
        lim["halt"].append(f"open risk {money(cur, lim['open_risk'])} is at your limit of {money(cur, lim['risk_cap'])}")
    return lim


def limit_play_reasons(lim, sz, S):
    if not S.get("f_limits") or not lim or lim["halt"]:
        return []
    if lim["open_risk"] + sz["loss"] > lim["risk_cap"]:
        return [f"taking it would lift open risk to {money(lim['cur'], lim['open_risk'] + sz['loss'])} "
                f"(limit {money(lim['cur'], lim['risk_cap'])})"]
    return []


# ---------------- take tracker ----------------
# The scan is a snapshot: a zone drops off it the moment price reaches the entry line, and every filter is
# judged again on each scan. The tracker keeps each TAKE once it has shown, then walks the candles after it to
# say what price did: not touched yet, filled, target, stopped, ran away without filling, or timed out.
FINAL_STATUSES = ("target", "stopped", "missed", "unclear", "expired", "timeout")
STATUS_TEXT = {"waiting": "Waiting for price to reach the entry", "filled": "Filled, trade is live",
               "timeout": "Filled but neither stop nor target was hit in the time allowed, closed at the last price",
               "target": "Hit the target", "stopped": "Hit the stop",
               "missed": "Ran to the target without filling the entry",
               "unclear": "Unclear, two levels were hit inside one candle", "expired": "Never filled in time"}


def zone_key(p):
    return f"{p['tkr']}|{p['side']}|{p['prox']}"


def latch_takes(log, rows, now=None):
    """Add every TAKE row that is not in the log yet. `log` is {key: record}. Returns the keys added."""
    now = now or datetime.now(timezone.utc)
    day = now.astimezone(SYD).date()
    added = []
    for p, sz, tech, flt in rows:
        if tech or flt:
            continue
        k = f"{zone_key(p)}|{day}"
        if k in log:
            continue
        log[k] = dict(key=k, zone=zone_key(p), tkr=p["tkr"], label=p.get("label", p["tkr"]),
                      market=p.get("market", ""), side=p["side"], entry=float(p["prox"]),
                      stop=float(sz["stop"]), target=float(p["tgt"]), rr=float(p["rr"]), score=int(p["score"]),
                      nd=int(p.get("nd", 2)), bar_min=int(p.get("bar_min", 15)), seen=now.isoformat(),
                      atr=float(p.get("atr15") or 0.0), kind=p.get("kind", ""),
                      seen_bar=p["bar_ts"].isoformat() if p.get("bar_ts") is not None else None,
                      seen_lo=p.get("bar_lo"), seen_hi=p.get("bar_hi"), seen_px=float(p["px"]),
                      status="waiting", fill=None, end=None, r=None, note="")
        added.append(k)
    return added


def _in_index_tz(ts, idx):
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(idx.tz) if idx.tz is not None else ts.tz_convert("UTC").tz_localize(None)


def _finish(rec, status, ts, note=""):
    rec["status"] = status
    rec["end"] = (ts.isoformat() if ts is not None else datetime.now(timezone.utc).isoformat())
    rec["note"] = note
    risk = abs(rec["entry"] - rec["stop"])
    if status == "target" and risk > 0:
        rec["r"] = round(abs(rec["target"] - rec["entry"]) / risk, 2)
    elif status == "stopped":
        rec["r"] = -1.0


def track_take(rec, bars, now=None, expire_min=120.0, fill_depth=0.0):
    """Walk the candles after the take first showed and update `rec` in place.
    `bars` needs High and Low columns and a datetime index. Inside one candle the order of the high and the
    low is unknown, so a candle that reaches both the stop and the target (or the entry and the target) is
    reported as unclear instead of being counted as a win or a loss."""
    if rec["status"] in FINAL_STATUSES:
        return rec
    now = now or datetime.now(timezone.utc)
    long = rec["side"] == "long"
    entry, stop, tgt = rec["entry"], rec["stop"], rec["target"]
    usable = bars is not None and len(bars) > 0 and isinstance(bars.index, pd.DatetimeIndex)
    seen_bar = _in_index_tz(rec["seen_bar"], bars.index) if usable and rec.get("seen_bar") else None
    pairs = zip(bars.index, bars[["Low", "High"]].itertuples(index=False)) if usable else ()
    for ts, row in pairs:
        lo, hi = float(row[0]), float(row[1])
        if seen_bar is not None:
            if ts < seen_bar:
                continue
            if ts == seen_bar:
                # the candle that was forming when the take showed: only a new extreme set after that counts
                if rec.get("seen_lo") is None or rec.get("seen_hi") is None:
                    continue
                lo = lo if lo < rec["seen_lo"] else float("inf")
                hi = hi if hi > rec["seen_hi"] else float("-inf")
        hit_stop = lo <= stop if long else hi >= stop
        ran = hi >= tgt if long else lo <= tgt
        if rec["status"] == "waiting":
            # fill_depth: a limit order is not always filled when price only touches the entry, so a backtest can
            # ask for price to trade a little through it (in price units). The live tracker uses 0.
            touched = lo <= entry - fill_depth if long else hi >= entry + fill_depth
            if touched:
                rec["fill"] = ts.isoformat()
                if ran:
                    _finish(rec, "unclear", ts, "entry and target both inside one candle")
                elif hit_stop:
                    _finish(rec, "stopped", ts)
                else:
                    rec["status"] = "filled"
            elif ran:
                _finish(rec, "missed", ts)
        else:
            if hit_stop and ran:
                _finish(rec, "unclear", ts, "stop and target both inside one candle")
            elif hit_stop:
                _finish(rec, "stopped", ts)
            elif ran:
                _finish(rec, "target", ts)
        if rec["status"] in FINAL_STATUSES:
            return rec
    if rec["status"] == "waiting":
        age = (now - datetime.fromisoformat(rec["seen"])).total_seconds() / 60
        if age > expire_min:
            _finish(rec, "expired", None, f"not touched within {expire_min:.0f} minutes")
    return rec


def why_changed(rec, rows, scanned=None):
    """Why a latched take is not a TAKE in the latest scan any more. None when it still is one."""
    if scanned is not None and rec["tkr"] not in scanned:
        return "not part of the latest scan"
    for p, sz, tech, flt in rows:
        if zone_key(p) == rec["zone"]:
            reasons_ = tech + flt
            return None if not reasons_ else "; ".join(reasons_[:2])
    return "zone left the scan: price reached it, or the zone or its target moved"


def take_stats(recs):
    """Counts and R for the tracked takes."""
    recs = list(recs)
    wins = [r for r in recs if r["status"] == "target"]
    losses = [r for r in recs if r["status"] == "stopped"]
    done = len(wins) + len(losses)
    return dict(n=len(recs), wins=len(wins), losses=len(losses), done=done,
                filled=sum(1 for r in recs if r.get("fill")),
                live=sum(1 for r in recs if r["status"] == "filled"),
                waiting=sum(1 for r in recs if r["status"] == "waiting"),
                missed=sum(1 for r in recs if r["status"] == "missed"),
                expired=sum(1 for r in recs if r["status"] == "expired"),
                unclear=sum(1 for r in recs if r["status"] == "unclear"),
                win_rate=(len(wins) / done * 100) if done else None,
                total_r=sum(r["r"] for r in wins + losses if r.get("r") is not None))


# ---------------- backtest ----------------
# Walk forward through history one step at a time. At each step the engine only sees candles that had already
# closed, builds plays and applies your filters exactly as the live scan does, latches each new TAKE, and then
# follows the candles after it to see what really happened. Nothing from the future is used to pick a take.
BT_WIN = 1500    # intraday bars shown to the engine at each step
BT_WARM = 200    # bars of history needed before the first step


def _naive(idx):
    idx = pd.DatetimeIndex(idx)
    return idx.tz_localize(None) if idx.tz is not None else idx


def _timeout(rec, rest):
    long = rec["side"] == "long"
    last = float(rest.Close.iloc[-1])
    risk = abs(rec["entry"] - rec["stop"])
    rec["status"] = "timeout"
    rec["end"] = rest.index[-1].isoformat()
    rec["r"] = round(((last - rec["entry"]) if long else (rec["entry"] - last)) / risk, 2) if risk > 0 else 0.0


def backtest_instrument(tk, label, market, kind, d15, d1h, dd, S, bar_min=5, days=20, step=12,
                        expire_min=60.0, hold_min=240.0, fill_depth_atr=0.0, start_ts=None):
    """Replay one instrument. Returns (finished take records, error text or None).
    days: how far back to start. step: bars between looks (12 five-minute bars is one look an hour).
    expire_min: an unfilled take is dropped after this long. hold_min: a filled trade is closed at the last
    price if neither stop nor target is hit in this time.
    fill_depth_atr: price must trade this far through the entry to count as filled (0 = a touch fills).
    start_ts: overrides `days` with an exact start time, so two runs can be lined up."""
    S = dict(S)
    S.update(f_events=False, f_news=False, f_earn=False, f_stale=False, f_index=False, f_limits=False)
    ctx = dict(events=[], status="", index=None)
    n = len(d15)
    if n < BT_WARM + 5 or len(d1h) < 10 or len(dd) < 26:
        return [], "not enough data"
    us = market in US_MARKETS
    nv = _naive(d15.index)
    h_end = _naive(d1h.index) + pd.Timedelta(hours=1)        # an hourly bar is only known once the hour ends
    d_day = _naive(dd.index).normalize()
    t0 = pd.Timestamp(start_ts).tz_localize(None) if start_ts is not None else nv[-1] - pd.Timedelta(days=days)
    start = max(BT_WARM, int(nv.searchsorted(t0)))
    per = int(S.get("per_inst", 3))
    exp_bars = max(1, math.ceil(expire_min / bar_min))
    hold_bars = max(1, math.ceil(hold_min / bar_min))
    log, out = {}, []
    for i in range(start, n - 2, max(1, int(step))):
        kh = int(h_end.searchsorted(nv[i], side="right"))
        kd = int(d_day.searchsorted(nv[i].normalize(), side="left"))   # finished days only
        if kh < 10 or kd < 26:
            continue
        w = d15.iloc[max(0, i - BT_WIN + 1): i + 1]
        try:
            plays = build_plays(tk, w, d1h.iloc[:kh], dd.iloc[:kd], None, S["max_dist"], kind=kind,
                                bar_min=bar_min)[:per]
        except Exception:  # noqa: BLE001
            continue
        if not plays:
            continue
        for p in plays:
            p["label"], p["market"] = label, market
            p["us"], p["news"], p["news_ok"], p["earn"] = us, [], True, []
        # money limits are not what is being tested, so size and leverage are made non-binding
        rows = evaluate(plays, 1e7, 1.0, {market: 1e6}, "USD", S, ctx, lambda c, a: 1.0, None)
        ts = d15.index[i].to_pydatetime()
        for k in latch_takes(log, rows, now=ts):
            rec = log[k]
            fut = d15.iloc[i + 1:]
            track_take(rec, fut.iloc[:exp_bars], now=ts + timedelta(minutes=expire_min + 1), expire_min=expire_min,
                       fill_depth=fill_depth_atr * (rec.get("atr") or 0.0))
            if rec["status"] == "filled":
                rest = fut[fut.index > pd.Timestamp(rec["fill"])].iloc[:hold_bars]
                track_take(rec, rest, now=ts)
                if rec["status"] == "filled":
                    if len(rest) < hold_bars:
                        rec["status"] = "open"          # the data ends before the trade resolves
                    else:
                        _timeout(rec, rest)
            rec["tf"] = f"{int(bar_min)}m"
            out.append(rec)
    return out, None


def backtest_net_r(rec, cost_atr=0.0):
    """R made by a filled trade after a trading cost, given in ATR of the scan bars. None when it never traded."""
    if rec["status"] not in ("target", "stopped", "timeout") or rec.get("r") is None:
        return None
    risk = abs(rec["entry"] - rec["stop"])
    cost = cost_atr * (rec.get("atr") or 0.0) / risk if risk > 0 else 0.0
    return float(rec["r"]) - cost


def wilson(k, n, z=1.96):
    """95% range for a win rate of k out of n."""
    if n <= 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - m) / d * 100, (c + m) / d * 100


def backtest_stats(recs, cost_atr=0.0):
    recs = list(recs)
    traded = sorted((r for r in recs if backtest_net_r(r, cost_atr) is not None),
                    key=lambda r: r.get("end") or r["seen"])
    net = [backtest_net_r(r, cost_atr) for r in traded]
    wins = sum(1 for r in traded if r["status"] == "target")
    losses = sum(1 for r in traded if r["status"] == "stopped")
    decided = wins + losses
    pos = sum(x for x in net if x > 0)
    neg = -sum(x for x in net if x < 0)
    curve, c, peak, dd_ = [], 0.0, 0.0, 0.0
    for x in net:
        c += x
        curve.append(c)
        peak = max(peak, c)
        dd_ = max(dd_, peak - c)
    ups = [x for x in net if x > 0]
    downs = [-x for x in net if x <= 0]
    be = (sum(downs) / len(downs)) / ((sum(ups) / len(ups)) + (sum(downs) / len(downs))) * 100 if ups and downs else None
    lo, hi = wilson(wins, decided)
    filled = sum(1 for r in recs if r.get("fill"))
    return dict(
        takes=len(recs), filled=filled, fill_rate=(filled / len(recs) * 100) if recs else None,
        traded=len(traded), wins=wins, losses=losses,
        timeouts=sum(1 for r in traded if r["status"] == "timeout"),
        missed=sum(1 for r in recs if r["status"] == "missed"), expired=sum(1 for r in recs if r["status"] == "expired"),
        unclear=sum(1 for r in recs if r["status"] == "unclear"), open=sum(1 for r in recs if r["status"] == "open"),
        win_rate=(wins / decided * 100) if decided else None, wr_lo=lo, wr_hi=hi, breakeven_wr=be,
        avg_r=(sum(net) / len(net)) if net else None,
        avg_r_gross=(sum(r["r"] for r in traded) / len(traded)) if traded else None,
        total_r=sum(net), profit_factor=(pos / neg) if neg > 0 else None, max_dd=dd_, curve=curve,
        avg_plan_rr=(sum(r["rr"] for r in traded) / len(traded)) if traded else None)


def backtest_breakdown(recs, keyfn, cost_atr=0.0):
    groups = {}
    for r in recs:
        groups.setdefault(keyfn(r), []).append(r)
    rows = []
    for k, g in sorted(groups.items(), key=lambda kv: str(kv[0])):
        s = backtest_stats(g, cost_atr)
        rows.append(dict(group=k, takes=s["takes"], traded=s["traded"], wins=s["wins"], losses=s["losses"],
                         win_rate=s["win_rate"], avg_r=s["avg_r"], total_r=s["total_r"]))
    return rows


def backtest_verdict(s):
    """(kind, plain sentence). Honest about sample size and about whether the win rate clears break-even."""
    n = s["traded"]
    if n < 30:
        return "few", (f"Only {n} trades closed, too few to judge either way. Run a longer period or more markets "
                       "before trusting anything below.")
    avg = s["avg_r"]
    if avg <= 0:
        return "none", (f"No edge shown. These takes lose {abs(avg):.2f}R per trade on average after costs "
                        f"over {n} trades.")
    if s["wr_lo"] is not None and s["breakeven_wr"] is not None and s["wr_lo"] < s["breakeven_wr"]:
        return "unproven", (f"Positive, {avg:+.2f}R per trade over {n} trades, but the win rate could still be below "
                            f"the {s['breakeven_wr']:.0f}% needed to break even. Not proven.")
    return "edge", (f"Positive: {avg:+.2f}R per trade over {n} trades, and the win rate range "
                    f"({s['wr_lo']:.0f}% to {s['wr_hi']:.0f}%) sits above break-even ({s['breakeven_wr']:.0f}%). "
                    "Still only history, and it assumes your orders fill at the zone price.")


# ---------------- alerts (ntfy.sh push notifications) ----------------
TOPIC_RE = re.compile(r"^[A-Za-z0-9_-]{6,64}$")


def valid_topic(topic):
    return bool(TOPIC_RE.match((topic or "").strip()))


def alert_lines(rows, limit=6):
    out = []
    for p, sz, _, _ in rows[:limit]:
        out.append(f"{'BUY' if p['side'] == 'long' else 'SELL SHORT'} {p['label']} @ {fmt(p['prox'], p['nd'])} "
                   f"stop {fmt(sz['stop'], p['nd'])} target {fmt(p['tgt'], p['nd'])} {p['rr']:.1f}R")
    if len(rows) > limit:
        out.append(f"and {len(rows) - limit} more")
    return out


def send_ntfy(topic, title, body, priority="default", timeout=8):
    topic = (topic or "").strip()
    if not valid_topic(topic):
        raise ValueError("Topic must be 6 to 64 letters, numbers, dashes or underscores.")
    req = urllib.request.Request(
        f"https://ntfy.sh/{topic}", data=body.encode("utf-8"), method="POST",
        headers={"Title": title.encode("ascii", "ignore").decode(), "Priority": priority,
                 "Tags": "chart_with_upwards_trend"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status == 200
