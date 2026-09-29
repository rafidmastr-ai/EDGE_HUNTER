"""Causal, scale-free features on M15 decision bars, identical for every symbol.

Every value at decision bar i uses only information known at that bar's close
(``Bars.close_time[i]``). Higher timeframes are aligned to their last *closed* bar.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

import numpy as np

from app.ml_edge.data import SymbolData, resample_activity
from app.research.ema_macd import macd
from app.research.intraday import DAY, Bars, align_to, atr, daily_levels, ema, resample, rsi

FEATURE_VERSION = 1
DECISION_MINUTES = 15
SESSION_START = 7 * 3600
LAST_ENTRY = 20 * 3600
NEW_YORK = ZoneInfo("America/New_York")
USD_QUOTED = ("EURUSD", "GBPUSD", "AUDUSD", "NZDUSD")


@dataclass(frozen=True)
class FeatureFrame:
    symbol: str
    close_time: np.ndarray  # decision time of each row (M15 bar close)
    x: np.ndarray  # rows x features (float32, NaN allowed)
    names: tuple[str, ...]
    atr: np.ndarray  # ATR14(M15) at the decision bar (label/barrier scale)
    tradable: np.ndarray  # inside the decision window with a valid ATR

    def select(self, mask: np.ndarray) -> "FeatureFrame":
        return FeatureFrame(self.symbol, self.close_time[mask], self.x[mask], self.names, self.atr[mask], self.tradable[mask])


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    finite = np.nan_to_num(values)
    csum = np.cumsum(np.r_[0.0, finite])
    count = np.cumsum(np.r_[0, np.isfinite(values).astype(int)])
    if len(values) >= window:
        n = count[window:] - count[:-window]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[window - 1 :] = np.where(n > window // 2, (csum[window:] - csum[:-window]) / n, np.nan)
    return out


def rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    mean = rolling_mean(values, window)
    sq = rolling_mean(values**2, window)
    return np.sqrt(np.clip(sq - mean**2, 0, None))


def lag(values: np.ndarray, k: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    if k < len(values):
        out[k:] = values[:-k]
    return out


def efficiency_ratio(close: np.ndarray, n: int) -> np.ndarray:
    change = np.abs(close - lag(close, n))
    path = rolling_mean(np.abs(np.diff(close, prepend=np.nan)), n) * n
    with np.errstate(invalid="ignore", divide="ignore"):
        return change / path


def vol_normalised_return(close: np.ndarray, k: int, vol_window: int) -> np.ndarray:
    log_close = np.log(close)
    step = np.diff(log_close, prepend=np.nan)
    sigma = rolling_std(step, vol_window)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (log_close - lag(log_close, k)) / (sigma * np.sqrt(k))


def same_slot_zscore(values: np.ndarray, slot: np.ndarray, lookback: int = 20) -> np.ndarray:
    """z-score of each value vs the previous ``lookback`` values in the same time-of-day slot."""
    out = np.full(len(values), np.nan)
    for s in np.unique(slot):
        idx = np.flatnonzero(slot == s)
        v = values[idx]
        prev_mean = lag(rolling_mean(v, lookback), 1)
        prev_std = lag(rolling_std(v, lookback), 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[idx] = (v - prev_mean) / prev_std
    return out


def _ny_dst(close_time: np.ndarray) -> np.ndarray:
    days = np.unique(close_time // DAY)
    flags = {}
    for day in days.tolist():
        date = datetime.fromtimestamp(day * DAY, tz=timezone.utc).date()
        local = datetime.combine(date, time(12, 0), tzinfo=NEW_YORK)
        flags[day] = 1.0 if local.dst() else 0.0
    return np.array([flags[d] for d in (close_time // DAY).tolist()])


def own_features(data: SymbolData) -> tuple[Bars, dict[str, np.ndarray], np.ndarray]:
    """Single-symbol features on M15 bars; returns (bars, features, atr15)."""
    m1 = data.m1
    b = resample(m1, DECISION_MINUTES)
    h1, h4, d1 = resample(m1, 60), resample(m1, 240), resample(m1, 1440)
    c = b.close
    a15 = atr(b, 14)
    a_h1, a_h4, a_d1 = atr(h1, 14), atr(h4, 14), atr(d1, 14)
    a_d1_at = align_to(b, d1, a_d1)
    a_h1_at = align_to(b, h1, a_h1)
    a_h4_at = align_to(b, h4, a_h4)
    f: dict[str, np.ndarray] = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for k in (1, 4, 16, 64):
            f[f"ret_m15_{k}"] = vol_normalised_return(c, k, 96)
        f["ret_h4_1"] = align_to(b, h4, vol_normalised_return(h4.close, 1, 30))
        f["ret_d1_1"] = align_to(b, d1, vol_normalised_return(d1.close, 1, 20))
        f["ret_d1_5"] = align_to(b, d1, vol_normalised_return(d1.close, 5, 20))
        f["atr_m15_over_d1"] = a15 / a_d1_at
        step = np.diff(np.log(c), prepend=np.nan)
        f["rv_1d_over_20d"] = rolling_std(step, 96) / rolling_std(step, 96 * 20)
        log_atr_pct = np.log(a15 / c)
        f["vol_level_z"] = (log_atr_pct - rolling_mean(log_atr_pct, 96 * 60)) / rolling_std(log_atr_pct, 96 * 60)
        f["er_h1_24"] = align_to(b, h1, efficiency_ratio(h1.close, 24))
        f["er_d1_10"] = align_to(b, d1, efficiency_ratio(d1.close, 10))
        f["er_m15_16"] = efficiency_ratio(c, 16)
        hi8 = np.array([np.nan] * len(h1))
        lo8 = np.array([np.nan] * len(h1))
        if len(h1) >= 8:
            from numpy.lib.stride_tricks import sliding_window_view

            hi8[7:] = sliding_window_view(h1.high, 8).max(axis=1)
            lo8[7:] = sliding_window_view(h1.low, 8).min(axis=1)
        f["h1_range8_over_atr"] = align_to(b, h1, (hi8 - lo8) / a_h1)
        for span in (20, 50, 200):
            f[f"dist_ema{span}_m15"] = (c - ema(c, span)) / a15
        for span in (20, 50, 200):
            f[f"dist_ema{span}_h1"] = align_to(b, h1, (h1.close - ema(h1.close, span)) / a_h1)
        for span in (50, 200):
            f[f"dist_ema{span}_h4"] = align_to(b, h4, (h4.close - ema(h4.close, span)) / a_h4)
        e200 = ema(c, 200)
        f["ema200_slope_m15"] = (e200 - lag(e200, 20)) / a15
        f["rsi_m15"] = rsi(c, 14)
        f["rsi_h1"] = align_to(b, h1, rsi(h1.close, 14))
        line, sig = macd(c)
        f["macd_hist_m15"] = (line - sig) / a15
        # today's range so far (UTC day), known at the bar close
        day = b.open_time // DAY
        starts = np.flatnonzero(np.r_[True, day[1:] != day[:-1]])
        day_high = np.empty(len(b))
        day_low = np.empty(len(b))
        for s, e in zip(starts, np.r_[starts[1:], len(b)]):
            day_high[s:e] = np.maximum.accumulate(b.high[s:e])
            day_low[s:e] = np.minimum.accumulate(b.low[s:e])
        f["day_position"] = (c - day_low) / (day_high - day_low)
        f["day_range_over_atr_d1"] = (day_high - day_low) / a_d1_at
        levels = daily_levels(m1, b)
        f["dist_pdh"] = (c - levels["pdh"]) / a15
        f["dist_pdl"] = (c - levels["pdl"]) / a15
        f["dist_asia_high"] = (c - levels["asia_high"]) / a15
        f["dist_asia_low"] = (c - levels["asia_low"]) / a15
        close_sod = b.close_time % DAY
        f["hour_sin"] = np.sin(2 * np.pi * close_sod / DAY)
        f["hour_cos"] = np.cos(2 * np.pi * close_sod / DAY)
        f["day_of_week"] = ((b.close_time // DAY + 3) % 7).astype(float)
        f["minutes_since_0700"] = (close_sod - SESSION_START) / 60.0
        f["us_dst"] = _ny_dst(b.close_time)
        volume, spread = resample_activity(data, DECISION_MINUTES)
        slot = (b.open_time % DAY) // (DECISION_MINUTES * 60)
        f["tick_volume_z"] = same_slot_zscore(np.log1p(volume), slot)
        f["spread_z"] = (spread - rolling_mean(spread, 96 * 20)) / rolling_std(spread, 96 * 20)
    return b, f, a15


def build_all(symbols: dict[str, SymbolData]) -> dict[str, FeatureFrame]:
    """Features for every symbol, including cross-market features shared by all."""
    base = {name: own_features(data) for name, data in symbols.items()}
    # cross-market series known at each M15 close (vol-normalised 4- and 16-bar returns)
    cross: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name, (bars, feats, _) in base.items():
        cross[name] = (bars.close_time, feats["ret_m15_4"], feats["ret_m15_16"])

    def at(name: str, times: np.ndarray, column: int) -> np.ndarray:
        source_time, *series = cross[name]
        index = np.searchsorted(source_time, times, "right") - 1
        out = np.full(len(times), np.nan)
        ok = index >= 0
        # stale if the other market's last close is more than 1 hour old
        ok &= np.where(ok, times - source_time[np.clip(index, 0, None)] <= 3600, False)
        out[ok] = series[column - 1][index[ok]]
        return out

    frames: dict[str, FeatureFrame] = {}
    for name, (bars, feats, a15) in base.items():
        times = bars.close_time
        usd = [n for n in USD_QUOTED if n in cross]
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if usd:
                feats["usd_strength_4"] = -np.nanmean(np.vstack([at(n, times, 1) for n in usd]), axis=0)
                feats["usd_strength_16"] = -np.nanmean(np.vstack([at(n, times, 2) for n in usd]), axis=0)
            risk = [n for n in ("AUDUSD", "EURJPY") if n in cross]
            if risk:
                feats["risk_on_4"] = np.nanmean(np.vstack([at(n, times, 1) for n in risk]), axis=0)
            if "XAUUSD" in cross:
                feats["gold_ret_4"] = at("XAUUSD", times, 1)
        feats["quote_is_usd"] = np.full(len(times), 1.0 if name.endswith("USD") else 0.0)
        names = tuple(sorted(feats))
        x = np.column_stack([feats[n] for n in names]).astype(np.float32)
        x[~np.isfinite(x)] = np.nan
        sod = times % DAY
        tradable = (sod > SESSION_START) & (sod <= LAST_ENTRY) & np.isfinite(a15) & (a15 > 0)
        frames[name] = FeatureFrame(name, times, x, names, a15, tradable)
    return frames


__all__ = ["DECISION_MINUTES", "FEATURE_VERSION", "FeatureFrame", "build_all", "own_features"]
