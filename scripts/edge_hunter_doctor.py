"""EDGE HUNTER doctor: why the app shows no analysis / no trade, and how to fix it.

Run from the project folder (same place as main.py):

    python scripts/edge_hunter_doctor.py            # all checks, including one Twelve Data request
    python scripts/edge_hunter_doctor.py --offline  # no network request

Each check prints OK or a problem with the fix (Arabic). Nothing is changed on disk.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LIVE_SYMBOLS = ("AUDUSD", "EURJPY", "EURUSD", "GBPUSD", "NZDUSD", "XAUUSD", "AUDJPY", "CADJPY")
EXPECTED_MODELS = 10
MIN_HISTORY_DAYS = 85
# text that only the current versions of these files contain (copied zip files left out = old code)
CODE_MARKERS = {
    "app/ml_edge/live.py": "_BudgetedProvider",
    "app/ml_edge/live_store.py": "def backfill_history",
    "app/web/analysis_service.py": "_add_model_hint",
    "app/web/app.py": "run_in_threadpool",
    "app/web/static/app.js": "renderRecommendationSource",
    "app/web/static/index.html": 'id="recommendation-source"',
    "app/providers/twelvedata.py": "bulk_max_bars",
    "app/ml_edge/data.py": "def find_m1_files",
}


@dataclass
class Report:
    ok: list[str] = field(default_factory=list)
    problems: list[tuple[str, str]] = field(default_factory=list)  # (problem, fix)

    def good(self, text: str) -> None:
        self.ok.append(text)

    def bad(self, problem: str, fix: str) -> None:
        self.problems.append((problem, fix))


def check_code(root: Path, report: Report) -> None:
    missing = []
    for rel, marker in CODE_MARKERS.items():
        path = root / rel
        if not path.exists() or marker not in path.read_text(encoding="utf-8", errors="ignore"):
            missing.append(rel)
    if missing:
        report.bad("ملفات قديمة أو ناقصة (لم تُنسخ آخر نسخة): " + "، ".join(missing),
                   "نفّذ git pull من الفرع claude/marhaba-wzzbl6 (الأفضل)، أو انسخ كل ملفات آخر zip مع الحفاظ على المسارات، ثم أعد تشغيل التطبيق.")
    else:
        report.good("ملفات البرنامج محدّثة.")


def check_settings(settings, report: Report) -> None:
    from app.providers.factory import build_live_provider

    if settings.data_mode != "live":
        report.bad(f"وضع البيانات = {settings.data_mode} (ليس live)",
                   "ضع EDGE_HUNTER_DATA_MODE=live في ملف .env ثم أعد تشغيل التطبيق.")
    else:
        report.good("وضع البيانات live.")
    provider = build_live_provider(settings)
    if not getattr(provider, "configured", False):
        report.bad("مزوّد Twelve Data غير مفعّل أو المفتاح غير صالح",
                   "في .env: EDGE_HUNTER_LIVE_PROVIDER=twelvedata و EDGE_HUNTER_LIVE_PROVIDER_ENABLED=true "
                   "و EDGE_HUNTER_LIVE_PROVIDER_API_KEY=<مفتاحك> (بدون مسافات أو علامات تنصيص).")
    else:
        report.good("مزوّد Twelve Data مفعّل والمفتاح بصيغة صحيحة.")
    if not settings.edge_ml_enabled:
        report.bad("نماذج EDGE ML معطّلة", "ضع EDGE_HUNTER_EDGE_ML_ENABLED=true في .env.")
    if settings.environment == "test":
        report.bad("EDGE_HUNTER_ENV=test: لا يعمل تحديث النماذج في هذا الوضع", "ضع EDGE_HUNTER_ENV=development في .env.")
    return provider


def check_provider(settings, provider, report: Report) -> None:
    from app.providers.models import LiveProviderError

    if not getattr(provider, "configured", False):
        return
    now = datetime.now(timezone.utc)
    try:
        bars = provider.get_ohlc("EURJPY", "M15", now - timedelta(days=4), now)
        report.good(f"اتصال Twelve Data يعمل ({len(bars)} شمعة EUR/JPY M15).")
    except LiveProviderError as exc:
        text = str(exc).lower()
        if "credit" in text:
            report.bad("نفد رصيد Twelve Data اليومي", "انتظر حتى 00:00 UTC (03:00 بغداد)، واجعل EDGE_HUNTER_EDGE_ML_REFRESH_MINUTES=30 أو 60.")
        elif exc.code in {"provider_401", "provider_403", "provider_http_401", "provider_http_403"}:
            report.bad("مفتاح Twelve Data مرفوض", "انسخ المفتاح الصحيح من حسابك في twelvedata.com إلى EDGE_HUNTER_LIVE_PROVIDER_API_KEY.")
        elif exc.code == "provider_unreachable":
            report.bad("لا يمكن الوصول إلى api.twelvedata.com", "تحقق من الإنترنت أو الجدار الناري أو البروكسي.")
        else:
            report.bad(f"خطأ من Twelve Data: {exc.code} — {exc}", "تحقق من المفتاح والخطة في حسابك.")
        return
    try:  # credits used today (free endpoint)
        request = urllib.request.Request(f"{settings.live_provider_url or 'https://api.twelvedata.com'}/api_usage",
                                         headers={"Authorization": f"apikey {settings.live_provider_api_key}"})
        with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310 - configured provider URL
            usage = json.loads(response.read().decode("utf-8"))
        used, limit = usage.get("daily_usage"), usage.get("plan_daily_limit")
        if used is not None and limit:
            line = f"رصيد Twelve Data اليوم: {used} من {limit} طلب."
            if used >= 0.9 * limit:
                report.bad(line + " (قارب على النفاد)", "قلّل EDGE_HUNTER_EDGE_ML_REFRESH_MINUTES (مثلاً 60) و EDGE_HUNTER_EDGE_ML_DAILY_REQUEST_BUDGET.")
            else:
                report.good(line)
    except Exception:
        pass


def check_models(settings, report: Report) -> None:
    from app.ml_edge.model import SymbolModel

    base = Path(settings.edge_ml_models_dir)
    loaded, errors = 0, []
    for folder in sorted(base.glob("*/*")):
        if not folder.is_dir():
            continue
        try:
            SymbolModel.load(folder)
            loaded += 1
        except Exception as exc:
            errors.append(f"{folder.parent.name}/{folder.name}: {exc}")
    if loaded < EXPECTED_MODELS:
        report.bad(f"النماذج المحمّلة {loaded} من {EXPECTED_MODELS} في {base}" + (f" ({'; '.join(errors)[:300]})" if errors else ""),
                   "انسخ مجلد models/edge_ml كاملاً من GitHub (ملفات model.joblib و manifest.json لكل نموذج).")
    else:
        report.good(f"النماذج: {loaded} نماذج تُحمَّل بنجاح.")


def _csv_span_days(files: list[Path]) -> float | None:
    def stamp(line: str):
        return datetime.fromisoformat(line.split(",", 1)[0][:19])

    try:
        with files[0].open(encoding="utf-8") as handle:
            next(handle)
            first = stamp(next(handle))
        with files[-1].open("rb") as handle:
            handle.seek(0, 2)
            handle.seek(max(0, handle.tell() - 400))
            last = stamp(handle.read().decode("utf-8", "ignore").strip().splitlines()[-1])
        return (last - first).total_seconds() / 86400
    except Exception:
        return None


def check_history(settings, report: Report, raw_dir: Path) -> None:
    from app.ml_edge.data import find_m1_files

    store = Path(settings.edge_ml_store_dir)
    missing = []
    for symbol in LIVE_SYMBOLS:
        npz = store / f"{symbol}_M1.npz"
        if npz.exists():
            import numpy as np

            t = np.load(npz)["t"]
            days = float(t[-1] - t[0]) / 86400 if len(t) else 0.0
            last = datetime.fromtimestamp(int(t[-1]), tz=timezone.utc) if len(t) else None
            if days >= MIN_HISTORY_DAYS:
                continue
            missing.append(f"{symbol} (المخزن {days:.0f} يوماً)")
            continue
        files = find_m1_files(raw_dir, symbol)
        span = _csv_span_days(files) if files else None
        if not files or (span is not None and span < MIN_HISTORY_DAYS):
            missing.append(f"{symbol} ({'لا توجد ملفات CSV' if not files else f'{span:.0f} يوماً فقط'})")
    if missing:
        report.bad("تاريخ M1 غير كافٍ للنماذج: " + "، ".join(missing),
                   f"الأسرع: انسخ مجلدات data/raw/<الرمز>/ (ملفات *_M1_*.csv و SOURCE.json) من GitHub ثم أعد تشغيل التطبيق. "
                   f"وإلا يجلبه التطبيق تلقائياً من Twelve Data (نحو 4 طلبات لكل زوج، مرة واحدة، 5–10 دقائق).")
    else:
        report.good(f"تاريخ M1 كافٍ (≥ {MIN_HISTORY_DAYS} يوماً) للرموز الثمانية.")


def run(root: Path = PROJECT_ROOT, *, offline: bool = False, settings=None, raw_dir: Path | None = None) -> Report:
    from config.config_hunter import load_settings

    report = Report()
    check_code(root, report)
    settings = settings or load_settings()
    provider = check_settings(settings, report)
    if not offline:
        check_provider(settings, provider, report)
    check_models(settings, report)
    check_history(settings, report, raw_dir or root / "data" / "raw")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="EDGE HUNTER doctor")
    parser.add_argument("--offline", action="store_true", help="skip the Twelve Data request")
    args = parser.parse_args()
    report = run(offline=args.offline)
    print("=" * 70)
    print("فحص EDGE HUNTER")
    print("=" * 70)
    for line in report.ok:
        print(f"[OK]  {line}")
    for problem, fix in report.problems:
        print(f"[!!]  {problem}\n      الحل: {fix}")
    print("-" * 70)
    print("لا توجد مشاكل." if not report.problems else f"عدد المشاكل: {len(report.problems)} — أصلحها بالترتيب ثم أعد تشغيل التطبيق وحدّث المتصفح (Ctrl+F5).")
    return 1 if report.problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
