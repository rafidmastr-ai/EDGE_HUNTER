"""Retrain the EDGE ML models and publish the enabled ones to the folder the app reads.

Steps:
  1. train   variant A  -> scripts/ml_edge_research.py        -> data/models/edge_ml/<SYM>
             variant B1 -> scripts/ml_edge_v2.py --variants B1 -> data/models/edge_ml_v2/B1/<SYM>
  2. publish only models whose manifest says "enabled": true to models/edge_ml/{A,B1}/<SYM>
             (+ B1 AUDJPY/CADJPY = the B1 EURJPY model applied unchanged), after a backup of
             models/edge_ml; every model is loaded (checksum) before the live folder is replaced,
             so a failed run never leaves the app without models.

Run from the project folder (Windows):

    .\\.venv\\Scripts\\python.exe scripts\\retrain_edge_ml.py                     # train + publish
    .\\.venv\\Scripts\\python.exe scripts\\retrain_edge_ml.py --publish-only      # publish what is already trained
    .\\.venv\\Scripts\\python.exe scripts\\retrain_edge_ml.py --dry-run           # show what would change
    .\\.venv\\Scripts\\python.exe scripts\\retrain_edge_ml.py --dev-end 2025-06-30  # newer training dates

Then restart the app (scripts/run_web.py).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SOURCES = {  # variant -> folder the training script writes to
    "A": PROJECT_ROOT / "data" / "models" / "edge_ml",
    "B1": PROJECT_ROOT / "data" / "models" / "edge_ml_v2" / "B1",
}
LIVE_DIR = PROJECT_ROOT / "models" / "edge_ml"
YEN_COPIES = {"AUDJPY": "EURJPY", "CADJPY": "EURJPY"}  # B1 target -> B1 source model
KEEP_FILES = ("README.md", "performance.json")
TRAIN_COMMANDS = {
    "A": ["scripts/ml_edge_research.py"],
    "B1": ["scripts/ml_edge_v2.py", "--variants", "B1"],
}


@dataclass
class PublishReport:
    published: list[str] = field(default_factory=list)  # "B1/EURJPY  symbol keep=0.05"
    skipped: list[str] = field(default_factory=list)  # disabled models
    removed: list[str] = field(default_factory=list)  # live before, not any more
    backup: Path | None = None


def _manifest(folder: Path) -> dict:
    return json.loads((folder / "manifest.json").read_text(encoding="utf-8"))


def _describe(key: str, manifest: dict) -> str:
    return f"{key:<11} {manifest['architecture']:<8} keep={manifest['keep']}"


def plan_publish(sources: dict[str, Path], yen_copies: bool = True) -> tuple[dict[str, Path], list[str]]:
    """{"A/AUDUSD": source folder, ...} of enabled models, and the disabled ones."""
    chosen: dict[str, Path] = {}
    skipped: list[str] = []
    for variant, base in sources.items():
        for folder in sorted(p for p in Path(base).glob("*") if (p / "manifest.json").exists()):
            key = f"{variant}/{folder.name}"
            if _manifest(folder).get("enabled"):
                chosen[key] = folder
            else:
                skipped.append(key)
    if yen_copies:
        for target, source in YEN_COPIES.items():
            if f"B1/{source}" in chosen and f"B1/{target}" not in chosen:
                chosen[f"B1/{target}"] = chosen[f"B1/{source}"]
    return chosen, skipped


def _write_yen_copy(folder: Path, target: str, source: str) -> None:
    manifest = _manifest(folder)
    manifest["symbol"] = target
    info = dict(manifest.get("info") or {})
    info["cross_pair"] = (f"B1 {source} model applied unchanged to {target} (no training on {target}); "
                          f"see docs/results/NEW_PAIR_{target}.md")
    info["source_model"] = f"models/edge_ml/B1/{source}"
    manifest["info"] = info  # model.joblib is unchanged, so model_sha256 stays valid
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")


def publish(sources: dict[str, Path] = SOURCES, live_dir: Path = LIVE_DIR, *, dry_run: bool = False,
            backup: bool = True, yen_copies: bool = True, now: datetime | None = None) -> PublishReport:
    from app.ml_edge.model import SymbolModel

    live_dir = Path(live_dir)
    chosen, skipped = plan_publish(sources, yen_copies)
    report = PublishReport(skipped=skipped)
    if not chosen:
        raise RuntimeError("لا يوجد أي نموذج مفعّل في مجلدات التدريب؛ لم يُغيَّر شيء.")
    before = {f"{p.parent.name}/{p.name}" for p in live_dir.glob("*/*") if p.is_dir()} if live_dir.exists() else set()
    report.removed = sorted(before - set(chosen))
    report.published = [_describe(k, _manifest(v)) for k, v in sorted(chosen.items())]
    if dry_run:
        return report

    staging = live_dir.with_name(live_dir.name + ".new")
    shutil.rmtree(staging, ignore_errors=True)
    try:
        for key, source in chosen.items():
            target = staging / key
            shutil.copytree(source, target)
            variant, symbol = key.split("/")
            if source.name != symbol:
                _write_yen_copy(target, symbol, source.name)
        for name in KEEP_FILES:
            if (live_dir / name).exists():
                shutil.copy2(live_dir / name, staging / name)
        for key in chosen:  # same check the app does at start-up
            model = SymbolModel.load(staging / key)
            if model.symbol != key.split("/")[1]:
                raise ValueError(f"{key}: manifest symbol is {model.symbol}")
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    if backup and live_dir.exists():
        stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d_%H%M%S")
        report.backup = live_dir.with_name(f"{live_dir.name}_backup_{stamp}")
        shutil.copytree(live_dir, report.backup)
    old = live_dir.with_name(live_dir.name + ".old")
    shutil.rmtree(old, ignore_errors=True)
    if live_dir.exists():
        live_dir.rename(old)
    staging.rename(live_dir)
    shutil.rmtree(old, ignore_errors=True)
    return report


def date_env(dev_end: str, oos_end: str | None = None) -> dict[str, str]:
    """Environment for the training scripts when the study dates move (see walkforward.study_dates)."""
    dev = datetime.fromisoformat(dev_end).replace(tzinfo=timezone.utc)
    oos = datetime.fromisoformat(oos_end).replace(tzinfo=timezone.utc) if oos_end else dev + timedelta(days=410)
    if oos <= dev:
        raise ValueError("--oos-end must be after --dev-end")
    forward = oos + timedelta(days=243)
    return {
        "EDGE_HUNTER_ML_DEV_END": dev.strftime("%Y-%m-%dT%H:%M:%S"),
        "EDGE_HUNTER_ML_OOS_END": oos.strftime("%Y-%m-%d"),
        "EDGE_HUNTER_ML_NEW_FORWARD": forward.strftime("%Y-%m-%d"),
        "EDGE_HUNTER_ML_DATA_END_REQUIRED": (forward + timedelta(days=30)).strftime("%Y-%m-%d"),
    }


def train(variants: list[str], env: dict[str, str]) -> int:
    for variant in variants:
        command = [sys.executable, *TRAIN_COMMANDS[variant]]
        print(f"\n=== تدريب {variant}: {' '.join(command[1:])} (قد يستغرق وقتاً طويلاً) ===", flush=True)
        code = subprocess.call(command, cwd=PROJECT_ROOT, env={**os.environ, **env})
        if code == 2 and variant == "B1":
            print("البيانات غير مكتملة (انظر الرسالة أعلاه): أضف ملفات M1 الناقصة إلى data/raw/<الرمز>/ "
                  "أو استخدم --dev-end بتاريخ أقدم. لم يُنشر أي نموذج.")
            return code
        if code != 0:
            print(f"فشل تدريب {variant} (رمز الخروج {code}). لم يُنشر أي نموذج.")
            return code
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Retrain EDGE ML models and publish the enabled ones")
    parser.add_argument("--publish-only", action="store_true", help="skip training, publish the models already trained")
    parser.add_argument("--dry-run", action="store_true", help="show what would be published; change nothing")
    parser.add_argument("--variants", default="A,B1", help="variants to train (default A,B1)")
    parser.add_argument("--dev-end", help="end of the training/validation period, e.g. 2025-06-30 (default: as published, 2023-11-14)")
    parser.add_argument("--oos-end", help="end of the out-of-sample test (default: dev-end + 410 days)")
    parser.add_argument("--no-backup", action="store_true", help="do not back up models/edge_ml")
    parser.add_argument("--no-yen-copies", action="store_true", help="do not publish B1 AUDJPY/CADJPY copies of B1 EURJPY")
    args = parser.parse_args(argv)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in TRAIN_COMMANDS]
    if unknown:
        parser.error(f"unknown variants: {unknown}")
    env = date_env(args.dev_end, args.oos_end) if args.dev_end else {}
    if env:
        print("تواريخ التدريب: " + ", ".join(f"{k}={v}" for k, v in env.items()))
    if not args.publish_only and not args.dry_run:
        code = train(variants, env)
        if code:
            return code

    try:
        report = publish(dry_run=args.dry_run, backup=not args.no_backup, yen_copies=not args.no_yen_copies)
    except Exception as exc:
        print(f"فشل النشر، ولم يتغيّر مجلد models/edge_ml: {exc}")
        return 1
    print("\n" + ("(تجربة فقط — لم يتغيّر شيء) " if args.dry_run else "") + f"النماذج المنشورة ({len(report.published)}):")
    for line in report.published:
        print("  " + line)
    if report.skipped:
        print("غير مفعّلة (لم تُنشر): " + ", ".join(report.skipped))
    if report.removed:
        print("كانت منشورة وأُزيلت الآن: " + ", ".join(report.removed))
    if report.backup:
        print(f"نسخة احتياطية من النماذج السابقة: {report.backup}")
    if not args.dry_run:
        print("أعد تشغيل التطبيق (scripts/run_web.py) ليستخدم النماذج الجديدة.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
