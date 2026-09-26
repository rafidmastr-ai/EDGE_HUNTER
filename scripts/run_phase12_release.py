"""Run the complete Phase 12 release checks and generate a machine-readable report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
REPORT_DIR = ROOT / "reports"


def run_tests(verbosity: int = 1) -> tuple[unittest.TestResult, str]:
    loader = unittest.TestLoader()
    suite = loader.discover(str(ROOT / "tests"))
    stream = sys.stdout
    runner = unittest.TextTestRunner(stream=stream, verbosity=verbosity)
    result = runner.run(suite)
    return result, ""


def run_compile_check() -> dict:
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "compileall", "-q", "app", "config", "scripts", "tests"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return {
        "name": "clean_installation_compile_check",
        "passed": completed.returncode == 0,
        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        "stderr": completed.stderr.strip(),
    }


def build_report(result: unittest.TestResult, compile_result: dict) -> dict:
    passed = result.testsRun - len(result.failures) - len(result.errors)
    return {
        "phase": "12",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "test_suite": {
            "tests_run": result.testsRun,
            "passed": passed,
            "failed": len(result.failures),
            "errors": len(result.errors),
            "successful": result.wasSuccessful(),
        },
        "preflight": compile_result,
        "release_readiness": (
            "AUTOMATED_PASS_BROWSER_QA_PENDING"
            if result.wasSuccessful() and compile_result["passed"]
            else "BLOCKED"
        ),
        "defects_fixed_in_phase": [
            "Carried forward and enforced the httpx2 dependency declaration in release manifests.",
            "Added an end-to-end regression covering trial expiry, server-side access denial, subscription-code redemption and restored analysis access.",
            "Added final static frontend checks for RTL, responsive breakpoints, copy controls and absence of embedded screenshots/market numbers.",
        ],
        "known_limitations": [
            "Browser visual QA requires a local Chromium/Playwright environment; static visual contracts are included in the automated suite.",
            "Current historical UI datasets expose XAUUSD and GBPUSD files, while the Phase 09 UI contract still lists EURUSD; missing-symbol handling remains an explicit data-unavailable state until the live provider phase changes the symbol source.",
        ],
        "failures": [str(item[0].id()) for item in result.failures],
        "errors": [str(item[0].id()) for item in result.errors],
    }


def write_reports(report: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "phase12_release_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    suite = report["test_suite"]
    pre = report["preflight"]
    md = [
        "# EDGE HUNTER — Phase 12 Release Readiness",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "## Automated test matrix",
        "",
        "| Area | Coverage |",
        "|---|---|",
        "| Clean installation | compileall + dependency manifest contract |",
        "| Configuration | environment contract and production constraints |",
        "| Database | migrations, isolation, public exposure guard |",
        "| CSV ingestion | real API analysis path on synthetic OHLC CSV |",
        "| Analysis | feature/signal integration regression |",
        "| Strategies | existing Classic/SMC/ICT suite |",
        "| Backtest | existing lifecycle/metric suite |",
        "| Research/Optimization | existing regression suites |",
        "| Confidence/Signal | existing selector/scoring/calibration suites |",
        "| API | auth + analysis + error contract |",
        "| Authentication/Subscription | full user journey |",
        "| Admin | secure login + code generation + dashboard |",
        "| Frontend | RTL/responsive/static contract + JS syntax |",
        "| Deployment/Security | existing Phase 11 security suite + public DB guard |",
        "| End-to-end | register → trial → analyze → expire → redeem → analyze |",
        "",
        "## Results",
        "",
        f"- Tests run: **{suite['tests_run']}**",
        f"- Passed: **{suite['passed']}**",
        f"- Failed: **{suite['failed']}**",
        f"- Errors: **{suite['errors']}**",
        f"- Compile check: **{'PASS' if pre['passed'] else 'FAIL'}**",
        f"- Release readiness: **{report['release_readiness']}**",
        "",
        "## Defects fixed",
        "",
    ]
    md.extend(f"- {item}" for item in report["defects_fixed_in_phase"])
    md.extend(["", "## Known limitations", ""])
    md.extend(f"- {item}" for item in report["known_limitations"])
    if report["failures"] or report["errors"]:
        md.extend(["", "## Failed tests", ""])
        md.extend(f"- `{item}`" for item in report["failures"] + report["errors"])
    (REPORT_DIR / "phase12_release_readiness.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="EDGE HUNTER Phase 12 release checks")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    print("=" * 68)
    print("          EDGE HUNTER - PHASE 12 RELEASE CHECK")
    print("=" * 68)
    compile_result = run_compile_check()
    print(f"CLEAN INSTALLATION / COMPILE: {'PASS' if compile_result['passed'] else 'FAIL'}")
    result, _ = run_tests(verbosity=0 if args.quiet else 1)
    report = build_report(result, compile_result)
    write_reports(report)
    suite = report["test_suite"]
    print()
    print(f"Tests run: {suite['tests_run']}")
    print(f"Passed: {suite['passed']}")
    print(f"Failed: {suite['failed']}")
    print(f"Errors: {suite['errors']}")
    print(f"RESULT: {'PHASE 12 PASS' if report['release_readiness'] != 'BLOCKED' else 'PHASE 12 FAILED'}")
    print(f"Reports: {REPORT_DIR}")
    return 0 if report["release_readiness"] == "AUTOMATED_PASS_BROWSER_QA_PENDING" else 1


if __name__ == "__main__":
    raise SystemExit(main())
