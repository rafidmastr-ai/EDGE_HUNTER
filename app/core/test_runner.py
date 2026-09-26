"""Project test runner used by the EDGE HUNTER application entry point."""

from __future__ import annotations

import unittest
from pathlib import Path


def run_project_tests() -> int:
    """Run the full unittest suite and return a process-style exit code."""
    project_root = Path(__file__).resolve().parents[2]
    tests_path = project_root / "tests"

    print("=" * 60)
    print("          EDGE HUNTER - AUTOMATED TESTS")
    print("=" * 60)
    print(f"Test directory: {tests_path}")
    print()

    loader = unittest.TestLoader()
    suite = loader.discover(str(tests_path))
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)

    print()
    print("-" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failed: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("RESULT: ALL TESTS PASSED")
        print("=" * 60)
        return 0

    print("RESULT: TESTS FAILED")
    print("=" * 60)
    return 1
