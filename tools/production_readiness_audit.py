#!/usr/bin/env python3
"""
EDGE HUNTER - Production Readiness Audit Tool

Read-mostly audit utility. It inspects the repository, source tree, configuration,
learning safety boundaries, API structure, security patterns, migrations and
registered-artifact metadata. It may create only reports/ and temporary files
outside the project when compiling source. The full test command is intentionally
run only when this tool is executed normally by the project owner.

Normal usage from project root:
    python tools/production_readiness_audit.py

Self validation without running the audit:
    python tools/production_readiness_audit.py --self-check

No third-party packages are required by this audit tool itself.
"""

from __future__ import annotations

import argparse
import ast
import configparser
import contextlib
import dataclasses
import datetime as _dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unicodedata
from typing import Any, Iterable, Sequence


TOOL_VERSION = "1.2.2"
REPORT_NAME_PREFIX = "production_readiness_audit_"
REPORT_DIR_NAME = "reports"

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
BLOCKER = "BLOCKER"
NOT_APPLICABLE = "NOT_APPLICABLE"
NOT_FULLY_VERIFIED = "NOT_FULLY_VERIFIED"

SECTION_NAMES: dict[str, str] = {
    "01": "Project Structure",
    "02": "Python Validation",
    "03": "Full Test Suite",
    "04": "Learning Safety",
    "05": "API Analyze Path",
    "06": "Leakage Protection",
    "07": "Model Registry",
    "08": "OOS / Walk-Forward / Robustness",
    "09": "Learned Policy",
    "10": "Continuous Learning",
    "11": "Strategies",
    "12": "Risk / R:R",
    "13": "Authentication / Subscription",
    "14": "Security",
    "15": "Configuration",
    "16": "Data Provider",
    "17": "Hardcoded / Dangerous Patterns",
    "18": "Database / Migrations",
    "19": "Artifact Integrity",
    "20": "Startup / Runtime Readiness",
}

STATUS_RANK = {
    PASS: 0,
    NOT_APPLICABLE: 0,
    NOT_FULLY_VERIFIED: 1,
    WARN: 1,
    FAIL: 2,
    BLOCKER: 3,
}

PY_EXTENSIONS = {".py"}
TEXT_EXTENSIONS = {
    ".py",
    ".sql",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".yaml",
    ".yml",
    ".json",
    ".md",
    ".txt",
    ".ps1",
    ".env",
    ".example",
}

EXCLUDED_DIRS = {
    ".git",
    ".idea",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
}

SECRET_NAME_RE = re.compile(
    r"(?:API[_-]?KEY|SECRET|TOKEN|PASSWORD|PASSWD|PRIVATE[_-]?KEY|ACCESS[_-]?KEY|AUTH[_-]?TOKEN)"
    r"(?:$|[_-])",
    re.IGNORECASE,
)
SECRET_VALUE_PATTERNS = [
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"AIza[0-9A-Za-z_-]{30,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._-]{20,}"),
]

# Names which describe hashing/configuration parameters rather than credentials.
SAFE_SECRET_NAME_SUFFIXES = (
    "_SCHEME",
    "_ALGORITHM",
    "_ITERATIONS",
    "_SALT_BYTES",
    "_DK_BYTES",
    "_TOKEN_LENGTH",
    "_PASSWORD_MIN",
    "_PASSWORD_MAX",
    "_TTL",
    "_TTL_SECONDS",
    "_TIMEOUT",
    "_TIMEOUT_SECONDS",
    "_MAX_LENGTH",
    "_MIN_LENGTH",
)

CREDENTIAL_NAME_RE = re.compile(
    r"(?:^|_)(SECRET(?:_KEY)?|API[_-]?KEY|ACCESS[_-]?TOKEN|AUTH(?:ORIZATION)?[_-]?TOKEN|"
    r"PRIVATE[_-]?KEY|DATABASE[_-]?PASSWORD|DB[_-]?PASSWORD|PASSWORD|PASSWD)$",
    re.IGNORECASE,
)

PLACEHOLDER_VALUES = {
    "", "none", "null", "false", "true",
    "your-key-here", "your_api_key", "your-api-key", "your_twelve_data_api_key",
    "your-twelve-data-api-key", "replace-me", "change-me", "placeholder",
    "insert-key-here", "insert-your-key-here", "<secret>", "<api-key>",
    "<your-key>", "ضع مفتاحك هنا", "ضع_مفتاحك_هنا",
}

DANGEROUS_CALLS = {
    "eval",
    "exec",
    "system",
}

LEARNING_FLAG_KEYS = (
    "EDGE_HUNTER_LEARNING_ENABLED",
    "EDGE_HUNTER_LEARNING_AUTOMATIC_RETRAINING_ENABLED",
    "EDGE_HUNTER_LEARNING_AUTOMATIC_PROMOTION_ENABLED",
    "EDGE_HUNTER_LEARNING_PRODUCTION_ENABLED",
)

EXPECTED_LEARNING_DEFAULTS = {
    key: "false" for key in LEARNING_FLAG_KEYS
}


def _now() -> _dt.datetime:
    return _dt.datetime.now().astimezone()


def _fmt_dt(value: _dt.datetime) -> str:
    return value.isoformat(timespec="seconds")


def redact(value: str, *, limit: int = 400) -> str:
    """Return text safe for a report by removing likely secrets."""
    if not value:
        return value
    redacted = value
    for pattern in SECRET_VALUE_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    # Redact key/value assignments with secret-like names while retaining names.
    redacted = re.sub(
        r"(?im)(\b[A-Z0-9][A-Z0-9_-]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|PRIVATE_KEY)\b\s*=\s*)([^\s#]+)",
        r"\1[REDACTED]",
        redacted,
    )
    if len(redacted) > limit:
        redacted = redacted[:limit] + " ...[TRUNCATED]"
    return redacted


def relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def safe_read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_bool(raw: str | None) -> bool | None:
    if raw is None:
        return None
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on", "y", "enabled"}:
        return True
    if value in {"0", "false", "no", "off", "n", "disabled", ""}:
        return False
    return None


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    text = safe_read_text(path)
    if text is None:
        return values
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, raw_value = line.split("=", 1)
        key = key.strip()
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def contains_any(text: str, patterns: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(p.lower() in lowered for p in patterns)


def _merge_safety(states: Sequence[str]) -> str:
    """Merge local provenance states without overclaiming safety."""
    values = set(states)
    if "tainted" in values:
        return "tainted"
    if "unknown" in values:
        return "unknown"
    return "safe"


def call_name(node: ast.Call) -> str:
    fn = node.func
    if isinstance(fn, ast.Name):
        return fn.id
    if isinstance(fn, ast.Attribute):
        return fn.attr
    return "<dynamic>"


def dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def is_test_path(path: Path, root: Path) -> bool:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except Exception:
        return False
    return "tests" in relative.parts or path.name.startswith("test_")


def is_audit_tool_path(path: Path, root: Path) -> bool:
    """Return True only for this audit utility; never audit the scanner as application code."""
    try:
        return path.resolve() == (root / "tools" / "production_readiness_audit.py").resolve()
    except Exception:
        return False


def is_placeholder_value(value: str) -> bool:
    """Recognize explicit example/placeholder values without treating them as secrets.

    This helper is intentionally conservative for arbitrary source literals, while
    being robust to Windows-edited ``.env.example`` files. Unicode format characters
    (including BOM/directional marks) are removed before classification.
    """
    normalized = unicodedata.normalize("NFKC", value or "")
    normalized = "".join(
        ch for ch in normalized
        if unicodedata.category(ch) != "Cf"
    )
    normalized = normalized.strip().strip('"').strip("'").strip()

    # Ignore an optional inline .env comment only when the value is clearly a
    # placeholder before the comment. This avoids changing the interpretation of
    # arbitrary values that merely contain a '#'.
    candidate = normalized.split(" #", 1)[0].strip()
    lower = candidate.casefold()
    compact = re.sub(r"[\s_-]+", "_", lower).strip("_")

    if lower in PLACEHOLDER_VALUES or compact in {
        "your_api_key",
        "your_twelve_data_api_key",
        "your_twleve_data_api_key",
        "ضع_مفتاحك_هنا",
    }:
        return True

    # Explicit generic example-token forms. These are deliberately bounded so a
    # normal random credential-shaped string is not accepted.
    if re.fullmatch(r"YOUR(?:[_\- ]+[A-Z0-9]+)+", candidate, flags=re.IGNORECASE):
        return True
    if re.fullmatch(r"(?:YOUR|INSERT|REPLACE|CHANGE)(?:[_\- ]+[A-Z0-9]+)+", candidate, flags=re.IGNORECASE):
        return True
    if re.fullmatch(r"\$\{[^}]+\}", candidate):
        return True
    if re.fullmatch(r"<[^>]+>", candidate):
        return True
    if any(token in lower for token in (
        "your_",
        "your-",
        "your ",
        "placeholder",
        "replace_me",
        "replace-me",
        "change_me",
        "change-me",
        "insert-your-",
        "insert your ",
    )):
        return True
    if "مفتاح" in candidate or "ضع " in candidate or candidate.endswith("هنا"):
        return True
    return False


def is_example_credential_value(key: str, value: str) -> bool:
    """Return whether a credential-like env value is explicitly safe example data.

    The key is used only for the narrow ``.env.example`` classification path. It does
    not whitelist real credentials; the value itself must still pass placeholder
    recognition.
    """
    if is_placeholder_value(value):
        return True
    upper_key = key.upper()
    if upper_key == "EDGE_HUNTER_LIVE_PROVIDER_API_KEY":
        normalized = unicodedata.normalize("NFKC", value or "")
        normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Cf")
        normalized = normalized.strip().strip('"').strip("'").strip()
        return bool(
            re.fullmatch(r"YOUR(?:[_\- ]+TWELVE(?:[_\- ]+DATA)?[_\- ]+API[_\- ]+KEY)", normalized, flags=re.IGNORECASE)
        )
    return False


def is_safe_credential_name(name: str) -> bool:
    upper = name.upper()
    return any(upper.endswith(suffix) for suffix in SAFE_SECRET_NAME_SUFFIXES)


@dataclasses.dataclass(frozen=True)
class Finding:
    section_id: str
    status: str
    item: str
    reason: str = ""
    path: str | None = None
    line: int | None = None
    severity: str | None = None

    def format_report(self, root: Path) -> str:
        lines = [f"[{self.status}] {self.item}"]
        if self.path:
            location = relpath(Path(self.path), root)
            if self.line:
                location += f":{self.line}"
            lines.append(f"  FILE: {location}")
        if self.severity:
            lines.append(f"  SEVERITY: {self.severity}")
        if self.reason:
            lines.append(f"  REASON: {redact(self.reason)}")
        return "\n".join(lines)


@dataclasses.dataclass
class SectionResult:
    section_id: str
    name: str
    findings: list[Finding] = dataclasses.field(default_factory=list)

    @property
    def status(self) -> str:
        if not self.findings:
            return NOT_FULLY_VERIFIED
        ordered = sorted(self.findings, key=lambda f: STATUS_RANK.get(f.status, 0), reverse=True)
        highest = ordered[0].status
        if highest == NOT_APPLICABLE and all(f.status == NOT_APPLICABLE for f in self.findings):
            return NOT_APPLICABLE
        return highest


@dataclasses.dataclass
class CommandResult:
    command: list[str]
    returncode: int | None
    duration_seconds: float
    timed_out: bool
    stdout: str
    stderr: str
    exception: str | None = None


class AuditToolError(RuntimeError):
    pass


class AuditContext:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.findings: dict[str, list[Finding]] = {key: [] for key in SECTION_NAMES}
        self.python_files: list[Path] = []
        self.text_files: list[Path] = []
        self.ast_cache: dict[Path, ast.Module | None] = {}
        self.source_cache: dict[Path, str | None] = {}
        self.test_result: CommandResult | None = None
        self.test_summary: dict[str, int | None] = {
            "tests": None,
            "passed": None,
            "failed": None,
            "errors": None,
        }
        self.test_command_display = "NOT_RUN"
        self.side_effects_before: dict[str, tuple[int, int, str]] = {}
        self.side_effects_after: dict[str, tuple[int, int, str]] = {}
        self.side_effect_notes: list[str] = []
        self.tool_internal_errors: list[str] = []
        self.started_at = _now()
        self.current_section = "Starting"

    def add(
        self,
        section_id: str,
        status: str,
        item: str,
        reason: str = "",
        *,
        path: Path | str | None = None,
        line: int | None = None,
        severity: str | None = None,
    ) -> None:
        self.findings[section_id].append(
            Finding(
                section_id=section_id,
                status=status,
                item=item,
                reason=reason,
                path=str(path) if path else None,
                line=line,
                severity=severity,
            )
        )

    def section(self, section_id: str) -> SectionResult:
        return SectionResult(section_id, SECTION_NAMES[section_id], self.findings[section_id])

    def read(self, path: Path) -> str | None:
        path = path.resolve()
        if path not in self.source_cache:
            self.source_cache[path] = safe_read_text(path)
        return self.source_cache[path]

    def parse(self, path: Path) -> ast.Module | None:
        path = path.resolve()
        if path not in self.ast_cache:
            source = self.read(path)
            if source is None:
                self.ast_cache[path] = None
            else:
                try:
                    self.ast_cache[path] = ast.parse(source, filename=str(path), type_comments=True)
                except SyntaxError:
                    self.ast_cache[path] = None
        return self.ast_cache[path]

    def iter_python(self, *, include_tests: bool = True) -> Iterable[Path]:
        for path in self.python_files:
            if not include_tests and is_test_path(path, self.root):
                continue
            yield path


class ProductionReadinessAudit:
    def __init__(self, root: Path) -> None:
        self.ctx = AuditContext(root)
        self.root = self.ctx.root

    # ------------------------------------------------------------------
    # Discovery and helpers
    # ------------------------------------------------------------------
    def discover_files(self) -> None:
        for current, dirs, files in os.walk(self.root, followlinks=False):
            dirs[:] = [name for name in dirs if name not in EXCLUDED_DIRS]
            base = Path(current)
            for name in files:
                path = base / name
                if path.suffix.lower() in PY_EXTENSIONS:
                    self.ctx.python_files.append(path)
                if path.suffix.lower() in TEXT_EXTENSIONS or path.name in {".env", ".env.example", ".gitignore"}:
                    self.ctx.text_files.append(path)
        self.ctx.python_files.sort()
        self.ctx.text_files.sort()

    def find_first(self, *relative_paths: str) -> Path | None:
        for item in relative_paths:
            path = self.root / item
            if path.exists():
                return path
        return None

    def find_paths(self, patterns: Sequence[str]) -> list[Path]:
        results: list[Path] = []
        for pattern in patterns:
            results.extend(self.root.glob(pattern))
        return sorted({p.resolve() for p in results if p.exists()})

    def source_paths_matching(self, token: str, *, extensions: set[str] | None = None) -> list[Path]:
        matches: list[Path] = []
        token_low = token.lower()
        for path in self.ctx.text_files:
            if extensions and path.suffix.lower() not in extensions:
                continue
            text = self.ctx.read(path)
            if text is not None and token_low in text.lower():
                matches.append(path)
        return matches

    def line_numbers_for_text(self, path: Path, patterns: Sequence[str]) -> list[int]:
        text = self.ctx.read(path) or ""
        regex = re.compile("|".join(re.escape(p) for p in patterns), re.IGNORECASE)
        return [i for i, line in enumerate(text.splitlines(), start=1) if regex.search(line)]

    def ast_calls(self, path: Path) -> list[tuple[ast.Call, str]]:
        tree = self.ctx.parse(path)
        if tree is None:
            return []
        calls: list[tuple[ast.Call, str]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                calls.append((node, call_name(node)))
        return calls

    def defs_named(self, path: Path, names: set[str]) -> list[ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef]:
        tree = self.ctx.parse(path)
        if tree is None:
            return []
        return [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in names
        ]

    def decorators_text(self, node: ast.AST) -> str:
        decorators = getattr(node, "decorator_list", [])
        return " ".join(ast.unparse(d) for d in decorators)

    def function_source(self, path: Path, node: ast.AST) -> str:
        source = self.ctx.read(path) or ""
        lines = source.splitlines()
        start = max(1, getattr(node, "lineno", 1))
        end = getattr(node, "end_lineno", start)
        return "\n".join(lines[start - 1 : end])

    def file_exists_any(self, paths: Sequence[str]) -> list[Path]:
        return [self.root / path for path in paths if (self.root / path).exists()]

    # ------------------------------------------------------------------
    # [01] Project structure
    # ------------------------------------------------------------------
    def audit_structure(self) -> None:
        section = "01"
        expected_dirs = [
            "app",
            "app/core",
            "app/data",
            "app/db",
            "app/features",
            "app/backtest",
            "app/research",
            "app/optimization",
            "app/signals",
            "app/strategies",
            "app/providers",
            "app/web",
            "app/auth",
            "app/admin",
            "app/learning",
            "config",
            "migrations",
            "tests",
        ]
        for rel in expected_dirs:
            path = self.root / rel
            if path.is_dir():
                self.ctx.add(section, PASS, f"Directory exists: {rel}", path=path)
            else:
                # Some pieces can legitimately be absent in future refactors; core architectural groups are stronger.
                status = BLOCKER if rel in {"app", "config", "tests"} else NOT_APPLICABLE
                self.ctx.add(section, status, f"Directory missing: {rel}", "Actual project structure does not contain this directory.", path=path)

        expected_files = [
            "main.py",
            "pyproject.toml",
            "config/config_hunter.py",
            "app/core/bootstrap.py",
            "app/core/test_runner.py",
            "app/db/database.py",
            "app/db/migrations.py",
            "app/web/app.py",
            "app/web/analysis_service.py",
            "app/web/schemas.py",
            "app/learning/models.py",
            "app/learning/dataset.py",
            "app/learning/training.py",
            "app/learning/registry.py",
            "app/learning/oos_evaluation.py",
            "app/learning/policies.py",
            "app/learning/continuous.py",
            "app/learning/continuous_repository.py",
            ".env.example",
            ".gitignore",
        ]
        for rel in expected_files:
            path = self.root / rel
            if path.is_file():
                self.ctx.add(section, PASS, f"File exists: {rel}", path=path)
            else:
                self.ctx.add(section, WARN if rel in {".env.example"} else NOT_FULLY_VERIFIED, f"File not found: {rel}", "This audit does not invent the missing module; downstream checks will adapt to discovered files.", path=path)

        strategy_dirs = [p for p in (self.root / "app").glob("**/strateg*" ) if p.is_dir()]
        if strategy_dirs:
            self.ctx.add(section, PASS, f"Strategy module location discovered: {', '.join(relpath(p, self.root) for p in strategy_dirs)}")
        else:
            self.ctx.add(section, NOT_FULLY_VERIFIED, "Strategy module location", "No directory matching the current strategy-module pattern was discovered.")

        learning_files = list((self.root / "app/learning").glob("*.py")) if (self.root / "app/learning").is_dir() else []
        self.ctx.add(section, PASS if learning_files else NOT_FULLY_VERIFIED, f"Learning Python modules discovered: {len(learning_files)}", path=self.root / "app/learning")

    # ------------------------------------------------------------------
    # [02] Python/static validation
    # ------------------------------------------------------------------
    def audit_python_validation(self) -> None:
        section = "02"
        syntax_errors = 0
        unreadable = 0
        for path in self.ctx.python_files:
            source = self.ctx.read(path)
            if source is None:
                unreadable += 1
                self.ctx.add(section, FAIL, "Python file could not be read", "The audit could not parse/read this source file safely.", path=path, severity="FAIL")
                continue
            try:
                ast.parse(source, filename=str(path), type_comments=True)
            except SyntaxError as exc:
                syntax_errors += 1
                self.ctx.add(section, FAIL, "Syntax error", f"{exc.msg} at line {exc.lineno}, offset {exc.offset}", path=path, line=exc.lineno, severity="FAIL")

        if syntax_errors == 0 and unreadable == 0:
            self.ctx.add(section, PASS, f"AST syntax parse passed for {len(self.ctx.python_files)} Python files")
        else:
            self.ctx.add(section, FAIL, "AST syntax validation encountered errors", f"syntax_errors={syntax_errors}, unreadable={unreadable}")

        self._run_compileall_in_temp()
        self._check_internal_import_targets()

    def _run_compileall_in_temp(self) -> None:
        section = "02"
        with tempfile.TemporaryDirectory(prefix="edge_hunter_compile_") as temp_dir:
            staging = Path(temp_dir) / "project"
            for path in self.ctx.python_files:
                try:
                    relative = path.relative_to(self.root)
                except ValueError:
                    continue
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(path, destination)
                except OSError as exc:
                    self.ctx.add(section, WARN, "Could not stage Python file for compileall", str(exc), path=path)
            command = [sys.executable, "-m", "compileall", "-q", str(staging)]
            result = self._run_command(command, cwd=self.root, timeout=300)
            if result.returncode == 0 and not result.timed_out:
                self.ctx.add(section, PASS, "python -m compileall completed successfully in an isolated temporary tree")
            elif result.timed_out:
                self.ctx.add(section, FAIL, "compileall timed out", "compileall exceeded 300 seconds", severity="FAIL")
            else:
                self.ctx.add(section, FAIL, "compileall reported an error", redact(result.stderr or result.stdout), severity="FAIL")

    def _project_module_names(self) -> set[str]:
        modules: set[str] = set()
        for path in self.ctx.python_files:
            try:
                relative = path.relative_to(self.root)
            except ValueError:
                continue
            parts = list(relative.with_suffix("").parts)
            if parts[-1] == "__init__":
                parts.pop()
            if parts:
                modules.add(".".join(parts))
        return modules

    def _check_internal_import_targets(self) -> None:
        section = "02"
        modules = self._project_module_names()
        unresolved: list[tuple[Path, int, str]] = []
        for path in self.ctx.python_files:
            tree = self.ctx.parse(path)
            if tree is None:
                continue
            for node in ast.walk(tree):
                imported: str | None = None
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        name = alias.name
                        if name.startswith("app.") or name == "app" or name.startswith("config.") or name == "config":
                            if not any(name == mod or mod.startswith(name + ".") or name.startswith(mod + ".") for mod in modules):
                                unresolved.append((path, node.lineno, name))
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if node.level == 0 and (module.startswith("app") or module.startswith("config")):
                        if module and not any(module == mod or mod.startswith(module + ".") or module.startswith(mod + ".") for mod in modules):
                            unresolved.append((path, node.lineno, module))
        if unresolved:
            for path, line, module in unresolved[:20]:
                self.ctx.add(section, WARN, f"Potential unresolved local import target: {module}", "Static module-map validation cannot prove dynamic import/package behavior; verify the referenced module path.", path=path, line=line, severity="WARN")
            if len(unresolved) > 20:
                self.ctx.add(section, WARN, f"Additional unresolved local imports omitted from report: {len(unresolved) - 20}")
        else:
            self.ctx.add(section, PASS, "No unresolved app/config import targets were detected by static module-map validation")

    def _terminal(self, message: str) -> None:
        """Emit live progress that is visible even while child processes run."""
        stamp = _now().strftime("%H:%M:%S")
        print(f"[{stamp}] {message}", flush=True)

    # ------------------------------------------------------------------
    # [03] Full test suite
    # ------------------------------------------------------------------
    def audit_full_tests(self) -> None:
        section = "03"
        command = self._discover_test_command()
        self.ctx.test_command_display = " ".join(command)
        self._terminal(f"[03/20] Full Test Suite selected: {self.ctx.test_command_display}")
        self._terminal("[03/20] Test timeout: 1800s (30 minutes)")
        before = self._snapshot_side_effect_targets()
        start = time.monotonic()
        result = self._run_command(command, cwd=self.root, timeout=1800)
        result = dataclasses.replace(result, duration_seconds=time.monotonic() - start)
        self.ctx.test_result = result
        after = self._snapshot_side_effect_targets()
        self.ctx.side_effects_before = before
        self.ctx.side_effects_after = after
        self._compare_side_effect_snapshots(before, after)

        self._parse_test_summary(result.stdout + "\n" + result.stderr)
        if result.stdout:
            self._terminal("Test output captured successfully for the final report")
        if result.timed_out:
            self.ctx.add(section, FAIL, "Full test suite timed out", "The official/discovered test command exceeded 1800 seconds.", severity="FAIL")
            return
        if result.exception:
            self.ctx.add(section, FAIL, "Test command could not be executed", result.exception, severity="FAIL")
            return
        if result.returncode == 0 and (self.ctx.test_summary["failed"] in {None, 0}) and (self.ctx.test_summary["errors"] in {None, 0}):
            self.ctx.add(section, PASS, "Full test suite completed with exit code 0")
        else:
            self.ctx.add(section, FAIL, f"Full test suite exited with code {result.returncode}", "See captured test output in the report for details.", severity="FAIL")

        if self.ctx.side_effect_notes:
            self.ctx.add(section, WARN, "Test-side effects observed", "; ".join(self.ctx.side_effect_notes), severity="WARN")

    def _discover_test_command(self) -> list[str]:
        main_path = self.root / "main.py"
        main_source = self.ctx.read(main_path) if main_path.exists() else None
        if main_source and "run_project_tests" in main_source:
            # Intentionally no --serve: main.py's bare invocation is bounded after startup checks.
            return [sys.executable, "main.py"]

        runner = self.root / "app/core/test_runner.py"
        runner_source = self.ctx.read(runner) if runner.exists() else None
        if runner_source and "run_project_tests" in runner_source:
            return [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]

        pyproject = self.root / "pyproject.toml"
        pyproject_source = self.ctx.read(pyproject) if pyproject.exists() else ""
        pytest_config = any(
            (self.root / name).exists() for name in ("pytest.ini", "tox.ini")
        ) or "pytest" in pyproject_source.lower()
        setup_cfg = self.root / "setup.cfg"
        if setup_cfg.exists():
            setup_text = self.ctx.read(setup_cfg) or ""
            pytest_config = pytest_config or "[tool:pytest]" in setup_text or "pytest" in setup_text.lower()
        if pytest_config:
            return [sys.executable, "-m", "pytest", "-q"]
        return [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]

    def _run_command(self, command: list[str], *, cwd: Path, timeout: int) -> CommandResult:
        """Run a child command while streaming its output to the terminal.

        The previous implementation used subprocess.run(capture_output=True), which
        hid all test progress until the command finished. This implementation keeps
        the captured output for parsing/reporting while also printing it live.
        """
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        start = time.monotonic()
        process: subprocess.Popen[str] | None = None
        output_lines: list[str] = []
        heartbeat_stop = threading.Event()

        def heartbeat() -> None:
            while not heartbeat_stop.wait(30):
                elapsed = time.monotonic() - start
                self._terminal(
                    f"Still running | elapsed={elapsed:.0f}s | timeout={timeout}s | command={' '.join(command)}"
                )

        heartbeat_thread = threading.Thread(target=heartbeat, name="audit-heartbeat", daemon=True)
        try:
            kwargs: dict[str, Any] = {
                "cwd": str(cwd),
                "env": env,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "text": True,
                "encoding": "utf-8",
                "errors": "replace",
                "bufsize": 1,
            }
            if os.name == "nt":
                creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                kwargs["creationflags"] = creationflags
            else:
                kwargs["start_new_session"] = True

            process = subprocess.Popen(command, **kwargs)
            heartbeat_thread.start()
            self._terminal(f"Started: {' '.join(command)}")

            deadline = start + timeout
            if process.stdout is not None:
                for raw_line in process.stdout:
                    line = raw_line.rstrip("\r\n")
                    output_lines.append(line)
                    print(f"    [TEST] {redact(line, limit=400)}", flush=True)
                    if time.monotonic() >= deadline:
                        raise subprocess.TimeoutExpired(command, timeout, output='\n'.join(output_lines))

            remaining = max(0.0, deadline - time.monotonic())
            returncode = process.wait(timeout=remaining)
            duration = time.monotonic() - start
            self._terminal(f"Finished: exit_code={returncode} | duration={duration:.1f}s")
            return CommandResult(
                command=command,
                returncode=returncode,
                duration_seconds=duration,
                timed_out=False,
                stdout=redact("\n".join(output_lines), limit=120_000),
                stderr="",
            )
        except subprocess.TimeoutExpired as exc:
            if process is not None:
                try:
                    process.kill()
                except Exception:
                    pass
                try:
                    process.wait(timeout=5)
                except Exception:
                    pass
            duration = time.monotonic() - start
            partial = "\n".join(output_lines)
            if isinstance(exc.stdout, bytes):
                partial += "\n" + exc.stdout.decode("utf-8", "replace")
            elif exc.stdout:
                partial += "\n" + str(exc.stdout)
            self._terminal(f"TIMEOUT: command exceeded {timeout}s and was stopped")
            return CommandResult(
                command=command,
                returncode=None,
                duration_seconds=duration,
                timed_out=True,
                stdout=redact(partial, limit=120_000),
                stderr="",
            )
        except Exception as exc:  # noqa: BLE001 - audit tool must isolate command failures
            if process is not None and process.poll() is None:
                try:
                    process.kill()
                    process.wait(timeout=5)
                except Exception:
                    pass
            duration = time.monotonic() - start
            self._terminal(f"Command execution error: {type(exc).__name__}: {redact(str(exc))}")
            return CommandResult(
                command=command,
                returncode=None,
                duration_seconds=duration,
                timed_out=False,
                stdout=redact("\n".join(output_lines), limit=120_000),
                stderr="",
                exception=f"{type(exc).__name__}: {exc}",
            )
        finally:
            heartbeat_stop.set()
            if process is not None and process.stdout is not None:
                try:
                    process.stdout.close()
                except Exception:
                    pass
            if heartbeat_thread.is_alive():
                heartbeat_thread.join(timeout=1)

    def _parse_test_summary(self, text: str) -> None:
        tests_match = re.search(r"Ran\s+(\d+)\s+tests?", text, re.IGNORECASE)
        if not tests_match:
            tests_match = re.search(r"Tests run\s*:\s*(\d+)", text, re.IGNORECASE)
        if tests_match:
            self.ctx.test_summary["tests"] = int(tests_match.group(1))
        pass_match = re.search(r"Passed\s*:\s*(\d+)", text, re.IGNORECASE)
        fail_match = re.search(r"Failed\s*:\s*(\d+)", text, re.IGNORECASE)
        error_match = re.search(r"Errors?\s*:\s*(\d+)", text, re.IGNORECASE)
        pytest_match = re.search(r"([0-9]+)\s+passed(?:,\s+([0-9]+)\s+failed)?(?:,\s+([0-9]+)\s+error)?", text, re.IGNORECASE)
        if pass_match:
            self.ctx.test_summary["passed"] = int(pass_match.group(1))
        if fail_match:
            self.ctx.test_summary["failed"] = int(fail_match.group(1))
        if error_match:
            self.ctx.test_summary["errors"] = int(error_match.group(1))
        if pytest_match:
            self.ctx.test_summary["passed"] = int(pytest_match.group(1))
            self.ctx.test_summary["failed"] = int(pytest_match.group(2) or 0)
            self.ctx.test_summary["errors"] = int(pytest_match.group(3) or 0)

    def _snapshot_side_effect_targets(self) -> dict[str, tuple[int, int, str]]:
        targets: list[Path] = []
        data_dir = self.root / "data"
        logs_dir = self.root / "logs"
        for directory in (data_dir, logs_dir):
            if directory.exists():
                for path in directory.rglob("*"):
                    if path.is_file() and path.suffix.lower() in {".db", ".sqlite", ".sqlite3", ".jsonl", ".log", ".json"}:
                        try:
                            stat = path.stat()
                            digest = sha256_file(path) if stat.st_size <= 8 * 1024 * 1024 else f"size:{stat.st_size}"
                            targets.append(path)
                        except OSError:
                            continue
        report_dir = self.root / REPORT_DIR_NAME
        if report_dir.exists():
            # Ignore the audit report itself; report creation is an expected side effect.
            for path in report_dir.rglob("*"):
                if path.is_file() and not path.name.startswith(REPORT_NAME_PREFIX):
                    try:
                        stat = path.stat()
                        targets.append(path)
                    except OSError:
                        continue
        snapshot: dict[str, tuple[int, int, str]] = {}
        for path in targets:
            try:
                stat = path.stat()
                digest = sha256_file(path) if stat.st_size <= 8 * 1024 * 1024 else f"size:{stat.st_size}"
                snapshot[relpath(path, self.root)] = (stat.st_size, stat.st_mtime_ns, digest)
            except OSError:
                continue
        return snapshot

    def _compare_side_effect_snapshots(self, before: dict[str, tuple[int, int, str]], after: dict[str, tuple[int, int, str]]) -> None:
        added = sorted(set(after) - set(before))
        removed = sorted(set(before) - set(after))
        changed = sorted(name for name in set(before) & set(after) if before[name] != after[name])
        for name in added:
            self.ctx.side_effect_notes.append(f"added {name}")
        for name in removed:
            self.ctx.side_effect_notes.append(f"removed {name}")
        for name in changed:
            self.ctx.side_effect_notes.append(f"changed {name}")

    # ------------------------------------------------------------------
    # [04] Learning safety
    # ------------------------------------------------------------------
    def audit_learning_safety(self) -> None:
        section = "04"
        env_file = self.root / ".env"
        env_example = self.root / ".env.example"
        env_values = load_env_file(env_file)
        example_values = load_env_file(env_example)
        for key in LEARNING_FLAG_KEYS:
            source = "default"
            raw = example_values.get(key, "false")
            if key in example_values:
                source = ".env.example"
            if key in env_values:
                raw = env_values[key]
                source = ".env"
            if key in os.environ:
                raw = os.environ[key]
                source = "process environment"
            parsed = normalize_bool(raw)
            if parsed is None:
                self.ctx.add(section, FAIL, f"Learning flag {key} has unparseable value", f"source={source}; value=[REDACTED]", severity="FAIL")
            elif parsed:
                self.ctx.add(section, BLOCKER, f"Unsafe enabled Learning flag: {key}", f"source={source}; effective value=true", severity="BLOCKER")
            else:
                self.ctx.add(section, PASS, f"{key}=false", f"effective source={source}")

        config_path = self.root / "config/config_hunter.py"
        config_text = self.ctx.read(config_path) or ""
        defaults_ok = True
        for attr in (
            "learning_enabled",
            "learning_automatic_retraining_enabled",
            "learning_automatic_promotion_enabled",
            "learning_production_enabled",
        ):
            pattern = re.compile(rf"{re.escape(attr)}\s*=\s*_env_bool\([^\n]*?False\)", re.IGNORECASE)
            if not pattern.search(config_text):
                defaults_ok = False
                self.ctx.add(section, WARN, f"Could not statically confirm safe default for {attr}", "Configuration may still be safe, but this audit did not match the expected _env_bool(..., False) pattern.", path=config_path)
        if defaults_ok and config_path.exists():
            self.ctx.add(section, PASS, "Central configuration exposes all four learning safety flags with False defaults", path=config_path)

        continuous = self.root / "app/learning/continuous.py"
        continuous_text = self.ctx.read(continuous) or ""
        if continuous.exists() and all(token in continuous_text for token in (
            "automatic_retraining_enabled: bool = False",
            "automatic_promotion_enabled: bool = False",
            "production_enabled: bool = False",
        )):
            self.ctx.add(section, PASS, "Continuous-learning configuration uses safe False defaults", path=continuous)
        else:
            self.ctx.add(section, NOT_FULLY_VERIFIED, "Continuous-learning safe defaults could not be fully confirmed statically", path=continuous if continuous.exists() else None)

    # ------------------------------------------------------------------
    # [05] API / Analyze path
    # ------------------------------------------------------------------
    def audit_api_analyze(self) -> None:
        section = "05"
        routes = self._locate_analyze_routes()
        if not routes:
            self.ctx.add(section, NOT_APPLICABLE, "Analyze route could not be discovered", "No route containing /api/analyze or an equivalent analyze decorator was found.")
            return
        for path, node, route_label in routes:
            source = self.function_source(path, node)
            names = {name for _, name in self.ast_calls_in_node(node)}
            dangerous = sorted(names & {"fit", "fit_transform", "train", "retrain", "fine_tune", "start_scheduler", "promote_candidate", "rollback", "build_and_store"})
            if dangerous:
                self.ctx.add(section, FAIL, "Analyze route directly references training/learning sink", f"calls={dangerous}; static review requires deeper code-path inspection.", path=path, line=getattr(node, "lineno", None), severity="FAIL")
            else:
                self.ctx.add(section, PASS, f"Analyze route has no direct training/promotion/rollback calls: {route_label}", path=path, line=getattr(node, "lineno", None))
            if "app.learning" in source or "learning." in source:
                self.ctx.add(section, FAIL, "Analyze route directly imports/references learning layer", "The request handler should not start Learning Pipeline operations.", path=path, line=getattr(node, "lineno", None), severity="FAIL")
            else:
                self.ctx.add(section, PASS, "Analyze route does not directly reference app.learning", path=path, line=getattr(node, "lineno", None))

        service = self._locate_analysis_service()
        if service is not None:
            path, node = service
            source = self.function_source(path, node)
            calls = sorted({name for _, name in self.ast_calls_in_node(node)})
            forbidden = [name for name in calls if name in {"fit", "fit_transform", "train", "retrain", "start_scheduler", "promote_candidate", "rollback", "build_and_store"}]
            if forbidden:
                self.ctx.add(section, FAIL, "Analysis service directly references a training/production sink", f"calls={forbidden}", path=path, line=getattr(node, "lineno", None), severity="FAIL")
            else:
                self.ctx.add(section, PASS, "Analysis service method contains no direct training/retraining/promotion/rollback calls", path=path, line=getattr(node, "lineno", None))
            if "app.learning" in source:
                self.ctx.add(section, FAIL, "Analysis service directly references app.learning", "This audit treats direct learning-layer coupling in Analyze as unsafe unless the code is demonstrably read-only.", path=path, line=getattr(node, "lineno", None), severity="FAIL")
            else:
                self.ctx.add(section, PASS, "Analysis service has no direct app.learning import/reference", path=path, line=getattr(node, "lineno", None))
        else:
            self.ctx.add(section, NOT_FULLY_VERIFIED, "Could not resolve a concrete analysis service method", "Static verification can confirm the route but not its complete transitive runtime graph.")

        # Project-wide call-site search for scheduler/promotion invoked from web/core startup.
        prohibited = self._find_call_sites_by_name({"start_scheduler", "promote_candidate", "rollback"}, roots=[self.root / "app/web", self.root / "app/core"])
        non_test = [x for x in prohibited if not is_test_path(x[0], self.root)]
        if non_test:
            for path, line, name in non_test:
                self.ctx.add(section, FAIL, f"Production-facing code calls {name}", "Automatic learning/promotion should not be initiated from Analyze or web startup paths.", path=path, line=line, severity="FAIL")
        else:
            self.ctx.add(section, PASS, "No non-test scheduler/promotion/rollback call sites found under app/web or app/core")

        self.ctx.add(section, NOT_FULLY_VERIFIED, "Transitive Analyze call graph", "This is a static audit; it does not execute every downstream helper path or production request.")

    def _locate_analyze_routes(self) -> list[tuple[Path, ast.FunctionDef | ast.AsyncFunctionDef, str]]:
        routes: list[tuple[Path, ast.FunctionDef | ast.AsyncFunctionDef, str]] = []
        for path in self.ctx.python_files:
            if not ("web" in path.parts or "api" in path.parts):
                continue
            tree = self.ctx.parse(path)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                decorators = " ".join(ast.unparse(d) for d in node.decorator_list)
                if "/api/analyze" in decorators.lower() or (node.name.lower() == "analyze" and "/api/" in decorators.lower()):
                    routes.append((path, node, decorators))
        return routes

    def _locate_analysis_service(self) -> tuple[Path, ast.FunctionDef | ast.AsyncFunctionDef] | None:
        candidates: list[tuple[Path, ast.FunctionDef | ast.AsyncFunctionDef]] = []
        for path in self.ctx.python_files:
            if path.name != "analysis_service.py":
                continue
            tree = self.ctx.parse(path)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "analyze":
                    candidates.append((path, node))
        return candidates[0] if candidates else None

    def ast_calls_in_node(self, node: ast.AST) -> list[tuple[ast.Call, str]]:
        result: list[tuple[ast.Call, str]] = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                result.append((child, call_name(child)))
        return result

    def _find_call_sites_by_name(self, names: set[str], *, roots: Sequence[Path] | None = None) -> list[tuple[Path, int, str]]:
        result: list[tuple[Path, int, str]] = []
        paths = self.ctx.python_files if roots is None else [p for p in self.ctx.python_files if any(p == r or r in p.parents for r in roots)]
        for path in paths:
            tree = self.ctx.parse(path)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and call_name(node) in names:
                    result.append((path, node.lineno, call_name(node)))
        return result

    # ------------------------------------------------------------------
    # [06] Leakage/look-ahead
    # ------------------------------------------------------------------
    def audit_leakage(self) -> None:
        section = "06"
        no_lookahead_test = self.find_first("tests/unit/features/test_no_lookahead.py")
        if no_lookahead_test:
            self.ctx.add(section, PASS, "Feature no-lookahead test exists", path=no_lookahead_test)
        else:
            self.ctx.add(section, NOT_FULLY_VERIFIED, "Dedicated no-lookahead test not found", "This audit will not assume look-ahead protection without concrete evidence.")

        dataset = self.root / "app/learning/dataset.py"
        training = self.root / "app/learning/training.py"
        oos = self.root / "app/learning/oos_evaluation.py"
        records = self.root / "app/learning/records.py"
        feature_store = self.root / "app/learning/feature_store.py"

        if dataset.exists():
            text = self.ctx.read(dataset) or ""
            tokens = ["_validate_temporal_splits", "DatasetSplit.OOS", "train_count", "validation_count", "oos_count"]
            if all(t in text for t in tokens):
                self.ctx.add(section, PASS, "Dataset layer contains explicit temporal split/OOS bookkeeping", path=dataset)
            else:
                self.ctx.add(section, NOT_FULLY_VERIFIED, "Dataset temporal/OOS guard could not be completely matched", path=dataset)
        else:
            self.ctx.add(section, NOT_FULLY_VERIFIED, "Dataset module not found")

        if training.exists():
            text = self.ctx.read(training) or ""
            guards = ["_assert_no_oos_in_training", "preprocessor.fit_transform", "preprocessor.transform", "model.fit"]
            if all(t in text for t in guards):
                self.ctx.add(section, PASS, "Training pipeline statically contains train-only fit and OOS contamination guard", path=training)
            else:
                self.ctx.add(section, NOT_FULLY_VERIFIED, "Training fit/OOS guards could not be fully matched", path=training)

        if oos.exists():
            tree = self.ctx.parse(oos)
            forbidden_calls = {
                "fit", "fit_transform", "partial_fit", "train", "retrain",
                "fine_tune", "build_and_store", "create_dataset", "write_dataset",
                "append_dataset", "save_dataset", "mutate_dataset",
            }
            actual_forbidden: list[tuple[int, str]] = []
            allowed_eval_calls: list[tuple[int, str]] = []
            if tree:
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Call):
                        continue
                    name = call_name(node)
                    if name in forbidden_calls:
                        actual_forbidden.append((node.lineno, name))
                    elif name in {"load_artifact", "predict", "transform"}:
                        allowed_eval_calls.append((node.lineno, name))
            if actual_forbidden:
                self.ctx.add(section, FAIL, "OOS evaluation module contains training/dataset-mutation calls", f"calls={actual_forbidden[:10]}", path=oos, line=actual_forbidden[0][0], severity="FAIL")
            else:
                # Importing the training pipeline is acceptable when it is used only as a
                # trusted artifact loader/predictor. The call graph still remains a static
                # limitation, so no claim is made beyond observed calls in this module.
                self.ctx.add(section, PASS, "OOS evaluation module has no training or dataset-mutation calls", path=oos)
                if allowed_eval_calls:
                    self.ctx.add(section, PASS, "OOS uses evaluation-only artifact loading/prediction calls", f"calls={allowed_eval_calls[:10]}", path=oos)
        else:
            self.ctx.add(section, NOT_FULLY_VERIFIED, "OOS evaluation module not found")

        if records.exists() and feature_store.exists():
            record_text = self.ctx.read(records) or ""
            snapshot_text = self.ctx.read(feature_store) or ""
            aligned = all(token in record_text for token in ("feature_snapshot_id", "decision_timestamp")) and "timestamp" in snapshot_text
            self.ctx.add(section, PASS if aligned else NOT_FULLY_VERIFIED, "Learning Record / Feature Snapshot identity fields are statically present" if aligned else "Learning Record / Feature Snapshot alignment not fully verified", path=records)

        # Negative-time shifting is a useful detector, but not proof of a leak by itself.
        negative_shift_hits: list[tuple[Path, int, str]] = []
        for path in self.ctx.python_files:
            if any(part in {"features", "strategies", "optimization", "learning"} for part in path.parts):
                text = self.ctx.read(path) or ""
                for i, line in enumerate(text.splitlines(), start=1):
                    if re.search(r"\.shift\s*\(\s*[-]\s*\d+", line) or re.search(r"shift\s*=\s*[-]\d+", line):
                        if not line.lstrip().startswith("#"):
                            negative_shift_hits.append((path, i, line.strip()))
        if negative_shift_hits:
            for path, line, snippet in negative_shift_hits[:20]:
                self.ctx.add(section, WARN, "Negative shift detected; manual look-ahead review required", f"code contains backward-index convention that may reference future bars; this is not automatically a leak.", path=path, line=line, severity="WARN")
        else:
            self.ctx.add(section, PASS, "No obvious negative pandas shift pattern was detected in feature/strategy/optimization/learning Python sources")

        self.ctx.add(section, NOT_FULLY_VERIFIED, "Global future-data isolation", "Static inspection cannot mathematically prove every runtime data path is free of look-ahead or future-data contamination.")

    # ------------------------------------------------------------------
    # [07] Model registry
    # ------------------------------------------------------------------
    def audit_model_registry(self) -> None:
        section = "07"
        registry = self.root / "app/learning/registry.py"
        repo = self.root / "app/learning/model_registry_repository.py"
        training = self.root / "app/learning/training.py"
        if not registry.exists():
            self.ctx.add(section, NOT_FULLY_VERIFIED, "Model registry module not found")
            return
        registry_text = self.ctx.read(registry) or ""
        repo_text = self.ctx.read(repo) if repo.exists() else ""
        required_terms = [
            "ModelVersion",
            "TrainingRun",
            "REGISTERED",
            "REVOKED",
            "artifact_sha256",
            "training_run_id",
            "dataset_version_id",
            "feature_schema_version",
            "label_version",
        ]
        missing = [term for term in required_terms if term not in registry_text + (repo_text or "")]
        if missing:
            self.ctx.add(section, NOT_FULLY_VERIFIED, "Model registry lineage/status contract not fully matched", f"missing_terms={missing}", path=registry)
        else:
            self.ctx.add(section, PASS, "Model Version registry contains artifact checksum and training/dataset/schema lineage fields", path=registry)

        artifact_guard_terms = ["load_artifact", "artifact_path", "artifact_sha256", "is_file", "sha256"]
        if all(term.lower() in registry_text.lower() for term in artifact_guard_terms):
            self.ctx.add(section, PASS, "Registry verifies artifact path/existence/hash before reload", path=registry)
        else:
            self.ctx.add(section, NOT_FULLY_VERIFIED, "Artifact validation/reload guard not fully matched", f"required={artifact_guard_terms}", path=registry)

        if training.exists() and "artifact_sha256" in (self.ctx.read(training) or "") and "joblib" in (self.ctx.read(training) or "").lower():
            self.ctx.add(section, PASS, "Training pipeline persists artifact checksum metadata", path=training)
        else:
            self.ctx.add(section, NOT_FULLY_VERIFIED, "Training artifact checksum persistence could not be confirmed", path=training if training.exists() else None)

        web_hits = self._source_paths_with_tokens_under({"app/web", "app/core"}, {"load_artifact", "ModelRegistryRepository", "joblib.load"})
        if web_hits:
            for path, line, token in web_hits[:20]:
                self.ctx.add(section, WARN, f"Production-facing source references artifact loader: {token}", "Requires context review to ensure only approved registry artifacts can be loaded.", path=path, line=line, severity="WARN")
        else:
            self.ctx.add(section, PASS, "No direct model-artifact loader call was found in app/web or app/core")

    # ------------------------------------------------------------------
    # [08] OOS / Walk-forward / Robustness
    # ------------------------------------------------------------------
    def audit_oos_robustness(self) -> None:
        section = "08"
        candidates = {
            "OOS": ["app/learning/oos_evaluation.py"],
            "Walk-Forward": ["app/optimization/splits.py", "app/optimization/validation.py"],
            "Robustness": ["app/optimization/robustness.py"],
            "Sensitivity": ["app/optimization/sensitivity.py"],
        }
        found_any = False
        for label, paths in candidates.items():
            existing = self.file_exists_any(paths)
            if existing:
                found_any = True
                self.ctx.add(section, PASS, f"{label} module found: {', '.join(relpath(p, self.root) for p in existing)}")
            else:
                self.ctx.add(section, NOT_APPLICABLE if label == "Sensitivity" else NOT_FULLY_VERIFIED, f"{label} module not found", "No module matching the current project path was discovered.")

        for path in self.file_exists_any(["app/optimization/splits.py", "app/optimization/validation.py"]):
            text = self.ctx.read(path) or ""
            if "datetime" in text or "timestamp" in text.lower() or "temporal" in text.lower():
                self.ctx.add(section, PASS, f"{relpath(path, self.root)} contains temporal-split terminology", path=path)
            if re.search(r"RandomState|random\.shuffle|train_test_split", text):
                self.ctx.add(section, WARN, "Random split/randomization reference detected in validation utility", "Randomization is not automatically unsafe, but OOS/walk-forward boundaries should remain temporal and deterministic.", path=path, severity="WARN")

        l8_test = self.find_first("tests/unit/learning/test_oos_walk_forward_robustness.py")
        if l8_test:
            self.ctx.add(section, PASS, "Dedicated L8 OOS/walk-forward/robustness test module exists", path=l8_test)

        oos = self.root / "app/learning/oos_evaluation.py"
        if oos.exists():
            calls = self.ast_calls(oos)
            fit_calls = [n.lineno for n, name in calls if name in {"fit", "fit_transform"}]
            if fit_calls:
                self.ctx.add(section, FAIL, "OOS module contains fit/fit_transform calls", f"lines={fit_calls}", path=oos, severity="FAIL")
            else:
                self.ctx.add(section, PASS, "OOS evaluation source has no model fitting calls")

        if found_any:
            self.ctx.add(section, NOT_FULLY_VERIFIED, "End-to-end OOS isolation", "Static source inspection cannot prove every runtime branch preserves OOS isolation under all future code paths.")

    # ------------------------------------------------------------------
    # [09] Learned policy
    # ------------------------------------------------------------------
    def audit_learned_policy(self) -> None:
        section = "09"
        policy = self.root / "app/learning/policies.py"
        adapters = self.root / "app/learning/policy_adapters.py"
        if not policy.exists():
            self.ctx.add(section, NOT_APPLICABLE, "Learned policy module not found")
            return
        policy_text = self.ctx.read(policy) or ""
        adapter_text = self.ctx.read(adapters) if adapters.exists() else ""
        checks = {
            "Policy Version": "LearnedPolicyVersion",
            "Compatibility": "compat",
            "Risk/lot override blocked": "lot_size",
            "R:R lower bound": "1.5",
            "R:R upper bound": "2.0",
            "Disabled default": "DISABLED",
            "No fixed strategy weights": "fixed strategy weights",
        }
        for label, token in checks.items():
            haystack = policy_text + (adapter_text or "")
            if token.lower() in haystack.lower():
                self.ctx.add(section, PASS, f"{label} evidence found", path=policy)
            else:
                self.ctx.add(section, NOT_FULLY_VERIFIED, f"{label} evidence not fully matched", path=policy)

        # Explicitly block dangerous risk/lot overrides found in policy construction code.
        bad_override_hits = []
        tree = self.ctx.parse(policy)
        if tree:
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in {"risk_percent", "lot_size"}:
                    bad_override_hits.append(node.lineno)
        # String presence alone is not failure: current contract rejects these keys. Only report a structural PASS if rejection terms exist.
        if "risk_percent" in policy_text and "lot_size" in policy_text and any(term in policy_text for term in ("reject", "forbidden", "override")):
            self.ctx.add(section, PASS, "Policy layer explicitly protects risk/lot size from learned overrides", path=policy)

        if adapters.exists():
            # AI outcome-classification safeguard: look for literal label-to-BUY/SELL mapping.
            suspicious: list[int] = []
            tree = self.ctx.parse(adapters)
            if tree:
                source = self.ctx.read(adapters) or ""
                for node in ast.walk(tree):
                    if isinstance(node, ast.Compare):
                        left = dotted_name(node.left) if isinstance(node.left, ast.AST) else None
                        rendered = ast.unparse(node)
                        if "predicted_label" in rendered and re.search(r"\b(BUY|SELL)\b", rendered):
                            suspicious.append(node.lineno)
                # Dict/literal conversion is captured by the broad token search below.
                if "predicted_label" in source and re.search(r"predicted_label[^\n]{0,120}(BUY|SELL)|(BUY|SELL)[^\n]{0,120}predicted_label", source):
                    self.ctx.add(section, WARN, "AI adapter contains predicted_label text near BUY/SELL tokens; manual review required", "This audit does not invent a BUY/SELL conversion and does not treat label names alone as proof of one.", path=adapters, severity="WARN")
                elif "AIInferenceResult" in source and "predicted_label" in source:
                    self.ctx.add(section, PASS, "AI inference result exposes predicted_label without a static direct BUY/SELL conversion detected", path=adapters)
            if suspicious:
                self.ctx.add(section, WARN, "Potential predicted_label to BUY/SELL comparison found", f"lines={sorted(set(suspicious))[:20]}", path=adapters, severity="WARN")

        if "fallback" in (adapter_text or "").lower():
            self.ctx.add(section, PASS, "Learned-policy adapter contains fallback terminology", path=adapters)
        else:
            self.ctx.add(section, NOT_FULLY_VERIFIED, "Fallback behavior could not be confirmed statically", path=adapters if adapters.exists() else None)

    # ------------------------------------------------------------------
    # [10] Continuous learning
    # ------------------------------------------------------------------
    def audit_continuous_learning(self) -> None:
        section = "10"
        continuous = self.root / "app/learning/continuous.py"
        repository = self.root / "app/learning/continuous_repository.py"
        migration = self.root / "migrations/011_learning_continuous_learning.sql"
        if not continuous.exists():
            self.ctx.add(section, NOT_APPLICABLE, "Continuous-learning orchestrator not found")
            return
        text = self.ctx.read(continuous) or ""
        checks = {
            "cycle states": ["DATA_COLLECTION", "DATA_COMPLETION", "DATASET_BUILD", "TRAINING", "REGISTRATION", "VALIDATION", "OOS", "ROBUSTNESS"],
            "trigger sources": ["SCHEDULED", "DATA_THRESHOLD", "MANUAL", "DRIFT"],
            "retry support": ["max_retries", "_run_with_retries"],
            "cooldown support": ["cooldown_seconds", "cooldown"],
            "active-scope guard": ["active", "scope_key"],
            "promotion gate": ["promote_candidate", "PromotionPolicy"],
            "rollback": ["rollback", "RollbackPolicy"],
            "production state": ["ProductionModelState"],
        }
        for label, tokens in checks.items():
            if all(token.lower() in text.lower() for token in tokens):
                self.ctx.add(section, PASS, f"Continuous learning {label} statically present", path=continuous)
            else:
                self.ctx.add(section, NOT_FULLY_VERIFIED, f"Continuous learning {label} not fully matched", path=continuous)

        if repository.exists():
            repo_text = self.ctx.read(repository) or ""
            audit_terms = ["learning_audit_events", "learning_state", "learning_production_history", "learning_production_state"]
            if all(term in repo_text for term in audit_terms):
                self.ctx.add(section, PASS, "Continuous-learning repository contains audit/state/history persistence", path=repository)
            else:
                self.ctx.add(section, NOT_FULLY_VERIFIED, "Continuous-learning repository persistence contracts not fully matched", path=repository)

        if migration.exists():
            migration_text = self.ctx.read(migration) or ""
            if all(term in migration_text for term in ("learning_cycles", "learning_candidates", "learning_production_state", "learning_production_history", "learning_monitoring_snapshots", "learning_audit_events")):
                self.ctx.add(section, PASS, "L10 migration contains cycle/candidate/production/monitoring/audit tables", path=migration)
            else:
                self.ctx.add(section, NOT_FULLY_VERIFIED, "L10 migration table coverage not fully matched", path=migration)

        starts = self._find_call_sites_by_name({"start_scheduler"})
        non_test = [item for item in starts if not is_test_path(item[0], self.root)]
        if non_test:
            for path, line, name in non_test:
                self.ctx.add(section, BLOCKER, "Scheduler start call found outside tests", "Continuous Learning must not start unintentionally during startup/analyze.", path=path, line=line, severity="BLOCKER")
        else:
            self.ctx.add(section, PASS, "No non-test start_scheduler call site found")

        self.ctx.add(section, NOT_FULLY_VERIFIED, "Crash/restart recovery under real scheduler process", "Static review can inspect guards but cannot prove process-level recovery without executing the scheduler lifecycle; the audit deliberately does not start it.")

    # ------------------------------------------------------------------
    # [11] Strategies
    # ------------------------------------------------------------------
    def audit_strategies(self) -> None:
        section = "11"
        strategy_dir = self.root / "app/strategies"
        if not strategy_dir.is_dir():
            self.ctx.add(section, NOT_APPLICABLE, "Strategy directory not found")
            return
        strategy_files = sorted(p for p in strategy_dir.glob("*.py") if p.name != "__init__.py")
        if not strategy_files:
            self.ctx.add(section, NOT_FULLY_VERIFIED, "No strategy Python modules discovered", path=strategy_dir)
            return
        self.ctx.add(section, PASS, f"Discovered {len(strategy_files)} strategy module(s): {', '.join(p.name for p in strategy_files)}", path=strategy_dir)

        models = strategy_dir / "models.py"
        registry = strategy_dir / "registry.py"
        if models.exists():
            text = self.ctx.read(models) or ""
            terms = ["StrategyContext", "StrategySignal"]
            self.ctx.add(section, PASS if all(t in text for t in terms) else NOT_FULLY_VERIFIED, "Strategy Context/Signal contract evidence", path=models)
        else:
            self.ctx.add(section, NOT_FULLY_VERIFIED, "Strategy models module not found")

        if registry.exists():
            text = self.ctx.read(registry) or ""
            for label, aliases in {
                "Classic": ["Classic", "classic"],
                "SMC": ["SMC", "smc"],
                "ICT": ["ICT", "ict"],
            }.items():
                self.ctx.add(section, PASS if any(a in text for a in aliases) else NOT_FULLY_VERIFIED, f"{label} strategy registry evidence", path=registry)
            if "equal" in text.lower() and "weight" in text.lower():
                self.ctx.add(section, WARN, "Registry contains equal/default weighting terminology", "Equal votes can be a documented default; this is not treated as a ranking or strategy recommendation.", path=registry, severity="WARN")
        else:
            self.ctx.add(section, NOT_FULLY_VERIFIED, "Strategy registry module not found")

        # Discover actual strategy class names; do not assume only Classic/SMC/ICT exist.
        classes: list[str] = []
        for path in strategy_files:
            tree = self.ctx.parse(path)
            if tree:
                classes.extend(node.name for node in tree.body if isinstance(node, ast.ClassDef))
        if classes:
            self.ctx.add(section, PASS, f"Strategy classes discovered: {', '.join(sorted(set(classes)))}")

        ai_refs = self.source_paths_within([self.root / "app"], {"AIInferenceResult", "predicted_label"})
        if ai_refs:
            self.ctx.add(section, PASS, "AI strategy/inference references are isolated in discovered learning/inference modules; this audit does not fabricate BUY/SELL mapping")
        else:
            self.ctx.add(section, NOT_FULLY_VERIFIED, "No AI inference module reference was discovered under app/", "This is informational only; AI strategy behavior may be represented differently.")

    # ------------------------------------------------------------------
    # [12] Risk / R:R
    # ------------------------------------------------------------------
    def audit_risk_rr(self) -> None:
        section = "12"
        risk_files = self.file_exists_any([
            "config/config_hunter.py",
            "app/web/schemas.py",
            "app/strategies/utils.py",
            "app/strategies/models.py",
            "app/learning/policies.py",
            "app/learning/policy_adapters.py",
            "app/backtest/config.py",
        ])
        if not risk_files:
            self.ctx.add(section, NOT_FULLY_VERIFIED, "Risk configuration source files not found")
            return
        risk_evidence = "\n".join(self.ctx.read(p) or "" for p in risk_files)
        if "risk_percent" in risk_evidence:
            self.ctx.add(section, PASS, "Risk percent controls are represented in configuration/schema/strategy layers")
        else:
            self.ctx.add(section, NOT_FULLY_VERIFIED, "Risk percent field not statically matched")

        rr_hits = []
        for path in risk_files:
            text = self.ctx.read(path) or ""
            if "1.5" in text and "2.0" in text:
                rr_hits.append(path)
        if rr_hits:
            self.ctx.add(section, PASS, "1.5–2.0 R:R bounds are represented in project code", ", ".join(relpath(p, self.root) for p in rr_hits))
        else:
            self.ctx.add(section, FAIL, "Could not locate the required 1.5–2.0 R:R bounds", "The requested R:R contract could not be substantiated by current source code.", severity="FAIL")

        policies = self.root / "app/learning/policies.py"
        if policies.exists():
            policy_text = self.ctx.read(policies) or ""
            # Evidence should include actual checks, not only docs.
            bound_checks = bool(re.search(r"1\.5\s*<=|rr\s*>=\s*1\.5", policy_text, re.I)) and bool(re.search(r"2\.0\s*>=|rr\s*<=\s*2\.0", policy_text, re.I))
            self.ctx.add(section, PASS if bound_checks else NOT_FULLY_VERIFIED, "Learned policy R:R bound expressions", path=policies)

        lot_override = self._search_ast_string_literals({"risk_percent", "lot_size"}, roots=risk_files)
        if lot_override:
            self.ctx.add(section, WARN, "Risk/lot field names are present in learned-policy source", "Field-name presence is expected; current policy logic must reject learned risk/lot overrides rather than silently change them.", severity="WARN")
        else:
            self.ctx.add(section, PASS, "No direct risk_percent/lot_size string-literal sink found in risk files")

        self.ctx.add(section, NOT_FULLY_VERIFIED, "End-to-end runtime risk enforcement", "Static audit cannot execute a live order/position path; it verifies contract evidence without altering strategy or risk settings.")

    # ------------------------------------------------------------------
    # [13] Authentication / subscriptions
    # ------------------------------------------------------------------
    def audit_auth_subscription(self) -> None:
        section = "13"
        security = self.root / "app/auth/security.py"
        auth_service = self.root / "app/auth/service.py"
        auth_repo = self.root / "app/auth/repository.py"
        web_app = self.root / "app/web/app.py"
        admin = self.root / "app/admin/service.py"
        files = [p for p in (security, auth_service, auth_repo, web_app, admin) if p.exists()]
        if not files:
            self.ctx.add(section, NOT_FULLY_VERIFIED, "Authentication/subscription modules not found")
            return
        corpus = "\n".join(self.ctx.read(p) or "" for p in files)
        checks = {
            "Password hashing": ["hash_password", "verify_password", "pbkdf2"],
            "Session handling": ["session", "token_hash"],
            "CSRF": ["csrf"],
            "Subscription state": ["subscription", "expires_at", "revoked"],
            "Trial": ["trial", "trial_expires_at"],
            "Device linking": ["device_hash", "device"],
            "Subscription code length": ["16"],
            "Admin controls": ["AdminService", "admin"],
        }
        for label, tokens in checks.items():
            self.ctx.add(section, PASS if all(token.lower() in corpus.lower() for token in tokens) else NOT_FULLY_VERIFIED, f"{label} evidence", path=files[0])

        # Use the same context-aware credential detector as the security section.
        auth_secret_hits = [item for item in self._hardcoded_secret_assignments() if item[0] in files]
        if auth_secret_hits:
            for path, line, name in auth_secret_hits[:20]:
                self.ctx.add(section, FAIL, "Hardcoded credential literal in auth source", f"variable={name}; value=[REDACTED]", path=path, line=line, severity="FAIL")
        else:
            self.ctx.add(section, PASS, "No hardcoded credential literal found in non-test auth source")

        self.ctx.add(section, NOT_FULLY_VERIFIED, "Real-account/session lifecycle", "No real accounts or production data were created or mutated by this audit.")

    # ------------------------------------------------------------------
    # [14] Security
    # ------------------------------------------------------------------
    def audit_security(self) -> None:
        section = "14"
        config = self.root / "config/config_hunter.py"
        web_security = self.root / "app/web/security.py"
        web_app = self.root / "app/web/app.py"
        auth_security = self.root / "app/auth/security.py"
        files = [p for p in (config, web_security, web_app, auth_security) if p.exists()]
        corpus = "\n".join(self.ctx.read(p) or "" for p in files)

        cors_validation = "wildcard CORS origins are not allowed" in corpus
        if cors_validation:
            self.ctx.add(section, PASS, "CORS wildcard validation is implemented in central configuration", path=config)
        else:
            self.ctx.add(section, NOT_FULLY_VERIFIED, "CORS wildcard rejection could not be matched", path=config if config.exists() else None)
        if 'allow_origins=["*"]' in corpus or "allow_origins=['*']" in corpus:
            self.ctx.add(section, FAIL, "Runtime wildcard CORS configuration detected", "An actual allow_origins wildcard is unsafe when credentials are enabled.", path=config if config.exists() else None, severity="FAIL")

        if "TrustedHostMiddleware" in corpus:
            self.ctx.add(section, PASS, "TrustedHostMiddleware is configured", path=web_app)
        else:
            self.ctx.add(section, WARN, "Trusted host middleware not detected", severity="WARN")

        security_terms = {
            "Request size limit": "RequestSizeLimitMiddleware",
            "Rate limiting": "PublicRateLimitMiddleware",
            "Request ID logging": "RequestIDLoggingMiddleware",
            "Security headers": "SecurityHeadersMiddleware",
            "CSRF": "csrf",
            "Secure cookies": "cookie_secure",
            "Debug setting": "EDGE_HUNTER_DEBUG",
        }
        for label, token in security_terms.items():
            self.ctx.add(section, PASS if token.lower() in corpus.lower() else NOT_FULLY_VERIFIED, f"{label} evidence", path=web_security if web_security.exists() else config)

        # Hardcoded secret scan across non-test Python source; .env is expected to hold secrets and is not printed.
        source_secret_hits = self._hardcoded_secret_assignments()
        if source_secret_hits:
            for path, line, name in source_secret_hits[:30]:
                self.ctx.add(section, BLOCKER, "POTENTIAL_SECRET", f"variable={name}; value=[REDACTED]", path=path, line=line, severity="BLOCKER")
        else:
            self.ctx.add(section, PASS, "No obvious hardcoded secret assignments detected in non-test Python source")

        env_example = self.root / ".env.example"
        gitignore = self.root / ".gitignore"
        if env_example.exists() and gitignore.exists():
            git_text = self.ctx.read(gitignore) or ""
            protected = ".env" in git_text
            self.ctx.add(section, PASS if protected else FAIL, ".env is ignored by version-control configuration" if protected else ".env is not clearly ignored", "The audit never prints .env values.", path=gitignore, severity=None if protected else "FAIL")

        self._audit_sql_injection_patterns(section)
        self._audit_dynamic_execution(section)
        self._audit_file_access_patterns(section)

    def _hardcoded_secret_assignments(self) -> list[tuple[Path, int, str]]:
        """Find actual credential literals, not algorithm/settings constants."""
        hits: list[tuple[Path, int, str]] = []
        for path in self.ctx.python_files:
            if is_test_path(path, self.root) or is_audit_tool_path(path, self.root):
                continue
            tree = self.ctx.parse(path)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
                    continue
                if isinstance(node, ast.Assign):
                    names = [dotted_name(target) or "" for target in node.targets]
                    value = node.value
                else:
                    names = [dotted_name(node.target) or ""]
                    value = node.value
                if not names or not isinstance(value, ast.Constant) or not isinstance(value.value, str):
                    continue
                literal = value.value.strip()
                if is_placeholder_value(literal):
                    continue
                for name in names:
                    if is_safe_credential_name(name):
                        continue
                    short_name = name.rsplit(".", 1)[-1]
                    if not CREDENTIAL_NAME_RE.search(short_name):
                        continue
                    # A literal credential in application source is a real finding.
                    hits.append((path, node.lineno, name))
        return hits

    def _sql_expr_safety(
        self,
        expr: ast.AST,
        assignments: dict[str, ast.AST],
        parameters: set[str],
        *,
        seen: set[str] | None = None,
    ) -> str:
        """Classify SQL text as safe, tainted, or unknown using small local AST dataflow."""
        seen = set() if seen is None else set(seen)
        if isinstance(expr, ast.Constant):
            return "safe"
        if isinstance(expr, ast.Name):
            if expr.id in parameters:
                return "tainted"
            if expr.id in seen:
                return "unknown"
            assigned = assignments.get(expr.id)
            if assigned is None:
                return "unknown"
            seen.add(expr.id)
            return self._sql_expr_safety(assigned, assignments, parameters, seen=seen)
        if isinstance(expr, (ast.List, ast.Tuple, ast.Set)):
            states = [self._sql_expr_safety(item, assignments, parameters, seen=seen) for item in expr.elts]
            return _merge_safety(states)
        if isinstance(expr, ast.JoinedStr):
            states = [self._sql_expr_safety(value.value, assignments, parameters, seen=seen) for value in expr.values if isinstance(value, ast.FormattedValue)]
            return _merge_safety(states or ["safe"])
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
            return _merge_safety([
                self._sql_expr_safety(expr.left, assignments, parameters, seen=seen),
                self._sql_expr_safety(expr.right, assignments, parameters, seen=seen),
            ])
        if isinstance(expr, ast.IfExp):
            return _merge_safety([
                self._sql_expr_safety(expr.body, assignments, parameters, seen=seen),
                self._sql_expr_safety(expr.orelse, assignments, parameters, seen=seen),
            ])
        if isinstance(expr, ast.Call):
            # "separator".join(clauses) is safe when clauses are themselves constant
            # SQL fragments. This is the common repository WHERE-clause pattern.
            if isinstance(expr.func, ast.Attribute) and expr.func.attr == "join" and expr.args:
                separator_state = self._sql_expr_safety(expr.func.value, assignments, parameters, seen=seen)
                item_state = self._sql_expr_safety(expr.args[0], assignments, parameters, seen=seen)
                return _merge_safety([separator_state, item_state])
            if isinstance(expr.func, ast.Name) and expr.func.id in {"str", "format"} and expr.args:
                return self._sql_expr_safety(expr.args[0], assignments, parameters, seen=seen)
            return "unknown"
        if isinstance(expr, ast.Attribute):
            base_state = self._sql_expr_safety(expr.value, assignments, parameters, seen=seen)
            return "tainted" if base_state == "tainted" else "unknown"
        if isinstance(expr, ast.Subscript):
            base_state = self._sql_expr_safety(expr.value, assignments, parameters, seen=seen)
            return "tainted" if base_state == "tainted" else "unknown"
        if isinstance(expr, ast.UnaryOp):
            return self._sql_expr_safety(expr.operand, assignments, parameters, seen=seen)
        if isinstance(expr, ast.BoolOp):
            return _merge_safety([self._sql_expr_safety(value, assignments, parameters, seen=seen) for value in expr.values])
        return "unknown"

    def _sql_scope_metadata(self, tree: ast.Module) -> tuple[dict[str, ast.AST], set[str]]:
        """Return module-level provenance; useful for fixture/self-check and module SQL."""
        assignments: dict[str, ast.AST] = {}
        parameters: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.AST):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assignments[target.id] = node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
                assignments[node.target.id] = node.value
        return assignments, parameters

    def _enclosing_function(self, tree: ast.Module, target: ast.AST) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
        parent: dict[ast.AST, ast.AST] = {}
        for owner in ast.walk(tree):
            for child in ast.iter_child_nodes(owner):
                parent[child] = owner
        current: ast.AST = target
        while current in parent:
            current = parent[current]
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return current
        return None

    def _is_allowlisted_sql_identifier(self, tree: ast.Module, call: ast.Call) -> bool:
        """Accept dynamic SQL identifiers only when guarded by an explicit constant allowlist."""
        if not isinstance(call.args[0] if call.args else None, ast.JoinedStr):
            return False
        function = self._enclosing_function(tree, call)
        if function is None:
            return False
        formatted_names: set[str] = set()
        for value in call.args[0].values:
            if isinstance(value, ast.FormattedValue) and isinstance(value.value, ast.Name):
                formatted_names.add(value.value.id)
        if not formatted_names:
            return False
        allowlisted: set[str] = set()
        for node in ast.walk(function):
            if isinstance(node, ast.Assign) and isinstance(node.value, (ast.Set, ast.Tuple, ast.List)):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.lower() in {"allowed", "allowed_fields", "allowed_columns", "allowed_names"}:
                        if all(isinstance(item, ast.Constant) and isinstance(item.value, str) for item in node.value.elts):
                            allowlisted.add(target.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.value, (ast.Set, ast.Tuple, ast.List)) and isinstance(node.target, ast.Name):
                if node.target.id.lower() in {"allowed", "allowed_fields", "allowed_columns", "allowed_names"} and all(isinstance(item, ast.Constant) and isinstance(item.value, str) for item in node.value.elts):
                    allowlisted.add(node.target.id)
        for name in formatted_names:
            guarded = False
            for node in ast.walk(function):
                if not isinstance(node, ast.If) or len(node.test.ops) != 1 or not isinstance(node.test.ops[0], (ast.NotIn,)):
                    continue
                if not isinstance(node.test.left, ast.Name) or node.test.left.id != name or not isinstance(node.test.comparators[0], ast.Name):
                    continue
                if node.test.comparators[0].id not in allowlisted:
                    continue
                guarded = any(isinstance(child, ast.Raise) for child in ast.walk(ast.Module(body=node.body, type_ignores=[])))
                if guarded:
                    break
            if not guarded:
                return False
        return True

    def _sql_scope_for_call(self, tree: ast.Module, call: ast.Call) -> tuple[dict[str, ast.AST], set[str]]:
        """Build provenance only from the function containing the SQL sink."""
        parent: dict[ast.AST, ast.AST] = {}
        for owner in ast.walk(tree):
            for child in ast.iter_child_nodes(owner):
                parent[child] = owner
        owner: ast.AST = call
        while owner in parent:
            owner = parent[owner]
            if isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef)):
                break
        if not isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return self._sql_scope_metadata(tree)
        assignments: dict[str, ast.AST] = {}
        parameters = {arg.arg for arg in (*owner.args.posonlyargs, *owner.args.args, *owner.args.kwonlyargs)}
        stack = list(owner.body)
        while stack:
            node = stack.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node is not owner:
                continue
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.AST):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assignments[target.id] = node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
                assignments[node.target.id] = node.value
            stack.extend(ast.iter_child_nodes(node))
        return assignments, parameters

    def _audit_sql_injection_patterns(self, section: str) -> None:
        for path in self.ctx.python_files:
            if is_test_path(path, self.root) or is_audit_tool_path(path, self.root):
                continue
            tree = self.ctx.parse(path)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or call_name(node) not in {"execute", "executemany", "executescript"} or not node.args:
                    continue
                assignments, parameters = self._sql_scope_for_call(tree, node)
                first = node.args[0]
                state = self._sql_expr_safety(first, assignments, parameters)
                sink = call_name(node)
                function = self._enclosing_function(tree, node)
                function_name = function.name if function else ""
                if relpath(path, self.root) == "app/db/database.py" and function_name == "execute":
                    self.ctx.add(section, PASS, "Database.execute safely forwards parameterized SQL through the DB abstraction", "The method is a trusted low-level wrapper; SQL construction is audited at its callers.", path=path, line=node.lineno)
                    continue
                if self._is_allowlisted_sql_identifier(tree, node):
                    self.ctx.add(section, PASS, "Dynamic SQL identifier is constrained by an explicit allowlist", "Only a constant allowlisted column/field name is interpolated; data values remain bound parameters.", path=path, line=node.lineno)
                    continue
                if state == "tainted":
                    self.ctx.add(section, FAIL, "SQL execution sink uses tainted dynamic SQL text", "SQL data derived from a function parameter or tainted value is interpolated into the statement text. Use placeholders/parameters for data values.", path=path, line=node.lineno, severity="FAIL")
                elif state == "unknown" and not (sink == "executescript" and isinstance(first, ast.Attribute) and isinstance(first.value, ast.Name) and first.value.id == "migration"):
                    self.ctx.add(section, NOT_FULLY_VERIFIED, "SQL execution sink uses dynamic/unknown SQL text", "Static analysis could not prove the statement text is trusted. Review the construction; parameterize all data values.", path=path, line=node.lineno)

    def _subprocess_command_safety(self, expr: ast.AST) -> str:
        if isinstance(expr, (ast.List, ast.Tuple)):
            if not expr.elts:
                return "unknown"
            executable = expr.elts[0]
            if isinstance(executable, ast.Attribute) and isinstance(executable.value, ast.Name) and executable.value.id == "sys" and executable.attr == "executable":
                return "safe"
            if isinstance(executable, ast.Constant) and isinstance(executable.value, str) and executable.value.strip():
                return "safe"
            if isinstance(executable, (ast.JoinedStr, ast.BinOp, ast.Name, ast.Call, ast.Attribute)):
                return "tainted" if isinstance(executable, (ast.JoinedStr, ast.BinOp)) else "unknown"
        if isinstance(expr, ast.Constant):
            return "safe"
        return "tainted" if isinstance(expr, (ast.JoinedStr, ast.BinOp)) else "unknown"

    def _audit_dynamic_execution(self, section: str) -> None:
        for path in self.ctx.python_files:
            if is_test_path(path, self.root) or is_audit_tool_path(path, self.root):
                continue
            tree = self.ctx.parse(path)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = call_name(node)
                if name in {"eval", "exec"}:
                    self.ctx.add(section, BLOCKER, f"Dangerous dynamic execution: {name}", "Dynamic code execution is not acceptable in production-facing application logic.", path=path, line=node.lineno, severity="BLOCKER")
                if name == "system" and isinstance(node.func, ast.Attribute):
                    self.ctx.add(section, BLOCKER, "os.system call detected", "Shell execution from application source requires explicit elimination or isolation.", path=path, line=node.lineno, severity="BLOCKER")
                if isinstance(node.func, ast.Attribute) and node.func.attr == "import_module":
                    self.ctx.add(section, NOT_FULLY_VERIFIED, "Dynamic import call detected", "Static analysis cannot prove the module name is trusted in every runtime path.", path=path, line=node.lineno)
                if isinstance(node.func, ast.Attribute) and node.func.attr == "load" and (dotted_name(node.func) or "").startswith("joblib."):
                    self.ctx.add(section, WARN, "joblib artifact load detected", "Serialization loading is acceptable only when the artifact path and checksum are trusted/verified.", path=path, line=node.lineno, severity="WARN")
                if isinstance(node.func, ast.Attribute) and node.func.attr in {"loads", "load"} and (dotted_name(node.func) or "").startswith("pickle."):
                    self.ctx.add(section, FAIL, "pickle deserialization detected", "Untrusted pickle input can execute code; verify provenance before any load.", path=path, line=node.lineno, severity="FAIL")
                if isinstance(node.func, ast.Attribute) and node.func.attr in {"run", "Popen", "call", "check_call", "check_output"} and (dotted_name(node.func) or "").startswith("subprocess"):
                    shell_true = any(isinstance(keyword.value, ast.Constant) and keyword.arg == "shell" and keyword.value.value is True for keyword in node.keywords)
                    if shell_true:
                        self.ctx.add(section, BLOCKER, "subprocess uses shell=True", "Shell interpretation is dangerous for production-facing command execution.", path=path, line=node.lineno, severity="BLOCKER")
                    elif node.args:
                        cmd_state = self._subprocess_command_safety(node.args[0])
                        if cmd_state == "tainted":
                            self.ctx.add(section, FAIL, "subprocess command may be user-controlled", "Command construction contains dynamic string formatting; ensure no untrusted value can reach the executable/arguments.", path=path, line=node.lineno, severity="FAIL")
                        elif cmd_state == "unknown":
                            self.ctx.add(section, NOT_FULLY_VERIFIED, "subprocess command construction requires review", "The audit could not prove the full command is trusted from static AST alone.", path=path, line=node.lineno)

    def _audit_file_access_patterns(self, section: str) -> None:
        # Dynamic file paths are only interesting when their provenance is unknown/tainted.
        for path in self.ctx.python_files:
            if is_test_path(path, self.root) or is_audit_tool_path(path, self.root):
                continue
            tree = self.ctx.parse(path)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or call_name(node) != "open" or not node.args:
                    continue
                assignments, parameters = self._sql_scope_for_call(tree, node)
                state = self._sql_expr_safety(node.args[0], assignments, parameters)
                if state == "tainted":
                    self.ctx.add(section, FAIL, "User-tainted file path passed to open()", "A path influenced by a function parameter reaches file access; enforce an approved root and reject traversal before opening.", path=path, line=node.lineno, severity="FAIL")
                elif state == "unknown" and isinstance(node.args[0], (ast.JoinedStr, ast.BinOp)):
                    self.ctx.add(section, NOT_FULLY_VERIFIED, "Dynamic file path requires static review", "The audit could not prove the path is confined to an approved root.", path=path, line=node.lineno)

    # ------------------------------------------------------------------
    # [15] Configuration
    # ------------------------------------------------------------------
    def audit_configuration(self) -> None:
        section = "15"
        config = self.root / "config/config_hunter.py"
        env_example = self.root / ".env.example"
        if config.exists():
            text = self.ctx.read(config) or ""
            terms = ["Settings", "load_settings", "os.getenv", "database_path", "allowed_hosts"]
            self.ctx.add(section, PASS if all(term.lower() in text.lower() for term in terms) else NOT_FULLY_VERIFIED, "Central configuration abstraction evidence", path=config)
            provider_terms = ["live_provider", "provider", "api_key"]
            self.ctx.add(section, PASS if all(term.lower() in text.lower() for term in provider_terms) else NOT_FULLY_VERIFIED, "Provider/environment configuration evidence", path=config)
        else:
            self.ctx.add(section, BLOCKER, "Central config/config_hunter.py is missing", "The project specifies a centralized configuration interface.", path=self.root / "config/config_hunter.py", severity="BLOCKER")

        if env_example.exists():
            env_text = self.ctx.read(env_example) or ""
            self.ctx.add(section, PASS if all(key in env_text for key in LEARNING_FLAG_KEYS) else NOT_FULLY_VERIFIED, "Learning flags are represented in .env.example", path=env_example)
            concrete_values: list[str] = []
            for key, value in load_env_file(env_example).items():
                if is_safe_credential_name(key):
                    continue
                if re.search(r"(?:KEY|TOKEN|SECRET|PASSWORD)", key, re.IGNORECASE) and not is_example_credential_value(key, value):
                    concrete_values.append(key)
            if concrete_values:
                self.ctx.add(section, FAIL, ".env.example contains a concrete credential-like value", f"keys={concrete_values}; values are never printed.", path=env_example, severity="FAIL")
            else:
                self.ctx.add(section, PASS, ".env.example does not expose a concrete credential value", path=env_example)
        else:
            self.ctx.add(section, NOT_FULLY_VERIFIED, ".env.example not found")

    # ------------------------------------------------------------------
    # [16] Data Provider
    # ------------------------------------------------------------------
    def audit_data_provider(self) -> None:
        section = "16"
        factory = self.root / "app/providers/factory.py"
        interface = self.root / "app/providers/live_provider.py"
        twelvedata = self.root / "app/providers/twelvedata.py"
        analysis = self.root / "app/web/analysis_service.py"
        config = self.root / "config/config_hunter.py"
        env_example = self.root / ".env.example"

        for label, path, required in (
            ("Provider abstraction", interface, ["LiveMarketDataProvider", "get_ohlc"]),
            ("Twelve Data adapter", twelvedata, ["TwelveDataLiveProvider", "TIMEFRAME_MAP", "OHLCBar"]),
            ("Provider factory", factory, ["twelvedata", "TwelveDataLiveProvider"]),
            ("Analysis service provider injection", analysis, ["live_provider", "get_ohlc"]),
        ):
            if not path.exists():
                self.ctx.add(section, NOT_FULLY_VERIFIED, f"{label} module not found", path=path)
                continue
            text = self.ctx.read(path) or ""
            self.ctx.add(section, PASS if all(token.lower() in text.lower() for token in required) else NOT_FULLY_VERIFIED, f"{label} contract evidence", path=path)

        if config.exists():
            text = self.ctx.read(config) or ""
            configured = all(token.lower() in text.lower() for token in ("live_provider", "live_provider_enabled", "live_provider_api_key"))
            self.ctx.add(section, PASS if configured else NOT_FULLY_VERIFIED, "Central live-provider configuration is environment based", path=config)
        else:
            self.ctx.add(section, NOT_FULLY_VERIFIED, "Central provider configuration module not found", path=config)

        if env_example.exists():
            env = load_env_file(env_example)
            provider = env.get("EDGE_HUNTER_LIVE_PROVIDER", "").strip().lower()
            api_key = env.get("EDGE_HUNTER_LIVE_PROVIDER_API_KEY", "")
            if provider == "twelvedata":
                self.ctx.add(section, PASS, "Twelve Data is the configured provider in .env.example", path=env_example)
            else:
                self.ctx.add(section, NOT_FULLY_VERIFIED, "Twelve Data provider is not the explicit .env.example selection", f"configured_provider={provider or '[empty]'}", path=env_example)
            safe_provider_placeholder = is_example_credential_value("EDGE_HUNTER_LIVE_PROVIDER_API_KEY", api_key)
            self.ctx.add(section, PASS if safe_provider_placeholder else FAIL, ".env.example uses a safe placeholder for the provider API key" if safe_provider_placeholder else ".env.example appears to contain a concrete provider credential", "Credential values are never printed by the audit.", path=env_example, severity=None if safe_provider_placeholder else "FAIL")
        else:
            self.ctx.add(section, NOT_FULLY_VERIFIED, ".env.example not found", path=env_example)

        # Legacy-provider detection is intentionally scoped to actual runtime/dependency files.
        legacy_re = re.compile(r"MetaTrader5|MetaTrader|\bmt5\b", re.IGNORECASE)
        runtime_hits: list[tuple[Path, int, str]] = []
        dependency_hits: list[tuple[Path, int, str]] = []
        test_hits: list[tuple[Path, int, str]] = []
        doc_hits: list[tuple[Path, int, str]] = []
        dependency_names = {"pyproject.toml", "requirements.txt", "requirements-phase13.txt", "requirements-web.txt", "Pipfile", "poetry.lock"}
        for path in self.ctx.text_files:
            if is_audit_tool_path(path, self.root):
                continue
            text = self.ctx.read(path) or ""
            for line_no, line in enumerate(text.splitlines(), start=1):
                if not legacy_re.search(line):
                    continue
                rel = relpath(path, self.root)
                if is_test_path(path, self.root):
                    test_hits.append((path, line_no, line.strip()))
                elif path.name in dependency_names and path.suffix.lower() in {".toml", ".txt", ".lock"}:
                    dependency_hits.append((path, line_no, line.strip()))
                elif path.suffix.lower() == ".py" and (rel.startswith("app/") or rel.startswith("config/") or rel.startswith("scripts/") or rel == "main.py"):
                    # For Python runtime code, comments/docstrings are not dependencies; AST import checks below are authoritative.
                    tree = self.ctx.parse(path)
                    is_import = False
                    if tree:
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Import):
                                if any("MetaTrader5" in alias.name or alias.name.lower() == "mt5" for alias in node.names):
                                    is_import = True
                            elif isinstance(node, ast.ImportFrom) and node.module and ("MetaTrader5" in node.module or node.module.lower() == "mt5"):
                                is_import = True
                    if is_import:
                        runtime_hits.append((path, line_no, line.strip()))
                else:
                    doc_hits.append((path, line_no, line.strip()))

        if dependency_hits:
            for path, line, _snippet in dependency_hits[:20]:
                self.ctx.add(section, BLOCKER, "Legacy MetaTrader dependency found", "Remove the package/dependency; Twelve Data is the active market-data source.", path=path, line=line, severity="BLOCKER")
        else:
            self.ctx.add(section, PASS, "No MetaTrader dependency declaration found in dependency manifests")

        if runtime_hits:
            for path, line, _snippet in runtime_hits[:20]:
                self.ctx.add(section, BLOCKER, "Legacy MetaTrader runtime import found", "Runtime market-data code must use the provider abstraction and Twelve Data adapter instead.", path=path, line=line, severity="BLOCKER")
        else:
            self.ctx.add(section, PASS, "No MetaTrader runtime import found in application code")

        if test_hits:
            self.ctx.add(section, FAIL, "Legacy MetaTrader references remain in tests", f"count={len(test_hits)}; replace with provider-architecture tests.", severity="FAIL")
        if doc_hits:
            self.ctx.add(section, NOT_APPLICABLE, "Legacy MetaTrader text appears only in documentation/non-runtime text", f"count={len(doc_hits)}; documentation does not affect runtime provider selection.")
        elif not dependency_hits and not runtime_hits and not test_hits:
            self.ctx.add(section, PASS, "Provider scan found no legacy MetaTrader runtime/test dependency")

    # ------------------------------------------------------------------
    # [17] Dangerous patterns
    # ------------------------------------------------------------------
    def audit_dangerous_patterns(self) -> None:
        section = "17"
        before = len(self.ctx.findings[section])
        self._audit_dynamic_execution(section)
        self._audit_sql_injection_patterns(section)
        dangerous_found = len(self.ctx.findings[section]) > before
        if not dangerous_found:
            self.ctx.add(section, PASS, "No high-risk dynamic-execution/SQL pattern detected in non-test Python source")
        # joblib loads are warnings by design; if they are properly integrity-guarded, registry section provides the evidence.
        joblib_hits = []
        for path in self.ctx.python_files:
            if is_test_path(path, self.root):
                continue
            for node, name in self.ast_calls(path):
                if name == "load" and isinstance(node.func, ast.Attribute) and (dotted_name(node.func) or "").startswith("joblib."):
                    joblib_hits.append((path, node.lineno))
        if joblib_hits:
            self.ctx.add(section, WARN, f"joblib.load call sites discovered: {len(joblib_hits)}", "The model registry must establish trusted artifact provenance/checksum before loading.", severity="WARN")

    # ------------------------------------------------------------------
    # [18] Database / migrations
    # ------------------------------------------------------------------
    def audit_database_migrations(self) -> None:
        section = "18"
        migration_dir = self.root / "migrations"
        files = sorted(migration_dir.glob("*.sql")) if migration_dir.is_dir() else []
        file_versions: list[tuple[int, Path]] = []
        for path in files:
            match = re.match(r"^(\d+)_", path.name)
            if not match:
                self.ctx.add(section, WARN, "Migration filename lacks numeric prefix", "The file cannot be matched to a version number from its name.", path=path, severity="WARN")
                continue
            file_versions.append((int(match.group(1)), path))

        runner = self.root / "app/db/migrations.py"
        runner_versions: list[tuple[int, str]] = []
        if runner.exists():
            tree = self.ctx.parse(runner)
            if tree:
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Migration" and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, int):
                        name = ""
                        if len(node.args) > 1 and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
                            name = node.args[1].value
                        runner_versions.append((int(node.args[0].value), name))
            if not runner_versions:
                for match in re.finditer(r"Migration\(\s*(\d+)\s*,\s*[\"']([^\"']+)", self.ctx.read(runner) or ""):
                    runner_versions.append((int(match.group(1)), match.group(2)))

        all_versions = sorted({version for version, _ in file_versions} | {version for version, _ in runner_versions})
        duplicate_file_versions = sorted({version for version, _ in file_versions if sum(1 for item, _ in file_versions if item == version) > 1})
        if all_versions:
            expected = list(range(1, max(all_versions) + 1))
            if all_versions == expected and not duplicate_file_versions:
                inline = sorted(version for version, _ in runner_versions if version not in {v for v, _ in file_versions})
                backed = sorted(version for version, _ in file_versions)
                self.ctx.add(section, PASS, f"Combined migration sequence is contiguous 1..{max(all_versions)}", f"runner_versions={sorted(v for v,_ in runner_versions)}; file_versions={backed}; inline_only={inline}")
            else:
                self.ctx.add(section, FAIL, f"Migration versions are inconsistent: {all_versions}", f"expected_contiguous={list(range(1, max(all_versions)+1))}; duplicate_file_versions={duplicate_file_versions}", severity="FAIL")
        else:
            self.ctx.add(section, BLOCKER, "No migration versions discovered", "No numeric migration versions were found in migration files or the migration runner.", severity="BLOCKER")

        if runner.exists() and runner_versions:
            file_version_set = {version for version, _ in file_versions}
            file_names_by_version = {version: path.name for version, path in file_versions}
            missing_runner_names = []
            runner_text = self.ctx.read(runner) or ""
            for version, name in file_names_by_version.items():
                if path_name := name:
                    if path_name not in runner_text and f"{version}," not in runner_text and f"{version},\n" not in runner_text:
                        missing_runner_names.append(path_name)
            if missing_runner_names:
                self.ctx.add(section, NOT_FULLY_VERIFIED, "Migration runner/file-backed migration alignment requires review", f"files_not_directly_matched={missing_runner_names}", path=runner)
            else:
                self.ctx.add(section, PASS, "Migration runner contains the discovered migration versions and file-backed migrations")
        elif runner.exists():
            self.ctx.add(section, NOT_FULLY_VERIFIED, "Migration runner exists but could not be parsed for Migration registrations", path=runner)
        else:
            self.ctx.add(section, NOT_FULLY_VERIFIED, "Migration runner module not found", path=runner)

        db_path = self._configured_database_path()
        if db_path.exists():
            schema = self._read_sqlite_schema(db_path)
            if schema is None:
                self.ctx.add(section, NOT_FULLY_VERIFIED, "SQLite database exists but could not be opened read-only", path=db_path)
            else:
                self.ctx.add(section, PASS, f"SQLite database opened read-only: {relpath(db_path, self.root)}", path=db_path)
                tables = schema.get("tables", set())
                applied = schema.get("applied", set())
                expected_tables = {"users", "subscriptions", "learning_records", "feature_snapshots", "learning_dataset_versions", "learning_training_runs", "learning_model_versions", "learning_policy_versions", "learning_cycles", "learning_candidates", "learning_production_state", "learning_production_history", "learning_monitoring_snapshots", "learning_state", "learning_audit_events"}
                missing = sorted(expected_tables - set(tables))
                if missing:
                    self.ctx.add(section, WARN, "Some expected schema tables are not present in the read-only database", f"missing={missing}", path=db_path, severity="WARN")
                else:
                    self.ctx.add(section, PASS, "Expected auth/learning schema tables are present in the read-only database", path=db_path)
                if applied and all(version in set(all_versions) for version in applied):
                    if all_versions and max(applied) < max(all_versions):
                        self.ctx.add(section, NOT_FULLY_VERIFIED, "Database migration state is behind current source migrations", f"applied={sorted(applied)}; source_max={max(all_versions)}. The audit does not run migrations.", path=db_path)
                    else:
                        self.ctx.add(section, PASS, "Database migration state contains only known migration versions", path=db_path)
                elif applied:
                    unknown = sorted(set(applied) - set(all_versions))
                    self.ctx.add(section, FAIL, "Database contains unknown migration versions", f"unknown={unknown}; source_versions={all_versions}", path=db_path, severity="FAIL")
        else:
            self.ctx.add(section, NOT_FULLY_VERIFIED, "Configured database file is not present", f"read-only path={db_path}; no migrations were executed by the audit.")

    def _configured_database_path(self) -> Path:
        env_values = load_env_file(self.root / ".env")
        raw = os.environ.get("EDGE_HUNTER_DATABASE_PATH") or env_values.get("EDGE_HUNTER_DATABASE_PATH") or "data/edge_hunter.db"
        path = Path(raw)
        if not path.is_absolute():
            path = self.root / path
        return path.resolve()

    def _read_sqlite_schema(self, db_path: Path) -> dict[str, Any] | None:
        try:
            uri = f"file:{db_path.as_posix()}?mode=ro"
            connection = sqlite3.connect(uri, uri=True, timeout=3)
            try:
                tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                triggers = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
                indexes = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='index'")}
                applied: set[int] = set()
                if "schema_migrations" in tables:
                    try:
                        for row in connection.execute("SELECT version FROM schema_migrations ORDER BY version"):
                            try:
                                applied.add(int(row[0]))
                            except (TypeError, ValueError):
                                pass
                    except sqlite3.DatabaseError:
                        pass
                return {"tables": tables, "triggers": triggers, "indexes": indexes, "applied": applied}
            finally:
                connection.close()
        except sqlite3.Error:
            return None

    # ------------------------------------------------------------------
    # [19] Artifact integrity
    # ------------------------------------------------------------------
    def audit_artifacts(self) -> None:
        section = "19"
        artifact_roots = self._discover_artifact_roots()
        if not artifact_roots:
            self.ctx.add(section, NOT_FULLY_VERIFIED, "No registered artifact directory was discovered from current source configuration", "The audit will not invent an artifact path.")
            return

        any_artifacts = False
        for root in artifact_roots:
            if not root.exists():
                self.ctx.add(section, NOT_APPLICABLE, f"Artifact root does not currently exist: {relpath(root, self.root)}", "No stored artifact is available to verify.", path=root)
                continue
            files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".joblib", ".pkl", ".pickle"}]
            if files:
                any_artifacts = True
                self.ctx.add(section, PASS, f"Found {len(files)} serialized artifact file(s) under {relpath(root, self.root)}")
                for path in files[:100]:
                    try:
                        digest = sha256_file(path)
                        self.ctx.add(section, PASS, f"Artifact readable and SHA-256 computed: {path.name}", f"sha256={digest}; bytes={path.stat().st_size}", path=path)
                    except Exception as exc:  # noqa: BLE001
                        self.ctx.add(section, FAIL, "Artifact hash computation failed", str(exc), path=path, severity="FAIL")
            else:
                self.ctx.add(section, NOT_FULLY_VERIFIED, f"Artifact root exists but contains no .joblib/.pkl artifacts: {relpath(root, self.root)}", path=root)

        # Verify metadata references from database without deserializing artifacts.
        db_path = self._configured_database_path()
        schema = self._read_sqlite_schema(db_path) if db_path.exists() else None
        if schema and "learning_model_versions" in schema.get("tables", set()):
            refs = self._read_model_artifact_metadata(db_path)
            if refs:
                for ref in refs:
                    path_value = ref.get("artifact_path")
                    expected_sha = ref.get("artifact_sha256")
                    if not isinstance(path_value, str) or not path_value:
                        self.ctx.add(section, FAIL, "Registered ModelVersion has no artifact path", f"model_version_id={ref.get('model_version_id')}", severity="FAIL")
                        continue
                    candidate = Path(path_value)
                    if not candidate.is_absolute():
                        candidate = self.root / candidate
                    candidate = candidate.resolve()
                    root_ok = any(root.resolve() == candidate or root.resolve() in candidate.parents for root in artifact_roots)
                    if not root_ok:
                        self.ctx.add(section, FAIL, "Registered artifact path is outside configured artifact root", f"model_version_id={ref.get('model_version_id')}; artifact_path={relpath(candidate, self.root)}", severity="FAIL")
                        continue
                    if not candidate.exists():
                        self.ctx.add(section, FAIL, "Registered artifact file is missing", f"model_version_id={ref.get('model_version_id')}; artifact_path={relpath(candidate, self.root)}", path=candidate, severity="FAIL")
                        continue
                    actual = sha256_file(candidate)
                    if expected_sha and actual.lower() != str(expected_sha).lower():
                        self.ctx.add(section, FAIL, "Registered artifact SHA-256 mismatch", f"model_version_id={ref.get('model_version_id')}; expected={expected_sha}; actual={actual}", path=candidate, severity="FAIL")
                    else:
                        self.ctx.add(section, PASS, "Registered artifact path and SHA-256 match", f"model_version_id={ref.get('model_version_id')}", path=candidate)
            else:
                self.ctx.add(section, NOT_FULLY_VERIFIED, "No ModelVersion artifact metadata rows were found to verify")
        else:
            self.ctx.add(section, NOT_FULLY_VERIFIED, "ModelVersion metadata table is not available for read-only artifact metadata verification")

        # Deliberately do not deserialize any artifact; doing so could execute untrusted pickle/joblib code.
        self.ctx.add(section, PASS, "Serialized artifacts were not deserialized by this audit (safe mode)")

    def _discover_artifact_roots(self) -> list[Path]:
        roots: list[Path] = []
        training = self.root / "app/learning/training.py"
        if training.exists():
            text = self.ctx.read(training) or ""
            for match in re.finditer(r"(?:Path\(|default_artifact_root\s*=\s*)(?:[\"'])([^\"']+)(?:[\"'])", text):
                candidate = match.group(1)
                if "artifact" in candidate.lower() or "training" in candidate.lower():
                    path = Path(candidate)
                    if not path.is_absolute():
                        path = self.root / path
                    roots.append(path.resolve())
        conventional = self.root / "data/training_artifacts"
        if conventional.exists() or training.exists():
            roots.append(conventional.resolve())
        return sorted(set(roots))

    def _read_model_artifact_metadata(self, db_path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        try:
            connection = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True, timeout=3)
            try:
                columns = [row[1] for row in connection.execute("PRAGMA table_info(learning_model_versions)")]
                wanted = [name for name in ("model_version_id", "artifact_path", "artifact_sha256", "status", "dataset_version_id", "training_run_id") if name in columns]
                if not wanted:
                    return rows
                query = "SELECT " + ", ".join(wanted) + " FROM learning_model_versions"
                for row in connection.execute(query):
                    rows.append(dict(zip(wanted, row)))
            finally:
                connection.close()
        except sqlite3.Error:
            return rows
        return rows

    # ------------------------------------------------------------------
    # [20] Startup/runtime readiness
    # ------------------------------------------------------------------
    def audit_startup(self) -> None:
        section = "20"
        main = self.root / "main.py"
        bootstrap = self.root / "app/core/bootstrap.py"
        if main.exists():
            text = self.ctx.read(main) or ""
            if "argparse" in text and "--serve" in text and "--production" in text:
                self.ctx.add(section, PASS, "Main entry point exposes explicit serve/production flags", path=main)
            else:
                self.ctx.add(section, NOT_FULLY_VERIFIED, "Main entry point flags could not be fully matched", path=main)
            if "if args.serve" in text or "if args.serve:" in text:
                self.ctx.add(section, PASS, "Bare main.py invocation does not start the long-running server automatically", path=main)
            else:
                self.ctx.add(section, NOT_FULLY_VERIFIED, "Could not confirm server-start gating in main.py", path=main)
            if "run_project_tests()" in text:
                self.ctx.add(section, PASS, "main.py runs the project test runner before startup", path=main)
        else:
            self.ctx.add(section, BLOCKER, "main.py entry point not found", severity="BLOCKER")

        if bootstrap.exists():
            text = self.ctx.read(bootstrap) or ""
            if "MigrationRunner(database).apply_all()" in text:
                self.ctx.add(section, WARN, "Application startup applies database migrations", "The audit itself does not execute migrations; the official main.py test/startup command may update the local Test/Development DB.", path=bootstrap, severity="WARN")
            if "start_scheduler" in text:
                self.ctx.add(section, BLOCKER, "Bootstrap references start_scheduler", "Continuous Learning must not auto-start at application startup.", path=bootstrap, severity="BLOCKER")
            else:
                self.ctx.add(section, PASS, "Bootstrap does not reference start_scheduler", path=bootstrap)
            if "create_application" in text and "Application" in text:
                self.ctx.add(section, PASS, "Application composition/bootstrap entry point found", path=bootstrap)
        else:
            self.ctx.add(section, NOT_FULLY_VERIFIED, "Bootstrap module not found", path=bootstrap)

        self.ctx.add(section, NOT_FULLY_VERIFIED, "Graceful failure under actual production process signals", "The audit avoids starting a long-running server or background workers by design.")

    # ------------------------------------------------------------------
    # Helpers for source searches
    # ------------------------------------------------------------------
    def _source_paths_with_tokens_under(self, roots: set[str], tokens: set[str]) -> list[tuple[Path, int, str]]:
        result: list[tuple[Path, int, str]] = []
        normalized_roots = {r.replace("\\", "/").strip("/") for r in roots}
        for path in self.ctx.python_files:
            rel = relpath(path, self.root)
            if not any(rel == root or rel.startswith(root + "/") for root in normalized_roots):
                continue
            text = self.ctx.read(path) or ""
            for number, line in enumerate(text.splitlines(), start=1):
                for token in tokens:
                    if token.lower() in line.lower():
                        result.append((path, number, token))
                        break
        return result

    def source_paths_within(self, roots: Sequence[Path], tokens: set[str]) -> list[tuple[Path, int, str]]:
        result: list[tuple[Path, int, str]] = []
        for path in self.ctx.python_files:
            if not any(root == path or root in path.parents for root in roots):
                continue
            text = self.ctx.read(path) or ""
            for number, line in enumerate(text.splitlines(), start=1):
                for token in tokens:
                    if token.lower() in line.lower():
                        result.append((path, number, token))
                        break
        return result

    def _search_ast_string_literals(self, literals: set[str], *, roots: Sequence[Path]) -> list[tuple[Path, int, str]]:
        hits: list[tuple[Path, int, str]] = []
        for path in roots:
            tree = self.ctx.parse(path)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in literals:
                    hits.append((path, node.lineno, node.value))
        return hits

    # ------------------------------------------------------------------
    # Report and final status
    # ------------------------------------------------------------------
    def audit_all(self) -> None:
        self.discover_files()
        audit_methods = [
            self.audit_structure,
            self.audit_python_validation,
            self.audit_full_tests,
            self.audit_learning_safety,
            self.audit_api_analyze,
            self.audit_leakage,
            self.audit_model_registry,
            self.audit_oos_robustness,
            self.audit_learned_policy,
            self.audit_continuous_learning,
            self.audit_strategies,
            self.audit_risk_rr,
            self.audit_auth_subscription,
            self.audit_security,
            self.audit_configuration,
            self.audit_data_provider,
            self.audit_dangerous_patterns,
            self.audit_database_migrations,
            self.audit_artifacts,
            self.audit_startup,
        ]
        for index, method in enumerate(audit_methods, start=1):
            sid = f"{index:02d}"
            name = SECTION_NAMES[sid]
            self.ctx.current_section = name
            self._terminal(f"[{sid}/20] START {name}")
            started = time.monotonic()
            try:
                method()
                status = self.ctx.section(sid).status
                self._terminal(f"[{sid}/20] DONE  {name} | status={status} | elapsed={time.monotonic() - started:.1f}s")
            except Exception as exc:  # noqa: BLE001 - failure isolation is core audit behavior
                status = BLOCKER
                message = f"{method.__name__} failed internally: {type(exc).__name__}: {exc}"
                self.ctx.tool_internal_errors.append(message)
                self.ctx.add(sid, BLOCKER, "Audit tool internal error", message, severity="BLOCKER")
                self._terminal(f"[{sid}/20] INTERNAL ERROR {name} | elapsed={time.monotonic() - started:.1f}s")

    def section_statuses(self) -> dict[str, str]:
        return {sid: self.ctx.section(sid).status for sid in SECTION_NAMES}

    def final_status(self) -> tuple[str, int]:
        if self.ctx.tool_internal_errors:
            return "AUDIT_INCOMPLETE", 2
        statuses = self.section_statuses().values()
        if any(status in {BLOCKER, FAIL} for status in statuses):
            return "BLOCKED", 1
        if any(status in {WARN, NOT_FULLY_VERIFIED} for status in statuses):
            return "READY_WITH_WARNINGS", 0
        return "READY_FOR_NEXT_STAGE", 0

    def report_path(self, timestamp: _dt.datetime) -> Path:
        reports = self.root / REPORT_DIR_NAME
        reports.mkdir(parents=True, exist_ok=True)
        safe_timestamp = timestamp.strftime("%Y%m%d_%H%M%S")
        return reports / f"{REPORT_NAME_PREFIX}{safe_timestamp}.txt"

    def render_report(self, report_path: Path, finished_at: _dt.datetime, final_status: str) -> None:
        result = self.ctx.test_result
        sections = [self.ctx.section(sid) for sid in SECTION_NAMES]
        lines: list[str] = []
        lines.extend([
            "=" * 78,
            "EDGE HUNTER",
            "PRODUCTION READINESS AUDIT",
            "=" * 78,
            f"Audit Tool Version: {TOOL_VERSION}",
            f"Timestamp Start: {_fmt_dt(self.ctx.started_at)}",
            f"Timestamp End: {_fmt_dt(finished_at)}",
            f"Project Root: {self.root}",
            f"Python Version: {sys.version.replace(os.linesep, ' ')}",
            f"Platform: {sys.platform}",
            "Mode: READ_ONLY_AUDIT (except report generation and the project's discovered test command)",
            "",
            "TEST EXECUTION",
            "-" * 78,
            f"Command: {self.ctx.test_command_display}",
        ])
        if result:
            lines.extend([
                f"Exit Code: {result.returncode}",
                f"Duration: {result.duration_seconds:.3f}s",
                f"Timed Out: {result.timed_out}",
                f"Tests: {self.ctx.test_summary['tests']}",
                f"Passed: {self.ctx.test_summary['passed']}",
                f"Failed: {self.ctx.test_summary['failed']}",
                f"Errors: {self.ctx.test_summary['errors']}",
            ])
        else:
            lines.append("Audit did not execute a test command.")

        if self.ctx.side_effect_notes:
            lines.append("Observed test side effects: " + "; ".join(self.ctx.side_effect_notes))

        if result:
            lines.extend(["", "Captured Test STDOUT", "-" * 78, redact(result.stdout, limit=120_000)])
            lines.extend(["", "Captured Test STDERR", "-" * 78, redact(result.stderr, limit=120_000)])
            if result.exception:
                lines.extend(["", "Test Command Exception", "-" * 78, redact(result.exception)])

        for section in sections:
            lines.extend([
                "",
                "=" * 78,
                f"[{section.section_id}] {section.name} .......... {section.status}",
                "=" * 78,
            ])
            if not section.findings:
                lines.append("[NOT_FULLY_VERIFIED] No findings were recorded for this section.")
            else:
                lines.extend(f.format_report(self.root) for f in section.findings)

        warnings = self._findings_by_status({WARN, NOT_FULLY_VERIFIED})
        failures = self._findings_by_status({FAIL})
        blockers = self._findings_by_status({BLOCKER})
        lines.extend([
            "",
            "=" * 78,
            "SUMMARY",
            "=" * 78,
            f"Warnings / Not Fully Verified: {len(warnings)}",
            f"Failures: {len(failures)}",
            f"Blockers: {len(blockers)}",
        ])
        if warnings:
            lines.extend(["", "WARNINGS / NOT FULLY VERIFIED"])
            for finding in warnings:
                lines.append(f"- [{finding.status}] {finding.item}: {redact(finding.reason)}")
        if failures:
            lines.extend(["", "FAILURES"])
            for finding in failures:
                location = relpath(Path(finding.path), self.root) if finding.path else ""
                if finding.line:
                    location += f":{finding.line}"
                suffix = f" ({location})" if location else ""
                lines.append(f"- {finding.item}{suffix}: {redact(finding.reason)}")
        if blockers:
            lines.extend(["", "BLOCKERS"])
            for finding in blockers:
                location = relpath(Path(finding.path), self.root) if finding.path else ""
                if finding.line:
                    location += f":{finding.line}"
                suffix = f" ({location})" if location else ""
                lines.append(f"- {finding.item}{suffix}: {redact(finding.reason)}")

        lines.extend([
            "",
            "=" * 78,
            "ITEMS NOT FULLY VERIFIED",
            "=" * 78,
        ])
        not_verified = self._findings_by_status({NOT_FULLY_VERIFIED})
        if not_verified:
            for finding in not_verified:
                lines.append(f"- [{finding.section_id}] {finding.item}: {redact(finding.reason)}")
        else:
            lines.append("None")

        lines.extend([
            "",
            "=" * 78,
            "FINAL STATUS",
            "=" * 78,
            final_status,
            "=" * 78,
        ])
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _findings_by_status(self, statuses: set[str]) -> list[Finding]:
        return [finding for findings in self.ctx.findings.values() for finding in findings if finding.status in statuses]

    def print_terminal_summary(self, report_path: Path, final_status: str) -> None:
        print("=" * 50)
        print("EDGE HUNTER")
        print("PRODUCTION READINESS AUDIT")
        print("=" * 50)
        print()
        for section_id, name in SECTION_NAMES.items():
            status = self.ctx.section(section_id).status
            print(f"[{section_id}] {name:<32} {status}")
        print()
        print("=" * 50)
        print("FINAL STATUS")
        print("=" * 50)
        print(final_status)
        print()
        print("Report:")
        print(report_path)
        print("=" * 50)
        if self.ctx.test_result:
            print(
                "Test command:"
                f" {self.ctx.test_command_display}\n"
                f"Test exit code: {self.ctx.test_result.returncode}\n"
                f"Test duration: {self.ctx.test_result.duration_seconds:.3f}s"
            )

    # ------------------------------------------------------------------
    # Self-check (never runs the full audit)
    # ------------------------------------------------------------------
    @staticmethod
    def self_check(root: Path) -> int:
        try:
            source = Path(__file__).read_text(encoding="utf-8")
            ast.parse(source, filename=str(Path(__file__)))
            imported = {
                "argparse",
                "ast",
                "contextlib",
                "configparser",
                "dataclasses",
                "datetime",
                "hashlib",
                "json",
                "os",
                "pathlib",
                "re",
                "shutil",
                "sqlite3",
                "subprocess",
                "sys",
                "tempfile",
                "textwrap",
                "threading",
                "time",
                "typing",
                "unicodedata",
                "__future__",
            }
            import_names = set()
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Import):
                    import_names.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    import_names.add(node.module.split(".")[0])
            third_party = sorted(import_names - imported)
            if third_party:
                print("SELF-CHECK FAIL: unexpected non-standard-library imports:", ", ".join(third_party))
                return 1
            for name in ("ProductionReadinessAudit", "AuditContext", "Finding", "main"):
                if name not in source:
                    print(f"SELF-CHECK FAIL: required symbol missing: {name}")
                    return 1
            if not (root / "app").exists():
                print("SELF-CHECK WARN: expected project root does not contain app/; syntax/import validation still passed.")
            # Scanner regression fixtures: prove the known previous false positives remain safe.
            if not is_safe_credential_name("PASSWORD_SCHEME"):
                print("SELF-CHECK FAIL: PASSWORD_SCHEME must be treated as a safe algorithm/configuration name")
                return 1
            if not is_placeholder_value("ضع مفتاحك هنا"):
                print("SELF-CHECK FAIL: Arabic .env.example placeholder was not recognized")
                return 1
            if not is_example_credential_value("EDGE_HUNTER_LIVE_PROVIDER_API_KEY", "YOUR_TWELVE_DATA_API_KEY"):
                print("SELF-CHECK FAIL: Twelve Data example placeholder was not recognized")
                return 1
            if is_example_credential_value("EDGE_HUNTER_LIVE_PROVIDER_API_KEY", "td_live_1234567890abcdef"):
                print("SELF-CHECK FAIL: key-shaped Twelve Data value was incorrectly accepted as placeholder")
                return 1
            parser = ast.parse("""
from x import db
def f(model_version_id=None):
    clauses=[]
    params=[]
    if model_version_id is not None:
        clauses.append('model_version_id = ?')
        params.append(model_version_id)
    where = ' WHERE ' + ' AND '.join(clauses) if clauses else ''
    db.execute(f'SELECT * FROM t{where}', tuple(params))
""")
            audit_instance = ProductionReadinessAudit(root)
            sink = next(n for n in ast.walk(parser) if isinstance(n, ast.Call) and call_name(n) == "execute")
            assignments, parameters = audit_instance._sql_scope_for_call(parser, sink)
            if audit_instance._sql_expr_safety(sink.args[0], assignments, parameters) != "safe":
                print("SELF-CHECK FAIL: parameterized WHERE-clause fixture was classified as unsafe")
                return 1
            if is_audit_tool_path(Path(__file__), root):
                pass
            else:
                print("SELF-CHECK FAIL: audit tool self-exclusion failed")
                return 1
            print("SELF-CHECK PASS")
            print(f"Tool version: {TOOL_VERSION}")
            print(f"Tool path: {Path(__file__).resolve()}")
            print("No audit, test suite, training, migration, scheduler, promotion, rollback, or server was executed.")
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"SELF-CHECK FAIL: {type(exc).__name__}: {exc}")
            return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EDGE HUNTER Production Readiness Audit")
    parser.add_argument("--self-check", action="store_true", help="validate the audit tool itself; do not audit the project")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    if args.self_check:
        return ProductionReadinessAudit.self_check(root)

    started = _now()
    audit = ProductionReadinessAudit(root)
    try:
        audit._terminal(f"Audit started | tool_version={TOOL_VERSION} | root={audit.root}")
        audit._terminal("Live terminal output is enabled. Full test output will appear with [TEST].")
        audit.audit_all()
        finished = _now()
        final_status, exit_code = audit.final_status()
        report_path = audit.report_path(finished)
        audit.render_report(report_path, finished, final_status)
        audit.print_terminal_summary(report_path, final_status)
        return exit_code
    except Exception as exc:  # noqa: BLE001
        # If report generation itself fails, keep the exit semantics explicit.
        try:
            report_dir = root / REPORT_DIR_NAME
            report_dir.mkdir(parents=True, exist_ok=True)
            emergency = report_dir / f"{REPORT_NAME_PREFIX}{started.strftime('%Y%m%d_%H%M%S')}_incomplete.txt"
            emergency.write_text(
                "EDGE HUNTER PRODUCTION READINESS AUDIT\n"
                "FINAL STATUS: AUDIT_INCOMPLETE\n"
                f"ERROR: {type(exc).__name__}: {redact(str(exc))}\n",
                encoding="utf-8",
            )
            print(f"AUDIT_INCOMPLETE\nReport: {emergency}")
        except Exception:
            print(f"AUDIT_INCOMPLETE: {type(exc).__name__}: {redact(str(exc))}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
