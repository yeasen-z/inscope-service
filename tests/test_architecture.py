"""架构约束测试：用 import 图钉死三个包的依赖方向。

允许：
    inscope_contract  <- inscope_server
    inscope_contract  <- inscope(client)

禁止：
    inscope_server <-> inscope(client)   互相 import
    inscope_contract  import 任何上层
"""
from __future__ import annotations

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

ALLOWED = {
    "inscope_contract": set(),
    "inscope_server": {"inscope_contract"},
    "inscope": {"inscope_contract"},
}


def _top_level_imports(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    return found


def _package_of(path: pathlib.Path) -> str | None:
    rel = path.relative_to(ROOT)
    head = rel.parts[0]
    return head if head in ALLOWED else None


def test_no_upward_or_sideways_imports():
    violations = []
    for package in ALLOWED:
        for path in (ROOT / package).rglob("*.py"):
            owner = _package_of(path)
            if owner is None:
                continue
            for imported in _top_level_imports(path):
                if imported not in ALLOWED:
                    continue
                if imported == owner:
                    continue
                if imported not in ALLOWED[owner]:
                    violations.append(f"{path.relative_to(ROOT)}: {owner} -> {imported}")
    assert not violations, "dependency direction violated:\n" + "\n".join(violations)


def test_contract_imports_nothing_from_upper_layers():
    offenders = []
    for path in (ROOT / "inscope_contract").rglob("*.py"):
        for imported in _top_level_imports(path):
            if imported in {"inscope_server", "inscope"}:
                offenders.append(f"{path.relative_to(ROOT)} -> {imported}")
    assert not offenders, "contract must not import upper layers:\n" + "\n".join(offenders)
