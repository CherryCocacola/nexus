"""
다언어 심볼 파서 단위 테스트 (Phase 10.0 확장, 2026-04-21).

검증 대상:
  1. ParserRegistry — 확장자 라우팅
  2. JavaScriptParser — function/arrow/class/method/async 변형
  3. TypeScriptParser — interface/type 추가
  4. GoParser — func/method(receiver)/struct/interface/type alias
  5. SymbolProjectIndexer — 다언어 혼합 트리 적재
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core.rag.parsers import (
    GoParser,
    JavaScriptParser,
    ParserRegistry,
    TypeScriptParser,
    build_default_registry,
)
from core.rag.symbol_indexer import SymbolProjectIndexer
from core.rag.symbol_store import SymbolStore


# ─────────────────────────────────────────────
# ParserRegistry
# ─────────────────────────────────────────────
def test_registry_routes_by_extension() -> None:
    reg = build_default_registry()
    assert reg.for_path("foo.py").language == "python"
    assert reg.for_path("bar.js").language == "javascript"
    assert reg.for_path("baz.jsx").language == "javascript"
    assert reg.for_path("qux.ts").language == "typescript"
    assert reg.for_path("quux.tsx").language == "typescript"
    assert reg.for_path("corge.go").language == "go"
    assert reg.for_path("unknown.rs") is None


def test_registry_later_registration_overrides() -> None:
    reg = ParserRegistry()
    reg.register(JavaScriptParser())
    # .ts가 JS에 의해 등록됐는지 확인 — 이 경우는 확장자 겹치지 않음
    assert reg.for_path("x.ts") is None
    reg.register(TypeScriptParser())
    assert reg.for_path("x.ts").language == "typescript"


def test_registry_supported_lists() -> None:
    reg = build_default_registry()
    exts = reg.supported_extensions()
    assert ".py" in exts and ".js" in exts and ".go" in exts
    langs = reg.supported_languages()
    assert {"python", "javascript", "typescript", "go"}.issubset(set(langs))


# ─────────────────────────────────────────────
# JavaScriptParser
# ─────────────────────────────────────────────
JS_SAMPLE = """\
/**
 * Sum two numbers.
 */
function sum(a, b) {
  return a + b;
}

export async function fetchUser(id) {
  return await api.get(id);
}

const multiply = (x, y) => x * y;
const asyncPing = async (url) => {
  return await fetch(url);
};

export default class Greeter {
  constructor(name) { this.name = name; }
  async hello(other) {
    return `hi ${other}`;
  }
  static create() {
    return new Greeter("world");
  }
}
"""


def test_js_parser_extracts_functions() -> None:
    entries = JavaScriptParser().parse(JS_SAMPLE, "sample.js")
    names = {(e.kind, e.name) for e in entries}
    assert ("function", "sum") in names
    assert ("async_function", "fetchUser") in names
    assert ("function", "multiply") in names
    assert ("async_function", "asyncPing") in names
    assert ("class", "Greeter") in names
    # 클래스 내부 메서드
    method_names = {(e.kind, e.qualified_name) for e in entries}
    assert ("method", "Greeter.constructor") in method_names
    assert ("async_method", "Greeter.hello") in method_names
    assert ("method", "Greeter.create") in method_names


def test_js_parser_captures_jsdoc() -> None:
    entries = JavaScriptParser().parse(JS_SAMPLE, "sample.js")
    sum_fn = next(e for e in entries if e.name == "sum")
    assert "Sum two numbers" in sum_fn.docstring


def test_js_parser_captures_static_tag() -> None:
    entries = JavaScriptParser().parse(JS_SAMPLE, "sample.js")
    create = next(e for e in entries if e.qualified_name == "Greeter.create")
    assert "static" in create.extra_tags


# ─────────────────────────────────────────────
# TypeScriptParser
# ─────────────────────────────────────────────
TS_SAMPLE = """\
interface User {
  id: number;
  name: string;
}

type ID = string | number;

export function greet(u: User): string {
  return "hi " + u.name;
}

class Repo {
  private cache: Map<ID, User> = new Map();
  async find(id: ID): Promise<User | undefined> {
    return this.cache.get(id);
  }
}
"""


def test_ts_parser_extracts_interface_and_type() -> None:
    entries = TypeScriptParser().parse(TS_SAMPLE, "sample.ts")
    qual = {(e.kind, e.qualified_name) for e in entries}
    assert ("interface", "User") in qual
    assert ("type", "ID") in qual
    assert ("function", "greet") in qual
    assert ("class", "Repo") in qual
    assert ("async_method", "Repo.find") in qual


def test_ts_parser_does_not_leak_interface_into_js_parser() -> None:
    """순수 JS 파서는 interface를 감지하지 않는다."""
    js_only = JavaScriptParser().parse("interface Foo {}", "x.js")
    assert all(e.kind != "interface" for e in js_only)


# ─────────────────────────────────────────────
# GoParser
# ─────────────────────────────────────────────
GO_SAMPLE = """\
// Package main is the entrypoint.
package main

// Greeter holds state.
type Greeter struct {
  name string
}

// Hello prints a greeting.
func (g *Greeter) Hello(other string) string {
  return "hi " + other
}

// Sum adds two integers.
func Sum(a int, b int) int {
  return a + b
}

type Reader interface {
  Read(p []byte) (int, error)
}

type ID = string
"""


def test_go_parser_extracts_funcs_methods_types() -> None:
    entries = GoParser().parse(GO_SAMPLE, "sample.go")
    pairs = {(e.kind, e.qualified_name) for e in entries}
    assert ("function", "Sum") in pairs
    assert ("method", "Greeter.Hello") in pairs
    assert ("struct", "Greeter") in pairs
    assert ("interface", "Reader") in pairs
    # type ID = string은 alias
    assert any(e.kind == "type" and e.name == "ID" for e in entries)


def test_go_parser_captures_doc_comment() -> None:
    entries = GoParser().parse(GO_SAMPLE, "sample.go")
    sum_fn = next(e for e in entries if e.name == "Sum")
    assert "Sum adds two integers" in sum_fn.docstring


def test_go_parser_tags_exported() -> None:
    entries = GoParser().parse(GO_SAMPLE, "sample.go")
    sum_fn = next(e for e in entries if e.name == "Sum")
    assert "exported" in sum_fn.extra_tags


# ─────────────────────────────────────────────
# SymbolProjectIndexer — 다언어 혼합 트리
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_project_indexer_handles_multiple_languages(tmp_path: Path) -> None:
    (tmp_path / "py_mod.py").write_text(
        "def hello(): pass\nclass Foo: pass\n", encoding="utf-8",
    )
    (tmp_path / "js_mod.js").write_text(JS_SAMPLE, encoding="utf-8")
    (tmp_path / "ts_mod.ts").write_text(TS_SAMPLE, encoding="utf-8")
    (tmp_path / "go_mod.go").write_text(GO_SAMPLE, encoding="utf-8")
    # 지원 안 하는 파일은 무시되어야 한다
    (tmp_path / "readme.md").write_text("# readme\n", encoding="utf-8")

    store = SymbolStore(pg_pool=None)
    indexer = SymbolProjectIndexer(store=store, embedder=None, project_source="t")
    stats = await indexer.index_project(tmp_path)

    # 4개 지원 파일, 각각 여러 심볼
    assert stats["files"] == 4
    assert stats["symbols"] > 10  # 합산 대략 20개 이상

    # 언어별 태그 확인
    tags_by_lang: dict[str, int] = {}
    for e in store._store.values():
        for t in e.tags:
            tags_by_lang[t] = tags_by_lang.get(t, 0) + 1
    assert tags_by_lang.get("python", 0) >= 2
    assert tags_by_lang.get("javascript", 0) >= 3
    assert tags_by_lang.get("typescript", 0) >= 3
    assert tags_by_lang.get("go", 0) >= 3


# ─────────────────────────────────────────────
# 하위 호환 — Python-only 경로 유지
# ─────────────────────────────────────────────
def test_python_parser_back_compat_via_extract_fn() -> None:
    """extract_symbols_from_source는 registry 없어도 .py에서 동작."""
    from core.rag.symbol_indexer import extract_symbols_from_source

    entries = extract_symbols_from_source(
        "def f(): pass\nclass C: pass\n",
        path="a.py", module="a",
    )
    names = {e.name for e in entries}
    assert "f" in names and "C" in names


def test_extract_without_parser_skips_unknown_extension() -> None:
    """registry 없이 .js 파일을 주면 자동 skip (하위 호환 정책)."""
    from core.rag.symbol_indexer import extract_symbols_from_source

    entries = extract_symbols_from_source(
        "function foo(){}", path="a.js", module="a",
    )
    assert entries == []
