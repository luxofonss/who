from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from loguru import logger
from tree_sitter import Language, Parser
from tree_sitter_languages import get_language
from adapters.gemini import Gemini  # Assume your own Gemini interface

JAVA_LANGUAGE: Language = get_language("java")
_PARSER = Parser()
_PARSER.set_language(JAVA_LANGUAGE)

CodeChunk = Dict[str, object]
DependencyGraph = Dict[str, Dict[str, List[str]]]

BLACKLIST_DIR = {
    "__pycache__", ".pytest_cache", ".venv", ".git", ".idea", "venv", "env",
    "node_modules", "dist", "build", ".vscode", ".github", ".gitlab", ".angular",
    "cdk.out", ".aws-sam", ".terraform",
}
WHITELIST_EXT = {".java"}


def _should_skip(path: Path) -> bool:
    parts = {p.name for p in path.parents}
    return bool(parts & BLACKLIST_DIR)


def list_java_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*.java"):
        if p.suffix in WHITELIST_EXT and not _should_skip(p):
            files.append(p)
    return files


def parse_project(root: Path) -> Tuple[List[CodeChunk], DependencyGraph]:
    chunks: List[CodeChunk] = []
    dep_graph: DependencyGraph = {}

    for file_path in list_java_files(root):
        try:
            text = file_path.read_text("utf-8")
        except UnicodeDecodeError:
            logger.warning(f"Could not read {file_path} – skipping")
            continue

        try:
            tree = _PARSER.parse(text.encode("utf-8"))
        except Exception as exc:
            logger.error(f"Parser error for {file_path}: {exc}")
            continue

        file_chunks, file_graph = _parse_file(file_path, tree, text)
        chunks.extend(file_chunks)
        dep_graph.update(file_graph)

    _populate_called_by(dep_graph)
    _attach_called_by_to_chunks(chunks, dep_graph)

    logger.info(f"parse_project: collected {len(chunks)} chunks ({len(dep_graph)} graph nodes)")
    return chunks, dep_graph


def _parse_file(file_path: Path, tree, source: str) -> Tuple[List[CodeChunk], DependencyGraph]:
    chunks: List[CodeChunk] = []
    graph: DependencyGraph = {}

    root_node = tree.root_node
    lines = source.splitlines()

    for class_node in [n for n in root_node.children if n.type == "class_declaration"]:
        class_name = _get_identifier(class_node, source) or file_path.stem
        chunk_type = _infer_chunk_type(class_node, source)
        endpoints = _extract_class_level_endpoints(class_node, source)
        field_map, _ = _extract_field_types_and_imports(class_node, source)

        method_nodes = []
        for child in class_node.children:
            if child.type == "class_body":
                method_nodes.extend([
                    n for n in child.children if n.type == "method_declaration"
                ])

        if not method_nodes:
            chunk = _build_chunk(
                file_path=file_path,
                class_name=class_name,
                method_name=None,
                chunk_type=chunk_type,
                start_byte=class_node.start_byte,
                end_byte=class_node.end_byte,
                start_point=class_node.start_point,
                end_point=class_node.end_point,
                content=source[class_node.start_byte:class_node.end_byte],
                calls=[],
                endpoints=endpoints if chunk_type == "controller" else [],
            )
            chunks.append(chunk)
            graph[f"{class_name}"] = {"calls": [], "called_by": []}
            continue

        for m_node in method_nodes:
            method_name = _get_identifier(m_node, source) or "unknown"
            calls = _extract_calls(m_node, source, class_name, field_map)
            chunk = _build_chunk(
                file_path=file_path,
                class_name=class_name,
                method_name=method_name,
                chunk_type=chunk_type,
                start_byte=m_node.start_byte,
                end_byte=m_node.end_byte,
                start_point=m_node.start_point,
                end_point=m_node.end_point,
                content=source[m_node.start_byte:m_node.end_byte],
                calls=calls,
                endpoints=endpoints if chunk_type == "controller" else [],
            )
            chunks.append(chunk)
            graph[f"{class_name}.{method_name}"] = {"calls": calls, "called_by": []}

    return chunks, graph


def _get_identifier(node, source: str) -> str | None:
    for child in node.children:
        if child.type == "identifier":
            return source[child.start_byte:child.end_byte]
    cursor = node.walk()
    while True:
        if cursor.node.type == "identifier":
            return source[cursor.node.start_byte:cursor.node.end_byte]
        if not cursor.goto_next_sibling():
            break
    return None


def _extract_calls(method_node, source: str, this_class: str, field_map: Dict[str, str]) -> List[str]:
    calls: set[str] = set()

    def walk(node):
        if node.type == "method_invocation":
            method_name_node = node.child_by_field_name("name")
            object_node = node.child_by_field_name("object")

            method_name = (
                source[method_name_node.start_byte:method_name_node.end_byte]
                if method_name_node else "unknown"
            )
            qualified = f"{this_class}.{method_name}"

            if object_node:
                obj_text = source[object_node.start_byte:object_node.end_byte].strip()
                if obj_text == "this":
                    qualified = f"{this_class}.{method_name}"
                elif obj_text in field_map:
                    qualified = f"{field_map[obj_text]}.{method_name}"
                elif obj_text.startswith("new "):
                    try:
                        parts = obj_text.split()
                        class_name = parts[1].split("(")[0]
                        qualified = f"{class_name}.{method_name}"
                    except Exception:
                        qualified = f"unknown.{method_name}"
                elif obj_text[0].isupper():
                    qualified = f"{obj_text}.{method_name}"
                else:
                    qualified = f"unknown.{method_name}"
            calls.add(qualified)

        for child in node.children:
            walk(child)

    walk(method_node)
    return sorted(calls)


def _infer_chunk_type(node, source: str) -> str:
    annotations = [c for c in node.children if c.type == "modifiers"]
    ann_text = "".join(source[c.start_byte:c.end_byte] for c in annotations)
    if "@Controller" in ann_text or "@RestController" in ann_text:
        return "controller"
    if "@Service" in ann_text:
        return "service"
    if "@Repository" in ann_text:
        return "repository"
    if "@Entity" in ann_text:
        return "entity"
    if "Filter" in ann_text:
        return "filter"
    return "other"


def _extract_class_level_endpoints(node, source: str):
    endpoints: List[Dict[str, str]] = []
    for child in node.children:
        if child.type == "modifiers":
            text = source[child.start_byte:child.end_byte]
            for ann in ["RequestMapping", "GetMapping", "PostMapping", "PutMapping", "DeleteMapping"]:
                if ann in text:
                    path = _extract_annotation_value(text, "value") or _extract_annotation_value(text, "path")
                    method = ann.replace("Mapping", "").upper()
                    endpoints.append({"path": path, "method": method})
    return endpoints


def _extract_annotation_value(text: str, key: str) -> str | None:
    marker = f"{key}="
    if marker not in text:
        return None
    try:
        sub = text.split(marker, 1)[1]
        first = sub.split(",", 1)[0]
        val = first.strip().strip("\"')(")
        return val
    except Exception:
        return None


def _extract_field_types_and_imports(class_node, source: str) -> Tuple[Dict[str, str], List[str]]:
    field_map = {}
    imports = []

    for node in class_node.children:
        if node.type == "field_declaration":
            type_node = node.child_by_field_name("type")
            name_node = node.child_by_field_name("declarator")
            if type_node and name_node:
                type_name = source[type_node.start_byte:type_node.end_byte].strip()
                var_name = source[name_node.start_byte:name_node.end_byte].strip().rstrip(";")
                field_map[var_name] = type_name

    for line in source.splitlines():
        if line.strip().startswith("import ") and line.strip().endswith(";"):
            imports.append(line.strip()[7:-1].strip())

    return field_map, imports


def _build_chunk(
    *,
    file_path: Path,
    class_name: str,
    method_name: str | None,
    chunk_type: str,
    start_byte: int,
    end_byte: int,
    start_point: Tuple[int, int],
    end_point: Tuple[int, int],
    content: str,
    calls: List[str],
    endpoints: List[Dict[str, str]],
) -> CodeChunk:
    chunk = {
        "file_path": str(file_path),
        "class_name": class_name,
        "method_name": method_name,
        "chunk_type": chunk_type,
        "calls": calls,
        "called_by": [],
        "line_start": start_point[0] + 1,
        "line_end": end_point[0] + 1,
        "content": content,
        "endpoints": endpoints,
    }
    chunk["summary"] = _summarise_chunk(chunk)
    return chunk


def _summarise_chunk(chunk: CodeChunk) -> str:
    prompt = (
        f"Summarise the following Java {chunk.get('chunk_type')} in ONE sentence.\n\n"
        + chunk["content"][:2000]
    )
    try:
        return ""  # Gemini().invoke(prompt).strip()
    except Exception as exc:
        logger.error("Gemini summary failed: %s", exc)
        return ""


def _populate_called_by(dep_graph: DependencyGraph):
    for caller, rel in dep_graph.items():
        for callee in rel["calls"]:
            if callee in dep_graph:
                dep_graph[callee]["called_by"].append(caller)


def _attach_called_by_to_chunks(chunks: List[CodeChunk], dep_graph: DependencyGraph):
    for chunk in chunks:
        if chunk["method_name"] is None:
            continue
        key = f"{chunk['class_name']}.{chunk['method_name']}"
        chunk["called_by"] = dep_graph.get(key, {}).get("called_by", [])
