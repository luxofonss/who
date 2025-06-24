from __future__ import annotations

"""Java source-code parser and analyser (Tree-sitter).

This module turns *.java files into fine-grained "chunks" (one per method or, if
no methods exist, per class) enriched with metadata required by downstream RAG
retrieval and dependency-graph traversal.

The heavy lifting is done with Tree-sitter.  We rely on the `tree_sitter` core
package plus `tree_sitter_languages` which bundles pre-compiled Java grammar –
this avoids the need for a system C tool-chain.
"""

from pathlib import Path
from typing import Dict, List, Tuple

from loguru import logger
from tree_sitter import Language, Parser
from tree_sitter_languages import get_language

# ---------------------------------------------------------------------------
# Initialise a *singleton* Parser instance for Java.
# ---------------------------------------------------------------------------
JAVA_LANGUAGE: Language = get_language("java")
_PARSER = Parser()
_PARSER.set_language(JAVA_LANGUAGE)

# ---------------------------------------------------------------------------
# Public data structures.
# ---------------------------------------------------------------------------

CodeChunk = Dict[str, object]
DependencyGraph = Dict[str, Dict[str, List[str]]]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_java_files(root: Path) -> List[Path]:
    """Return all *.java files under *root*."""
    return [p for p in root.rglob("*.java") if p.is_file()]


def parse_project(root: Path) -> Tuple[List[CodeChunk], DependencyGraph]:
    """Parse *root* and return (chunks, dependency_graph).

    The dependency graph maps a fully-qualified `Class.method` string to:
        {"calls": [...], "called_by": [...]}
    """
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
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Parser error for {file_path}: {exc}")
            continue

        file_chunks, file_graph = _parse_file(file_path, tree, text)
        chunks.extend(file_chunks)
        dep_graph.update(file_graph)

    _populate_called_by(dep_graph)
    _attach_called_by_to_chunks(chunks, dep_graph)

    logger.info(
        "parse_project: collected %d chunks (%d graph nodes)",
        len(chunks),
        len(dep_graph),
    )
    return chunks, dep_graph

# ---------------------------------------------------------------------------
# Internal helpers.
# ---------------------------------------------------------------------------

def _parse_file(
    file_path: Path, tree, source: str
) -> Tuple[List[CodeChunk], DependencyGraph]:
    """Return (chunks, dep_graph) extracted from *file_path*."""
    chunks: List[CodeChunk] = []
    graph: DependencyGraph = {}

    root_node = tree.root_node
    lines = source.splitlines()

    # Iterate over top-level class declarations.
    for class_node in [n for n in root_node.children if n.type == "class_declaration"]:
        class_name = _get_identifier(class_node, source) or file_path.stem
        chunk_type = _infer_chunk_type(class_node, source)
        endpoints = _extract_class_level_endpoints(class_node, source)
        field_map, _ = _extract_field_types_and_imports(class_node, source)

        method_nodes = [n for n in class_node.children if n.type == "method_declaration"]
        if not method_nodes:
            # Fallback – whole class as one chunk.
            chunk = _build_chunk(
                file_path=file_path,
                class_name=class_name,
                method_name=None,
                chunk_type=chunk_type,
                start_byte=class_node.start_byte,
                end_byte=class_node.end_byte,
                start_point=class_node.start_point,
                end_point=class_node.end_point,
                content=source[class_node.start_byte : class_node.end_byte],
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
                content=source[m_node.start_byte : m_node.end_byte],
                calls=calls,
                endpoints=endpoints if chunk_type == "controller" else [],
            )
            chunks.append(chunk)
            graph[f"{class_name}.{method_name}"] = {"calls": calls, "called_by": []}

    return chunks, graph

def _extract_field_types_and_imports(class_node, source: str) -> Tuple[Dict[str, str], List[str]]:
    """Extract class fields (var name -> class name) and imports."""
    field_map = {}  # var_name -> class_name
    imports = []

    # Tìm các field trong class
    for node in class_node.children:
        if node.type == "field_declaration":
            # Ex: private AuthService authService;
            type_node = node.child_by_field_name("type")
            name_node = node.child_by_field_name("declarator")
            if type_node and name_node:
                type_name = source[type_node.start_byte:type_node.end_byte].strip()
                var_name = source[name_node.start_byte:name_node.end_byte].strip().rstrip(";")
                field_map[var_name] = type_name

    # Tìm các dòng import trong toàn source
    for line in source.splitlines():
        line = line.strip()
        if line.startswith("import ") and line.endswith(";"):
            imports.append(line[len("import "):-1].strip())

    return field_map, imports


def _get_identifier(node, source: str) -> str | None:
    """Return the identifier text for *node* or ``None``.

    This is mainly used for class_declaration and method_declaration nodes.
    """
    for child in node.children:
        if child.type == "identifier":
            ident = source[child.start_byte : child.end_byte]
            if ident:
                return ident
    logger.debug("Identifier not found for node type=%s at=%s", node.type, node.start_point)
    return None


def _infer_chunk_type(node, source: str) -> str:
    # Look for annotations immediately preceding the class node.
    annotations = [c for c in node.children if c.type == "modifiers"]
    ann_text = "".join(source[c.start_byte : c.end_byte] for c in annotations)
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
    """Return list of endpoint dicts found on class annotations."""
    endpoints: List[Dict[str, str]] = []
    for child in node.children:
        if child.type == "modifiers":
            text = source[child.start_byte : child.end_byte]
            for ann in ["RequestMapping", "GetMapping", "PostMapping", "PutMapping", "DeleteMapping"]:
                if ann in text:
                    path = _extract_annotation_value(text, "value") or _extract_annotation_value(text, "path")
                    method = ann.replace("Mapping", "").upper()
                    endpoints.append({"path": path, "method": method})
    return endpoints


def _extract_annotation_value(text: str, key: str) -> str | None:
    # Very naive extraction; covers the common cases for demo purposes.
    marker = f"{key}="
    if marker not in text:
        return None
    try:
        sub = text.split(marker, 1)[1]
        first = sub.split(",", 1)[0]
        val = first.strip().strip("\"')(")  # remove quotes / paren
        return val
    except Exception:  # noqa: BLE001
        return None


def _extract_calls(
    method_node, source: str, this_class: str, field_map: Dict[str, str]
) -> List[str]:
    """Return a deduplicated list of fully-qualified call names found in the method body."""
    calls: set[str] = set()

    def walk(node):
        if node.type == "method_invocation":
            method_name_node = node.child_by_field_name("name")
            object_node = node.child_by_field_name("object")

            method_name = (
                source[method_name_node.start_byte:method_name_node.end_byte]
                if method_name_node
                else "unknown"
            )

            qualified = f"{this_class}.{method_name}"  # default

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
            else:
                qualified = f"{this_class}.{method_name}"

            calls.add(qualified)

        for child in node.children:
            walk(child)

    walk(method_node)
    return sorted(calls)



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
    return {
        "file_path": str(file_path),
        "class_name": class_name,
        "method_name": method_name,
        "chunk_type": chunk_type,
        "calls": calls,
        "called_by": [],  # will be filled later
        "line_start": start_point[0] + 1,
        "line_end": end_point[0] + 1,
        "content": content,
        "endpoints": endpoints,
    }


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