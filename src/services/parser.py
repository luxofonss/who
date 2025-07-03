from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Tuple, List, Dict, Optional

from loguru import logger
from tree_sitter import Language, Parser
from tree_sitter_languages import get_language
from utils.constant import JAVA_STANDARD_TYPES, GENERIC_TYPE_VARS
# from adapters.gemini import Gemini  # Assume your own Gemini interface

JAVA_LANGUAGE: Language = get_language("java")
_PARSER = Parser()
_PARSER.set_language(JAVA_LANGUAGE)
EXCLUDE_CHUNK_TYPE = ["package_declaration", "import_declaration"]

CodeChunk = Dict[str, object]
DependencyGraph = Dict[str, Dict[str, List[str]]]

BLACKLIST_DIR = {
    ".git", ".idea", "env", ".github", ".gitlab", 
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
    _populate_extends_and_implements_by(chunks, dep_graph)

    logger.info(f"parse_project: collected {len(chunks)} chunks ({len(dep_graph)} graph nodes)")
    return chunks, dep_graph


def _parse_file(file_path: Path, tree, source: str) -> Tuple[List[CodeChunk], DependencyGraph]:
    chunks: List[CodeChunk] = []
    graph: DependencyGraph = {}

    root_node = tree.root_node

    for class_node in [n for n in root_node.children if n.type not in EXCLUDE_CHUNK_TYPE]:
        class_name = _get_identifier(class_node, source) or file_path.stem
        chunk_type = _infer_chunk_type(class_node, source)
        class_endpoints = _extract_class_level_endpoints(class_node, source)
        class_hierarchy = _extract_class_hierarchy(class_node, source)

        field_map = _extract_fields(class_node, source)

        method_nodes = []
        for child in class_node.children:
            if child.type == "class_body" or child.type == "interface_body" or child.type == "enum_body":
                method_nodes.extend([
                    n for n in child.children if n.type == "method_declaration"
                ])
    
        logger.info(f"class_node: {class_name}")
        logger.info(f"method_nodes: {method_nodes}")
        chunk = _build_chunk(
            file_path=file_path,
            class_name=class_name,
            method_name=None,
            chunk_type=chunk_type,
            start_point=class_node.start_point,
            end_point=class_node.end_point,
            content=source[class_node.start_byte:class_node.end_byte],
            calls=[],
            endpoints=[{"path": ep, "method": "REQUEST"} for ep in class_endpoints] if chunk_type == "controller" else [],
            extends=class_hierarchy['extends'],
            implements=class_hierarchy['implements'],
        )
        chunks.append(chunk)
        graph[f"{class_name}"] = {
            "calls": [], 
            "called_by": [], 
            "extends": class_hierarchy['extends'], 
            "implements": class_hierarchy['implements'],
            "extended_by": [],
            "implemented_by": []
        }

        for m_node in method_nodes:
            method_name = _get_identifier(m_node, source) or "unknown"
            logger.info(f"method_name: {method_name}")
            param_map = _extract_param_types(m_node, source)
            vars = _extract_vars(m_node)
            vars = [var for var in vars if var not in JAVA_STANDARD_TYPES and var not in GENERIC_TYPE_VARS]
            calls = _extract_calls(m_node, source, class_name, {**field_map, **param_map})

            endpoint = _extract_method_endpoint(m_node, source)
            endpoints = []
            if endpoint:
                path, method = endpoint
                endpoints.append({"path": path, "method": method})

            chunk = _build_chunk(
                file_path=file_path,
                class_name=class_name,
                method_name=method_name,
                chunk_type=chunk_type,
                start_point=m_node.start_point,
                end_point=m_node.end_point,
                content=source[m_node.start_byte:m_node.end_byte],
                calls=calls,
                endpoints=endpoints,
                extends=None,
                implements=[],
                vars=vars,
            )
            chunks.append(chunk)
            graph[f"{class_name}.{method_name}"] = {"calls": calls, "called_by": []}

    return chunks, graph

def _extract_vars(node) -> List[str]:
    q = """
    (
        (type_identifier) @vars
    )
    """
    captures = JAVA_LANGUAGE.query(q).captures(node);
    res = []

    for node, capture_name in captures:
        logger.info(f"capture_name: {capture_name}, text: {node.text.decode()}")
        res.append(node.text.decode())

    return res

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


def _extract_calls(method_node, source: str, this_class: str, var_types: Dict[str, str]) -> List[str]:
    calls: set[str] = set()

    def walk(node):
        if node.type == "method_invocation":
            method_name_node = node.child_by_field_name("name")
            object_node = node.child_by_field_name("object")

            method_name = (
                source[method_name_node.start_byte:method_name_node.end_byte]
                if method_name_node else "unknown"
            )

            qualified = f"unknown.{method_name}"  # fallback

            logger.info(f"var_types: {var_types}")

            if object_node:
                obj_text = _resolve_object_name(object_node, source).strip()

                if obj_text == "this":
                    qualified = f"{this_class}.{method_name}"
                elif obj_text in var_types:
                    qualified = f"{var_types[obj_text]}.{method_name}"
                elif "." in obj_text:
                    # Try to resolve root var in a chain like repo.a.b -> repo
                    root = obj_text.split(".", 1)[0]
                    if root in var_types:
                        qualified = f"{var_types[root]}.{method_name}"
                    else:
                        logger.warning(f"Unresolved object: '{obj_text}' -> '{method_name}'")
                        qualified = f"unknown.{method_name}"
                elif obj_text[0].isupper():
                    qualified = f"{obj_text}.{method_name}"
                else:
                    logger.warning(f"Unknown object: '{obj_text}' in method '{method_name}'")
                    qualified = f"unknown.{method_name}"
            else:
                qualified = f"{this_class}.{method_name}"

            calls.add(qualified)

        for child in node.children:
            walk(child)

    walk(method_node)
    return sorted(calls)

def _resolve_object_name(node, source: str) -> str:
    """Resolves the left-hand-side of a method call or field access."""
    if node is None:
        return "unknown"

    if node.type == "identifier":
        return source[node.start_byte:node.end_byte]

    if node.type == "field_access":
        obj = _resolve_object_name(node.child_by_field_name("object"), source)
        field = node.child_by_field_name("field")
        if field:
            return f"{obj}.{source[field.start_byte:field.end_byte]}"
        return obj

    if node.type == "method_invocation":
        return _resolve_object_name(node.child_by_field_name("object"), source)

    return "unknown"

def _extract_param_types(method_node, source: str) -> Dict[str, str]:
    param_map = {}
    for child in method_node.children:
        if child.type == "formal_parameters":
            for param in child.children:
                if param.type == "formal_parameter":
                    type_node = param.child_by_field_name("type")
                    name_node = param.child_by_field_name("name")
                    if type_node and name_node:
                        typename = source[type_node.start_byte:type_node.end_byte].strip()
                        name = source[name_node.start_byte:name_node.end_byte].strip()
                        param_map[name] = typename
    return param_map

def _infer_chunk_type(node, source: str) -> str:
    annotations = [c for c in node.children if c.type == "modifiers"]
    ann_text = "".join(source[c.start_byte:c.end_byte].lower() for c in annotations)
    if "@controller" in ann_text or "@restController" in ann_text:
        return "controller"
    if "@service" in ann_text:
        return "service"
    if "@repository" in ann_text:
        return "repository"
    if "@entity" in ann_text:
        return "entity"
    if "filter" in ann_text:
        return "filter"
    if "configuration" in ann_text:
        return "configuration"
    if "component" in ann_text:
        return "component"
    if "bean" in ann_text:
        return "bean"
    if "interface" in ann_text:
        return "interface"
    if "abstract" in ann_text:
        return "abstract_class"
    
    return "other"


def _extract_class_level_endpoints(class_node, source: str) -> List[str]:
    paths = []
    for child in class_node.children:
        if child.type == "modifiers":
            for grandchild in child.children:
                if grandchild.type != "annotation":
                    continue


            text = source[child.start_byte:child.end_byte]
            if "@RequestMapping" in text:
                val = _extract_annotation_value(text, "value") or _extract_annotation_value(text, "path")
                if val:
                    paths.append(val)
    return paths

def _extract_annotation_value(annotation_text: str, param_name: str) -> str | None:
    # Handle value = "/path", path = "/path", or short form like @GetMapping("/users")
    pattern = rf'{param_name}\s*=\s*(?:"([^"]+)"|\{{?"([^"]+)"\}}?)'
    match = re.search(pattern, annotation_text)
    if match:
        return match.group(1) or match.group(2)
    # Handle short form (e.g., @GetMapping("/users"))
    if param_name == "value" and '"' in annotation_text:
        match = re.search(r'"([^"]+)"', annotation_text)
        return match.group(1) if match else None
    return None

def _extract_method_endpoint(method_node, source: str) -> Tuple[str, str] | None:
    method_path = ""
    http_method = "REQUEST"

    for child in method_node.children:
        if child.type == "modifiers":
            for grandchild in child.children:
                if grandchild.type == "annotation":
                    text = source[grandchild.start_byte:grandchild.end_byte]
                    for ann in ["GetMapping", "PostMapping", "PutMapping", "DeleteMapping", "RequestMapping"]:
                        if f"@{ann}" in text:
                            if ann != "RequestMapping":
                                http_method = ann.replace("Mapping", "").upper()
                            else:
                                # Check for method parameter in RequestMapping
                                method_match = re.search(r'method\s*=\s*RequestMethod\.(\w+)', text)
                                if method_match:
                                    http_method = method_match.group(1).upper()
                            method_path = _extract_annotation_value(text, "value") or _extract_annotation_value(text, "path") or ""
                            break

    if not method_path and http_method == "REQUEST":
        return None

    class_path = ""
    parent = method_node.parent
    while parent:
        if parent.type == "class_declaration":
            for child in parent.children:
                if child.type == "modifiers":
                    for grandchild in child.children:
                        if grandchild.type == "annotation":
                            text = source[grandchild.start_byte:grandchild.end_byte]
                            if "@RequestMapping" in text:
                                class_path = _extract_annotation_value(text, "value") or _extract_annotation_value(text, "path") or ""
            break
        parent = parent.parent

    # Robust path combination
    if class_path and method_path:
        full_path = (class_path.rstrip("/") + "/" + method_path.lstrip("/")).rstrip("/")
    elif method_path:
        full_path = method_path.rstrip("/")
    else:
        full_path = class_path.rstrip("/")
    return full_path or "/", http_method

def _extract_fields(class_node, source: str) -> Dict[str, str]:
    field_map: Dict[str, str] = {}

    field_nodes = []
    for child in class_node.children:
        if child.type == "class_body":
            field_nodes.extend([
                n for n in child.children if n.type == "field_declaration"
            ])

    for field_node in field_nodes:
        type_node = field_node.child_by_field_name("type")
        if not type_node:
            continue

        # Extract base type (ignore generic parameters)
        raw_type = source[type_node.start_byte:type_node.end_byte].strip()
        logger.info(f"raw_type: {raw_type}")
        type_name = raw_type.split("<")[0].strip()
        logger.info(f"type_name: {type_name}")

        # Support multiple declarators (e.g., `String a, b;`)
        declarator_nodes = [
            n for n in field_node.children if n.type == "variable_declarator"
        ]
        for decl in declarator_nodes:
            logger.info(f"decl: {decl}")
            name_node = decl.child_by_field_name("name")
            if name_node:
                var_name = source[name_node.start_byte:name_node.end_byte].strip()
                field_map[var_name] = type_name

    return field_map


def _build_chunk(
    *,
    file_path: Path,
    class_name: str,
    method_name: str | None,
    chunk_type: str,
    start_point: Tuple[int, int],
    end_point: Tuple[int, int],
    content: str,
    calls: List[str],
    endpoints: List[Dict[str, str]],
    extends: Optional[str] = None,
    implements: Optional[List[str]] = None,
    vars: List[str] = []
) -> CodeChunk:
    line_start = start_point[0] + 1
    line_end = end_point[0] + 1
    chunk_id = f"{file_path}::{class_name}::{method_name or ''}::{line_start}::{line_end}"
    chunk = {
        "id": chunk_id,
        "file_path": str(file_path),
        "class_name": class_name,
        "method_name": method_name,
        "chunk_type": chunk_type,
        "calls": calls,
        "called_by": [],
        "line_start": line_start,
        "line_end": line_end,
        "content": content,
        "endpoints": endpoints,
        "extends": extends,
        "implements": implements or [],
        "extended_by": [],
        "implemented_by": [],
        "vars": vars
    }

    # TODO: add summary, now i dont have enough model to do this hehe
    chunk["summary"] = _summarise_chunk(chunk)
    return chunk


def _summarise_chunk(chunk: CodeChunk) -> str:
    # TODO: enable this when we have enough money to call model
    # prompt = (
    #     f"Summarise the following Java {chunk.get('chunk_type')} in ONE sentence.\n\n"
    #     + chunk["content"][:2000]
    # )
    # try:
    #     return Gemini().invoke(prompt).strip()
    # except Exception as exc:
    #     logger.error("Gemini summary failed: %s", exc)
    #     return ""
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

def _populate_extends_and_implements_by(chunks: List[CodeChunk], dep_graph: DependencyGraph):
    # First, populate the graph relationships
    for chunk in chunks:
        this_class = str(chunk.get("class_name", ""))
        if not this_class:
            continue

        implements = chunk.get("implements", [])
        if isinstance(implements, list):
            for parent in implements:
                dep_graph.setdefault(str(parent), {}).setdefault("implemented_by", []).append(this_class)

        extends = chunk.get("extends")
        if extends:
            dep_graph.setdefault(str(extends), {}).setdefault("extended_by", []).append(this_class)
    
    # Then, populate the chunk relationships
    for chunk in chunks:
        class_name = str(chunk.get("class_name", ""))
        if class_name in dep_graph:
            chunk["extended_by"] = dep_graph[class_name].get("extended_by", [])
            chunk["implemented_by"] = dep_graph[class_name].get("implemented_by", [])

def _extract_class_hierarchy(class_node, source: str) -> dict:
    extends = None
    implements = []

    # --- Extract superclass ---
    superclass_node = class_node.child_by_field_name("superclass")
    if superclass_node:
        logger.info(f"superclass_node: {superclass_node}")
        # Try field_name first
        type_id_node = superclass_node.child_by_field_name("type_identifier")
        if not type_id_node:
            # Fallback: iterate to find a 'type_identifier' manually
            type_id_node = next(
                (child for child in superclass_node.children if child.type == "type_identifier"),
                None
            )
        if type_id_node:
            extends = source[type_id_node.start_byte:type_id_node.end_byte]

    # --- Extract implemented interfaces ---
    interfaces_node = class_node.child_by_field_name("interfaces")
    if interfaces_node:
        logger.info(f"interfaces_node: {interfaces_node}")
        # Fallback: no "type_list" field name, so find manually
        type_list = next(
            (child for child in interfaces_node.children if child.type == "type_list"),
            None
        )
        if type_list:
            for child in type_list.children:
                if child.type == "type_identifier":
                    implements.append(source[child.start_byte:child.end_byte])

    return {
        "extends": extends,
        "implements": implements
    }
