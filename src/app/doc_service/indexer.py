from __future__ import annotations

import html
import json
import re
import unicodedata
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from app.doc_service.models import DocumentHeading, DocumentMeta, IndexedDocument

FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
HEADING_RE = re.compile(r"^(#{2,3})\s+(.+?)\s*$", re.MULTILINE)
WORD_RE = re.compile(r"[A-Za-z0-9\u3131-\u318E\uAC00-\uD7A3]+")
NON_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip())


def slugify(text: str) -> str:
    normalized = _normalize_text(text).lower()
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    source = ascii_only or normalized
    source = source.replace("_", "-").replace(" ", "-")
    slug = NON_SLUG_RE.sub("-", source).strip("-")
    return slug or "document"


def _parse_scalar(raw: str):
    value = raw.strip()
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered.lstrip("-").isdigit():
        return int(lowered)
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(chunk.strip()) for chunk in inner.split(",")]
    return value


def parse_front_matter(markdown: str) -> tuple[dict, str]:
    match = FRONT_MATTER_RE.match(markdown)
    if not match:
        return {}, markdown
    body = markdown[match.end():]
    raw_meta = match.group(1)
    meta: dict = {}
    for line in raw_meta.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        meta[key.strip()] = _parse_scalar(value)
    return meta, body


def extract_headings(markdown_body: str) -> list[DocumentHeading]:
    headings: list[DocumentHeading] = []
    used: dict[str, int] = {}
    for match in HEADING_RE.finditer(markdown_body):
        level = len(match.group(1))
        text = _normalize_text(match.group(2).strip())
        base_anchor = slugify(text)
        count = used.get(base_anchor, 0)
        used[base_anchor] = count + 1
        anchor = base_anchor if count == 0 else f"{base_anchor}-{count + 1}"
        headings.append(DocumentHeading(level=level, text=text, anchor=anchor))
    return headings


def _extract_title(markdown_body: str, fallback: str) -> str:
    for line in markdown_body.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return _normalize_text(stripped[2:].strip()) or fallback
    return fallback


def _extract_summary(markdown_body: str, fallback: str) -> str:
    for line in markdown_body.splitlines():
        stripped = _normalize_text(line)
        if stripped and not stripped.startswith("#"):
            return stripped
    return fallback


def _derive_category(source_path: str) -> str:
    path = Path(source_path)
    if len(path.parts) > 1:
        return slugify(path.parts[0]) or "general"
    return "general"


def _as_tags(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    return [text]


def _reading_minutes(word_count: int) -> int:
    if word_count <= 0:
        return 1
    return max(1, (word_count + 249) // 250)


def _iso_utc_from_stat(path: Path) -> str:
    ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return ts.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _resolve_slug(candidate: str, used: dict[str, int]) -> str:
    base = slugify(candidate)
    count = used.get(base, 0)
    used[base] = count + 1
    return base if count == 0 else f"{base}-{count + 1}"


class DocumentIndexer:
    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.docs_dir = project_dir / "docs"
        self.guide_dir = project_dir / "guide"
        self.uploads_dir = project_dir / "uploads" / "docs"
        self.index_dir = project_dir / "docs_index"
        self.documents_index_path = self.index_dir / "documents.json"
        self.search_index_path = self.index_dir / "search.json"

    def scan_paths(self) -> list[Path]:
        files: list[Path] = []
        for root in (self.guide_dir, self.docs_dir, self.uploads_dir):
            if not root.exists():
                continue
            files.extend(path for path in root.rglob("*.md") if path.is_file())
        return sorted(files, key=lambda path: path.relative_to(self.project_dir).as_posix().lower())

    def build(self) -> list[IndexedDocument]:
        used_slugs: dict[str, int] = {}
        indexed: list[IndexedDocument] = []
        for path in self.scan_paths():
            indexed.append(self._build_document(path, used_slugs))
        self._write_indexes(indexed)
        return indexed

    def _build_document(self, path: Path, used_slugs: dict[str, int]) -> IndexedDocument:
        markdown = path.read_text(encoding="utf-8")
        front_matter, body = parse_front_matter(markdown)
        relative_path = path.relative_to(self.project_dir).as_posix()
        fallback_title = _normalize_text(path.stem.replace("-", " ").replace("_", " ")) or path.stem
        title = _normalize_text(str(front_matter.get("title") or _extract_title(body, fallback_title)))
        summary = _normalize_text(str(front_matter.get("summary") or _extract_summary(body, title)))
        slug = _resolve_slug(str(front_matter.get("slug") or path.stem or title), used_slugs)
        headings = extract_headings(body)
        category = slugify(str(front_matter.get("category") or _derive_category(relative_path))) or "general"
        tags = [slugify(tag).strip("-") or tag for tag in _as_tags(front_matter.get("tags"))]
        is_public = bool(front_matter.get("is_public", True))
        order = int(front_matter.get("order", 0) or 0)
        words = WORD_RE.findall(body)
        search_blob = "\n".join(
            [
                title,
                summary,
                " ".join(tags),
                " ".join(heading.text for heading in headings),
                re.sub(r"\s+", " ", body),
            ]
        )
        meta = DocumentMeta(
            slug=slug,
            title=title,
            summary=summary,
            source_path=relative_path,
            public_url=f"/p/{slug}",
            category=category,
            tags=tags,
            is_public=is_public,
            updated_at=_iso_utc_from_stat(path),
            word_count=len(words),
            reading_minutes=_reading_minutes(len(words)),
            headings=headings,
            order=order,
        )
        return IndexedDocument(meta=meta, markdown=markdown, search_text=search_blob)

    def _write_indexes(self, indexed_documents: list[IndexedDocument]) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        documents_payload = [doc.meta.to_summary_dict() | {
            "source_path": doc.meta.source_path,
            "is_public": doc.meta.is_public,
            "headings": [heading.to_dict() for heading in doc.meta.headings],
            "order": doc.meta.order,
        } for doc in indexed_documents]
        search_payload = [
            {
                "slug": doc.meta.slug,
                "title": doc.meta.title,
                "summary": doc.meta.summary,
                "category": doc.meta.category,
                "tags": doc.meta.tags,
                "headings": [heading.to_dict() for heading in doc.meta.headings],
                "body": re.sub(r"\s+", " ", doc.markdown),
                "updated_at": doc.meta.updated_at,
                "order": doc.meta.order,
                "source_path": doc.meta.source_path,
            }
            for doc in indexed_documents
        ]
        self.documents_index_path.write_text(
            json.dumps(documents_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.search_index_path.write_text(
            json.dumps(search_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
