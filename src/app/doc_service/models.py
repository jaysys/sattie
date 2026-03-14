from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class DocumentHeading:
    level: int
    text: str
    anchor: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DocumentMeta:
    slug: str
    title: str
    summary: str
    source_path: str
    public_url: str
    category: str
    tags: list[str] = field(default_factory=list)
    is_public: bool = True
    updated_at: str = ""
    word_count: int = 0
    reading_minutes: int = 1
    headings: list[DocumentHeading] = field(default_factory=list)
    order: int = 0

    def to_summary_dict(self) -> dict:
        return {
            "slug": self.slug,
            "title": self.title,
            "summary": self.summary,
            "category": self.category,
            "tags": list(self.tags),
            "public_url": self.public_url,
            "updated_at": self.updated_at,
            "word_count": self.word_count,
            "reading_minutes": self.reading_minutes,
        }

    def to_detail_dict(self, html: str) -> dict:
        payload = asdict(self)
        payload["headings"] = [heading.to_dict() for heading in self.headings]
        payload["html"] = html
        payload["raw_markdown_url"] = f"/api/docs/{self.slug}/raw"
        return payload


@dataclass
class IndexedDocument:
    meta: DocumentMeta
    markdown: str
    search_text: str
