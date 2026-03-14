from __future__ import annotations

import html
import os
import posixpath
import re
import unicodedata
from dataclasses import asdict
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Optional

from fastapi import HTTPException

from app.doc_service.indexer import DocumentIndexer, extract_headings, parse_front_matter
from app.doc_service.models import DocumentMeta, IndexedDocument
from app.doc_service.seo import build_seo_head
from app.md_viwer import render_markdown_html

SEARCH_WEIGHTS = {
    "title": 10,
    "tags": 7,
    "summary": 5,
    "headings": 4,
    "body": 2,
}
INVALID_FILENAME_RE = re.compile(r"[^A-Za-z0-9._\-\u3131-\u318E\uAC00-\uD7A3 ]+")
H2_H3_RE = re.compile(r"<h([23])>(.*?)</h\1>")
ANCHOR_HREF_RE = re.compile(r'(<a\s+[^>]*href=")([^"]+)(")', re.IGNORECASE)
KST = timezone(timedelta(hours=9))


class DocumentService:
    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.indexer = DocumentIndexer(project_dir)
        self.docs_by_slug: dict[str, IndexedDocument] = {}
        self.last_indexed_at = ""
        self._load()

    def _load(self) -> None:
        indexed_documents = self.indexer.build()
        self.docs_by_slug = {doc.meta.slug: doc for doc in indexed_documents}
        self.last_indexed_at = max((doc.meta.updated_at for doc in indexed_documents), default="")

    def _format_kst(self, value: str) -> str:
        if not value:
            return "-"
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(KST).strftime("%Y-%m-%d %H:%M KST")

    def reindex(self) -> dict:
        try:
            self._load()
        except Exception as exc:
            raise HTTPException(status_code=500, detail={"ok": False, "error_code": "INDEX_BUILD_FAILED", "message": str(exc)}) from exc
        return {
            "ok": True,
            "documents_count": len(self.docs_by_slug),
            "reindexed_at": self.last_indexed_at,
        }

    def _public_documents(self) -> list[IndexedDocument]:
        return [doc for doc in self.docs_by_slug.values() if doc.meta.is_public]

    def list_documents(
        self,
        category: Optional[str] = None,
        tag: Optional[str] = None,
        sort: str = "latest",
        page: int = 1,
        page_size: int = 20,
    ) -> dict:
        if sort not in {"latest", "popular", "title"}:
            raise HTTPException(
                status_code=400,
                detail={"ok": False, "error_code": "INVALID_SORT", "message": "Unsupported sort value"},
            )
        docs = self._public_documents()
        if category:
            docs = [doc for doc in docs if doc.meta.category == category]
        if tag:
            docs = [doc for doc in docs if tag in doc.meta.tags]
        docs = self._sort_documents(docs, sort)
        total = len(docs)
        start = max(0, (page - 1) * page_size)
        end = start + page_size
        return {
            "ok": True,
            "count": total,
            "page": page,
            "page_size": page_size,
            "items": [doc.meta.to_summary_dict() for doc in docs[start:end]],
        }

    def search_documents(
        self,
        query: str,
        category: Optional[str] = None,
        tag: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> dict:
        if not query.strip():
            raise HTTPException(
                status_code=400,
                detail={"ok": False, "error_code": "EMPTY_QUERY", "message": "Query must not be empty"},
            )
        qq = query.strip().lower()
        scored: list[tuple[float, list[str], IndexedDocument]] = []
        for doc in self._public_documents():
            if category and doc.meta.category != category:
                continue
            if tag and tag not in doc.meta.tags:
                continue
            score = 0
            matched_fields: list[str] = []
            if qq in doc.meta.title.lower():
                score += SEARCH_WEIGHTS["title"]
                matched_fields.append("title")
            if any(qq in item.lower() for item in doc.meta.tags):
                score += SEARCH_WEIGHTS["tags"]
                matched_fields.append("tags")
            if qq in doc.meta.summary.lower():
                score += SEARCH_WEIGHTS["summary"]
                matched_fields.append("summary")
            if any(qq in heading.text.lower() for heading in doc.meta.headings):
                score += SEARCH_WEIGHTS["headings"]
                matched_fields.append("headings")
            body_text = doc.markdown.lower()
            if qq in body_text:
                score += SEARCH_WEIGHTS["body"]
                matched_fields.append("body")
            if score == 0:
                continue
            matched_fields = sorted(set(matched_fields), key=lambda item: SEARCH_WEIGHTS[item], reverse=True)
            scored.append((float(score), matched_fields, doc))
        scored.sort(key=lambda item: (item[0], item[2].meta.updated_at), reverse=True)
        total = len(scored)
        start = max(0, (page - 1) * page_size)
        end = start + page_size
        items = []
        for score, matched_fields, doc in scored[start:end]:
            payload = doc.meta.to_summary_dict()
            payload["score"] = score
            payload["matched_fields"] = matched_fields
            items.append(payload)
        return {
            "ok": True,
            "query": query,
            "count": total,
            "page": page,
            "page_size": page_size,
            "items": items,
        }

    def get_document_detail(self, slug: str) -> dict:
        doc = self.docs_by_slug.get(slug)
        if doc is None:
            raise HTTPException(status_code=404, detail={"ok": False, "error_code": "DOC_NOT_FOUND", "message": "Document not found"})
        rendered_html = self._render_document_html(doc)
        related_documents = [item.meta.to_summary_dict() for item in self._related_documents(doc)]
        return {
            "ok": True,
            "document": doc.meta.to_detail_dict(rendered_html),
            "related_documents": related_documents,
        }

    def get_document_raw(self, slug: str) -> str:
        doc = self.docs_by_slug.get(slug)
        if doc is None:
            raise HTTPException(status_code=404, detail={"ok": False, "error_code": "DOC_NOT_FOUND", "message": "Document not found"})
        return doc.markdown

    def upload_document(self, filename: str, content: str) -> tuple[int, dict]:
        safe_name = self._sanitize_md_filename(filename)
        target_dir = self.indexer.uploads_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / safe_name
        payload_bytes = content.encode("utf-8")
        if len(payload_bytes) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail={"ok": False, "error_code": "FILE_TOO_LARGE", "message": "File too large (max 10MB)"},
            )
        target_path.write_bytes(payload_bytes)
        status_code = 201
        try:
            result = self.reindex()
            slug = self._slug_for_source_path(target_path.relative_to(self.project_dir).as_posix())
            return status_code, {
                "ok": True,
                "file_name": safe_name,
                "size_bytes": len(payload_bytes),
                "slug": slug,
                "public_url": f"/p/{slug}",
                "reindexed": True,
                "message": "Upload completed and index updated.",
                "admin_url": "/admin/docs",
                **result,
            }
        except HTTPException:
            fallback_slug = self._fallback_slug_from_filename(safe_name)
            return status_code, {
                "ok": True,
                "file_name": safe_name,
                "size_bytes": len(payload_bytes),
                "slug": fallback_slug,
                "public_url": f"/p/{fallback_slug}",
                "reindexed": False,
                "message": "Upload completed, but indexing is delayed. Check /admin/docs.",
                "admin_url": "/admin/docs",
            }

    def preview_document(self, filename: str, content: str) -> dict:
        safe_name = self._sanitize_md_filename(filename or "preview.md")
        front_matter, body = parse_front_matter(content)
        title = str(front_matter.get("title") or self._preview_title(body, safe_name))
        summary = str(front_matter.get("summary") or self._preview_summary(body, title))
        headings = [heading.to_dict() for heading in extract_headings(body)]
        html_body = render_markdown_html(content)
        return {
            "ok": True,
            "file_name": safe_name,
            "title": title,
            "summary": summary,
            "headings": headings,
            "html": html_body,
        }

    def admin_summary(self) -> dict:
        documents = sorted(self.docs_by_slug.values(), key=lambda doc: doc.meta.updated_at, reverse=True)
        items = [
            {
                "title": doc.meta.title,
                "slug": doc.meta.slug,
                "category": doc.meta.category,
                "is_public": doc.meta.is_public,
                "updated_at": doc.meta.updated_at,
                "source_path": doc.meta.source_path,
            }
            for doc in documents
        ]
        return {
            "total_docs": len(documents),
            "published_docs": sum(1 for doc in documents if doc.meta.is_public),
            "hidden_docs": sum(1 for doc in documents if not doc.meta.is_public),
            "last_indexed_at": self.last_indexed_at,
            "items": items,
        }

    def render_landing_page(self) -> str:
        featured = self._sort_documents(self._public_documents(), "latest")[:5]
        featured_items = "".join(
            "<li>"
            f"<a href='{html.escape(item.meta.public_url, quote=True)}'>{html.escape(item.meta.title)}</a>"
            f" <span style='color:#667085;'>({html.escape(self._format_kst(item.meta.updated_at))})</span>"
            "</li>"
            for item in featured
        ) or "<li>샘플 문서 준비 중</li>"
        head = build_seo_head(
            title="Sattie Docs Hub",
            description="문서 허브, 업로드, 운영 화면으로 이동하는 랜딩 페이지",
            canonical_url="/",
        )
        return (
            "<html><head><meta charset='utf-8'/>"
            f"{head}"
            "<style>body{font-family:sans-serif;padding:32px;line-height:1.6;background:linear-gradient(180deg,#f6f8fb 0%,#eef3ff 100%);}main{max-width:1320px;margin:0 auto;}ul{padding-left:20px;}a{color:#0f5cdd;text-decoration:none;}section{margin:24px 0;padding:20px;border:1px solid #d6dbe3;border-radius:16px;background:#fff;} .hero{padding:28px;border-radius:24px;background:linear-gradient(135deg,#0f172a,#1d4ed8);color:#fff;} .hero a{color:#fff;} .cta{display:flex;gap:12px;flex-wrap:wrap;margin-top:16px;} .cta a{display:inline-block;padding:12px 16px;border-radius:12px;background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.24);} .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;} .card{padding:16px;border-radius:16px;background:#fff;border:1px solid #d6dbe3;}</style>"
            "</head><body><main>"
            "<section class='hero'>"
            "<h1 style='margin-top:0;'>마크다운 파일 기반 블로그</h1>"
            "<p>복잡한 수식, 차트들도 아주 깔끔하게 보여주는 Sattie Document Viewer 서비스</p>"
            "<div class='cta'><a href='/hub'>문서 보기</a><a href='/upload'>문서 업로드</a></div>"
            "</section>"
            "<section><h2>주요 경로</h2><ul>"
            "<li><a href='/hub'>문서 허브</a></li>"
            "<li><a href='/upload'>문서 업로드</a></li>"
            "<li><a href='/admin/docs'>운영자 화면</a></li>"
            "<li><a href='/guide'>기존 Guide</a></li>"
            "<li><a href='/ui'>기존 위성영상 UI</a></li>"
            "</ul></section>"
            f"<section><h2>최근 문서</h2><ul>{featured_items}</ul></section>"
            "</main></body></html>"
        )

    def render_hub_page(self, category: Optional[str] = None, tag: Optional[str] = None, sort: str = "latest", q: str = "") -> str:
        public_docs = self._public_documents()
        categories = sorted({doc.meta.category for doc in public_docs})
        category_tabs = ["<a href='/hub' style='display:inline-flex;align-items:center;justify-content:center;height:44px;padding:0 14px;border-radius:999px;border:1px solid #cbd5e1;background:#fff;box-sizing:border-box;'>All</a>"]
        for item in categories:
            current = "background:#0f5cdd;color:#fff;border-color:#0f5cdd;" if category == item else "background:#fff;"
            category_tabs.append(
                f"<a href='/hub/category/{html.escape(item, quote=True)}' style='display:inline-flex;align-items:center;justify-content:center;height:44px;padding:0 14px;border-radius:999px;border:1px solid #cbd5e1;box-sizing:border-box;{current}'>{html.escape(item)}</a>"
            )
        if q.strip():
            result = self.search_documents(q, category=category, tag=tag, page=1, page_size=20)
            title = "문서 검색 결과"
            items = result["items"]
        else:
            result = self.list_documents(category=category, tag=tag, sort=sort, page=1, page_size=20)
            title = "문서 허브"
            items = result["items"]
        featured_docs = self._sort_documents(public_docs, "popular")[:3]
        recent_docs = self._sort_documents(public_docs, "latest")[:5]
        cards = []
        for item in items:
            cards.append(
                "<li style='padding:16px 0;border-bottom:1px solid #e5e7eb;list-style:none;'>"
                f"<h3 style='margin:0 0 8px;'><a href='{html.escape(item['public_url'], quote=True)}'>{html.escape(item['title'])}</a></h3>"
                f"<p style='margin:0 0 8px;color:#475467;'>{html.escape(item['summary'])}</p>"
                f"<p style='margin:0;font-size:14px;color:#667085;'>카테고리: {html.escape(item['category'])} | 읽기시간: {item['reading_minutes']}분</p>"
                "</li>"
            )
        cards_html = ("<ul style='margin:0;padding:0;background:#fff;border-top:1px solid #e5e7eb;border-bottom:1px solid #e5e7eb;'>" + "".join(cards) + "</ul>") if cards else "<p>표시할 문서가 없습니다.</p>"
        featured_html = "".join(
            f"<li><a href='{html.escape(item.meta.public_url, quote=True)}'>{html.escape(item.meta.title)}</a></li>"
            for item in featured_docs
        ) or "<li>문서 없음</li>"
        recent_html = "".join(
            f"<li><a href='{html.escape(item.meta.public_url, quote=True)}'>{html.escape(item.meta.title)}</a> <span style='color:#667085;'>{html.escape(self._format_kst(item.meta.updated_at))}</span></li>"
            for item in recent_docs
        ) or "<li>문서 없음</li>"
        head = build_seo_head(title=title, description="문서 허브 화면", canonical_url="/hub")
        return (
            "<html><head><meta charset='utf-8'/>"
            f"{head}"
            "<style>body{font-family:sans-serif;padding:24px;background:#f7f8fa;}main{max-width:1320px;margin:0 auto;}input,select,button{height:44px;padding:0 12px;border:1px solid #cbd5e1;border-radius:10px;box-sizing:border-box;line-height:44px;margin:0;}button{background:#0f5cdd;color:#fff;border:none;line-height:1;}section.cards{display:block;} .filters{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:nowrap;margin:12px 0 20px;min-height:44px;} .filters form{display:flex;align-items:center;justify-content:flex-end;gap:12px;flex:1;flex-wrap:nowrap;margin:0;} .filters input[name='q']{width:280px;flex:0 0 280px;} .filters input[name='tag']{width:140px;flex:0 0 140px;} .filters select{width:120px;flex:0 0 120px;} .tabs{display:flex;gap:10px;flex-wrap:wrap;align-items:center;flex:0 0 auto;min-height:44px;} .tabs a{line-height:1;} .grid{display:grid;grid-template-columns:280px 1fr;gap:20px;} .panel{background:#fff;border:1px solid #d6dbe3;border-radius:16px;padding:18px;} @media(max-width:900px){.filters{align-items:stretch;justify-content:flex-start;flex-wrap:wrap;}.filters form{width:100%;justify-content:flex-start;flex-wrap:wrap;}.filters input[name='q'],.filters input[name='tag'],.filters select{width:100%;flex:1 1 auto;}.grid{grid-template-columns:1fr;}}</style>"
            "</head><body><main>"
            "<p><a href='/'>메인으로</a> | <a href='/upload'>업로드</a> | <a href='/admin/docs'>운영자</a></p>"
            f"<h1>{html.escape(title)}</h1>"
            "<div class='filters'>"
            f"<div class='tabs'>{''.join(category_tabs)}</div>"
            "<form method='get' action='/hub'>"
            f"<input type='text' name='q' placeholder='검색어 입력' value='{html.escape(q, quote=True)}'/>"
            f"<input type='text' name='tag' placeholder='tag' value='{html.escape(tag or '', quote=True)}'/>"
            f"<select name='sort'><option value='latest'{' selected' if sort == 'latest' else ''}>latest</option><option value='popular'{' selected' if sort == 'popular' else ''}>popular</option><option value='title'{' selected' if sort == 'title' else ''}>title</option></select>"
            "<button type='submit'>조회</button>"
            "</form></div>"
            "<div class='grid'>"
            "<div class='panel'>"
            "<h2 style='margin-top:0;'>탐색</h2>"
            "<h3>Featured Documents</h3><ul>"
            f"{featured_html}"
            "</ul><h3>Recent Updates</h3><ul>"
            f"{recent_html}"
            "</ul></div>"
            "<div>"
            f"<p>문서 수: {result['count']}</p>"
            f"<section class='cards'>{cards_html}</section>"
            "</div></div>"
            "</main></body></html>"
        )

    def render_detail_page(self, slug: str) -> str:
        payload = self.get_document_detail(slug)
        document = payload["document"]
        current_doc = self.docs_by_slug[slug]
        category_related = "".join(
            f"<li><a href='{html.escape(item.meta.public_url, quote=True)}'>{html.escape(item.meta.title)}</a></li>"
            for item in self._related_documents(current_doc)
        ) or "<li>카테고리 내 다른 문서 없음</li>"
        head = build_seo_head(
            title=document["title"],
            description=document["summary"],
            canonical_url=document["public_url"],
        )
        return (
            "<html><head><meta charset='utf-8'/>"
            f"{head}"
            "<style>body{font-family:sans-serif;margin:0;background:#f5f7fb;overflow-x:hidden;}main{max-width:1320px;margin:0 auto;padding:24px;display:grid;grid-template-columns:240px minmax(0,1fr);gap:24px;}aside,article{background:#fff;border:1px solid #d8dee7;border-radius:16px;padding:20px;min-width:0;}article img{max-width:100%;}article table{display:block;max-width:100%;overflow:auto;white-space:nowrap;}article p,article li,article h1,article h2,article h3,article h4,article h5,article h6,article td,article th{overflow-wrap:anywhere;word-break:break-word;}pre{overflow:auto;position:relative;background:#0f172a;color:#e2e8f0;padding:16px;border-radius:12px;max-width:100%;}article pre.mermaid,article .mermaid{background:#fff !important;color:inherit;border:1px solid #d8dee7;}article .mermaid svg{background:transparent !important;}article .mermaid svg,.chart-zoomable{cursor:zoom-in;}article .math-block{margin:14px 0;overflow:auto;background:transparent;border:none;border-radius:0;padding:0;color:#111827;text-align:left;max-width:100%;}article .math-block mjx-container,article .math-block svg{background:transparent !important;}article .math-block mjx-container[display='true']{display:block !important;text-align:left !important;margin:0 !important;}pre button.copy-btn{position:absolute;top:10px;right:10px;padding:6px 10px;border:none;border-radius:8px;background:#e2e8f0;color:#111827;cursor:pointer;} article h2[id],article h3[id]{scroll-margin-top:90px;position:relative;} article h2[id] .heading-copy,article h3[id] .heading-copy{margin-left:8px;font-size:12px;color:#0f5cdd;text-decoration:none;} #imgModal{position:fixed;inset:0;background:rgba(0,0,0,.85);display:none;align-items:center;justify-content:center;z-index:9999;padding:24px;}#imgModal.show{display:flex;}#imgModal img{max-width:95vw;max-height:90vh;width:auto;height:auto;border-radius:8px;background:#111;}#imgModal .close{position:absolute;top:12px;right:16px;color:#fff;font-size:30px;line-height:1;cursor:pointer;}#chartModal{position:fixed;inset:0;background:rgba(255,255,255,.98);display:none;align-items:center;justify-content:center;z-index:10000;padding:28px;box-sizing:border-box;}#chartModal.show{display:flex;}#chartModal .chart-shell{width:min(96vw,1600px);height:min(92vh,980px);display:flex;flex-direction:column;background:#fff;border:1px solid #d8dee7;border-radius:28px;box-shadow:0 30px 80px rgba(15,23,42,.12);overflow:hidden;}#chartModal .chart-toolbar{display:flex;gap:14px;align-items:center;justify-content:flex-end;flex-wrap:wrap;padding:6px 12px;color:#111827;background:#fff;text-align:right;min-height:36px;width:50%;align-self:flex-end;box-sizing:border-box;order:2;}#chartModal .chart-toolbar button{border:none;background:transparent;color:#0f5cdd;border-radius:0;padding:0;cursor:pointer;text-decoration:none;font:inherit;line-height:1.2;}#chartModal .chart-toolbar button:hover{text-decoration:underline;}#chartModal .chart-stage{flex:1;position:relative;overflow:hidden;cursor:grab;padding:24px;display:flex;align-items:center;justify-content:center;background:#fff;order:1;}#chartModal .chart-stage.dragging{cursor:grabbing;}#chartModal .chart-canvas{position:absolute;inset:24px;display:flex;align-items:center;justify-content:center;}#chartModal .chart-canvas svg{display:block;width:100%;height:100%;max-width:none !important;max-height:none !important;}@media(max-width:900px){main{grid-template-columns:1fr;padding:14px;}aside{order:2;}article{order:1;}aside,article{padding:16px;}article table{font-size:14px;}}</style>"
            "<script>window.MathJax={tex:{inlineMath:[['$','$'],['\\\\(','\\\\)']],displayMath:[['$$','$$'],['\\\\[','\\\\]']],processEscapes:true},startup:{typeset:false}};</script>"
            "<script>window.addEventListener('load',function(){var enhanceCharts=function(){var chartModal=document.getElementById('chartModal');var chartStage=chartModal?chartModal.querySelector('.chart-stage'):null;var chartCanvas=chartModal?chartModal.querySelector('.chart-canvas'):null;var chartState={svg:null,base:null,current:null,dragging:false,startX:0,startY:0,startViewBox:null};var cloneBox=function(box){return box?{x:box.x,y:box.y,w:box.w,h:box.h}:null;};var readViewBox=function(svg){var vb=svg.getAttribute('viewBox');if(vb){var p=vb.trim().split(/\\s+/).map(Number);if(p.length===4&&p.every(function(n){return Number.isFinite(n);})){return {x:p[0],y:p[1],w:p[2],h:p[3]};}}var width=svg.viewBox&&svg.viewBox.baseVal&&svg.viewBox.baseVal.width?svg.viewBox.baseVal.width:0;var height=svg.viewBox&&svg.viewBox.baseVal&&svg.viewBox.baseVal.height?svg.viewBox.baseVal.height:0;if(!(width>0&&height>0)){var bbox=svg.getBBox();width=bbox.width||1000;height=bbox.height||800;return {x:bbox.x||0,y:bbox.y||0,w:width,h:height};}return {x:0,y:0,w:width,h:height};};var applyViewBox=function(){if(chartState.svg&&chartState.current){chartState.svg.setAttribute('viewBox',[chartState.current.x,chartState.current.y,chartState.current.w,chartState.current.h].join(' '));}};var resetChart=function(){chartState.current=cloneBox(chartState.base);applyViewBox();};var zoomChart=function(factor,originX,originY){if(!chartState.current||!chartStage)return;var rect=chartStage.getBoundingClientRect();var ox=typeof originX==='number'?originX:rect.width/2;var oy=typeof originY==='number'?originY:rect.height/2;var rx=Math.max(0,Math.min(1,(ox-24)/Math.max(1,rect.width-48)));var ry=Math.max(0,Math.min(1,(oy-24)/Math.max(1,rect.height-48)));var nextW=chartState.current.w/factor;var nextH=chartState.current.h/factor;chartState.current={x:chartState.current.x+(chartState.current.w-nextW)*rx,y:chartState.current.y+(chartState.current.h-nextH)*ry,w:nextW,h:nextH};applyViewBox();};var openChart=function(svg){if(!chartModal||!chartCanvas||!svg)return;chartCanvas.innerHTML='';chartState.svg=svg.cloneNode(true);chartState.svg.removeAttribute('style');chartState.svg.setAttribute('preserveAspectRatio','xMidYMin meet');chartCanvas.appendChild(chartState.svg);chartState.base=readViewBox(chartState.svg);chartModal.classList.add('show');chartModal.setAttribute('aria-hidden','false');resetChart();};var closeChart=function(){if(!chartModal||!chartCanvas)return;chartModal.classList.remove('show');chartModal.setAttribute('aria-hidden','true');chartCanvas.innerHTML='';chartState.svg=null;chartState.base=null;chartState.current=null;chartState.dragging=false;if(chartStage)chartStage.classList.remove('dragging');};document.querySelectorAll('.mermaid svg').forEach(function(svg){if(svg.dataset.zoomBound==='1')return;svg.dataset.zoomBound='1';svg.classList.add('chart-zoomable');svg.addEventListener('click',function(){openChart(svg);});});if(chartModal&&!chartModal.dataset.bound){chartModal.dataset.bound='1';chartModal.querySelectorAll('button[data-action]').forEach(function(btn){btn.addEventListener('click',function(){var action=btn.getAttribute('data-action');if(action==='close'){closeChart();}else if(action==='zoom-in'){zoomChart(1.2);}else if(action==='zoom-out'){zoomChart(1/1.2);}else if(action==='reset'){resetChart();}});});chartStage.addEventListener('mousedown',function(e){if(!chartState.current)return;chartState.dragging=true;chartState.startX=e.clientX;chartState.startY=e.clientY;chartState.startViewBox=cloneBox(chartState.current);chartStage.classList.add('dragging');});window.addEventListener('mousemove',function(e){if(!chartState.dragging||!chartState.current||!chartStage)return;var rect=chartStage.getBoundingClientRect();var dx=e.clientX-chartState.startX;var dy=e.clientY-chartState.startY;chartState.current.x=chartState.startViewBox.x-(dx/Math.max(1,rect.width-48))*chartState.startViewBox.w;chartState.current.y=chartState.startViewBox.y-(dy/Math.max(1,rect.height-48))*chartState.startViewBox.h;applyViewBox();});window.addEventListener('mouseup',function(){chartState.dragging=false;chartState.startViewBox=null;if(chartStage)chartStage.classList.remove('dragging');});chartStage.addEventListener('wheel',function(e){e.preventDefault();var rect=chartStage.getBoundingClientRect();zoomChart(e.deltaY<0?1.12:1/1.12,e.clientX-rect.left,e.clientY-rect.top);},{passive:false});chartModal.addEventListener('click',function(e){if(e.target===chartModal){closeChart();}});window.addEventListener('keydown',function(e){if(e.key==='Escape'){closeChart();}});} };var mj=document.createElement('script');mj.src='https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js';mj.async=true;mj.onload=function(){if(window.MathJax&&window.MathJax.typesetPromise){window.MathJax.typesetPromise().catch(function(){});}};document.head.appendChild(mj);var mm=document.createElement('script');mm.src='https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js';mm.async=true;mm.onload=function(){if(window.mermaid){window.mermaid.initialize({startOnLoad:true,securityLevel:'loose'});window.mermaid.run({querySelector:'.mermaid'}).then(enhanceCharts).catch(function(){});}else{enhanceCharts();}};document.head.appendChild(mm);document.querySelectorAll('pre > code').forEach(function(code){var pre=code.parentElement;var btn=document.createElement('button');btn.textContent='Copy';btn.className='copy-btn';btn.type='button';btn.onclick=function(){navigator.clipboard.writeText(code.textContent||'');btn.textContent='Copied';setTimeout(function(){btn.textContent='Copy';},1200);};pre.appendChild(btn);});document.querySelectorAll('article h2[id], article h3[id]').forEach(function(node){var link=document.createElement('a');link.href='#'+node.id;link.textContent='링크 복사';link.className='heading-copy';link.onclick=function(ev){ev.preventDefault();navigator.clipboard.writeText(window.location.origin+window.location.pathname+'#'+node.id);};node.appendChild(link);});var modal=document.getElementById('imgModal');var modalImg=document.getElementById('imgModalImg');document.querySelectorAll('article img').forEach(function(img){img.style.cursor='zoom-in';img.addEventListener('click',function(){modalImg.src=img.src;modal.classList.add('show');});});modal.addEventListener('click',function(){modal.classList.remove('show');modalImg.src='';});enhanceCharts();});</script>"
            "</head><body>"
            "<main>"
            "<aside>"
            "<p><a href='/hub'>문서 허브</a></p>"
            "<h3>카테고리 문서</h3><ul>"
            f"{category_related}"
            "</ul>"
            "</aside>"
            "<article>"
            f"<h1>{html.escape(document['title'])}</h1>"
            f"<p>{html.escape(document['summary'])}</p>"
            f"<p style='color:#667085;'>카테고리: {html.escape(document['category'])} | 읽기시간: {document['reading_minutes']}분 | 원문: <a href='{html.escape(document['raw_markdown_url'], quote=True)}'>다운로드</a></p>"
            f"{document['html']}"
            "</article>"
            "</main>"
            "<div id='imgModal'><span class='close'>&times;</span><img id='imgModalImg' alt='preview'/></div>"
            "<div id='chartModal' aria-hidden='true'><div class='chart-shell'><div class='chart-toolbar'><button type='button' data-action='zoom-in'>확대</button><button type='button' data-action='zoom-out'>축소</button><button type='button' data-action='reset'>원위치</button><button type='button' data-action='close'>닫기</button></div><div class='chart-stage'><div class='chart-canvas'></div></div></div></div>"
            "<script>(function(){const chartModal=document.getElementById('chartModal');if(!chartModal){return;}const chartStage=chartModal.querySelector('.chart-stage');const chartCanvas=chartModal.querySelector('.chart-canvas');const state={svg:null,base:null,current:null,dragging:false,startX:0,startY:0,startViewBox:null};const cloneBox=function(box){return box?{x:box.x,y:box.y,w:box.w,h:box.h}:null;};const readViewBox=function(svg){var vb=svg.getAttribute('viewBox');if(vb){var p=vb.trim().split(/\\s+/).map(Number);if(p.length===4&&p.every(Number.isFinite)){return{x:p[0],y:p[1],w:p[2],h:p[3]};}}var bbox=svg.getBBox();return{x:bbox.x||0,y:bbox.y||0,w:bbox.width||1000,h:bbox.height||800};};const applyViewBox=function(){if(state.svg&&state.current){state.svg.setAttribute('viewBox',[state.current.x,state.current.y,state.current.w,state.current.h].join(' '));}};const reset=function(){state.current=cloneBox(state.base);applyViewBox();};const open=function(svg){chartCanvas.innerHTML='';state.svg=svg.cloneNode(true);state.svg.removeAttribute('style');state.svg.setAttribute('preserveAspectRatio','xMidYMin meet');chartCanvas.appendChild(state.svg);state.base=readViewBox(state.svg);chartModal.classList.add('show');chartModal.setAttribute('aria-hidden','false');reset();};const close=function(){chartModal.classList.remove('show');chartModal.setAttribute('aria-hidden','true');chartCanvas.innerHTML='';state.svg=null;state.base=null;state.current=null;state.dragging=false;chartStage.classList.remove('dragging');};const zoom=function(factor,clientX,clientY){if(!state.current){return;}var rect=chartStage.getBoundingClientRect();var rx=(typeof clientX==='number')?(clientX-rect.left-24)/Math.max(1,rect.width-48):0.5;var ry=(typeof clientY==='number')?(clientY-rect.top-24)/Math.max(1,rect.height-48):0.5;rx=Math.max(0,Math.min(1,rx));ry=Math.max(0,Math.min(1,ry));var nextW=state.current.w/factor;var nextH=state.current.h/factor;state.current={x:state.current.x+(state.current.w-nextW)*rx,y:state.current.y+(state.current.h-nextH)*ry,w:nextW,h:nextH};applyViewBox();};document.addEventListener('click',function(e){var btn=e.target.closest('#chartModal button[data-action]');if(btn){e.preventDefault();e.stopPropagation();var action=btn.getAttribute('data-action');if(action==='close'){close();}else if(action==='zoom-in'){zoom(1.2);}else if(action==='zoom-out'){zoom(1/1.2);}else if(action==='reset'){reset();}return;}var svg=e.target.closest('.mermaid svg');if(svg&&!chartModal.contains(svg)){e.preventDefault();e.stopPropagation();open(svg);return;}if(e.target===chartModal){close();}},true);chartStage.addEventListener('pointerdown',function(e){if(!state.current){return;}state.dragging=true;state.startX=e.clientX;state.startY=e.clientY;state.startViewBox=cloneBox(state.current);chartStage.classList.add('dragging');chartStage.setPointerCapture(e.pointerId);});chartStage.addEventListener('pointermove',function(e){if(!state.dragging||!state.current){return;}var rect=chartStage.getBoundingClientRect();var dx=e.clientX-state.startX;var dy=e.clientY-state.startY;state.current.x=state.startViewBox.x-(dx/Math.max(1,rect.width-48))*state.startViewBox.w;state.current.y=state.startViewBox.y-(dy/Math.max(1,rect.height-48))*state.startViewBox.h;applyViewBox();});chartStage.addEventListener('pointerup',function(e){state.dragging=false;state.startViewBox=null;chartStage.classList.remove('dragging');try{chartStage.releasePointerCapture(e.pointerId);}catch(_e){}});chartStage.addEventListener('pointercancel',function(){state.dragging=false;state.startViewBox=null;chartStage.classList.remove('dragging');});chartStage.addEventListener('wheel',function(e){if(!state.current){return;}e.preventDefault();zoom(e.deltaY<0?1.12:1/1.12,e.clientX,e.clientY);},{passive:false});window.addEventListener('keydown',function(e){if(e.key==='Escape'){close();}});})();</script>"
            "</body></html>"
        )

    def render_upload_page(self) -> str:
        head = build_seo_head(title="문서 업로드", description="Markdown 문서를 업로드합니다.", canonical_url="/upload")
        return (
            "<html><head><meta charset='utf-8'/>"
            f"{head}"
            "<style>body{font-family:sans-serif;padding:24px;background:#f7f8fa;}main{max-width:1320px;margin:0 auto;}textarea,input,button{width:100%;box-sizing:border-box;margin:10px 0;padding:12px;border-radius:10px;border:1px solid #cbd5e1;}button{background:#0f5cdd;color:#fff;border:none;}button.secondary{background:#fff;color:#0f172a;border:1px solid #cbd5e1;} .layout{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:20px;} .panel{background:#fff;border:1px solid #d6dbe3;border-radius:16px;padding:24px;} .toolbar{display:flex;gap:10px;flex-wrap:wrap;} .toolbar button{width:auto;padding:10px 14px;} #result{white-space:pre-wrap;background:#f8fafc;padding:12px;border-radius:10px;display:none;} #status{margin-top:8px;color:#475467;} #previewMeta{margin:0 0 16px;color:#667085;} #previewBody{min-height:360px;border:1px solid #e5e7eb;border-radius:12px;padding:20px;background:#fff;} #previewBody pre{overflow:auto;background:#0f172a;color:#e2e8f0;padding:16px;border-radius:12px;} #previewBody pre.mermaid,#previewBody .mermaid{background:#fff !important;color:inherit;border:1px solid #d8dee7;padding:12px;border-radius:12px;} #previewBody .math-block{margin:14px 0;} #previewBody img{max-width:100%;} @media(max-width:960px){.layout{grid-template-columns:1fr;}}</style>"
            "<script>let previewTimer=null;async function previewDoc(){const filename=(document.getElementById('filename').value||'preview.md').trim()||'preview.md';const content=document.getElementById('content').value;const status=document.getElementById('status');const meta=document.getElementById('previewMeta');const body=document.getElementById('previewBody');if(!content.trim()){meta.textContent='미리보기 내용이 없습니다.';body.innerHTML='';return;}status.textContent='미리보기 생성 중...';const res=await fetch('/api/docs/preview',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({filename,content})});const data=await res.json();if(!res.ok){status.textContent=data.message||('미리보기 실패 ('+res.status+')');meta.textContent='';body.innerHTML='';return;}meta.textContent=(data.title||filename)+' | headings '+(data.headings?data.headings.length:0)+'개';body.innerHTML=data.html;status.textContent='미리보기 업데이트 완료';}function queuePreview(){clearTimeout(previewTimer);previewTimer=setTimeout(previewDoc,250);}async function selectFile(ev){const file=(ev.target.files||[])[0];if(!file){return;}document.getElementById('filename').value=file.name;document.getElementById('status').textContent='파일 읽는 중...';const text=await file.text();document.getElementById('content').value=text;document.getElementById('status').textContent='파일 선택 완료';queuePreview();}async function uploadDoc(){const filename=document.getElementById('filename').value;const content=document.getElementById('content').value;const result=document.getElementById('result');const status=document.getElementById('status');status.textContent='업로드 중...';result.style.display='none';const res=await fetch('/api/docs/upload',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({filename,content})});const data=await res.json();result.style.display='block';result.textContent=JSON.stringify(data,null,2);status.textContent=data.message||('완료 ('+res.status+')');if(data.reindexed&&data.public_url){window.location.href=data.public_url;}}window.addEventListener('load',function(){document.getElementById('picker').addEventListener('change',selectFile);document.getElementById('content').addEventListener('input',queuePreview);document.getElementById('filename').addEventListener('input',queuePreview);});</script>"
            "</head><body><main>"
            "<p><a href='/hub'>문서 허브</a></p>"
            "<h1>문서 업로드</h1>"
            "<div class='layout'>"
            "<section class='panel'>"
            "<h2 style='margin-top:0;'>업로드 입력</h2>"
            "<input id='filename' placeholder='example.md'/>"
            "<input id='picker' type='file' accept='.md,text/markdown'/>"
            "<textarea id='content' rows='20' placeholder='# title'></textarea>"
            "<div class='toolbar'><button type='button' class='secondary' onclick='previewDoc()'>미리보기</button><button type='button' onclick='uploadDoc()'>업로드</button></div>"
            "<p id='status'>`.md` 파일을 선택하거나 내용을 직접 입력한 뒤 미리보기와 업로드를 사용할 수 있습니다.</p>"
            "<pre id='result'></pre>"
            "</section>"
            "<section class='panel'>"
            "<h2 style='margin-top:0;'>미리보기</h2>"
            "<p id='previewMeta'>미리보기 내용이 없습니다.</p>"
            "<div id='previewBody'></div>"
            "</section>"
            "</div>"
            "</main></body></html>"
        )

    def render_admin_page(self) -> str:
        summary = self.admin_summary()
        rows = []
        for item in summary["items"]:
            rows.append(
                "<tr>"
                f"<td>{html.escape(item['title'])}</td>"
                f"<td>{html.escape(item['slug'])}</td>"
                f"<td>{html.escape(item['category'])}</td>"
                f"<td>{'published' if item['is_public'] else 'hidden'}</td>"
                f"<td>{html.escape(self._format_kst(item['updated_at']))}</td>"
                f"<td>{html.escape(item['source_path'])}</td>"
                "</tr>"
            )
        table_rows = "".join(rows) or "<tr><td colspan='6'>문서 없음</td></tr>"
        head = build_seo_head(title="운영자 문서 관리", description="문서 인덱스 상태와 재색인을 관리합니다.", canonical_url="/admin/docs")
        return (
            "<html><head><meta charset='utf-8'/>"
            f"{head}"
            "<style>body{font-family:sans-serif;padding:24px;background:#f7f8fa;}main{max-width:1320px;margin:0 auto;}section{background:#fff;border:1px solid #d6dbe3;border-radius:16px;padding:20px;margin-bottom:16px;}table{width:100%;border-collapse:collapse;}th,td{border-bottom:1px solid #e5e7eb;padding:10px;text-align:left;}button{padding:10px 14px;border:none;border-radius:10px;background:#0f5cdd;color:#fff;margin-right:8px;}#adminStatus{white-space:pre-wrap;background:#f8fafc;padding:12px;border-radius:10px;display:none;}</style>"
            "<script>async function reindexDocs(){const res=await fetch('/api/docs/reindex',{method:'POST'});const data=await res.json();const box=document.getElementById('adminStatus');box.style.display='block';box.textContent=JSON.stringify(data,null,2);if(res.ok){window.location.reload();}} function refreshAdmin(){window.location.reload();}</script>"
            "</head><body><main>"
            "<p><a href='/hub'>문서 허브</a></p>"
            "<h1>운영자 문서 관리</h1>"
            "<section><div style='display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;'>"
            f"<div><strong>Total Docs</strong><p>{summary['total_docs']}</p></div>"
            f"<div><strong>Published</strong><p>{summary['published_docs']}</p></div>"
            f"<div><strong>Hidden</strong><p>{summary['hidden_docs']}</p></div>"
            f"<div><strong>Last Indexed</strong><p>{html.escape(self._format_kst(summary['last_indexed_at']))}</p></div>"
            "</div></section>"
            "<section><button type='button' onclick='reindexDocs()'>Reindex All</button><button type='button' onclick='refreshAdmin()'>Refresh</button><pre id='adminStatus'></pre></section>"
            "<section><table><thead><tr><th>Title</th><th>Slug</th><th>Category</th><th>Status</th><th>Updated</th><th>Source</th></tr></thead><tbody>"
            f"{table_rows}"
            "</tbody></table></section>"
            "</main></body></html>"
        )

    def _slug_for_source_path(self, source_path: str) -> str:
        for slug, document in self.docs_by_slug.items():
            if document.meta.source_path == source_path:
                return slug
        return self._fallback_slug_from_filename(Path(source_path).name)

    def _fallback_slug_from_filename(self, filename: str) -> str:
        name = Path(filename).stem
        return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "document"

    def _sanitize_md_filename(self, name: str) -> str:
        base = unicodedata.normalize("NFC", Path(name).name).strip()
        if not base:
            raise HTTPException(
                status_code=400,
                detail={"ok": False, "error_code": "INVALID_FILENAME", "message": "Filename is empty"},
            )
        if not base.lower().endswith(".md"):
            raise HTTPException(
                status_code=400,
                detail={"ok": False, "error_code": "INVALID_UPLOAD_TYPE", "message": "Only .md upload is allowed"},
            )
        safe = INVALID_FILENAME_RE.sub("_", base).strip()
        if not safe or safe in {".", ".."}:
            raise HTTPException(
                status_code=400,
                detail={"ok": False, "error_code": "INVALID_FILENAME", "message": "Invalid filename"},
            )
        return safe

    def _preview_title(self, markdown_body: str, fallback_filename: str) -> str:
        for line in markdown_body.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                return stripped[2:].strip() or Path(fallback_filename).stem
        return Path(fallback_filename).stem

    def _preview_summary(self, markdown_body: str, fallback: str) -> str:
        for line in markdown_body.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                return stripped
        return fallback

    def _sort_documents(self, docs: list[IndexedDocument], sort: str) -> list[IndexedDocument]:
        if sort == "title":
            return sorted(docs, key=lambda doc: doc.meta.title.lower())
        if sort == "popular":
            docs = sorted(docs, key=lambda doc: doc.meta.updated_at, reverse=True)
            return sorted(docs, key=lambda doc: doc.meta.order)
        return sorted(docs, key=lambda doc: doc.meta.updated_at, reverse=True)

    def _render_document_html(self, doc: IndexedDocument) -> str:
        html_text = render_markdown_html(doc.markdown)
        heading_iter = iter(doc.meta.headings)

        def replace_heading(match: re.Match[str]) -> str:
            heading = next(heading_iter, None)
            if heading is None:
                return match.group(0)
            level = match.group(1)
            content = match.group(2)
            return f"<h{level} id=\"{html.escape(heading.anchor, quote=True)}\">{content}</h{level}>"

        rendered = H2_H3_RE.sub(replace_heading, html_text)
        return self._rewrite_internal_links(rendered, doc.meta.source_path)

    def _rewrite_internal_links(self, html_text: str, source_path: str) -> str:
        def replace_href(match: re.Match[str]) -> str:
            prefix, href, suffix = match.groups()
            rewritten = self._resolve_internal_doc_url(source_path, href)
            if rewritten is None:
                return match.group(0)
            return f"{prefix}{html.escape(rewritten, quote=True)}{suffix}"

        return ANCHOR_HREF_RE.sub(replace_href, html_text)

    def _resolve_internal_doc_url(self, source_path: str, href: str) -> Optional[str]:
        target = href.strip()
        if not target or target.startswith(("#", "http://", "https://", "mailto:", "tel:", "/")):
            return None
        if target.lower().startswith("javascript:"):
            return None

        if "#" in target:
            path_part, fragment = target.split("#", 1)
            fragment_suffix = f"#{fragment}"
        else:
            path_part = target
            fragment_suffix = ""
        if "?" in path_part:
            path_only, query = path_part.split("?", 1)
            query_suffix = f"?{query}"
        else:
            path_only = path_part
            query_suffix = ""

        current_dir = PurePosixPath(source_path).parent.as_posix()
        combined = posixpath.normpath(posixpath.join(current_dir, path_only))
        if combined.startswith("../"):
            return None

        for doc in self.docs_by_slug.values():
            if doc.meta.source_path == combined:
                return f"{doc.meta.public_url}{query_suffix}{fragment_suffix}"
        return None

    def _related_documents(self, doc: IndexedDocument) -> list[IndexedDocument]:
        related = [
            item for item in self._public_documents()
            if item.meta.slug != doc.meta.slug and item.meta.category == doc.meta.category
        ]
        return sorted(related, key=lambda item: item.meta.updated_at, reverse=True)[:5]
