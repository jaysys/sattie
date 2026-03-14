from __future__ import annotations

import html


def build_seo_head(title: str, description: str, canonical_url: str) -> str:
    safe_title = html.escape(title)
    safe_desc = html.escape(description)
    safe_canonical = html.escape(canonical_url, quote=True)
    return (
        f"<title>{safe_title}</title>"
        f"<meta name='description' content='{safe_desc}'/>"
        f"<link rel='canonical' href='{safe_canonical}'/>"
        f"<meta property='og:title' content='{safe_title}'/>"
        f"<meta property='og:description' content='{safe_desc}'/>"
        f"<meta property='og:url' content='{safe_canonical}'/>"
        f"<meta name='twitter:card' content='summary'/>"
    )
