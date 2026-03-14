from __future__ import annotations

import html
import re


def render_markdown_html(md_text: str) -> str:
    # Lightweight renderer without extra dependencies.
    raw_lines = md_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    ref_defs: dict[str, tuple[str, str]] = {}
    ref_pat = re.compile(r'^\s*\[([^\]]+)\]:\s+(\S+)(?:\s+"([^"]*)")?\s*$')
    lines: list[str] = []
    for ln in raw_lines:
        m = ref_pat.match(ln)
        if m:
            key = m.group(1).strip().lower()
            url = m.group(2).strip()
            title = (m.group(3) or "").strip()
            ref_defs[key] = (url, title)
            continue
        lines.append(ln)
    out: list[str] = []
    in_code = False
    code_lang = ""
    code_lines: list[str] = []
    in_ul = False
    in_ol = False

    def looks_like_formula_block(text: str) -> bool:
        sample = text.strip()
        if not sample:
            return False
        markers = ("≈", "=", "tan(", "sin(", "cos(", "theta_", "d_", "h *", "km", "deg")
        return any(marker in sample for marker in markers)

    def convert_formula_text_to_tex(text: str) -> str:
        tex = text.strip()
        tex = re.sub(r"\btheta_([A-Za-z0-9]+)\b", r"\\theta_{\1}", tex)
        tex = re.sub(r"\b([A-Za-z])_([A-Za-z0-9]+)\b", r"\1_{\2}", tex)
        tex = re.sub(r"(?<=\d)°", r"^\\circ", tex)
        tex = re.sub(r"\btan\(", r"\\tan(", tex)
        tex = re.sub(r"\bsin\(", r"\\sin(", tex)
        tex = re.sub(r"\bcos\(", r"\\cos(", tex)
        tex = re.sub(r"\s\*\s", r" \\cdot ", tex)
        return tex

    def close_lists() -> None:
        nonlocal in_ul, in_ol
        if in_ul:
            out.append("</ul>")
            in_ul = False
        if in_ol:
            out.append("</ol>")
            in_ol = False

    def _render_plain_with_links(text: str) -> str:
        # Inline link: [text](url "title")
        image_pat = re.compile(r'!\[([^\]]*)\]\((\S+?)(?:\s+"([^"]*)")?\)')
        image_ref_pat = re.compile(r'!\[([^\]]*)\]\[([^\]]+)\]')
        inline_pat = re.compile(r'\[([^\]]+)\]\((\S+?)(?:\s+"([^"]*)")?\)')
        ref_pat2 = re.compile(r'\[([^\]]+)\]\[([^\]]+)\]')
        auto_url_pat = re.compile(r'(?<!["\'>])(https?://[^\s<]+)')

        def replace_image_ref(match: re.Match[str]) -> str:
            alt = html.escape(match.group(1))
            label = match.group(2).strip().lower()
            found = ref_defs.get(label)
            if not found:
                return alt
            url, title = found
            t_attr = f' title="{html.escape(title, quote=True)}"' if title else ""
            return (
                f"<a href=\"{html.escape(url, quote=True)}\" class='md-image-link' data-full=\"{html.escape(url, quote=True)}\" target='_blank' rel='noopener noreferrer'>"
                f'<img src="{html.escape(url, quote=True)}" alt="{alt}"{t_attr} '
                "loading='lazy' class='md-image'/>"
                "</a>"
            )

        def replace_image(match: re.Match[str]) -> str:
            alt = html.escape(match.group(1))
            url = html.escape(match.group(2), quote=True)
            title = (match.group(3) or "").strip()
            t_attr = f' title="{html.escape(title, quote=True)}"' if title else ""
            return (
                f"<a href=\"{url}\" class='md-image-link' data-full=\"{url}\" target='_blank' rel='noopener noreferrer'>"
                f'<img src="{url}" alt="{alt}"{t_attr} '
                "loading='lazy' class='md-image'/>"
                "</a>"
            )

        def replace_ref(match: re.Match[str]) -> str:
            label = match.group(2).strip().lower()
            found = ref_defs.get(label)
            text_ = html.escape(match.group(1))
            if not found:
                return text_
            url, title = found
            t_attr = f' title="{html.escape(title, quote=True)}"' if title else ""
            return f'<a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener noreferrer"{t_attr}>{text_}</a>'

        def replace_inline(match: re.Match[str]) -> str:
            text_ = html.escape(match.group(1))
            url = html.escape(match.group(2), quote=True)
            title = (match.group(3) or "").strip()
            t_attr = f' title="{html.escape(title, quote=True)}"' if title else ""
            return f'<a href="{url}" target="_blank" rel="noopener noreferrer"{t_attr}>{text_}</a>'

        # Escape first, then re-insert anchors through unique placeholders.
        anchors: list[str] = []

        def stash_anchor(a: str) -> str:
            anchors.append(a)
            return f"__ANCHOR_{len(anchors)-1}__"

        tmp = text
        tmp = image_ref_pat.sub(lambda m: stash_anchor(replace_image_ref(m)), tmp)
        tmp = image_pat.sub(lambda m: stash_anchor(replace_image(m)), tmp)
        tmp = ref_pat2.sub(lambda m: stash_anchor(replace_ref(m)), tmp)
        tmp = inline_pat.sub(lambda m: stash_anchor(replace_inline(m)), tmp)
        tmp = auto_url_pat.sub(
            lambda m: stash_anchor(
                f'<a href="{html.escape(m.group(1), quote=True)}" target="_blank" rel="noopener noreferrer">{html.escape(m.group(1))}</a>'
            ),
            tmp,
        )
        esc = html.escape(tmp)
        esc = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", esc)
        for i, a in enumerate(anchors):
            esc = esc.replace(html.escape(f"__ANCHOR_{i}__"), a)
        return esc

    def render_inline(text: str) -> str:
        # Preserve inline code spans and leave math delimiters ($, $$, \(...\), \[...\]) untouched.
        parts = text.split("`")
        if len(parts) == 1:
            return _render_plain_with_links(text)
        out_parts: list[str] = []
        for idx, part in enumerate(parts):
            if idx % 2 == 1:
                out_parts.append("<code>" + html.escape(part) + "</code>")
            else:
                out_parts.append(_render_plain_with_links(part))
        return "".join(out_parts)

    def render_table_cell(text: str) -> str:
        rendered = render_inline(text)
        rendered = re.sub(r"&lt;br\s*/?&gt;", "<br>", rendered, flags=re.IGNORECASE)
        return rendered

    def split_table_row(s: str) -> list[str]:
        t = s.strip()
        if t.startswith("|"):
            t = t[1:]
        if t.endswith("|"):
            t = t[:-1]
        return [c.strip() for c in t.split("|")]

    def is_table_sep_row(s: str) -> bool:
        cells = split_table_row(s)
        if not cells:
            return False
        for c in cells:
            x = c.replace(":", "").replace("-", "").strip()
            if x != "" or "-" not in c:
                return False
        return True

    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\n")
        stripped = line.strip()
        # Block math: $$ ... $$ (multiline)
        if stripped == "$$":
            close_lists()
            i += 1
            math_lines: list[str] = []
            while i < len(lines) and lines[i].strip() != "$$":
                math_lines.append(lines[i])
                i += 1
            if i < len(lines) and lines[i].strip() == "$$":
                i += 1
            expr = "\n".join(math_lines)
            out.append("<div class='math-block'>$$\n" + html.escape(expr) + "\n$$</div>")
            continue
        # Block math fallback: [ ... ] (multiline)
        # Some docs use bracket-only lines to wrap formulas.
        if stripped == "[":
            close_lists()
            i += 1
            math_lines = []
            while i < len(lines) and lines[i].strip() != "]":
                math_lines.append(lines[i])
                i += 1
            if i < len(lines) and lines[i].strip() == "]":
                i += 1
            expr = "\n".join(math_lines).strip()
            out.append("<div class='math-block'>$$\n" + html.escape(expr) + "\n$$</div>")
            continue
        if stripped.startswith("```"):
            if in_code:
                code_text = "\n".join(code_lines)
                if code_lang.lower() == "mermaid":
                    out.append("<pre class='mermaid'>" + html.escape(code_text) + "</pre>")
                elif code_lang.lower() in {"text", "math", "formula"} and looks_like_formula_block(code_text):
                    out.append("<div class='math-block'>$$\n" + html.escape(convert_formula_text_to_tex(code_text)) + "\n$$</div>")
                else:
                    out.append("<pre><code>" + html.escape(code_text) + "</code></pre>")
                code_lines = []
                code_lang = ""
                in_code = False
            else:
                close_lists()
                in_code = True
                code_lang = stripped[3:].strip()
            i += 1
            continue
        if in_code:
            code_lines.append(line)
            i += 1
            continue
        if not stripped:
            close_lists()
            i += 1
            continue
        if stripped == "---":
            close_lists()
            out.append("<hr/>")
            i += 1
            continue
        if stripped.lower() in {"<br>", "<br/>", "<br />", "<br><br>", "<br/><br/>", "<br /><br />"}:
            close_lists()
            out.append(stripped)
            i += 1
            continue
        # Markdown table: header row + separator row + body rows.
        if "|" in stripped and (i + 1) < len(lines):
            next_line = lines[i + 1].strip()
            if "|" in next_line and is_table_sep_row(next_line):
                close_lists()
                headers = split_table_row(stripped)
                out.append("<table style='border-collapse:collapse;width:auto;max-width:100%;margin:12px 0;'>")
                out.append("<thead><tr>")
                for h in headers:
                    out.append(
                        "<th style='border:1px solid #ddd;padding:8px;background:#f7f7f7;text-align:left;'>"
                        + render_table_cell(h)
                        + "</th>"
                    )
                out.append("</tr></thead>")
                out.append("<tbody>")
                i += 2
                while i < len(lines):
                    row_line = lines[i].strip()
                    if not row_line or "|" not in row_line:
                        break
                    cells = split_table_row(row_line)
                    out.append("<tr>")
                    for c in cells:
                        out.append("<td style='border:1px solid #ddd;padding:8px;vertical-align:top;'>" + render_table_cell(c) + "</td>")
                    out.append("</tr>")
                    i += 1
                out.append("</tbody></table>")
                continue
        if stripped.startswith("#"):
            close_lists()
            lvl = min(6, len(stripped) - len(stripped.lstrip("#")))
            content = stripped[lvl:].strip()
            out.append(f"<h{lvl}>{render_inline(content)}</h{lvl}>")
            i += 1
            continue
        if stripped.startswith("- ") or stripped.startswith("* "):
            if in_ol:
                out.append("</ol>")
                in_ol = False
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            out.append("<li>" + render_inline(stripped[2:].strip()) + "</li>")
            i += 1
            continue
        if len(stripped) >= 3 and stripped[0].isdigit() and ". " in stripped:
            prefix, content = stripped.split(". ", 1)
            if prefix.isdigit():
                if in_ul:
                    out.append("</ul>")
                    in_ul = False
                if not in_ol:
                    out.append("<ol>")
                    in_ol = True
                out.append("<li>" + render_inline(content.strip()) + "</li>")
                i += 1
                continue
        close_lists()
        out.append("<p>" + render_inline(stripped) + "</p>")
        i += 1

    if in_code:
        code_text = "\n".join(code_lines)
        if code_lang.lower() == "mermaid":
            out.append("<pre class='mermaid'>" + html.escape(code_text) + "</pre>")
        elif code_lang.lower() in {"text", "math", "formula"} and looks_like_formula_block(code_text):
            out.append("<div class='math-block'>$$\n" + html.escape(convert_formula_text_to_tex(code_text)) + "\n$$</div>")
        else:
            out.append("<pre><code>" + html.escape(code_text) + "</code></pre>")
    if in_ul:
        out.append("</ul>")
    if in_ol:
        out.append("</ol>")
    return "\n".join(out)


def render_guide_page(rel: str, md_text: str) -> str:
    body = render_markdown_html(md_text)
    return (
        "<html><head>"
        "<meta charset='utf-8'/>"
        "<style>"
        "body{font-family:sans-serif;padding:20px;line-height:1.6;overflow-x:hidden;}"
        "body,main{max-width:100%;}"
        "table{display:block;max-width:100%;overflow:auto;white-space:nowrap;}"
        "pre{background:#f7f7f8;border:1px solid #e5e7eb;border-radius:8px;padding:12px;overflow:auto;}"
        "code{background:#f3f4f6;border-radius:4px;padding:2px 4px;}"
        ".math-block{margin:14px 0;overflow:auto;background:transparent;border:none;border-radius:0;padding:0;text-align:left;}"
        ".math-block mjx-container,.math-block svg{background:transparent !important;}"
        ".math-block mjx-container[display='true']{display:block !important;text-align:left !important;margin:0 !important;}"
        "pre.mermaid,.mermaid{background:#fff !important;color:inherit;border:1px solid #e5e7eb;border-radius:8px;padding:12px;}"
        ".mermaid svg{background:transparent !important;}"
        ".mermaid svg,.chart-zoomable{cursor:zoom-in;}"
        "p,li,h1,h2,h3,h4,h5,h6,td,th{overflow-wrap:anywhere;word-break:break-word;}"
        ".md-image-link{display:inline-block;margin:10px 0;}"
        ".md-image{display:block;max-width:min(520px,100%);width:100%;height:auto;border:1px solid #ddd;border-radius:8px;background:#fff;cursor:zoom-in;}"
        "#imgModal{position:fixed;inset:0;background:rgba(0,0,0,.85);display:none;align-items:center;justify-content:center;z-index:9999;padding:24px;}"
        "#imgModal.show{display:flex;}"
        "#imgModal img{max-width:95vw;max-height:90vh;width:auto;height:auto;border-radius:8px;box-shadow:0 10px 30px rgba(0,0,0,.45);background:#111;}"
        "#imgModal .close{position:absolute;top:12px;right:16px;color:#fff;font-size:30px;line-height:1;cursor:pointer;}"
        "#chartModal{position:fixed;inset:0;background:rgba(255,255,255,.98);display:none;align-items:center;justify-content:center;z-index:10000;padding:28px;box-sizing:border-box;}"
        "#chartModal.show{display:flex;}"
        "#chartModal .chart-shell{width:min(96vw,1600px);height:min(92vh,980px);display:flex;flex-direction:column;background:#fff;border:1px solid #d8dee7;border-radius:28px;box-shadow:0 30px 80px rgba(15,23,42,.12);overflow:hidden;}"
        "#chartModal .chart-toolbar{display:flex;gap:14px;align-items:center;justify-content:flex-end;flex-wrap:wrap;padding:6px 12px;color:#111827;background:#fff;text-align:right;min-height:36px;width:50%;align-self:flex-end;box-sizing:border-box;order:2;}"
        "#chartModal .chart-toolbar button{border:none;background:transparent;color:#0f5cdd;border-radius:0;padding:0;cursor:pointer;text-decoration:none;font:inherit;line-height:1.2;}"
        "#chartModal .chart-toolbar button:hover{text-decoration:underline;}"
        "#chartModal .chart-stage{flex:1;position:relative;overflow:hidden;cursor:grab;padding:24px;display:flex;align-items:center;justify-content:center;background:#fff;order:1;}"
        "#chartModal .chart-stage.dragging{cursor:grabbing;}"
        "#chartModal .chart-canvas{position:absolute;inset:24px;display:flex;align-items:center;justify-content:center;}"
        "#chartModal .chart-canvas svg{display:block;width:100%;height:100%;max-width:none !important;max-height:none !important;}"
        "@media(max-width:900px){body{padding:14px;}table{font-size:14px;}pre{padding:10px;}}"
        "</style>"
        "<script>"
        "window.MathJax={"
        "tex:{inlineMath:[['$','$'],['\\\\(','\\\\)']],displayMath:[['$$','$$'],['\\\\[','\\\\]']],processEscapes:true},"
        "chtml:{displayAlign:'left',displayIndent:'0'},"
        "svg:{displayAlign:'left',displayIndent:'0',fontCache:'global'},"
        "startup:{typeset:false}"
        "};"
        "</script>"
        "<script>"
        "window.addEventListener('load',function(){"
        "  var sources=["
        "    'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js',"
        "    'https://unpkg.com/mathjax@3/es5/tex-svg.js',"
        "    'https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-svg.min.js'"
        "  ];"
        "  var loadMathJax=function(idx){"
        "    if(idx>=sources.length){return;}"
        "    var s=document.createElement('script');"
        "    s.async=true; s.src=sources[idx];"
        "    s.onerror=function(){loadMathJax(idx+1);};"
        "    document.head.appendChild(s);"
        "  };"
        "  loadMathJax(0);"
        "  var tryTypeset=function(){"
        "    if(window.MathJax && window.MathJax.typesetPromise){"
        "      window.MathJax.typesetPromise().catch(function(){});"
        "      return true;"
        "    }"
        "    return false;"
        "  };"
        "  if(!tryTypeset()){"
        "    var n=0; var t=setInterval(function(){n+=1; if(tryTypeset()||n>50){clearInterval(t);}},200);"
        "  }"
        "  var mermaidSources=["
        "    'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js',"
        "    'https://unpkg.com/mermaid@10/dist/mermaid.min.js',"
        "    'https://cdnjs.cloudflare.com/ajax/libs/mermaid/10.9.1/mermaid.min.js'"
        "  ];"
        "  var renderMermaid=function(){"
        "    if(!window.mermaid){return false;}"
        "    try{window.mermaid.initialize({startOnLoad:false,securityLevel:'loose'});window.mermaid.run({querySelector:'.mermaid'});return true;}catch(e){return false;}"
        "  };"
        "  var loadMermaid=function(idx){"
        "    if(idx>=mermaidSources.length){return;}"
        "    var s=document.createElement('script');"
        "    s.async=true; s.src=mermaidSources[idx];"
        "    s.onload=function(){ if(!renderMermaid()){ setTimeout(renderMermaid, 200); } };"
        "    s.onerror=function(){ loadMermaid(idx+1); };"
        "    document.head.appendChild(s);"
        "  };"
        "  if(!renderMermaid()){"
        "    loadMermaid(0);"
        "    var k=0; var mt=setInterval(function(){k+=1; if(renderMermaid()||k>50){clearInterval(mt);}},200);"
        "  }"
        "});"
        "</script>"
        "</head><body>"
        "<p><a href='/guide'>&larr; 목록으로</a></p>"
        f"<h2>{html.escape(rel)}</h2>"
        "<hr/>"
        f"{body}"
        "<div id='imgModal' aria-hidden='true'><span class='close' title='닫기'>&times;</span><img alt='preview'/></div>"
        "<div id='chartModal' aria-hidden='true'>"
        "<div class='chart-shell'>"
        "<div class='chart-toolbar'><button type='button' data-action='zoom-in'>확대</button><button type='button' data-action='zoom-out'>축소</button><button type='button' data-action='reset'>원위치</button><button type='button' data-action='close'>닫기</button></div>"
        "<div class='chart-stage'><div class='chart-canvas'></div></div>"
        "</div>"
        "</div>"
        "<script>"
        "(function(){"
        "  const modal=document.getElementById('imgModal'); if(!modal) return;"
        "  const modalImg=modal.querySelector('img'); const close=modal.querySelector('.close');"
        "  document.querySelectorAll('.md-image-link').forEach(function(a){"
        "    a.addEventListener('click', function(e){"
        "      e.preventDefault(); const src=a.getAttribute('data-full')||a.getAttribute('href'); if(!src) return;"
        "      modalImg.setAttribute('src', src); modal.classList.add('show'); modal.setAttribute('aria-hidden','false');"
        "    });"
        "  });"
        "  const hide=function(){modal.classList.remove('show'); modal.setAttribute('aria-hidden','true'); modalImg.setAttribute('src','');};"
        "  if(close) close.addEventListener('click', hide);"
        "  modal.addEventListener('click', function(e){ if(e.target===modal) hide(); });"
        "  window.addEventListener('keydown', function(e){ if(e.key==='Escape') hide(); });"
        "  const chartModal=document.getElementById('chartModal');"
        "  const chartStage=chartModal?chartModal.querySelector('.chart-stage'):null;"
        "  const chartCanvas=chartModal?chartModal.querySelector('.chart-canvas'):null;"
        "  let chartState={svg:null,base:null,current:null,dragging:false,startX:0,startY:0,startViewBox:null};"
        "  const cloneBox=function(box){ return box?{x:box.x,y:box.y,w:box.w,h:box.h}:null; };"
        "  const readViewBox=function(svg){ var vb=svg.getAttribute('viewBox'); if(vb){ var p=vb.trim().split(/\\s+/).map(Number); if(p.length===4&&p.every(function(n){return Number.isFinite(n);})){ return {x:p[0],y:p[1],w:p[2],h:p[3]}; } } var width=svg.viewBox&&svg.viewBox.baseVal&&svg.viewBox.baseVal.width?svg.viewBox.baseVal.width:0; var height=svg.viewBox&&svg.viewBox.baseVal&&svg.viewBox.baseVal.height?svg.viewBox.baseVal.height:0; if(!(width>0&&height>0)){ var bbox=svg.getBBox(); width=bbox.width||1000; height=bbox.height||800; return {x:bbox.x||0,y:bbox.y||0,w:width,h:height}; } return {x:0,y:0,w:width,h:height}; };"
        "  const applyViewBox=function(){ if(chartState.svg&&chartState.current){ chartState.svg.setAttribute('viewBox',[chartState.current.x,chartState.current.y,chartState.current.w,chartState.current.h].join(' ')); } };"
        "  const resetChart=function(){ chartState.current=cloneBox(chartState.base); applyViewBox(); };"
        "  const zoomChart=function(factor,originX,originY){ if(!chartState.current||!chartStage) return; var rect=chartStage.getBoundingClientRect(); var ox=typeof originX==='number'?originX:rect.width/2; var oy=typeof originY==='number'?originY:rect.height/2; var rx=Math.max(0,Math.min(1,(ox-24)/Math.max(1,rect.width-48))); var ry=Math.max(0,Math.min(1,(oy-24)/Math.max(1,rect.height-48))); var nextW=chartState.current.w/factor; var nextH=chartState.current.h/factor; chartState.current={x:chartState.current.x+(chartState.current.w-nextW)*rx,y:chartState.current.y+(chartState.current.h-nextH)*ry,w:nextW,h:nextH}; applyViewBox(); };"
        "  const openChart=function(svg){ if(!chartModal||!chartCanvas||!svg) return; chartCanvas.innerHTML=''; chartState.svg=svg.cloneNode(true); chartState.svg.removeAttribute('style'); chartState.svg.setAttribute('preserveAspectRatio','xMidYMin meet'); chartCanvas.appendChild(chartState.svg); chartState.base=readViewBox(chartState.svg); chartModal.classList.add('show'); chartModal.setAttribute('aria-hidden','false'); resetChart(); };"
        "  const closeChart=function(){ if(!chartModal||!chartCanvas) return; chartModal.classList.remove('show'); chartModal.setAttribute('aria-hidden','true'); chartCanvas.innerHTML=''; chartState.svg=null; chartState.base=null; chartState.current=null; chartState.dragging=false; if(chartStage) chartStage.classList.remove('dragging'); };"
        "  document.querySelectorAll('.mermaid svg').forEach(function(svg){ svg.classList.add('chart-zoomable'); svg.addEventListener('click', function(){ openChart(svg); }); });"
        "  if(chartModal){"
        "    chartModal.querySelectorAll('button[data-action]').forEach(function(btn){ btn.addEventListener('click', function(){ var action=btn.getAttribute('data-action'); if(action==='close'){ closeChart(); } else if(action==='zoom-in'){ zoomChart(1.2); } else if(action==='zoom-out'){ zoomChart(1/1.2); } else if(action==='reset'){ resetChart(); } }); });"
        "    chartStage.addEventListener('mousedown', function(e){ if(!chartState.current) return; chartState.dragging=true; chartState.startX=e.clientX; chartState.startY=e.clientY; chartState.startViewBox=cloneBox(chartState.current); chartStage.classList.add('dragging'); });"
        "    window.addEventListener('mousemove', function(e){ if(!chartState.dragging||!chartState.current||!chartStage) return; var rect=chartStage.getBoundingClientRect(); var dx=e.clientX-chartState.startX; var dy=e.clientY-chartState.startY; chartState.current.x=chartState.startViewBox.x-(dx/Math.max(1,rect.width-48))*chartState.startViewBox.w; chartState.current.y=chartState.startViewBox.y-(dy/Math.max(1,rect.height-48))*chartState.startViewBox.h; applyViewBox(); });"
        "    window.addEventListener('mouseup', function(){ chartState.dragging=false; chartState.startViewBox=null; if(chartStage) chartStage.classList.remove('dragging'); });"
        "    chartStage.addEventListener('wheel', function(e){ e.preventDefault(); zoomChart(e.deltaY<0?1.12:1/1.12, e.clientX-chartStage.getBoundingClientRect().left, e.clientY-chartStage.getBoundingClientRect().top); }, {passive:false});"
        "    chartModal.addEventListener('click', function(e){ if(e.target===chartModal){ closeChart(); } });"
        "    window.addEventListener('keydown', function(e){ if(e.key==='Escape') closeChart(); });"
        "  }"
        "})();"
        "</script>"
        "<script>"
        "(function(){"
        "  const chartModal=document.getElementById('chartModal');"
        "  if(!chartModal){return;}"
        "  const chartStage=chartModal.querySelector('.chart-stage');"
        "  const chartCanvas=chartModal.querySelector('.chart-canvas');"
        "  const state={svg:null,base:null,current:null,dragging:false,startX:0,startY:0,startViewBox:null};"
        "  const cloneBox=function(box){return box?{x:box.x,y:box.y,w:box.w,h:box.h}:null;};"
        "  const readViewBox=function(svg){var vb=svg.getAttribute('viewBox'); if(vb){var p=vb.trim().split(/\\s+/).map(Number); if(p.length===4&&p.every(Number.isFinite)){return {x:p[0],y:p[1],w:p[2],h:p[3]};}} var bbox=svg.getBBox(); return {x:bbox.x||0,y:bbox.y||0,w:bbox.width||1000,h:bbox.height||800};};"
        "  const applyViewBox=function(){ if(state.svg&&state.current){ state.svg.setAttribute('viewBox',[state.current.x,state.current.y,state.current.w,state.current.h].join(' ')); } };"
        "  const reset=function(){ state.current=cloneBox(state.base); applyViewBox(); };"
        "  const open=function(svg){ chartCanvas.innerHTML=''; state.svg=svg.cloneNode(true); state.svg.removeAttribute('style'); state.svg.setAttribute('preserveAspectRatio','xMidYMin meet'); chartCanvas.appendChild(state.svg); state.base=readViewBox(state.svg); chartModal.classList.add('show'); chartModal.setAttribute('aria-hidden','false'); reset(); };"
        "  const close=function(){ chartModal.classList.remove('show'); chartModal.setAttribute('aria-hidden','true'); chartCanvas.innerHTML=''; state.svg=null; state.base=null; state.current=null; state.dragging=false; chartStage.classList.remove('dragging'); };"
        "  const zoom=function(factor,clientX,clientY){ if(!state.current){return;} var rect=chartStage.getBoundingClientRect(); var rx=(typeof clientX==='number')?(clientX-rect.left-24)/Math.max(1,rect.width-48):0.5; var ry=(typeof clientY==='number')?(clientY-rect.top-24)/Math.max(1,rect.height-48):0.5; rx=Math.max(0,Math.min(1,rx)); ry=Math.max(0,Math.min(1,ry)); var nextW=state.current.w/factor; var nextH=state.current.h/factor; state.current={x:state.current.x+(state.current.w-nextW)*rx,y:state.current.y+(state.current.h-nextH)*ry,w:nextW,h:nextH}; applyViewBox(); };"
        "  document.addEventListener('click', function(e){ var btn=e.target.closest('#chartModal button[data-action]'); if(btn){ e.preventDefault(); e.stopPropagation(); var action=btn.getAttribute('data-action'); if(action==='close'){close();} else if(action==='zoom-in'){zoom(1.2);} else if(action==='zoom-out'){zoom(1/1.2);} else if(action==='reset'){reset();} return; } var svg=e.target.closest('.mermaid svg'); if(svg && !chartModal.contains(svg)){ e.preventDefault(); e.stopPropagation(); open(svg); return; } if(e.target===chartModal){ close(); } }, true);"
        "  chartStage.addEventListener('pointerdown', function(e){ if(!state.current){return;} state.dragging=true; state.startX=e.clientX; state.startY=e.clientY; state.startViewBox=cloneBox(state.current); chartStage.classList.add('dragging'); chartStage.setPointerCapture(e.pointerId); });"
        "  chartStage.addEventListener('pointermove', function(e){ if(!state.dragging||!state.current){return;} var rect=chartStage.getBoundingClientRect(); var dx=e.clientX-state.startX; var dy=e.clientY-state.startY; state.current.x=state.startViewBox.x-(dx/Math.max(1,rect.width-48))*state.startViewBox.w; state.current.y=state.startViewBox.y-(dy/Math.max(1,rect.height-48))*state.startViewBox.h; applyViewBox(); });"
        "  chartStage.addEventListener('pointerup', function(e){ state.dragging=false; state.startViewBox=null; chartStage.classList.remove('dragging'); try{chartStage.releasePointerCapture(e.pointerId);}catch(_e){} });"
        "  chartStage.addEventListener('pointercancel', function(){ state.dragging=false; state.startViewBox=null; chartStage.classList.remove('dragging'); });"
        "  chartStage.addEventListener('wheel', function(e){ if(!state.current){return;} e.preventDefault(); zoom(e.deltaY<0?1.12:1/1.12,e.clientX,e.clientY); }, {passive:false});"
        "  window.addEventListener('keydown', function(e){ if(e.key==='Escape'){ close(); } });"
        "})();"
        "</script>"
        "</body></html>"
    )
