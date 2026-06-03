# -*- coding: utf-8 -*-
"""
Markdown → PDF with LaTeX rendered via pdflatex (MiKTeX).
"""

import os, re, io, hashlib, subprocess, shutil
import markdown
import fitz  # PyMuPDF

# ── Paths ────────────────────────────────────────────────────────────────────
MD_PATH   = r"c:\Users\weize jie\Desktop\test\graph DeepLearning\report\T3former_综合阅读报告.md"
PDF_PATH  = r"c:\Users\weize jie\Desktop\test\graph DeepLearning\report\T3former_综合阅读报告.pdf"
IMG_DIR   = os.path.join(os.path.dirname(MD_PATH), "_latex_imgs")
PDFLATEX  = r"C:\Users\weize jie\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe"
os.makedirs(IMG_DIR, exist_ok=True)

# MiKTeX env: skip update checker prompt that would block pdflatex
ENV = dict(os.environ)
ENV['MIKTEX_DISABLE_UPDATE_CHECKER'] = '1'

# ── LaTeX → PNG via pdflatex ───────────────────────────────────────────────
LATEX_TEMPLATE = (
    r"\documentclass[border=2pt,png,tightpage]{standalone}"
    r"\usepackage{amsmath}\usepackage{amssymb}"
    r"\begin{document}$#{FORMULA}#$\end{document}"
)


def latex_to_png(formula: str, dpi: int = 150) -> str:
    """
    Render LaTeX formula to PNG using MiKTeX pdflatex.
    Returns absolute path to the PNG file.
    """
    safe = formula.replace('\n', ' ').strip()
    key  = hashlib.md5((f"{safe}|{dpi}").encode()).hexdigest()
    png_path = os.path.join(IMG_DIR, f"{key}.png")

    if os.path.exists(png_path):
        return os.path.abspath(png_path)

    # Build standalone .tex
    tex_src = LATEX_TEMPLATE.replace('#{FORMULA}#', safe)
    tex_src = tex_src.encode('utf-8') if False else tex_src

    jobname = f"m_{key}"
    work_dir = os.path.join(IMG_DIR, jobname)
    os.makedirs(work_dir, exist_ok=True)

    tex_path = os.path.join(work_dir, f"{jobname}.tex")
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(tex_src)

    # Run pdflatex twice (needed for standalone package)
    try:
        for _ in range(2):
            r = subprocess.run(
                [PDFLATEX,
                 "-interaction=nonstopmode",
                 "-halt-on-error",
                 "-output-directory", work_dir,
                 tex_path],
                capture_output=True, text=True,
                timeout=90, env=ENV
            )
    except Exception as e:
        raise RuntimeError(f"pdflatex subprocess error: {e}")

    pdf_path = os.path.join(work_dir, f"{jobname}.pdf")
    if not os.path.exists(pdf_path):
        raise RuntimeError(f"pdflatex failed to produce PDF.\n"
                          f"stdout: {r.stdout[-300:]}\nstderr: {r.stderr[-300:]}")

    # PDF page → PNG
    doc = fitz.open(pdf_path)
    page = doc[0]
    scale = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    doc.close()
    pix.save(png_path)

    shutil.rmtree(work_dir, ignore_errors=True)
    return os.path.abspath(png_path)


def math_to_img_tag(formula: str, is_block: bool = False, dpi: int = 160) -> str:
    png = latex_to_png(formula, dpi=dpi)
    align_val = "middle"
    if is_block:
        tag = (f'<img src="file:///{png}" alt="math" '
               f'style="display:block;margin-top:0.6em;margin-bottom:0.6em;margin-left:auto;margin-right:auto;" />')
    else:
        tag = (f'<img src="file:///{png}" alt="math" '
               f'style="display:inline-block;vertical-align:{align_val};margin-top:0;margin-bottom:0;margin-left:2px;margin-right:2px;" />')
    return tag


# ── Markdown → HTML ─────────────────────────────────────────────────────────
def convert_md_to_html(md_path: str) -> str:
    with open(md_path, 'r', encoding='utf-8') as f:
        md_text = f.read()

    title = "T3former 综合技术报告"
    for line in md_text.split('\n'):
        if line.startswith('# '):
            title = line[2:].strip()
            break

    # Step 1: protect fenced code blocks
    fenced = []
    def _pf(m):
        fenced.append(m.group(0))
        return f"\x00F{len(fenced)-1}\x00"
    md_text = re.sub(r'```[\s\S]*?```', _pf, md_text)

    # Step 2: protect inline code
    inline = []
    def _pi(m):
        inline.append(m.group(0))
        return f"\x00I{len(inline)-1}\x00"
    md_text = re.sub(r'`[^`\n]+`', _pi, md_text)

    # Step 3: block LaTeX $$...$$ → <img>
    def _bl(m):
        return math_to_img_tag(m.group(1).strip(), is_block=True, dpi=180)
    md_text = re.sub(r'\$\$([\s\S]+?)\$\$', _bl, md_text)

    # Step 4: inline LaTeX $...$ → <img>  (skip currency patterns)
    def _il(m):
        raw = m.group(1)
        if re.match(r'^\s*\$[\d,.kKmM]+\s*$', raw):
            return m.group(0)
        return math_to_img_tag(raw, is_block=False, dpi=140)

    md_text = re.sub(r'(?<!\$)\$(?!\$)([^\$]+)\$', _il, md_text)

    # Step 5: restore
    for i, code in enumerate(inline):
        md_text = md_text.replace(f"\x00I{i}\x00", code)
    for i, block in enumerate(fenced):
        md_text = md_text.replace(f"\x00F{i}\x00", block)

    # Step 6: markdown → html
    md = markdown.Markdown(
        extensions=['tables', 'fenced_code', 'codehilite',
                    'nl2br', 'sane_lists'],
        extension_configs={
            'codehilite': {'css_class': 'highlight', 'guess_lang': False},
        }
    )
    body_html = md.convert(md_text)

    # Step 7: wrap with CSS
    css = """
    body {
        font-family: "Times New Roman", "SimSun", serif;
        font-size: 10.5pt; line-height: 1.7;
        color: #1a1a1a; margin: 0; padding: 0;
    }
    h1, h2, h3, h4 {
        font-family: "SimHei", "Microsoft YaHei", sans-serif;
        color: #1a1a2e; page-break-after: avoid;
    }
    h1 { font-size: 17pt; border-bottom: 2px solid #1a1a2e;
         padding-bottom: 6px; margin: 1.5em 0 0.5em; }
    h2 { font-size: 13.5pt; border-bottom: 1px solid #aaa;
         padding-bottom: 4px; margin: 1.3em 0 0.4em; }
    h3 { font-size: 11.5pt; margin: 1.1em 0 0.3em; }
    h4 { font-size: 10.5pt; font-style: italic; margin: 1em 0 0.3em; }
    p  { margin: 0.4em 0; text-align: justify; }
    table { border-collapse: collapse; width: 100%; margin: 0.8em 0;
            font-size: 9.5pt; page-break-inside: avoid; }
    th { background: #2c3e50; color: white; padding: 6px 10px;
         text-align: left; font-weight: bold; }
    td { border: 1px solid #bdc3c7; padding: 5px 9px; vertical-align: top; }
    tr:nth-child(even) td { background: #f4f6f9; }
    code { font-family: Consolas, monospace; font-size: 8.5pt;
           background: #f4f4f4; padding: 1px 4px; border-radius: 3px; color: #c7254e; }
    pre  { background: #f8f8f8; border: 1px solid #ddd; border-radius: 4px;
           padding: 8px 12px; font-size: 8.5pt; overflow-x: auto;
           page-break-inside: avoid; margin: 0.6em 0; }
    pre code { background: none; padding: 0; color: #333; }
    blockquote { border-left: 4px solid #3498db; margin: 0.6em 0;
                 padding: 0.4em 1em; background: #f0f8ff; }
    hr { border: none; border-top: 1px solid #ccc; margin: 1em 0; }
    ul, ol { margin: 0.4em 0; padding-left: 1.6em; }
    li { margin: 0.25em 0; }
    strong { font-weight: bold; }
    em { font-style: italic; }

    """

    return (f'<!DOCTYPE html>\n<html lang="zh-CN">\n'
            f'<head><meta charset="UTF-8">\n'
            f'<title>{title}</title>\n'
            f'<style>{css}</style>\n'
            f'</head>\n<body>{body_html}</body>\n</html>')


# ── HTML → PDF ──────────────────────────────────────────────────────────────
def html_to_pdf(html_content: str, output_path: str):
    html_path = output_path.replace('.pdf', '_tmp.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    out_buf = io.BytesIO()
    writer  = fitz.DocumentWriter(out_buf)
    rect    = fitz.paper_rect('A4')
    M       = 36
    inner   = fitz.Rect(M, M, rect.width - M, rect.height - M)

    story = fitz.Story(
        html=html_content,
        archive=os.path.dirname(os.path.abspath(html_path)),
        user_css="body { font-size: 10.5pt; }"
    )

    more = 1
    while more:
        dev = writer.begin_page(rect)
        more, _ = story.place(inner)
        story.draw(dev)
        writer.end_page()

    writer.close()
    out_buf.seek(0)

    doc = fitz.open('pdf', out_buf.read())
    doc.save(output_path, garbage=4, deflate=True)
    doc.close()
    os.remove(html_path)


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("[1/2] Markdown → HTML (LaTeX → PNG via pdflatex) ...")
    html = convert_md_to_html(MD_PATH)
    n = html.count('<img src="file:///')
    print(f"      {n} LaTeX formulas to render")

    print("[2/2] HTML → PDF (PyMuPDF) ...")
    html_to_pdf(html, PDF_PATH)

    kb = os.path.getsize(PDF_PATH) // 1024
    print(f"\n[DONE] {PDF_PATH}")
    print(f"       {kb} KB  |  {n} formulas")
