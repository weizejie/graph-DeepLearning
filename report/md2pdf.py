import fitz
import markdown
import re

md_file = r'c:/Users/weize jie/Desktop/test/graph DeepLearning/report/T3Former_代码解析报告.md'
pdf_file = r'c:/Users/weize jie/Desktop/test/graph DeepLearning/report/T3Former_代码解析报告.pdf'

with open(md_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Remove all markdown links to avoid pymupdf anchor errors
content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)

# Remove code reference blocks (startLine:endLine:filepath format)
content = re.sub(r'```\d+:\d+:[^\n]+\n', '', content)

md = markdown.Markdown(extensions=['tables', 'fenced_code', 'codehilite'])
html_body = md.convert(content)

css = """
body {
    font-family: 'SimSun', 'Microsoft YaHei', 'Arial', sans-serif;
    font-size: 10.5pt;
    line-height: 1.65;
    margin: 2.2cm;
    color: #1a1a1a;
}
h1 { font-size: 18pt; font-weight: bold; margin-top: 20pt; margin-bottom: 6pt;
     border-bottom: 2px solid #2c3e50; padding-bottom: 4pt; color: #2c3e50; }
h2 { font-size: 13.5pt; font-weight: bold; margin-top: 14pt; color: #2c3e50; }
h3 { font-size: 11.5pt; font-weight: bold; margin-top: 10pt; color: #34495e; }
h4 { font-size: 10.5pt; font-weight: bold; margin-top: 8pt; }
pre {
    background-color: #f6f6f6;
    border: 1px solid #ddd;
    border-radius: 4px;
    padding: 7pt;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 8pt;
    white-space: pre-wrap;
    word-wrap: break-word;
    margin: 8pt 0;
}
code {
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 8.5pt;
    background-color: #f0f0f0;
    padding: 1pt 3pt;
    border-radius: 2px;
}
pre code { background: none; padding: 0; }
table {
    border-collapse: collapse;
    width: 100%;
    margin: 10pt 0;
}
th {
    background-color: #2c3e50;
    color: white;
    padding: 5pt 8pt;
    text-align: left;
    font-weight: bold;
    font-size: 9.5pt;
}
td {
    border: 1px solid #ccc;
    padding: 4pt 7pt;
    font-size: 9.5pt;
}
tr:nth-child(even) { background-color: #f5f5f5; }
blockquote {
    border-left: 4px solid #2980b9;
    margin: 8pt 0;
    padding: 4pt 12pt;
    background-color: #eef6fb;
    color: #2c3e50;
    font-size: 10pt;
}
strong { font-weight: bold; }
em { font-style: italic; }
hr { border: none; border-top: 1px solid #ddd; margin: 12pt 0; }
"""

def rectfn(rect_num, filled):
    mediabox = fitz.Rect(0, 0, 595, 842)  # A4 page in points
    rect = fitz.Rect(45, 45, 550, 820)     # content area
    ctm = fitz.Identity
    return mediabox, rect, ctm

story = fitz.Story(html_body, user_css=css)

with fitz.DocumentWriter(pdf_file) as writer:
    more = story.write(writer, rectfn)
    while more:
        more = story.write(writer, rectfn)

import os
print(f"PDF saved: {pdf_file}")
print(f"File size: {os.path.getsize(pdf_file):,} bytes")
