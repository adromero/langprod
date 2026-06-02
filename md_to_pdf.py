#!/usr/bin/env python3
"""Convert a markdown file to PDF using markdown + weasyprint."""
import sys
import markdown
from weasyprint import HTML

def convert(md_path, pdf_path):
    with open(md_path) as f:
        md_text = f.read()

    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "toc"],
    )

    full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
    @page {{
        size: letter;
        margin: 1in;
    }}
    body {{
        font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
        font-size: 11pt;
        line-height: 1.5;
        color: #1a1a1a;
        max-width: 100%;
    }}
    h1 {{
        font-size: 18pt;
        margin-top: 0;
        margin-bottom: 0.5em;
        page-break-after: avoid;
    }}
    h2 {{
        font-size: 14pt;
        margin-top: 1.5em;
        margin-bottom: 0.5em;
        border-bottom: 1px solid #ccc;
        padding-bottom: 0.2em;
        page-break-after: avoid;
    }}
    h3 {{
        font-size: 12pt;
        margin-top: 1.2em;
        margin-bottom: 0.4em;
        page-break-after: avoid;
    }}
    p {{
        margin: 0.5em 0;
        text-align: justify;
    }}
    table {{
        border-collapse: collapse;
        width: 100%;
        margin: 1em 0;
        font-size: 10pt;
    }}
    th, td {{
        border: 1px solid #ccc;
        padding: 6px 10px;
        text-align: left;
    }}
    th {{
        background-color: #f5f5f5;
        font-weight: bold;
    }}
    code {{
        font-family: "Courier New", monospace;
        font-size: 9.5pt;
        background-color: #f5f5f5;
        padding: 1px 4px;
        border-radius: 2px;
    }}
    pre {{
        background-color: #f5f5f5;
        padding: 12px;
        border-radius: 4px;
        overflow-x: auto;
        font-size: 9pt;
        line-height: 1.4;
    }}
    pre code {{
        background: none;
        padding: 0;
    }}
    blockquote {{
        border-left: 3px solid #ccc;
        margin-left: 0;
        padding-left: 1em;
        color: #555;
    }}
    hr {{
        border: none;
        border-top: 1px solid #ccc;
        margin: 2em 0;
    }}
    strong {{
        font-weight: 600;
    }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""

    HTML(string=full_html).write_pdf(pdf_path)
    print(f"PDF written to {pdf_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python md_to_pdf.py input.md output.pdf")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
