import pypdf

import sys
import os

pdf_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "paper.pdf")
if not os.path.exists(pdf_path) and os.path.exists("paper/paper.pdf"):
    pdf_path = "paper/paper.pdf"

reader = pypdf.PdfReader(pdf_path)
print(f"Inspecting: {pdf_path}")
print(f"Total pages: {len(reader.pages)}")
for i, page in enumerate(reader.pages):
    txt = page.extract_text()
    first_line = txt.splitlines()[0] if txt.splitlines() else ""
    last_line = txt.splitlines()[-1] if txt.splitlines() else ""
    print(f"Page {i+1} ({len(txt)} chars):")
    first_line = (first_line[:60]).encode('ascii', 'replace').decode('ascii')
    last_line = (last_line[:60]).encode('ascii', 'replace').decode('ascii')
    print(f"   First: {first_line}")
    print(f"   Last:  {last_line}")
    flags = []
    if "Empirical Results" in txt: flags.append("Sec 4")
    if "Table 1:" in txt: flags.append("Table 1")
    if "Table 2:" in txt: flags.append("Table 2")
    if "Discussion" in txt: flags.append("Sec 5")
    if "Threats to Validity" in txt: flags.append("Sec 6")
    if "Conclusion" in txt: flags.append("Sec 7")
    if "References" in txt: flags.append("References")
    print(f"   Contains: {', '.join(flags)}")
