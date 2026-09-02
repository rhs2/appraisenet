"""Build the paper: tables and macros from the study artifacts, PDF with tectonic,
DOCX with pandoc, and the HTML rendition for the project page (docs/paper.html).
The built PDF is also copied into docs/ so GitHub Pages serves it.

    python paper/build.py
"""
from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
DOCS = ROOT / "docs"
TITLE = "AppraiseNet: Calibrated Used-Vehicle Price Estimation with Classical, Deep and Hybrid Learners"
SHORT = "AppraiseNet"
FILE = "Sium_Finstuen_2026_AppraiseNet"
AUTHOR_LINE = "Rakibul Hasan Sium and Drew Finstuen, Lykios (sio@lykios.com, drew@lykios.com)"


def run(cmd: list[str], cwd: Path) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def tex_escape(text: str) -> str:
    return text.replace("&", "\\&").replace("%", "\\%").replace("#", "\\#")


def prepare_plain_source(src: Path, dst: Path) -> Path:
    """pandoc drops IEEEtran-specific macros and thebibliography, so for the Word
    and HTML builds: expand macros/tables, number captions, resolve refs and cites."""
    tex = src.read_text(encoding="utf-8")
    tex = re.sub(r"\\IfFileExists\{title\.tex\}\{\\input\{title\}\}\{.*?\}\n", "", tex, count=1, flags=re.S)
    tex = tex.replace("\\title{\\PaperTitle}", f"\\title{{{tex_escape(TITLE)}}}")
    macros = (PAPER / "tables" / "macros.tex").read_text(encoding="utf-8")
    for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{(.*)\}", macros):
        tex = re.sub(r"\\" + m.group(1) + r"\b(\{\})?", m.group(2).replace("\\", "\\\\"), tex)
    tex = tex.replace("\\input{tables/macros}\n", "")
    tex = re.sub(r"\\input\{tables/(\w+)\}",
                 lambda mm: (PAPER / "tables" / f"{mm.group(1)}.tex").read_text(encoding="utf-8"), tex)
    tex = tex.replace("\\begin{table*}", "\\begin{table}").replace("\\end{table*}", "\\end{table}")
    tex = tex.replace("\\begin{figure*}", "\\begin{figure}").replace("\\end{figure*}", "\\end{figure}")
    counts = {"table": 0, "figure": 0}
    labels: dict[str, str] = {}
    out, pos = [], 0
    for env in re.finditer(r"\\begin\{(table|figure)\}", tex):
        kind = env.group(1)
        end = tex.find(f"\\end{{{kind}}}", env.end())
        cap = tex.find("\\caption{", env.end())
        if cap == -1 or cap > end:
            continue
        counts[kind] += 1
        lab = re.search(r"\\label\{([^}]+)\}", tex[env.end():end])
        if lab:
            labels[lab.group(1)] = str(counts[kind])
        prefix = f"Table {counts['table']}. " if kind == "table" else f"Fig. {counts['figure']}. "
        out.append(tex[pos:cap + len("\\caption{")] + prefix)
        pos = cap + len("\\caption{")
    tex = "".join(out) + tex[pos:]
    for n, sec in enumerate(re.finditer(r"\\section\{[^}]*\}\s*(?:\\label\{([^}]+)\})?", tex), start=1):
        if sec.group(1):
            labels[sec.group(1)] = str(n)
    tex = re.sub(r"(~?)\\ref\{([^}]+)\}", lambda mm: (" " if mm.group(1) else "") + labels.get(mm.group(2), "?"), tex)
    m = re.search(r"\\begin\{thebibliography\}\{\d+\}(.*?)\\end\{thebibliography\}", tex, flags=re.S)
    items = re.findall(r"\\bibitem\{([^}]+)\}\s*(.*?)(?=\\bibitem\{|$)", m.group(1), flags=re.S) if m else []
    order = {key: i + 1 for i, (key, _) in enumerate(items)}

    def cite(match):
        keys = [k.strip() for k in match.group(2).split(",")]
        return (" " if match.group(1) else "") + "[" + ", ".join(str(order.get(k, "?")) for k in keys) + "]"

    tex = re.sub(r"(~?)\\cite\{([^}]+)\}", cite, tex)
    refs = ("\\section*{References}\n\\begin{enumerate}\n"
            + "\n".join(f"\\item {body.strip()}" for _, body in items) + "\n\\end{enumerate}\n")
    tex = re.sub(r"\\begin\{thebibliography\}\{\d+\}.*?\\end\{thebibliography\}", lambda _: refs, tex, flags=re.S)
    tex = re.sub(r"\\IEEEPARstart\{(\w)\}\{(\w+)\}", r"\1\2", tex)
    tex = re.sub(r"\\begin\{IEEEkeywords\}\s*(.*?)\s*\\end\{IEEEkeywords\}",
                 r"\\noindent\\textbf{Index Terms:} \1", tex, flags=re.S)
    tex = re.sub(r"\\markboth\{[^}]*\}\{[^}]*\}", "", tex)
    dst.write_text(tex, encoding="utf-8")
    return dst


SITENAV = ('<nav class="sitenav"><a href="index.html">Project page</a> '
           '<a href="paper.html" class="active">Paper</a> '
           f'<a href="{FILE}.pdf">PDF</a> '
           '<a href="https://github.com/rhs2/appraisenet">Code</a></nav>\n')


def _stamp_pdf_links(pdf: Path) -> None:
    """Point every PDF link at the freshly built file.

    The paper keeps one filename for its whole life, so a reader who downloaded an
    earlier edition, and the Pages CDN in front of them, will happily serve that
    cached copy forever. Stamping a content digest on the link changes the URL
    whenever the bytes change, which is the only thing a cache reacts to.
    """
    digest = hashlib.md5(pdf.read_bytes()).hexdigest()[:8]
    pattern = re.compile(re.escape(FILE) + r"\.pdf(\?v=[0-9a-f]+)?")
    for path in (DOCS / "index.html", DOCS / "paper.html", ROOT / "README.md"):
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        stamped = pattern.sub(f"{FILE}.pdf?v={digest}", text)
        if stamped != text:
            path.write_text(stamped, encoding="utf-8")
    print(f"PDF links stamped ?v={digest}")


def main() -> int:
    run([sys.executable, str(PAPER / "make_tables.py")], ROOT)
    title_tex = PAPER / "title.tex"
    title_tex.write_text(f"\\newcommand{{\\PaperTitle}}{{{tex_escape(TITLE)}}}\n"
                         f"\\newcommand{{\\PaperShortTitle}}{{{tex_escape(SHORT)}}}\n", encoding="utf-8")
    if shutil.which("tectonic"):
        run(["tectonic", "--keep-logs", "main.tex"], PAPER)
    else:
        for _ in range(2):
            run(["pdflatex", "-interaction=nonstopmode", "main.tex"], PAPER)
    pdf = PAPER / f"{FILE}.pdf"
    shutil.move(PAPER / "main.pdf", pdf)
    print(f"PDF  -> {pdf.name} ({pdf.stat().st_size // 1024} KB)")
    DOCS.mkdir(exist_ok=True)
    shutil.copy(pdf, DOCS / pdf.name)

    if shutil.which("pandoc"):
        plain = prepare_plain_source(PAPER / "main.tex", PAPER / "main_plain.tex")
        docx = PAPER / f"{FILE}.docx"
        run(["pandoc", plain.name, "-o", docx.name, "--from=latex",
             "--resource-path=.:figures:../reports/figures",
             f"--metadata=title:{TITLE}", f"--metadata=author:{AUTHOR_LINE}"], PAPER)
        print(f"DOCX -> {docx.name} ({docx.stat().st_size // 1024} KB)")
        html = DOCS / "paper.html"
        run(["pandoc", plain.name, "-o", str(html), "--from=latex", "--standalone", "--mathjax",
             "--css=style.css", "--resource-path=.:figures:../reports/figures",
             "--extract-media=" + str(DOCS / "media"),
             f"--metadata=title:{TITLE}", f"--metadata=author:{AUTHOR_LINE}"], PAPER)
        text = html.read_text(encoding="utf-8")
        text = text.replace("<body>", "<body>\n" + SITENAV, 1)
        # pandoc's --extract-media writes absolute local paths; make them site-relative
        text = text.replace(str(DOCS) + "/", "")
        html.write_text(text, encoding="utf-8")
        print(f"HTML -> {html.relative_to(ROOT)}")
        plain.unlink(missing_ok=True)
    else:
        print("pandoc not found; DOCX and HTML skipped")
    _stamp_pdf_links(pdf)   # last: pandoc rewrites paper.html and would drop the digest
    for junk in ("main.log", "main.aux", "title.tex"):
        (PAPER / junk).unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
