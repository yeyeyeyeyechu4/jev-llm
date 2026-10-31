"""Post-process a pandoc-generated .docx into a submission-style manuscript.

- A4 page, 2.5 cm margins
- body text Times New Roman 11 pt, double-spaced, justified
- headings Times New Roman bold, single-spaced
- tables 8.5 pt, single-spaced
- continuous line numbers and centred page numbers
- bracketed citation markers ([1], [2,3]) turned into superscripts (Nature style)

Usage: python postprocess_docx.py <docx> [<docx> ...] [--cjk]
Run with the bundled python that has python-docx installed.
"""
import copy
import re
import sys

import docx
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

CITATION = re.compile(r"\[(\d+(?:\s*[-,–]\s*\d+)*)\]")


def set_run_font(run, name, size, bold=None, cjk=False):
    run.font.name = name
    run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    rfonts.set(qn("w:ascii"), name)
    rfonts.set(qn("w:hAnsi"), name)
    if cjk:
        rfonts.set(qn("w:eastAsia"), "SimSun")


def set_style_font(style, name, size, bold=None, cjk=False):
    font = style.font
    font.name = name
    font.size = Pt(size)
    if bold is not None:
        font.bold = bold
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    rfonts.set(qn("w:ascii"), name)
    rfonts.set(qn("w:hAnsi"), name)
    if cjk:
        rfonts.set(qn("w:eastAsia"), "SimSun")


def add_line_numbers(section):
    sect_pr = section._sectPr
    for existing in sect_pr.findall(qn("w:lnNumType")):
        sect_pr.remove(existing)
    ln = OxmlElement("w:lnNumType")
    ln.set(qn("w:countBy"), "1")
    ln.set(qn("w:restart"), "continuous")
    ln.set(qn("w:distance"), "360")
    sect_pr.append(ln)


def add_page_number(section, cjk=False):
    footer = section.footer
    footer.is_linked_to_previous = False
    p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    p.text = ""
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    for el in (fld_begin, instr, fld_sep, text, fld_end):
        run._r.append(el)
    run.font.size = Pt(9)
    run.font.name = "Times New Roman"


def superscript_citations(paragraph):
    """Split runs so that bracketed citation markers become superscript."""
    for run in list(paragraph.runs):
        text = run.text
        if not CITATION.search(text):
            continue
        pieces = []
        pos = 0
        for m in CITATION.finditer(text):
            if m.start() > pos:
                pieces.append((text[pos:m.start()], False))
            pieces.append((m.group(0), True))
            pos = m.end()
        if pos < len(text):
            pieces.append((text[pos:], False))
        if not any(flag for _, flag in pieces):
            continue
        parent = run._element.getparent()
        anchor = run._element
        template = copy.deepcopy(run._element)
        for piece_text, is_cite in pieces:
            new_el = copy.deepcopy(template)
            for t in new_el.findall(qn("w:t")):
                new_el.remove(t)
            for br in new_el.findall(qn("w:br")):
                new_el.remove(br)
            t = OxmlElement("w:t")
            t.set(qn("xml:space"), "preserve")
            t.text = piece_text
            new_el.append(t)
            if is_cite:
                rpr = new_el.find(qn("w:rPr"))
                if rpr is None:
                    rpr = OxmlElement("w:rPr")
                    new_el.insert(0, rpr)
                vert = OxmlElement("w:vertAlign")
                vert.set(qn("w:val"), "superscript")
                rpr.append(vert)
            parent.insert(parent.index(anchor) + 1, new_el)
            anchor = new_el
        parent.remove(run._element)


def style_table(table, body_font, cjk, usable_cm=16.0):
    """Fixed-width layout, content-proportional columns, small font, no row splits."""
    ncols = len(table.columns)
    if ncols == 0:
        return
    # font size by width of the table
    size = 8.5 if ncols <= 6 else (8.0 if ncols <= 8 else 7.5)
    # measure content (in characters, CJK counted as 2)
    weight = [1.0] * ncols
    for row in table.rows:
        for j, cell in enumerate(row.cells):
            if j >= ncols:
                continue
            txt = cell.text.strip()
            w = sum(2 if ord(ch) > 0x2E80 else 1 for ch in txt)
            weight[j] = max(weight[j], min(w, 34))
    total = sum(weight) or 1.0
    widths = [max(1.5, usable_cm * w / total) for w in weight]
    scale = usable_cm / sum(widths)
    widths = [w * scale for w in widths]

    table.autofit = False
    tbl = table._tbl
    tblPr = tbl.tblPr
    # table width 100 %
    for el in tblPr.findall(qn("w:tblW")):
        tblPr.remove(el)
    tblW = OxmlElement("w:tblW")
    tblW.set(qn("w:type"), "pct")
    tblW.set(qn("w:w"), "5000")
    tblPr.append(tblW)
    # fixed layout
    for el in tblPr.findall(qn("w:tblLayout")):
        tblPr.remove(el)
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tblPr.append(layout)
    # cell margins (small)
    for el in tblPr.findall(qn("w:tblCellMar")):
        tblPr.remove(el)
    mar = OxmlElement("w:tblCellMar")
    for side, val in (("left", "57"), ("right", "57"), ("top", "11"), ("bottom", "11")):
        e = OxmlElement(f"w:{side}")
        e.set(qn("w:w"), val)
        e.set(qn("w:type"), "dxa")
        mar.append(e)
    tblPr.append(mar)

    for j, w in enumerate(widths):
        table.columns[j].width = Cm(w)
    for row in table.rows:
        trPr = row._tr.get_or_add_trPr()
        cant = OxmlElement("w:cantSplit")
        trPr.append(cant)
        for j, cell in enumerate(row.cells):
            if j < len(widths):
                cell.width = Cm(widths[j])
            for p in cell.paragraphs:
                p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
                p.paragraph_format.space_after = Pt(0)
                for run in p.runs:
                    set_run_font(run, body_font, size, cjk=cjk)
    # bold header row
    for cell in table.rows[0].cells:
        for p in cell.paragraphs:
            for run in p.runs:
                run.font.bold = True


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cjk = "--cjk" in sys.argv
    for path in args:
        doc = docx.Document(path)
        styles = doc.styles
        body_font = "SimSun" if cjk else "Times New Roman"
        heading_font = "SimHei" if cjk else "Times New Roman"
        for name, size, bold in [("Normal", 11, None), ("Body Text", 11, None),
                                 ("First Paragraph", 11, None), ("Compact", 9, None),
                                 ("Title", 16, True), ("Subtitle", 12, False),
                                 ("Heading 1", 14, True), ("Heading 2", 12, True),
                                 ("Heading 3", 11, True)]:
            try:
                set_style_font(styles[name], heading_font if name.startswith("Heading") or name in ("Title", "Subtitle") else body_font, size, bold, cjk)
            except KeyError:
                pass
        for section in doc.sections:
            section.page_width = Cm(21.0)
            section.page_height = Cm(29.7)
            section.top_margin = Cm(2.5)
            section.bottom_margin = Cm(2.5)
            section.left_margin = Cm(2.5)
            section.right_margin = Cm(2.5)
            add_line_numbers(section)
            add_page_number(section, cjk)
        for p in doc.paragraphs:
            style_name = p.style.name if p.style is not None else ""
            is_heading = style_name.startswith("Heading") or style_name in ("Title", "Subtitle")
            pf = p.paragraph_format
            if is_heading:
                pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
                pf.space_before = Pt(10 if style_name != "Title" else 0)
                pf.space_after = Pt(4)
            elif style_name == "Compact":
                pf.line_spacing_rule = WD_LINE_SPACING.SINGLE
            else:
                pf.line_spacing = 2.0
                pf.space_after = Pt(0)
                p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY if style_name in (
                    "Body Text", "First Paragraph", "Normal") else p.alignment
            superscript_citations(p)
        if cjk and doc.paragraphs and doc.paragraphs[0].style is not None and doc.paragraphs[0].style.name == "Heading 1":
            for run in doc.paragraphs[0].runs:
                set_run_font(run, heading_font, 16, bold=True, cjk=cjk)
        for p in doc.paragraphs:
            if re.match(r"^\d{1,2}\.\s", p.text) and ("doi.org" in p.text or "arXiv" in p.text
                                                      or "https://" in p.text or "PhysioNet" in p.text
                                                      or "Kaggle" in p.text or "National Center" in p.text):
                p.paragraph_format.left_indent = Cm(0.75)
                p.paragraph_format.first_line_indent = Cm(-0.75)
        for table in doc.tables:
            style_table(table, body_font, cjk)
        # keep each figure on the same page as its caption
        for p in doc.paragraphs:
            if p._element.findall(".//" + qn("w:drawing")):
                p.paragraph_format.keep_with_next = True
        doc.save(path)
        print("styled", path)


if __name__ == "__main__":
    main()
