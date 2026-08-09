"""Check the MSSP manuscript against the constraints that do not need LaTeX.

No TeX toolchain was available where this manuscript was written, so it has
never been compiled. This checks what can be checked from the source:
Elsevier's stated limits (abstract length, Highlights count and character
budget), and the internal consistency a compile would otherwise catch late
(undefined citations, unused bib entries, missing figures, dangling refs).

It does NOT replace compiling. Run `latexmk -pdf manuscript.tex` before
submitting.

Usage: python docs/paper/mssp/check_submission.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ABSTRACT_MAX_WORDS = 250        # MSSP guide for authors
HL_MIN, HL_MAX = 3, 5           # Elsevier: 3 to 5 bullet points
HL_MAX_CHARS = 85               # Elsevier: max 85 characters incl. spaces

fails: list[str] = []
warns: list[str] = []


def check(ok: bool, msg: str, hard: bool = True) -> None:
    (print(f"  [PASS] {msg}") if ok else
     (fails if hard else warns).append(msg))
    if not ok:
        print(f"  [{'FAIL' if hard else 'WARN'}] {msg}")


def strip_tex(s: str) -> str:
    """Rough de-TeX for word counting: drop commands but keep their text."""
    s = re.sub(r"%.*", "", s)
    s = re.sub(r"\$[^$]*\$", "x", s)                 # inline math -> one word
    s = re.sub(r"\\[a-zA-Z]+\s*(\[[^]]*\])?", " ", s)
    return re.sub(r"[{}~\\]", " ", s)


def main() -> int:
    tex = (HERE / "manuscript.tex").read_text()
    bib = (HERE / "references.bib").read_text()

    print("Abstract")
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    if not m:
        check(False, "abstract block found")
    else:
        n = len(strip_tex(m.group(1)).split())
        check(n <= ABSTRACT_MAX_WORDS,
              f"abstract is {n} words (limit {ABSTRACT_MAX_WORDS})")

    print("\nHighlights")
    m = re.search(r"\\begin\{highlights\}(.*?)\\end\{highlights\}", tex, re.S)
    hl_file = (HERE / "highlights.txt").read_text().strip().splitlines()
    if not m:
        check(False, "highlights block found in manuscript")
    else:
        items = [strip_tex(i).strip()
                 for i in re.findall(r"\\item\s*(.*)", m.group(1))]
        check(HL_MIN <= len(items) <= HL_MAX,
              f"{len(items)} highlights (allowed {HL_MIN}-{HL_MAX})")
        for i, it in enumerate(items, 1):
            # count the plain-text form, which is what Elsevier's form takes
            plain = hl_file[i - 1] if i <= len(hl_file) else it
            check(len(plain) <= HL_MAX_CHARS,
                  f"highlight {i} is {len(plain)} chars "
                  f"(limit {HL_MAX_CHARS}): {plain[:60]}...")
        check(len(hl_file) == len(items),
              f"highlights.txt has {len(hl_file)} lines, "
              f"manuscript has {len(items)}")

    print("\nCitations and bibliography")
    cited = set()
    for grp in re.findall(r"\\cite\{([^}]*)\}", tex):
        cited.update(k.strip() for k in grp.split(","))
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    check(not (cited - keys),
          f"every \\cite resolves to a bib entry (missing: "
          f"{sorted(cited - keys) or 'none'})")
    check(not (keys - cited),
          f"every bib entry is cited (uncited: "
          f"{sorted(keys - cited) or 'none'})")
    for key in sorted(keys):
        entry = re.search(r"@\w+\{" + re.escape(key) + r",(.*?)\n\}", bib, re.S)
        body = entry.group(1) if entry else ""
        check("doi" in body.lower(), f"{key} has a DOI", hard=False)

    print("\nCross-references")
    labels = set(re.findall(r"\\label\{([^}]*)\}", tex))
    # \eqref and \autoref count as references too; missing them made every
    # equation label look unused
    refs = set(re.findall(r"\\(?:eq|auto|c)?ref\{([^}]*)\}", tex))
    check(not (refs - labels),
          f"every \\ref has a \\label (dangling: "
          f"{sorted(refs - labels) or 'none'})")
    check(not (labels - refs),
          f"every \\label is referenced (unused: "
          f"{sorted(labels - refs) or 'none'})", hard=False)

    print("\nFigures")
    for g in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]*)\}", tex):
        check((HERE / g).exists(), f"figure file present: {g}")

    print("\nSubmission TODOs left in the source")
    todos = re.findall(r"%%\s*TODO[^\n]*", tex)
    for t in todos:
        print(f"  [TODO] {t.lstrip('% ').strip()}")
    check(True, f"{len(todos)} TODO markers (expected before submission)")

    print()
    if fails:
        print(f"{len(fails)} FAIL, {len(warns)} WARN")
        return 1
    print(f"all hard checks pass, {len(warns)} WARN")
    print("NOTE: not a substitute for compiling. Run `latexmk -pdf "
          "manuscript.tex`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
