# MSSP submission package

Manuscript formatted for *Mechanical Systems and Signal Processing*
(Elsevier). The working draft it was built from is
[`../draft.md`](../draft.md); that stays the readable version, this is the
submission artefact.

```
manuscript.tex        elsarticle, preprint 12pt, elsarticle-num
references.bib        six entries, locators verified against publishers
highlights.txt        the same five bullets, plain text for the submission form
figs/                 fig1-3, copied from ../figs
check_submission.py   the constraints that can be checked without LaTeX
```

## ⚠️ Never compiled

No TeX toolchain was available in the environment this was written in, so
**`manuscript.tex` has not been compiled even once.** Expect the ordinary
first-compile problems: missing packages, a float that will not place, an
overfull line. Before doing anything else with it:

```bash
latexmk -pdf manuscript.tex
```

`check_submission.py` covers what can be verified from the source alone —
Elsevier's stated limits, and the internal consistency a compile would
otherwise surface late:

```
$ python check_submission.py
  [PASS] abstract is 233 words (limit 250)
  [PASS] 5 highlights (allowed 3-5), longest 69 chars (limit 85)
  [PASS] every \cite resolves; every bib entry cited; all six have DOIs
  [PASS] every \ref has a \label; all three figure files present
```

It is not a substitute for compiling.

## Journal requirements this was built against

| Requirement | Source | Status |
|---|---|---|
| Reference style `elsarticle-num` | MSSP guide for authors | applied |
| Abstract ≈250 words | MSSP guide for authors | 233 |
| Highlights, 3–5 bullets, ≤85 chars incl. spaces | Elsevier standard | 5, longest 69 |
| Declaration of competing interest | Elsevier standard | present |
| CRediT authorship statement | Elsevier standard | present, one author |
| Data availability statement | Elsevier standard | present, needs archival DOI |

The guide-for-authors page itself is blocked from this environment; the
requirements above come from the journal's published guide as reported in
search results and from Elsevier's cross-journal standards. **Re-read the
live guide before submitting** — in particular whether MSSP currently wants
a graphical abstract, and its declared limits on figure count and colour
charges.

## Before submitting

Four `%% TODO` markers are left in the source deliberately, and
`check_submission.py` lists them on every run:

1. **Author list, affiliations, ORCIDs.** Currently one placeholder author
   with a KETI affiliation, taken from the repository account. This is a
   guess and certainly needs correcting.
2. **CRediT statement** — currently assigns every role to that one author.
3. **Data availability** — replace the bare GitHub URL with an archived DOI
   (Zenodo or similar). Elsevier prefers a persistent identifier, and a
   moving branch is not one.
4. **Acknowledgements / funding** — fill in, or state that no funding was
   received. Elsevier requires the statement either way.

## Substantive gaps, not formatting ones

Formatting is the easy part and it is done. The manuscript still has the
open scientific items recorded in `../draft.md` §8 and in
`../README.md`, and none of them is fixed by a template:

- **no reference sensor** — every claim about the real clip rests on video
  alone;
- **the field case is a negative result** — no positive demonstration that
  the restoration works under good conditions;
- **the repeat check is unexercised** on real data;
- two reference details still to check against primary sources rather than
  secondary citations (`../draft.md`, "Verification status").

A reviewer will raise the first of these. It is worth deciding how to answer
before submitting, rather than after.
