from __future__ import annotations

import fitz
from build_atlas import Paper, get_pdf, pdf_caption_candidates

PAPERS = [
    Paper("PlaNet", "1811.04551", "ICML", 2019, "world", 1),
    Paper("EfficientZero", "2111.00210", "NeurIPS", 2021, "world", 1),
    Paper("FitVid", "2106.13195", "ICLR", 2022, "world", 1),
    Paper("STORM", "2310.09615", "NeurIPS", 2023, "world", 1),
    Paper("PaLM-E", "2303.03378", "ICML", 2023, "embodied", 1),
    Paper("ACT", "2304.13705", "RSS", 2023, "embodied", 1),
    Paper("Code as Policies", "2209.07753", "ICRA", 2023, "embodied", 1),
]

for paper in PAPERS:
    print("\n" + "=" * 100)
    print(paper.arxiv_id, paper.title)
    try:
        path = get_pdf(paper)
        doc = fitz.open(path)
        candidates = pdf_caption_candidates(doc, paper)
    except Exception as exc:
        print("PDF ERROR:", type(exc).__name__, exc)
        continue
    candidates = sorted(candidates, key=lambda x: (x["figure_number"] if x["figure_number"] is not None else 999, x["page"], x["rect"][1]))
    for item in candidates:
        cap = item["caption"].replace("\n", " ")
        print(f"FIG={item['figure_number']!s:>3} page={item['page']+1:>2} score={item['score']:>5.1f} rect={tuple(round(v,1) for v in item['rect'])} :: {cap[:700]}")
