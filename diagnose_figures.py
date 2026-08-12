from __future__ import annotations

from build_atlas import Paper, html_figure_candidates

PAPERS = [
    Paper("PlaNet", "1811.04551", "ICML", 2019, "world", 1),
    Paper("EfficientZero", "2111.00210", "NeurIPS", 2021, "world", 1),
    Paper("FitVid", "2106.13195", "ICLR", 2022, "world", 1),
    Paper("MCVD", "2205.09853", "NeurIPS", 2022, "world", 1),
    Paper("STORM", "2310.09615", "NeurIPS", 2023, "world", 1),
    Paper("PaLM-E", "2303.03378", "ICML", 2023, "embodied", 1),
    Paper("VIMA", "2210.03094", "ICML", 2023, "embodied", 1),
    Paper("Open X-Embodiment", "2310.08864", "ICRA", 2024, "embodied", 1),
    Paper("Octo", "2405.12213", "RSS", 2024, "embodied", 1),
    Paper("ACT", "2304.13705", "RSS", 2023, "embodied", 1),
    Paper("SayCan", "2204.01691", "CoRL", 2022, "embodied", 1),
    Paper("Code as Policies", "2209.07753", "ICRA", 2023, "embodied", 1),
    Paper("MVP", "2203.06173", "RSS", 2022, "embodied", 1),
    Paper("Voyager", "2305.16291", "NeurIPS", 2023, "embodied", 1),
]

for paper in PAPERS:
    print("\n" + "=" * 100)
    print(paper.arxiv_id, paper.title)
    try:
        candidates = html_figure_candidates(paper)
    except Exception as exc:
        print("HTML ERROR:", type(exc).__name__, exc)
        continue
    candidates = sorted(candidates, key=lambda x: (x["figure_number"] if x["figure_number"] is not None else 999, x["order"]))
    for item in candidates:
        cap = item["caption"].replace("\n", " ")
        print(f"FIG={item['figure_number']!s:>3} order={item['order']:>2} score={item['score']:>5.1f} images={len(item['srcs']):>2} :: {cap[:500]}")
