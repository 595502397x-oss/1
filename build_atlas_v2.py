from __future__ import annotations

import re
import unicodedata

import build_atlas as ba


# Papers whose arXiv HTML conversion omits or mis-ranks the main method figure.
FORCE_PDF_FIGURE = {
    "1811.04551": 2,   # PlaNet latent dynamics designs
    "2111.00210": 2,   # EfficientZero self-supervised consistency
    "2106.13195": 2,   # FitVid architecture
    "2310.09615": 2,   # STORM structure and imagination
    "2303.03378": 1,   # PaLM-E multimodal architecture
    "2304.13705": 4,   # ACT architecture
    "2209.07753": 1,   # Code as Policies pipeline
}

# Main method figures that are available as clean original images in arXiv HTML.
STRICT_HTML_FIGURE = {
    "2303.07109": 1,   # TWM architecture
    "2205.09853": 3,   # MCVD U-Net/noise prediction diagram
    "2210.03094": 3,   # VIMA architecture
    "2405.12213": 0,   # Octo architecture is numbered Fig. 0 in the HTML conversion
    "2204.01691": 1,   # SayCan grounding overview
    "2203.06173": 2,   # MVP pretraining-to-control pipeline
    "2305.16291": 2,   # Voyager components
    "2411.04983": 1,   # DINO-WM training, inference, and planning overview
}


def _replace_and_reorder_papers() -> None:
    # Correct a mistyped arXiv ID for TWM, and replace the adjacent I-JEPA paper
    # with DINO-WM so the atlas includes the exact direction shown by the user.
    dino = next(p for p in ba.WORLD_PAPERS if p.arxiv_id == "2411.04983")
    world = []
    for paper in ba.WORLD_PAPERS:
        if paper.arxiv_id == "2303.07142":
            world.append(
                ba.Paper(
                    "Transformer-based World Models Are Happy With 100k Interactions",
                    "2303.07109",
                    "ICLR",
                    2023,
                    "潜在世界模型与规划",
                    1,
                )
            )
        elif paper.arxiv_id == "2301.08243":
            world.append(dino)
        elif paper.arxiv_id == "2411.04983":
            continue
        else:
            world.append(paper)
    ba.WORLD_PAPERS = world

    # Open X-Embodiment's main figures are dataset/performance summaries rather
    # than a method-logic diagram. Replace it with UniPi's video-policy pipeline.
    unipi = next(p for p in ba.EMBODIED_PAPERS if p.arxiv_id == "2302.00111")
    embodied = []
    for paper in ba.EMBODIED_PAPERS:
        if paper.arxiv_id == "2310.08864":
            embodied.append(unipi)
        elif paper.arxiv_id == "2302.00111":
            continue
        else:
            embodied.append(paper)
    ba.EMBODIED_PAPERS = embodied


_replace_and_reorder_papers()


_original_html_candidates = ba.html_figure_candidates
_original_pdf_candidates = ba.pdf_caption_candidates
_original_extract_one = ba.extract_one
_original_build_pdf = ba.build_pdf


def html_figure_candidates_strict(paper: ba.Paper) -> list[dict]:
    candidates = _original_html_candidates(paper)
    desired = STRICT_HTML_FIGURE.get(paper.arxiv_id)
    if desired is None:
        return candidates
    for item in candidates:
        if item["figure_number"] == desired:
            item["score"] += 2000
    return sorted(candidates, key=lambda x: x["score"], reverse=True)


def pdf_caption_candidates_strict(doc, paper: ba.Paper) -> list[dict]:
    candidates = _original_pdf_candidates(doc, paper)
    desired = FORCE_PDF_FIGURE.get(paper.arxiv_id)
    if desired is None:
        return candidates

    start_pattern = re.compile(rf"^(?:Figure|Fig\.)\s*{desired}\s*[:.]", re.I)
    for item in candidates:
        if item["figure_number"] != desired:
            continue
        item["score"] += 2000
        text = item["caption"].lstrip()
        if start_pattern.search(text):
            item["score"] += 500
        if len(text) <= 600:
            item["score"] += 50
    return sorted(candidates, key=lambda x: x["score"], reverse=True)


def extract_one_strict(paper: ba.Paper) -> ba.SelectedFigure:
    if paper.arxiv_id in FORCE_PDF_FIGURE:
        desired = FORCE_PDF_FIGURE[paper.arxiv_id]
        print(f"\n[{paper.venue} {paper.year}] {paper.title} ({paper.arxiv_id})", flush=True)
        print(f"  Forced PDF extraction: Figure {desired}", flush=True)
        return ba.extract_from_pdf(paper)
    return _original_extract_one(paper)


def sanitize_caption(text: str) -> str:
    # ReportLab's CJK font can display unsupported math glyphs as unrelated CJK
    # characters. Keep the caption readable by retaining its English wording and
    # replacing TeX-heavy fragments with spaces; the original math remains visible
    # inside the figure itself.
    text = unicodedata.normalize("NFKD", text or "")
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\\[A-Za-z]+\*?", " ", text)
    text = text.replace("{", " ").replace("}", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_pdf_sanitized(selected: list[ba.SelectedFigure]):
    for item in selected:
        item.caption = sanitize_caption(item.caption)
    return _original_build_pdf(selected)


ba.html_figure_candidates = html_figure_candidates_strict
ba.pdf_caption_candidates = pdf_caption_candidates_strict
ba.extract_one = extract_one_strict
ba.build_pdf = build_pdf_sanitized


if __name__ == "__main__":
    ba.main()
