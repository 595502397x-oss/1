from __future__ import annotations

import base64
import csv
import io
import json
import math
import os
import re
import textwrap
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageChops, ImageOps
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import cairosvg
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
FIG_DIR = WORK / "figures"
PDF_DIR = WORK / "pdfs"
OUT = ROOT / "output"
for p in (WORK, FIG_DIR, PDF_DIR, OUT):
    p.mkdir(parents=True, exist_ok=True)

TARGET_WORLD = 26
TARGET_EMBODIED = 26
MIN_TOTAL = 50


@dataclass(frozen=True)
class Paper:
    title: str
    arxiv_id: str
    venue: str
    year: int
    category: str
    preferred_figure: int = 1


@dataclass
class SelectedFigure:
    title: str
    arxiv_id: str
    venue: str
    year: int
    category: str
    figure_number: int | None
    caption: str
    source_url: str
    extraction_method: str
    image_file: str
    score: float


WORLD_PAPERS = [
    Paper("Learning Latent Dynamics for Planning from Pixels (PlaNet)", "1811.04551", "ICML", 2019, "潜在世界模型与规划", 2),
    Paper("Dream to Control: Learning Behaviors by Latent Imagination", "1912.01603", "ICLR", 2020, "潜在世界模型与规划", 1),
    Paper("Model-Based Reinforcement Learning for Atari (SimPLe)", "1903.00374", "ICLR", 2020, "潜在世界模型与规划", 1),
    Paper("Mastering Atari with Discrete World Models (DreamerV2)", "2010.02193", "ICLR", 2021, "潜在世界模型与规划", 1),
    Paper("Mastering Atari Games with Limited Data (EfficientZero)", "2111.00210", "NeurIPS", 2021, "搜索式世界模型", 2),
    Paper("VideoGPT: Video Generation using VQ-VAE and Transformers", "2104.10157", "NeurIPS", 2021, "生成式视频世界模型", 1),
    Paper("FitVid: Overfitting in Pixel-Level Video Prediction", "2106.13195", "ICLR", 2022, "生成式视频世界模型", 1),
    Paper("Temporal Difference Learning for Model Predictive Control (TD-MPC)", "2203.04955", "ICML", 2022, "潜在世界模型与规划", 1),
    Paper("Denoised MDPs: Learning World Models Better Than the World Itself", "2206.15477", "ICML", 2022, "世界模型表示学习", 1),
    Paper("DayDreamer: World Models for Physical Robot Learning", "2206.14176", "CoRL", 2022, "潜在世界模型与规划", 1),
    Paper("Masked World Models for Visual Control", "2206.14244", "CoRL", 2022, "世界模型表示学习", 1),
    Paper("MCVD: Masked Conditional Video Diffusion for Prediction, Generation, and Interpolation", "2205.09853", "NeurIPS", 2022, "生成式视频世界模型", 1),
    Paper("Transformers are Sample-Efficient World Models (IRIS)", "2209.00588", "ICLR", 2023, "潜在世界模型与规划", 1),
    Paper("Transformer-based World Models Are Happy With 100k Interactions", "2303.07142", "ICLR", 2023, "潜在世界模型与规划", 1),
    Paper("MAGVIT: Masked Generative Video Transformer", "2212.05199", "CVPR", 2023, "生成式视频世界模型", 1),
    Paper("Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA)", "2301.08243", "CVPR", 2023, "世界模型表示学习", 1),
    Paper("Multi-View Masked World Models for Visual Robotic Manipulation", "2302.02408", "ICML", 2023, "世界模型表示学习", 1),
    Paper("Learning World Models with Identifiable Factorization", "2306.06561", "NeurIPS", 2023, "世界模型表示学习", 1),
    Paper("STORM: Efficient Stochastic Transformer based World Models for Reinforcement Learning", "2310.09615", "NeurIPS", 2023, "潜在世界模型与规划", 1),
    Paper("TD-MPC2: Scalable, Robust World Models for Continuous Control", "2310.16828", "ICLR", 2024, "潜在世界模型与规划", 1),
    Paper("Learning Interactive Real-World Simulators (UniSim)", "2310.06114", "ICLR", 2024, "可交互世界模拟器", 1),
    Paper("Language Model Beats Diffusion — Tokenizer is Key to Visual Generation (MAGVIT-v2)", "2310.05737", "ICLR", 2024, "生成式视频世界模型", 1),
    Paper("Driving into the Future: Multiview Visual Forecasting and Planning with World Model for Autonomous Driving (Drive-WM)", "2311.17918", "CVPR", 2024, "驾驶世界模型", 1),
    Paper("Photorealistic Video Generation with Diffusion Models (W.A.L.T)", "2312.06662", "CVPR", 2024, "生成式视频世界模型", 1),
    Paper("Genie: Generative Interactive Environments", "2402.15391", "ICML", 2024, "可交互世界模拟器", 1),
    Paper("WorldDreamer: Towards General World Models for Video Generation via Predicting Masked Tokens", "2401.09985", "ECCV", 2024, "生成式视频世界模型", 1),
    Paper("DriveDreamer: Towards Real-world-driven World Models for Autonomous Driving", "2309.09777", "ECCV", 2024, "驾驶世界模型", 1),
    Paper("DIAMOND: Diffusion As a Model of Environment Dreams", "2405.12399", "NeurIPS", 2024, "生成式视频世界模型", 1),
    Paper("EfficientZero V2: Mastering Discrete and Continuous Control with Limited Data", "2403.00564", "ICML", 2024, "搜索式世界模型", 2),
    Paper("iVideoGPT: Interactive VideoGPTs are Scalable World Models", "2405.15223", "NeurIPS", 2024, "可交互世界模拟器", 1),
    Paper("DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning", "2411.04983", "ICML", 2025, "世界模型表示学习", 1),
    Paper("Phenaki: Variable Length Video Generation from Open Domain Textual Descriptions", "2210.02399", "ICLR", 2023, "生成式视频世界模型", 1),
    Paper("TECO: Temporally Consistent Transformers for Video Generation", "2305.04966", "ICML", 2023, "生成式视频世界模型", 1),
]


EMBODIED_PAPERS = [
    Paper("Transporter Networks: Rearranging the Visual World for Robotic Manipulation", "2010.14406", "CoRL", 2020, "机器人策略与动作生成", 2),
    Paper("CLIPort: What and Where Pathways for Robotic Manipulation", "2109.12098", "CoRL", 2021, "视觉语言机器人策略", 2),
    Paper("RT-1: Robotics Transformer for Real-World Control at Scale", "2212.06817", "RSS", 2023, "视觉语言机器人策略", 2),
    Paper("RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control", "2307.15818", "CoRL", 2023, "视觉语言机器人策略", 1),
    Paper("PaLM-E: An Embodied Multimodal Language Model", "2303.03378", "ICML", 2023, "视觉语言机器人策略", 1),
    Paper("VIMA: General Robot Manipulation with Multimodal Prompts", "2210.03094", "ICML", 2023, "视觉语言机器人策略", 2),
    Paper("Open X-Embodiment: Robotic Learning Datasets and RT-X Models", "2310.08864", "ICRA", 2024, "视觉语言机器人策略", 1),
    Paper("Octo: An Open-Source Generalist Robot Policy", "2405.12213", "RSS", 2024, "视觉语言机器人策略", 1),
    Paper("OpenVLA: An Open-Source Vision-Language-Action Model", "2406.09246", "CoRL", 2024, "视觉语言机器人策略", 1),
    Paper("LEO: An Embodied Generalist Agent in 3D World", "2311.12871", "ICML", 2024, "具身多模态智能体", 1),
    Paper("3D-VLA: A 3D Vision-Language-Action Generative World Model", "2403.09631", "ICML", 2024, "视觉语言机器人策略", 1),
    Paper("Diffusion Policy: Visuomotor Policy Learning via Action Diffusion", "2303.04137", "RSS", 2023, "机器人策略与动作生成", 2),
    Paper("Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT)", "2304.13705", "RSS", 2023, "机器人策略与动作生成", 1),
    Paper("PerAct: Multi-Task 6D Robotic Manipulation via Perceiver-Actor", "2209.05451", "CoRL", 2022, "机器人策略与动作生成", 2),
    Paper("Do As I Can, Not As I Say: Grounding Language in Robotic Affordances (SayCan)", "2204.01691", "CoRL", 2022, "分层规划与执行", 2),
    Paper("SuSIE: Zero-Shot Robotic Manipulation with Pretrained Image-Editing Diffusion Models", "2310.10639", "ICLR", 2024, "分层规划与执行", 1),
    Paper("Code as Policies: Language Model Programs for Embodied Control", "2209.07753", "ICRA", 2023, "分层规划与执行", 2),
    Paper("Inner Monologue: Embodied Reasoning through Planning with Language Models", "2207.05608", "CoRL", 2022, "分层规划与执行", 1),
    Paper("Socratic Models: Composing Zero-Shot Multimodal Reasoning with Language", "2204.00598", "ICLR", 2023, "具身多模态智能体", 1),
    Paper("Where are we in the search for an Artificial Visual Cortex for Embodied Intelligence? (VC-1)", "2303.18240", "NeurIPS", 2023, "具身视觉表示", 1),
    Paper("LIV: Language-Image Representations and Rewards for Robotic Control", "2306.00958", "ICML", 2023, "具身视觉表示", 1),
    Paper("MVP: Pre-training Video Encoders for Robot Learning", "2203.06173", "RSS", 2022, "具身视觉表示", 1),
    Paper("R3M: A Universal Visual Representation for Robot Manipulation", "2203.12601", "CoRL", 2022, "具身视觉表示", 1),
    Paper("MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge", "2206.08853", "NeurIPS", 2022, "开放世界具身智能体", 1),
    Paper("Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos", "2206.11795", "NeurIPS", 2022, "开放世界具身智能体", 1),
    Paper("Voyager: An Open-Ended Embodied Agent with Large Language Models", "2305.16291", "NeurIPS", 2023, "开放世界具身智能体", 1),
    Paper("STEVE-1: A Generative Model for Text-to-Behavior in Minecraft", "2306.00937", "NeurIPS", 2023, "开放世界具身智能体", 1),
    Paper("Learning Universal Policies via Text-Guided Video Generation (UniPi)", "2302.00111", "NeurIPS", 2023, "分层规划与执行", 1),
    Paper("ViNT: A Foundation Model for Visual Navigation", "2306.14846", "CoRL", 2023, "视觉导航", 1),
    Paper("NoMaD: Goal Masked Diffusion Policies for Navigation and Exploration", "2310.07896", "ICRA", 2024, "视觉导航", 1),
    Paper("Mobile ALOHA: Learning Bimanual Mobile Manipulation with Low-Cost Whole-Body Teleoperation", "2401.02117", "CoRL", 2024, "机器人策略与动作生成", 1),
    Paper("Universal Manipulation Interface: In-The-Wild Robot Teaching Without In-The-Wild Robots", "2402.10329", "CoRL", 2024, "机器人策略与动作生成", 1),
    Paper("GNM: A General Navigation Model to Drive Any Robot", "2210.03370", "ICRA", 2023, "视觉导航", 1),
    Paper("VIP: Towards Universal Visual Reward and Representation via Value-Implicit Pre-Training", "2210.00030", "ICLR", 2023, "具身视觉表示", 1),
    Paper("GR-1: Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation", "2312.13139", "ICLR", 2024, "视觉语言机器人策略", 1),
    Paper("CALVIN: A Benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks", "2112.03227", "CoRL", 2021, "具身任务与评测", 1),
    Paper("Habitat 2.0: Training Home Assistants to Rearrange their Habitat", "2106.14405", "NeurIPS", 2021, "具身任务与评测", 1),
    Paper("ALFRED: A Benchmark for Interpreting Grounded Instructions for Everyday Tasks", "1912.01734", "CVPR", 2020, "具身任务与评测", 1),
    Paper("Language-Driven Representation Learning for Robotics (Voltron)", "2302.12766", "CoRL", 2022, "具身视觉表示", 1),
    Paper("BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark with 1,000 Everyday Activities and Realistic Simulation", "2403.09227", "CoRL", 2022, "具身任务与评测", 1),
]


POSITIVE = {
    "overview": 10,
    "architecture": 10,
    "framework": 9,
    "pipeline": 9,
    "method": 6,
    "approach": 5,
    "schematic": 10,
    "diagram": 9,
    "workflow": 9,
    "training": 5,
    "inference": 5,
    "planning": 6,
    "world model": 7,
    "policy": 4,
    "agent": 3,
    "tokenizer": 4,
    "model": 3,
    "latent": 3,
    "prediction": 3,
    "interaction": 3,
    "system": 4,
    "components": 3,
    "algorithm": 3,
}

NEGATIVE = {
    "ablation": -12,
    "performance": -9,
    "quantitative": -9,
    "comparison": -7,
    "results": -7,
    "success rate": -8,
    "learning curve": -12,
    "reward curve": -12,
    "benchmark": -4,
    "qualitative": -3,
    "generated samples": -5,
    "examples": -2,
    "visualization": -2,
}


def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5,
        read=5,
        connect=5,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8))
    s.headers.update(
        {
            "User-Agent": "ResearchFigureAtlas/1.0 (academic figure extraction; contact via repository owner)",
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    return s


SESSION = make_session()


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def parse_figure_number(text: str) -> int | None:
    m = re.search(r"(?:figure|fig\.?)[\s\u00a0]*(\d+)", text, flags=re.I)
    return int(m.group(1)) if m else None


def caption_score(caption: str, figure_number: int | None, preferred: int) -> float:
    t = caption.lower()
    score = 0.0
    for key, value in POSITIVE.items():
        if key in t:
            score += value
    for key, value in NEGATIVE.items():
        if key in t:
            score += value
    if figure_number == preferred:
        score += 13
    elif figure_number == 1:
        score += 7
    elif figure_number == 2:
        score += 5
    elif figure_number == 3:
        score += 2
    if len(caption) > 30:
        score += 1
    return score


def fetch_bytes(url: str, timeout: int = 60) -> tuple[bytes, str]:
    if url.startswith("data:"):
        header, data = url.split(",", 1)
        if ";base64" in header:
            return base64.b64decode(data), header.split(";")[0].replace("data:", "")
        return data.encode("utf-8"), header.split(";")[0].replace("data:", "")
    r = SESSION.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content, r.headers.get("content-type", "")


def image_from_bytes(data: bytes, content_type: str, source_url: str) -> Image.Image:
    lower = source_url.lower().split("?")[0]
    stripped = data.lstrip()
    if "svg" in content_type or lower.endswith(".svg") or stripped.startswith(b"<svg") or b"<svg" in stripped[:500]:
        png = cairosvg.svg2png(bytestring=data, output_width=2200)
        return Image.open(io.BytesIO(png)).convert("RGB")
    if "pdf" in content_type or lower.endswith(".pdf") or data.startswith(b"%PDF"):
        doc = fitz.open(stream=data, filetype="pdf")
        page = doc[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(2.4, 2.4), alpha=False)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    im = Image.open(io.BytesIO(data))
    im.load()
    if im.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", im.size, "white")
        alpha = im.getchannel("A")
        bg.paste(im.convert("RGB"), mask=alpha)
        return bg
    return im.convert("RGB")


def trim_white(im: Image.Image) -> Image.Image:
    rgb = im.convert("RGB")
    bg = Image.new("RGB", rgb.size, (255, 255, 255))
    diff = ImageChops.difference(rgb, bg).convert("L")
    diff = diff.point(lambda p: 0 if p < 10 else p)
    bbox = diff.getbbox()
    if not bbox:
        return rgb
    left, top, right, bottom = bbox
    pad = max(8, int(min(rgb.size) * 0.015))
    return rgb.crop((max(0, left - pad), max(0, top - pad), min(rgb.width, right + pad), min(rgb.height, bottom + pad)))


def normalize_image(im: Image.Image) -> Image.Image:
    im = ImageOps.exif_transpose(im).convert("RGB")
    im = trim_white(im)
    max_w, max_h = 2200, 1500
    ratio = min(1.0, max_w / im.width, max_h / im.height)
    if ratio < 1.0:
        im = im.resize((max(1, int(im.width * ratio)), max(1, int(im.height * ratio))), Image.Resampling.LANCZOS)
    return im


def compose_images(images: list[Image.Image]) -> Image.Image:
    images = [normalize_image(i) for i in images]
    if len(images) == 1:
        return images[0]
    gap = 18
    if len(images) <= 3:
        target_h = min(850, max(im.height for im in images))
        scaled = []
        for im in images:
            s = min(1.0, target_h / im.height)
            scaled.append(im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.Resampling.LANCZOS))
        width = sum(i.width for i in scaled) + gap * (len(scaled) - 1)
        height = max(i.height for i in scaled)
        canvas_im = Image.new("RGB", (width, height), "white")
        x = 0
        for im in scaled:
            canvas_im.paste(im, (x, (height - im.height) // 2))
            x += im.width + gap
        return normalize_image(canvas_im)
    cols = 2
    rows = math.ceil(len(images) / cols)
    cell_w = min(1000, max(im.width for im in images))
    cell_h = min(650, max(im.height for im in images))
    canvas_im = Image.new("RGB", (cols * cell_w + gap, rows * cell_h + gap * (rows - 1)), "white")
    for idx, im in enumerate(images):
        s = min(cell_w / im.width, cell_h / im.height, 1.0)
        r = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.Resampling.LANCZOS)
        col, row = idx % cols, idx // cols
        x = col * (cell_w + gap) + (cell_w - r.width) // 2
        y = row * (cell_h + gap) + (cell_h - r.height) // 2
        canvas_im.paste(r, (x, y))
    return normalize_image(canvas_im)


def save_candidate(im: Image.Image, paper: Paper, suffix: str) -> Path:
    im = normalize_image(im)
    if im.width < 350 or im.height < 120:
        raise ValueError(f"image too small: {im.size}")
    aspect = im.width / max(1, im.height)
    if aspect > 10.0 or aspect < 0.10:
        raise ValueError(f"extreme aspect ratio: {aspect:.2f}")
    path = FIG_DIR / f"{paper.arxiv_id.replace('.', '_')}_{suffix}.jpg"
    im.save(path, "JPEG", quality=90, optimize=True, progressive=True)
    return path


def html_figure_candidates(paper: Paper) -> list[dict]:
    url = f"https://arxiv.org/html/{paper.arxiv_id}"
    r = SESSION.get(url, timeout=60)
    r.raise_for_status()
    if "Internal Error" in r.text[:2000] or len(r.text) < 5000:
        raise RuntimeError("arXiv HTML conversion unavailable")
    soup = BeautifulSoup(r.text, "lxml")
    candidates: list[dict] = []
    for order, fig in enumerate(soup.find_all("figure"), start=1):
        cap = fig.find("figcaption")
        caption = clean_text(cap.get_text(" ", strip=True) if cap else "")
        number = parse_figure_number(caption)
        srcs = []
        for img in fig.find_all("img"):
            src = img.get("src") or img.get("data-src")
            if src:
                abs_src = urljoin(r.url, src)
                if abs_src not in srcs:
                    srcs.append(abs_src)
        inline_svg = fig.find("svg")
        if not srcs and inline_svg is None:
            continue
        score = caption_score(caption, number, paper.preferred_figure)
        if order <= 3:
            score += max(0, 3 - order)
        candidates.append(
            {
                "caption": caption,
                "figure_number": number,
                "score": score,
                "srcs": srcs,
                "inline_svg": str(inline_svg) if inline_svg is not None else None,
                "page_url": r.url,
                "order": order,
            }
        )
    return sorted(candidates, key=lambda x: x["score"], reverse=True)


def extract_from_html(paper: Paper) -> SelectedFigure:
    candidates = html_figure_candidates(paper)
    if not candidates:
        raise RuntimeError("no HTML figures")
    errors = []
    for rank, cand in enumerate(candidates[:7], start=1):
        try:
            ims: list[Image.Image] = []
            if cand["inline_svg"]:
                ims.append(image_from_bytes(cand["inline_svg"].encode("utf-8"), "image/svg+xml", cand["page_url"]))
            for src in cand["srcs"][:8]:
                data, ctype = fetch_bytes(src)
                ims.append(image_from_bytes(data, ctype, src))
            if not ims:
                continue
            composed = compose_images(ims)
            area_bonus = min(7.0, max(0.0, math.log2(max(1, composed.width * composed.height) / 180000.0)))
            final_score = cand["score"] + area_bonus
            path = save_candidate(composed, paper, f"html_{rank}")
            return SelectedFigure(
                title=paper.title,
                arxiv_id=paper.arxiv_id,
                venue=paper.venue,
                year=paper.year,
                category=paper.category,
                figure_number=cand["figure_number"],
                caption=cand["caption"],
                source_url=cand["page_url"],
                extraction_method="arXiv HTML 原图",
                image_file=str(path),
                score=final_score,
            )
        except Exception as exc:
            errors.append(f"candidate {rank}: {type(exc).__name__}: {exc}")
    raise RuntimeError("; ".join(errors[-4:]) or "HTML candidates failed")


def get_pdf(paper: Paper) -> Path:
    path = PDF_DIR / f"{paper.arxiv_id.replace('.', '_')}.pdf"
    if path.exists() and path.stat().st_size > 10000:
        return path
    errors = []
    for url in (f"https://arxiv.org/pdf/{paper.arxiv_id}", f"https://export.arxiv.org/pdf/{paper.arxiv_id}"):
        try:
            data, _ = fetch_bytes(url, timeout=90)
            if not data.startswith(b"%PDF"):
                raise ValueError("response is not PDF")
            path.write_bytes(data)
            return path
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError(" | ".join(errors))


def pdf_caption_candidates(doc: fitz.Document, paper: Paper) -> list[dict]:
    out = []
    for pno in range(min(len(doc), 12)):
        page = doc[pno]
        for block in page.get_text("blocks"):
            x0, y0, x1, y1, text = block[:5]
            text = clean_text(text)
            number = parse_figure_number(text)
            if number is None:
                continue
            if not re.search(r"(?:figure|fig\.)\s*\d+", text, flags=re.I):
                continue
            score = caption_score(text, number, paper.preferred_figure)
            score += max(0, 5 - pno * 0.5)
            out.append(
                {
                    "page": pno,
                    "rect": (x0, y0, x1, y1),
                    "caption": text,
                    "figure_number": number,
                    "score": score,
                }
            )
    return sorted(out, key=lambda x: x["score"], reverse=True)


def extract_from_pdf(paper: Paper) -> SelectedFigure:
    pdf_path = get_pdf(paper)
    doc = fitz.open(pdf_path)
    candidates = pdf_caption_candidates(doc, paper)
    if not candidates:
        candidates = [
            {
                "page": 0,
                "rect": (40, doc[0].rect.height * 0.60, doc[0].rect.width - 40, doc[0].rect.height * 0.66),
                "caption": "First-page method overview crop",
                "figure_number": None,
                "score": 0.0,
            }
        ]
    errors = []
    for rank, cand in enumerate(candidates[:10], start=1):
        try:
            page = doc[cand["page"]]
            pw, ph = page.rect.width, page.rect.height
            x0, y0, x1, _ = cand["rect"]
            caption_width = x1 - x0
            if caption_width > pw * 0.62:
                cx0, cx1 = pw * 0.045, pw * 0.955
            else:
                margin = 10
                cx0, cx1 = max(0, x0 - margin), min(pw, x1 + margin)
            cy1 = max(80, y0 - 4)
            crop_h = min(ph * 0.48, max(230, cy1 * 0.78))
            cy0 = max(0, cy1 - crop_h)
            clip = fitz.Rect(cx0, cy0, cx1, cy1)
            pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), clip=clip, alpha=False)
            mode = "RGB" if pix.n < 4 else "RGBA"
            im = Image.frombytes(mode, [pix.width, pix.height], pix.samples).convert("RGB")
            path = save_candidate(im, paper, f"pdf_{rank}")
            return SelectedFigure(
                title=paper.title,
                arxiv_id=paper.arxiv_id,
                venue=paper.venue,
                year=paper.year,
                category=paper.category,
                figure_number=cand["figure_number"],
                caption=cand["caption"],
                source_url=f"https://arxiv.org/pdf/{paper.arxiv_id}",
                extraction_method=f"PDF 第 {cand['page'] + 1} 页裁切",
                image_file=str(path),
                score=float(cand["score"]),
            )
        except Exception as exc:
            errors.append(f"candidate {rank}: {type(exc).__name__}: {exc}")
    raise RuntimeError("; ".join(errors[-4:]) or "PDF crop failed")


def extract_one(paper: Paper) -> SelectedFigure:
    print(f"\n[{paper.venue} {paper.year}] {paper.title} ({paper.arxiv_id})", flush=True)
    html_error = None
    try:
        fig = extract_from_html(paper)
        print(f"  OK HTML: Figure {fig.figure_number}, score={fig.score:.1f}", flush=True)
        return fig
    except Exception as exc:
        html_error = f"{type(exc).__name__}: {exc}"
        print(f"  HTML failed: {html_error}", flush=True)
    try:
        fig = extract_from_pdf(paper)
        print(f"  OK PDF: Figure {fig.figure_number}, score={fig.score:.1f}", flush=True)
        return fig
    except Exception as exc:
        raise RuntimeError(f"HTML [{html_error}] | PDF [{type(exc).__name__}: {exc}]") from exc


def logic_sentence(category: str) -> str:
    mapping = {
        "潜在世界模型与规划": "观测先被编码成隐状态；动作推动隐状态向未来演化，预测的奖励或价值再用于规划和策略更新。",
        "搜索式世界模型": "模型只预测搜索所需的状态、奖励和价值，树搜索或轨迹优化据此比较候选动作并选择下一步。",
        "生成式视频世界模型": "图把视频压缩成离散或连续的潜在表示，再用时序模型按条件生成未来帧或可交互轨迹。",
        "世界模型表示学习": "图通过遮挡预测、特征预测或因子分解约束潜在表示，使其保留状态转移和控制真正需要的信息。",
        "可交互世界模拟器": "初始画面与动作条件共同进入生成模型，模型连续产生可被人或智能体控制的未来视觉状态。",
        "驾驶世界模型": "多相机观测被统一编码，模型预测未来场景并把预测结果交给轨迹规划或驾驶决策模块。",
        "机器人策略与动作生成": "视觉与机器人状态进入策略网络，模型直接生成连续动作、动作块或末端执行器轨迹。",
        "视觉语言机器人策略": "图像、语言指令和机器人状态被放入共享表示或同一序列，再解码为可执行动作。",
        "分层规划与执行": "高层模型先生成子目标、代码或技能步骤，低层控制器逐步执行，并根据新观测继续规划。",
        "具身多模态智能体": "多个预训练模型通过共享文本或特征接口协作，把感知、推理和动作选择串成完整闭环。",
        "具身视觉表示": "大规模视频先训练视觉表示或奖励函数，下游机器人再在冻结或微调后的特征上学习控制。",
        "开放世界具身智能体": "智能体把感知、记忆、技能库和规划器连接起来，在长时间交互中不断提出并完成新目标。",
        "视觉导航": "当前观测与目标被映射成局部路径或动作分布，模型比较可达性后滚动选择下一段运动。",
        "具身任务与评测": "图把任务定义、场景、传感器、动作接口与评测指标组织成统一的训练和测试闭环。",
    }
    return mapping.get(category, "沿着图中的箭头可直接读出输入、内部状态、预测量与动作输出之间的依赖关系。")


def truncate(text: str, limit: int) -> str:
    text = clean_text(text)
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def wrap_chars(text: str, max_chars: int) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def wrap_words(text: str, width: int) -> list[str]:
    return textwrap.wrap(clean_text(text), width=width, break_long_words=False, break_on_hyphens=False)


def fit_title(c: canvas.Canvas, text: str, x: float, y: float, max_width: float, font: str, start_size: float = 15.0) -> float:
    size = start_size
    while size > 9 and pdfmetrics.stringWidth(text, font, size) > max_width:
        size -= 0.5
    c.setFont(font, size)
    c.drawString(x, y, text)
    return size


def draw_cover(c: canvas.Canvas, count: int, font: str) -> None:
    w, h = landscape(A4)
    c.setFont(font, 27)
    c.drawCentredString(w / 2, h - 115, "具身智能与世界模型：数学逻辑图集")
    c.setFont(font, 15)
    c.drawCentredString(w / 2, h - 153, f"从顶会论文中筛选并截取 {count} 张方法结构、训练/推理流程与规划关系图")
    c.setLineWidth(1)
    c.line(92, h - 176, w - 92, h - 176)

    notes = [
        "覆盖方向：潜在世界模型、视频生成与交互模拟器、视觉语言动作模型、机器人策略、分层规划、视觉导航。",
        "会议范围：ICML、ICLR、NeurIPS、CVPR、ECCV、RSS、CoRL、ICRA。",
        "选图规则：优先选择能沿箭头读出变量、状态转移、损失、预测与动作依赖关系的总览图或结构图；排除纯结果曲线。",
        "每页包含：论文与会议、原图、简明中文逻辑说明、原始图注、arXiv 来源。",
    ]
    y = h - 225
    c.setFont(font, 12)
    for note in notes:
        for line in wrap_chars(note, 54):
            c.drawString(115, y, "• " + line)
            y -= 24
        y -= 4

    c.setFont(font, 10)
    c.drawString(92, 77, "版权说明：图像版权归原作者及出版方。本图集仅用于研究检索、教学与个人学习，来源逐页标注。")
    c.drawString(92, 55, "生成日期：2026-08-12")
    c.showPage()


def draw_index(c: canvas.Canvas, selected: list[SelectedFigure], font: str) -> None:
    w, h = landscape(A4)
    per_page = 26
    per_col = 13
    for page_start in range(0, len(selected), per_page):
        chunk = selected[page_start : page_start + per_page]
        c.setFont(font, 20)
        c.drawString(45, h - 44, "目录")
        c.setFont(font, 9)
        c.drawRightString(w - 45, h - 42, f"{page_start + 1}–{page_start + len(chunk)} / {len(selected)}")
        for local_idx, fig in enumerate(chunk):
            col = local_idx // per_col
            row = local_idx % per_col
            x = 45 + col * (w / 2 - 12)
            y = h - 80 - row * 37
            global_idx = page_start + local_idx + 1
            c.setFont(font, 9.5)
            title = truncate(fig.title, 55)
            c.drawString(x, y, f"{global_idx:02d}. {title}")
            c.setFont(font, 8)
            c.drawString(x + 21, y - 13, f"{fig.venue} {fig.year} · {fig.category} · arXiv:{fig.arxiv_id}")
        c.showPage()


def draw_figure_page(c: canvas.Canvas, fig: SelectedFigure, idx: int, total: int, font: str) -> None:
    w, h = landscape(A4)
    left, right = 36, w - 36
    fit_title(c, f"{idx:02d}. {fig.title}", left, h - 34, right - left - 50, font, 14.5)
    c.setFont(font, 9)
    fig_label = f"Figure {fig.figure_number}" if fig.figure_number is not None else "Figure"
    meta = f"{fig.venue} {fig.year} · {fig.category} · {fig_label} · arXiv:{fig.arxiv_id} · {fig.extraction_method}"
    c.drawString(left, h - 53, truncate(meta, 145))
    c.drawRightString(right, h - 34, f"{idx}/{total}")
    c.setLineWidth(0.5)
    c.line(left, h - 62, right, h - 62)

    image = Image.open(fig.image_file)
    iw, ih = image.size
    box_x, box_y = left, 155
    box_w, box_h = right - left, h - 230
    scale = min(box_w / iw, box_h / ih)
    draw_w, draw_h = iw * scale, ih * scale
    x = box_x + (box_w - draw_w) / 2
    y = box_y + (box_h - draw_h) / 2
    c.drawImage(ImageReader(image), x, y, width=draw_w, height=draw_h, preserveAspectRatio=True, mask="auto")

    c.setLineWidth(0.4)
    c.line(left, 143, right, 143)
    c.setFont(font, 10.5)
    logic = "图中逻辑：" + logic_sentence(fig.category)
    logic_lines = wrap_chars(logic, 64)[:2]
    yy = 126
    for line in logic_lines:
        c.drawString(left, yy, line)
        yy -= 16

    c.setFont(font, 8.2)
    caption = truncate(fig.caption or "原图注未能从 HTML/PDF 文本中稳定提取。", 360)
    cap_lines = wrap_words("原图注：" + caption, 125)[:3]
    yy = 88
    for line in cap_lines:
        c.drawString(left, yy, line)
        yy -= 12

    source = f"来源：https://arxiv.org/abs/{fig.arxiv_id}"
    c.setFont(font, 8)
    c.drawString(left, 31, source)
    c.linkURL(source.replace("来源：", ""), (left, 24, left + 260, 40), relative=0)
    c.showPage()


def build_pdf(selected: list[SelectedFigure]) -> Path:
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    font = "STSong-Light"
    out_path = OUT / "embodied_world_model_visual_logic_atlas_52.pdf"
    c = canvas.Canvas(str(out_path), pagesize=landscape(A4), pageCompression=1)
    c.setTitle("具身智能与世界模型：数学逻辑图集")
    c.setAuthor("OpenAI research assistant")
    c.setSubject("Top-conference embodied AI and world-model method figures")
    draw_cover(c, len(selected), font)
    draw_index(c, selected, font)
    for idx, fig in enumerate(selected, start=1):
        draw_figure_page(c, fig, idx, len(selected), font)
    c.save()
    return out_path


def write_sources(selected: list[SelectedFigure], errors: list[dict], pdf_path: Path) -> None:
    csv_path = OUT / "sources.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index",
            "title",
            "venue",
            "year",
            "category",
            "arxiv_id",
            "figure_number",
            "caption",
            "source_url",
            "extraction_method",
        ])
        for idx, item in enumerate(selected, start=1):
            writer.writerow([
                idx,
                item.title,
                item.venue,
                item.year,
                item.category,
                item.arxiv_id,
                item.figure_number or "",
                item.caption,
                item.source_url,
                item.extraction_method,
            ])
    report = {
        "generated_at": "2026-08-12",
        "target_world": TARGET_WORLD,
        "target_embodied": TARGET_EMBODIED,
        "selected_count": len(selected),
        "pdf": str(pdf_path.relative_to(ROOT)),
        "pdf_size_bytes": pdf_path.stat().st_size,
        "selected": [asdict(x) for x in selected],
        "errors": errors,
    }
    (OUT / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def collect_group(papers: Iterable[Paper], quota: int, group_name: str, errors: list[dict]) -> list[SelectedFigure]:
    selected: list[SelectedFigure] = []
    for paper in papers:
        if len(selected) >= quota:
            break
        try:
            selected.append(extract_one(paper))
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED: {message}", flush=True)
            errors.append({"group": group_name, "title": paper.title, "arxiv_id": paper.arxiv_id, "error": message})
        time.sleep(0.9)
    return selected


def main() -> None:
    errors: list[dict] = []
    world = collect_group(WORLD_PAPERS, TARGET_WORLD, "world_model", errors)
    embodied = collect_group(EMBODIED_PAPERS, TARGET_EMBODIED, "embodied", errors)
    selected = world + embodied

    if len(selected) < MIN_TOTAL:
        raise RuntimeError(
            f"Only {len(selected)} figures were extracted; minimum is {MIN_TOTAL}. "
            f"World={len(world)}, embodied={len(embodied)}. See errors in workflow log."
        )

    pdf_path = build_pdf(selected)
    write_sources(selected, errors, pdf_path)
    print("\nBuild complete")
    print(f"  figures: {len(selected)}")
    print(f"  PDF: {pdf_path} ({pdf_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  sources: {OUT / 'sources.csv'}")
    print(f"  report: {OUT / 'build_report.json'}")


if __name__ == "__main__":
    main()
