#!/usr/bin/env python3
from __future__ import annotations

import json
import mimetypes
import re
import time
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "gui_agent_rl_survey.md"
OUT = ROOT / "figures.json"
FIGURE_DIR = ROOT / "figures"

STRONG_KEYWORDS = [
    "pipeline",
    "framework",
    "architecture",
    "overview",
    "training framework",
    "overall framework",
    "overall training",
    "method overview",
    "system overview",
    "paradigm",
    "paradigms",
    "design",
]

WEAK_KEYWORDS = ["method", "workflow", "system", "training", "model", "agent", "gui"]

SEVERE_NEGATIVE_KEYWORDS = [
    "table",
    "results",
    "result",
    "performance",
    "benchmark",
    "quantitative",
    "comparison",
    "ablation",
    "dataset",
    "statistics",
    "gpu hours",
    "data scaling",
]

NEGATIVE_KEYWORDS = [
    "training dynamics",
    "domain distribution",
    "source distribution",
    "layout shift",
    "action space",
    "case study",
    "failure mode",
    "failure modes",
    "reward curve",
    "actor entropy",
]


def paper_links() -> list[dict[str, str]]:
    text = SOURCE.read_text(encoding="utf-8")
    rows = []
    for m in re.finditer(r"^###\s+(\d+)\.\s+(.+?)\n\n链接：(.+)$", text, re.M):
        number = int(m.group(1))
        title = m.group(2).strip()
        links = re.findall(r"https://arxiv\.org/abs/(\d{4}\.\d{4,5})", m.group(3))
        if links:
            rows.append({"number": number, "title": title, "arxiv": links[0]})
        else:
            rows.append({"number": number, "title": title, "arxiv": ""})
    return rows


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def caption_score(caption: str, area: int) -> int:
    lower = caption.lower()
    score = 0
    for i, kw in enumerate(STRONG_KEYWORDS):
        if kw in lower:
            score += 55 - i
    for i, kw in enumerate(WEAK_KEYWORDS):
        if kw in lower:
            score += 16 - i
    if re.search(r"\bfigure\s*1\b|\bfig\.\s*1\b", lower):
        score += 16
    if re.search(r"\bfigure\s*2\b|\bfig\.\s*2\b", lower):
        score += 8
    if "teaser" in lower:
        score += 12
    for kw in SEVERE_NEGATIVE_KEYWORDS:
        if kw in lower:
            score -= 130
    for kw in NEGATIVE_KEYWORDS:
        if kw in lower:
            score -= 36
    if lower.startswith("table"):
        score -= 90
    if area:
        score += min(area // 50000, 12)
    return score


def resolve_image_src(arxiv_id: str, final_url: str, raw_src: str) -> str:
    src = raw_src.strip()
    if re.match(r"^\d{4}\.\d{4,5}v\d+/", src):
        return urljoin("https://arxiv.org/html/", src)
    if re.match(r"^\d{4}\.\d{4,5}/", src):
        return urljoin("https://arxiv.org/html/", src)
    return urljoin(f"{final_url}/", src)


def candidates(arxiv_id: str) -> list[dict[str, str | int]]:
    if not arxiv_id:
        return []
    url = f"https://arxiv.org/html/{arxiv_id}"
    req = Request(url, headers={"User-Agent": "codex"})
    try:
        with urlopen(req, timeout=25) as res:
            final_url = res.geturl().rstrip("/")
            html = res.read().decode("utf-8", "ignore")
    except Exception:
        return []
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for fig in soup.select("figure"):
        img = fig.find("img")
        if not img or not img.get("src"):
            continue
        caption = clean(fig.get_text(" "))
        src = resolve_image_src(arxiv_id, final_url, img["src"])
        width = int(img.get("width") or 0)
        height = int(img.get("height") or 0)
        area = width * height
        score = caption_score(caption, area)
        out.append(
            {
                "src": src,
                "caption": caption[:900],
                "score": score,
                "width": width,
                "height": height,
            }
        )
    out.sort(key=lambda x: int(x["score"]), reverse=True)
    return out[:5]


def local_extension(url: str, content_type: str) -> str:
    ext = Path(url.split("?", 1)[0]).suffix.lower()
    if ext in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
        return ext
    guessed = mimetypes.guess_extension(content_type.split(";", 1)[0].strip())
    if guessed in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
        return guessed
    return ".png"


def download_selected(number: int, cands: list[dict[str, str | int]]) -> None:
    FIGURE_DIR.mkdir(exist_ok=True)
    for old in FIGURE_DIR.glob(f"{number:02d}.*"):
        old.unlink()
    if not cands:
        return
    selected = cands[0]
    if int(selected.get("score") or 0) < 25:
        return
    req = Request(str(selected["src"]), headers={"User-Agent": "codex"})
    try:
        with urlopen(req, timeout=25) as res:
            content_type = res.headers.get("Content-Type", "")
            payload = res.read(8_000_001)
            if len(payload) > 8_000_000 or not content_type.startswith("image/"):
                return
    except Exception:
        return
    ext = local_extension(str(selected["src"]), content_type)
    local = FIGURE_DIR / f"{number:02d}{ext}"
    local.write_bytes(payload)
    selected["local_src"] = f"figures/{local.name}"


def apply_manual_overrides(number: int, cands: list[dict[str, str | int]]) -> None:
    if number == 12:
        for cand in cands:
            caption = str(cand.get("caption") or "").lower()
            if "overview of ui-agile" in caption:
                cand["score"] = int(cand.get("score") or 0) + 220
    if number == 37:
        for cand in cands:
            caption = str(cand.get("caption") or "").lower()
            if "propose, simulate, select" in caption:
                cand["score"] = int(cand.get("score") or 0) + 220


def main() -> None:
    data = {}
    for row in paper_links():
        print(row["number"], row["arxiv"], row["title"])
        cands = candidates(row["arxiv"])
        apply_manual_overrides(int(row["number"]), cands)
        cands.sort(key=lambda x: int(x["score"]), reverse=True)
        download_selected(int(row["number"]), cands)
        data[str(row["number"])] = {
            "title": row["title"],
            "arxiv": row["arxiv"],
            "candidates": cands,
        }
        time.sleep(0.35)
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
