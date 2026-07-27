"""Full-corpus BM25 retriever over rag.jsonl paragraph chunks."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class Chunk:
    paper_id: str
    paper_name: str
    section: str
    text: str
    chunk_id: str

    def to_dict(self) -> dict:
        return asdict(self)


def build_chunks(rag_dataset: list[dict]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for paper in rag_dataset:
        paper_id = str(paper.get("id", ""))
        paper_name = paper.get("paper_name") or paper.get("title") or paper_id
        sections = paper.get("section_name") or []
        paragraphs = paper.get("paragraphs") or []
        for s_idx, (section, paras) in enumerate(zip(sections, paragraphs)):
            section_name = (section or "Section").strip()
            if not isinstance(paras, list):
                continue
            for p_idx, para in enumerate(paras):
                if not isinstance(para, str):
                    continue
                text = para.strip()
                if len(text) < 40:
                    continue
                chunks.append(
                    Chunk(
                        paper_id=paper_id,
                        paper_name=paper_name,
                        section=section_name,
                        text=text,
                        chunk_id=f"{paper_id}:{s_idx}:{p_idx}",
                    )
                )
    return chunks


class CorpusRetriever:
    def __init__(self, rag_dataset: list[dict]):
        self.chunks = build_chunks(rag_dataset)
        if not self.chunks:
            raise ValueError("No chunks built from rag dataset")
        tokenized = [tokenize(c.text) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        scores = self.bm25.get_scores(tokenize(query))
        # argsort descending without numpy dependency on order quirks
        ranked = sorted(
            range(len(scores)),
            key=lambda i: float(scores[i]),
            reverse=True,
        )[:top_k]
        results = []
        for rank, idx in enumerate(ranked):
            item = self.chunks[idx].to_dict()
            item["score"] = float(scores[idx])
            item["rank"] = rank
            results.append(item)
        return results


def format_retrieved_context(retrieved: list[dict], max_chars: int = 6000) -> str:
    parts: list[str] = []
    total = 0
    for i, hit in enumerate(retrieved, start=1):
        block = (
            f"[{i}] Paper: {hit['paper_name']} | Section: {hit['section']}\n"
            f"{hit['text']}"
        )
        if total + len(block) + 2 > max_chars:
            remain = max_chars - total - 2
            if remain > 100:
                parts.append(block[:remain] + "…")
            break
        parts.append(block)
        total += len(block) + 2
    return "\n\n".join(parts)
