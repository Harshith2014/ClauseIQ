"""Golden dataset for RAG evaluation.

Curated ground-truth QA pairs built from the 4 real legal documents:
  - exhibit101.pdf          (Verdisys / O'Keefe Executive Employment Agreement)
  - EMPLOYMENT-AGREEMENT.pdf  (Jaipur India template)
  - EMPLOYMENT_AGREEMENT (3).pdf  (LetsVenture India template)
  - sample-standard_contract.pdf  (Australia / Purple Nike)

Also includes 3 unanswerable questions (no answer exists in any document)
to test hallucination resistance.

Usage:
    from eval.golden_dataset import load_golden, save_golden, generate_llm_pairs
    pairs = load_golden()                       # load existing golden.json
    pairs = generate_llm_pairs(chunks, n=15)    # LLM-generate additional pairs
    save_golden(pairs)                          # write / overwrite golden.json
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import GOLDEN_DATASET_PATH, OLLAMA_MODEL, OLLAMA_BASE_URL

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class GoldenPair:
    id: str
    question: str
    ground_truth: str
    source_files: list[str]           # which PDFs contain the answer
    source_pages: list[int]           # 0-based page numbers
    category: str                     # salary | termination | non_compete | etc.
    unanswerable: bool = False        # True -> correct answer is "not found"
    toc_artifact_risk: bool = False   # True -> TOC chunk may appear; flag if cited
    multi_jurisdiction: bool = False  # True -> answer must cite ALL listed sources
    expected_sources: list[str] = field(default_factory=list)  # for multi_jurisdiction
    notes: str = ""


# ---------------------------------------------------------------------------
# Curated ground-truth pairs (hand-verified against real documents)
# ---------------------------------------------------------------------------

CURATED_PAIRS: list[GoldenPair] = [
    # ------------------------------------------------------------------
    # SALARY / COMPENSATION
    # ------------------------------------------------------------------
    GoldenPair(
        id="q01",
        question="What is the annual base salary in the Verdisys employment agreement?",
        ground_truth=(
            "$175,000 in year 1, $195,000 in year 2, and $215,000 in year 3."
        ),
        source_files=["exhibit101.pdf"],
        source_pages=[1],
        category="salary",
    ),
    GoldenPair(
        id="q02",
        question="What is the sign-on bonus amount in the Verdisys agreement?",
        ground_truth="A one-time sign-on bonus of $40,000.",
        source_files=["exhibit101.pdf"],
        source_pages=[1],
        category="salary",
    ),
    GoldenPair(
        id="q03",
        question="What is the car allowance provided to the employee in exhibit101?",
        ground_truth="$1,000 per month.",
        source_files=["exhibit101.pdf"],
        source_pages=[1],
        category="compensation",
    ),
    GoldenPair(
        id="q04",
        question="What is the maximum annual performance bonus as a percentage of base salary?",
        ground_truth="Up to 50% of base salary.",
        source_files=["exhibit101.pdf"],
        source_pages=[1],
        category="salary",
    ),
    # ------------------------------------------------------------------
    # TERMINATION / NOTICE
    # ------------------------------------------------------------------
    GoldenPair(
        id="q05",
        question="How many days notice is required before the end of the term to prevent renewal in the Verdisys agreement?",
        ground_truth="60 days written notice prior to the end of the then-current term.",
        source_files=["exhibit101.pdf"],
        source_pages=[0],
        category="termination",
    ),
    GoldenPair(
        id="q06",
        question="What is the termination notice period in the LetsVenture employment agreement?",
        ground_truth=(
            "Two months written notice, or two months salary in lieu of notice."
        ),
        source_files=["EMPLOYMENT_AGREEMENT (3).pdf"],
        source_pages=[1],
        category="termination",
    ),
    GoldenPair(
        id="q07",
        question="Can the employer in the Jaipur agreement terminate immediately without notice?",
        ground_truth=(
            "Yes. The employer can terminate the employee immediately without notice "
            "for breach or misconduct."
        ),
        source_files=["EMPLOYMENT-AGREEMENT.pdf"],
        source_pages=[2, 3],
        category="termination",
    ),
    # ------------------------------------------------------------------
    # SEVERANCE
    # ------------------------------------------------------------------
    GoldenPair(
        id="q08",
        question="What severance is provided in the Verdisys agreement upon termination without cause?",
        ground_truth=(
            "Base Compensation for the remaining term of the agreement (up to 6 months), "
            "paid as a lump sum within 30 days, plus COBRA continuation for 6 months "
            "and accelerated vesting of stock options."
        ),
        source_files=["exhibit101.pdf"],
        source_pages=[2],
        category="severance",
    ),
    GoldenPair(
        id="q09",
        question="Is severance paid if the employee is terminated in the Jaipur agreement for not meeting performance criteria?",
        ground_truth=(
            "No. The Jaipur agreement explicitly states that no severance is paid "
            "if the employee is terminated for breach or failure to meet performance criteria."
        ),
        source_files=["EMPLOYMENT-AGREEMENT.pdf"],
        source_pages=[3],
        category="severance",
    ),
    # ------------------------------------------------------------------
    # NON-COMPETE
    # ------------------------------------------------------------------
    GoldenPair(
        id="q10",
        question="What is the non-compete duration in the Jaipur employment agreement?",
        ground_truth=(
            "Up to 3 years after termination, with damages governed by sections 73-74 "
            "of the Indian Contract Act."
        ),
        source_files=["EMPLOYMENT-AGREEMENT.pdf"],
        source_pages=[3, 4],
        category="non_compete",
    ),
    # ------------------------------------------------------------------
    # GOVERNING LAW  (toc_artifact_risk = True for q11)
    # ------------------------------------------------------------------
    GoldenPair(
        id="q11",
        question="What governing law applies to the Verdisys employment agreement?",
        ground_truth="The laws of the State of Texas.",
        source_files=["exhibit101.pdf"],
        source_pages=[3],
        category="governing_law",
        toc_artifact_risk=False,  # exhibit101 only; no TOC issue on that doc
    ),
    GoldenPair(
        id="q12",
        question="What governing law applies to the sample standard contract (Australian agreement)?",
        ground_truth=(
            "The laws of Victoria, Australia."
        ),
        source_files=["sample-standard_contract.pdf"],
        source_pages=[9],
        category="governing_law",
        toc_artifact_risk=True,
        notes=(
            "The sample-standard_contract TOC on p.1 lists 'Governing Law' as a section "
            "header. If the answer cites page 1 / the TOC entry rather than the actual "
            "clause (p.9-10), flag as a TOC artifact faithfulness failure."
        ),
    ),
    # ------------------------------------------------------------------
    # JURISDICTION — multi-jurisdiction query
    # ------------------------------------------------------------------
    GoldenPair(
        id="q13",
        question="Which jurisdiction or governing law applies to each of the employment agreements in the dataset?",
        ground_truth=(
            "exhibit101.pdf: State of Texas (USA). "
            "EMPLOYMENT-AGREEMENT.pdf: Jaipur, India. "
            "EMPLOYMENT_AGREEMENT (3).pdf: India (courts jurisdiction blank). "
            "sample-standard_contract.pdf: Victoria, Australia."
        ),
        source_files=[
            "exhibit101.pdf",
            "EMPLOYMENT-AGREEMENT.pdf",
            "EMPLOYMENT_AGREEMENT (3).pdf",
            "sample-standard_contract.pdf",
        ],
        source_pages=[3, 5, 3, 9],
        category="governing_law",
        multi_jurisdiction=True,
        expected_sources=[
            "exhibit101.pdf",
            "EMPLOYMENT-AGREEMENT.pdf",
            "sample-standard_contract.pdf",
        ],
        notes=(
            "Full credit only if the answer cites Texas (exhibit101), Jaipur (EMPLOYMENT-AGREEMENT), "
            "and Victoria/Australia (sample-standard_contract). Missing any document = incomplete."
        ),
    ),
    # ------------------------------------------------------------------
    # CONFIDENTIALITY / IP
    # ------------------------------------------------------------------
    GoldenPair(
        id="q14",
        question="What happens to intellectual property created during employment in the Australian contract?",
        ground_truth=(
            "The employee assigns all intellectual property rights to the employer "
            "for any IP created in connection with their duties, including work done "
            "outside business hours if employer resources were used."
        ),
        source_files=["sample-standard_contract.pdf"],
        source_pages=[5, 6],
        category="intellectual_property",
    ),
    GoldenPair(
        id="q15",
        question="What happens to intellectual property created during employment in the Jaipur agreement?",
        ground_truth=(
            "All software and hardware created during employment is the exclusive "
            "property of the company."
        ),
        source_files=["EMPLOYMENT-AGREEMENT.pdf"],
        source_pages=[4],
        category="intellectual_property",
    ),
    GoldenPair(
        id="q16",
        question="Do confidentiality obligations survive termination in the Australian contract?",
        ground_truth=(
            "Yes. Confidentiality obligations explicitly survive termination of the agreement."
        ),
        source_files=["sample-standard_contract.pdf"],
        source_pages=[4, 5],
        category="confidentiality",
    ),
    # ------------------------------------------------------------------
    # PROBATIONARY PERIOD
    # ------------------------------------------------------------------
    GoldenPair(
        id="q17",
        question="Is there a probationary period in the Verdisys agreement?",
        ground_truth=(
            "The Verdisys agreement does not specify a probationary period."
        ),
        source_files=["exhibit101.pdf"],
        source_pages=[0],
        category="probation",
    ),
    # ------------------------------------------------------------------
    # VACATION / LEAVE
    # ------------------------------------------------------------------
    GoldenPair(
        id="q18",
        question="How many weeks of paid vacation does the employee receive in the Verdisys agreement?",
        ground_truth="No less than 4 weeks paid vacation per calendar year.",
        source_files=["exhibit101.pdf"],
        source_pages=[2],
        category="leave",
    ),
    GoldenPair(
        id="q19",
        question="How is leave governed in the LetsVenture employment agreement?",
        ground_truth=(
            "Leave and holidays are governed by the company's policies, "
            "which are not specified within the agreement itself."
        ),
        source_files=["EMPLOYMENT_AGREEMENT (3).pdf"],
        source_pages=[1, 2],
        category="leave",
    ),
    # ------------------------------------------------------------------
    # PENALTY / BREACH
    # ------------------------------------------------------------------
    GoldenPair(
        id="q20",
        question="What financial penalty applies if the employee leaves before 2 years in the Jaipur agreement?",
        ground_truth="A penalty of Rs. 2 lakhs (Rs. 200,000).",
        source_files=["EMPLOYMENT-AGREEMENT.pdf"],
        source_pages=[3],
        category="penalties",
    ),
    # ------------------------------------------------------------------
    # DISPUTE RESOLUTION
    # ------------------------------------------------------------------
    GoldenPair(
        id="q21",
        question="How are disputes resolved under the Verdisys employment agreement?",
        ground_truth=(
            "Through binding arbitration under the rules of the American Arbitration "
            "Association (AAA)."
        ),
        source_files=["exhibit101.pdf"],
        source_pages=[3, 4],
        category="dispute_resolution",
    ),
    GoldenPair(
        id="q22",
        question="Who has final authority over administrative disputes in the Jaipur employment agreement?",
        ground_truth=(
            "The company's board of directors; the board's decision is final and binding "
            "for internal administrative matters, with courts of Jaipur having jurisdiction."
        ),
        source_files=["EMPLOYMENT-AGREEMENT.pdf"],
        source_pages=[5],
        category="dispute_resolution",
    ),
    # ------------------------------------------------------------------
    # UNANSWERABLE QUESTIONS  (system must output "not found" / "cannot determine")
    # ------------------------------------------------------------------
    GoldenPair(
        id="u01",
        question="What is the employee stock option strike price in the Verdisys agreement?",
        ground_truth=(
            "The documents do not specify the stock option strike price. "
            "This information is not found in any of the indexed agreements."
        ),
        source_files=[],
        source_pages=[],
        category="compensation",
        unanswerable=True,
        notes=(
            "The Verdisys agreement mentions accelerated vesting of options but does not "
            "state a strike price. A correct answer must acknowledge the information is absent."
        ),
    ),
    GoldenPair(
        id="u02",
        question="What is the health insurance premium contribution by the employer?",
        ground_truth=(
            "None of the indexed documents specify a health insurance premium contribution. "
            "This information is not found in the dataset."
        ),
        source_files=[],
        source_pages=[],
        category="benefits",
        unanswerable=True,
        notes=(
            "None of the 4 documents mention employer health insurance contributions. "
            "COBRA is mentioned in exhibit101 but only as a severance benefit, not a "
            "regular premium. Hallucinating a number here is a faithfulness failure."
        ),
    ),
    GoldenPair(
        id="u03",
        question="What is the employee's job title in the LetsVenture employment agreement?",
        ground_truth=(
            "The LetsVenture agreement template does not specify the employee's job title; "
            "the field is left blank in the document."
        ),
        source_files=["EMPLOYMENT_AGREEMENT (3).pdf"],
        source_pages=[0],
        category="employment_terms",
        unanswerable=True,
        notes=(
            "LetsVenture agreement is a template with blanks. Job title field is blank. "
            "Model must say 'not specified' rather than invent a title."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def save_golden(pairs: list[GoldenPair], path: Path = GOLDEN_DATASET_PATH) -> None:
    """Serialise golden pairs to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(p) for p in pairs]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(pairs)} golden pairs to {path}")


def load_golden(path: Path = GOLDEN_DATASET_PATH) -> list[GoldenPair]:
    """Load golden pairs from JSON. Falls back to CURATED_PAIRS if file absent."""
    if not path.exists():
        print(f"[golden_dataset] {path} not found — returning curated pairs only.")
        return list(CURATED_PAIRS)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pairs = []
    for d in data:
        d.setdefault("toc_artifact_risk", False)
        d.setdefault("multi_jurisdiction", False)
        d.setdefault("expected_sources", [])
        d.setdefault("unanswerable", False)
        d.setdefault("notes", "")
        pairs.append(GoldenPair(**d))
    return pairs


# ---------------------------------------------------------------------------
# LLM-generated additional pairs
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str, model: str = OLLAMA_MODEL, base_url: str = OLLAMA_BASE_URL) -> str:
    """Minimal synchronous Ollama call (no extra deps)."""
    import urllib.request
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 512},
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["response"].strip()


def generate_llm_pairs(
    chunks,  # list[Document]
    n: int = 15,
    start_id: int = 100,
    model: str = OLLAMA_MODEL,
) -> list[GoldenPair]:
    """Use Ollama to generate n additional QA pairs from random chunks.

    Each chunk is sent to the LLM with a prompt asking for one question +
    factual answer grounded strictly in the chunk text.  Failures are skipped.
    """
    import random

    PROMPT_TMPL = (
        "You are building a QA evaluation dataset for a legal document RAG system.\n"
        "Read the following clause from a legal document and generate exactly ONE "
        "question-answer pair that tests specific factual retrieval.\n\n"
        "Rules:\n"
        "- The question must be answerable solely from this clause.\n"
        "- The answer must be a direct factual statement (1-3 sentences max).\n"
        "- Do NOT ask vague or generic questions.\n"
        "- Output ONLY valid JSON on a single line: "
        '{"question": "...", "answer": "..."}\n\n'
        "Clause (from {source}, page {page}):\n{text}\n\nJSON:"
    )

    sampled = random.sample(chunks, min(n * 2, len(chunks)))  # oversample, skip failures
    pairs: list[GoldenPair] = []
    pid = start_id

    for doc in sampled:
        if len(pairs) >= n:
            break
        text = doc.page_content.strip()
        if len(text) < 80:
            continue
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", 0)
        prompt = PROMPT_TMPL.format(source=source, page=page, text=text[:600])
        try:
            raw = _call_ollama(prompt, model=model)
            # Extract JSON — find first { ... } block
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start == -1 or end == 0:
                continue
            obj = json.loads(raw[start:end])
            q = obj.get("question", "").strip()
            a = obj.get("answer", "").strip()
            if not q or not a or len(q) < 10 or len(a) < 5:
                continue
            pairs.append(GoldenPair(
                id=f"llm{pid:03d}",
                question=q,
                ground_truth=a,
                source_files=[source],
                source_pages=[page],
                category="llm_generated",
            ))
            pid += 1
            time.sleep(0.2)  # gentle rate limiting
        except Exception as exc:
            print(f"  [generate_llm_pairs] skipped chunk ({exc})")
            continue

    print(f"Generated {len(pairs)} LLM pairs.")
    return pairs


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from ingestion.pdf_loader import load_pdfs
    from ingestion.chunker import chunk_documents
    from config import RAW_DIR

    parser = argparse.ArgumentParser(description="Build golden evaluation dataset")
    parser.add_argument("--llm-extra", type=int, default=10,
                        help="Number of LLM-generated pairs to add (default: 10)")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM generation; save curated pairs only")
    args = parser.parse_args()

    pairs = list(CURATED_PAIRS)
    print(f"Curated pairs: {len(pairs)}")

    if not args.no_llm and args.llm_extra > 0:
        print(f"Generating {args.llm_extra} LLM pairs from real documents...")
        docs = load_pdfs(RAW_DIR)
        chunks = chunk_documents(docs, chunk_size=512)
        extra = generate_llm_pairs(chunks, n=args.llm_extra, start_id=100)
        pairs.extend(extra)

    save_golden(pairs)
    print(f"Total: {len(pairs)} pairs  ({sum(1 for p in pairs if p.unanswerable)} unanswerable)")
