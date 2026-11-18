"""Download open article full text for a pinned five-question LitQA2 pilot."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

ELIGIBILITY = Path(__file__).with_name("europe-pmc-open-fulltext.json")
FULL_TEXT_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"


async def fetch_article(
    client: httpx.AsyncClient, record: dict[str, str], output_dir: Path
) -> dict[str, str | int]:
    """Save the source XML and body text, keeping provenance for the local index."""
    url = FULL_TEXT_URL.format(pmcid=record["pmcid"])
    response = await client.get(url)
    response.raise_for_status()
    root = ET.fromstring(response.content)
    body = root.find(".//body")
    if body is None:
        raise ValueError(f"No article body found for {record['pmcid']}")
    article_title = root.find(".//article-title")
    title = (
        " ".join(article_title.itertext()).strip()
        if article_title is not None
        else record["doi"]
    )
    text = re.sub(r"\s+", " ", " ".join(body.itertext())).strip()
    if len(text) < 5_000:
        raise ValueError(f"Article body is unexpectedly short for {record['pmcid']}")
    source_dir = output_dir / "source_xml"
    source_dir.mkdir(exist_ok=True)
    (source_dir / f"{record['id']}.xml").write_bytes(response.content)
    (output_dir / f"{record['id']}.txt").write_text(
        f"Title: {title}\nDOI: {record['doi']}\n\n{text}\n", encoding="utf-8"
    )
    return {
        **record,
        "title": title,
        "source_url": url,
        "xml_sha256": hashlib.sha256(response.content).hexdigest(),
        "body_characters": len(text),
    }


async def prepare(questions: Path, output_dir: Path) -> list[dict[str, str | int]]:
    selected = {
        json.loads(line)["question_id"]
        for line in questions.read_text(encoding="utf-8").splitlines()
    }
    eligible = {
        record["id"]: record
        for record in json.loads(ELIGIBILITY.read_text(encoding="utf-8"))["records"]
    }
    if len(selected) != 5 or not selected <= set(eligible):
        raise ValueError("Expected five pinned questions with open full-text sources")
    output_dir.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=45) as client:
        records = await asyncio.gather(
            *(fetch_article(client, eligible[qid], output_dir) for qid in sorted(selected))
        )
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["file_location", "docname", "citation", "title", "doi"]
        )
        writer.writeheader()
        for record in records:
            writer.writerow({
                "file_location": f"{record['id']}.txt",
                "docname": str(record["id"])[:8],
                "citation": f"{record['title']} ({record['doi']})",
                "title": record["title"],
                "doi": record["doi"],
            })
    (output_dir / "sources.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    records = asyncio.run(prepare(args.questions, args.output_dir))
    print(f"Prepared {len(records)} open full-text articles in {args.output_dir}")


if __name__ == "__main__":
    main()
