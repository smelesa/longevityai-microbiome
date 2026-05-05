#!/usr/bin/env python3
"""
LongevityAI-Microbiome — Phase 2: PubMed Scraping (using curl)
NCBI E-utilities API — free, rate limit: 3 req/sec
"""

import subprocess
import time
import json
import re
import pandas as pd
from pathlib import Path

RATE_LIMIT_DELAY = 0.34
BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

def curl_get(url, params=None):
    """Make GET request via curl"""
    cmd = ["curl", "-s", "-G", url]
    if params:
        for k, v in params.items():
            cmd.extend(["--data-urlencode", f"{k}={v}"])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return result.stdout

def esearch(query, max_results=100):
    url = f"{BASE_URL}/esearch.fcgi"
    xml = curl_get(url, {
        "db": "pubmed",
        "term": query,
        "retmax": str(max_results),
        "retmode": "json",
        "datetype": "pdat",
        "sort": "relevance"
    })
    try:
        data = json.loads(xml)
        return data.get("esearchresult", {}).get("idlist", [])
    except:
        return []

def efetch(pmids):
    url = f"{BASE_URL}/efetch.fcgi"
    ids = ",".join(pmids)
    xml = curl_get(url, {
        "db": "pubmed",
        "id": ids,
        "rettype": "abstract",
        "retmode": "xml"
    })
    return xml

def parse_abstracts_xml(xml_text):
    articles = []
    article_blocks = re.split(r'<PubmedArticle>', xml_text)
    
    for block in article_blocks:
        if '<PMID' not in block:
            continue
        
        pmid_match = re.search(r'<PMID[^>]*>([^<]+)</PMID>', block)
        if not pmid_match:
            continue
        pmid = pmid_match.group(1).strip()
        
        title_match = re.search(r'<ArticleTitle>([^<]+)</ArticleTitle>', block)
        title = title_match.group(1).strip() if title_match else ""
        
        # Abstract may have multiple AbstractText tags
        abstract_matches = re.findall(r'<AbstractText[^>]*>([^<]+)</AbstractText>', block)
        abstract = " ".join(abstract_matches[:3]).strip()
        
        author_blocks = re.findall(r'<Author[^>]*>.*?<LastName>([^<]+)</LastName>', block, re.DOTALL)
        authors = ", ".join(author_blocks[:5])
        
        journal_match = re.search(r'<Journal>.*?<Title>([^<]+)</Title>', block, re.DOTALL)
        journal = journal_match.group(1).strip() if journal_match else ""
        
        year_match = re.search(r'<PubDate>.*?<Year>(\d{4})</Year>', block, re.DOTALL)
        year = year_match.group(1).strip() if year_match else ""
        
        articles.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "journal": journal,
            "year": year
        })
    
    return articles

SEARCH_QUERIES = [
    ("microbiome longevity", 100),
    ("Akkermansia muciniphila longevity", 80),
    ("gut microbiome aging clock", 60),
    ("butyrate producing bacteria longevity human", 60),
    ("fecal microbiota transplantation aging", 50),
    ("Bifidobacterium longevity", 50),
    ("microbiome centenarians gut", 40),
    ("gut brain axis longevity", 40),
]

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

from datetime import datetime
print("=" * 60)
print("LongevityAI-Microbiome — Phase 2: PubMed Scraping")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

all_articles = []
seen_pmids = set()

for query, max_results in SEARCH_QUERIES:
    print(f"[{query[:45]:45}] max={max_results}...", end=" ", flush=True)
    
    try:
        pmids = esearch(query, max_results)
        print(f"found {len(pmids)} PMIDs", end=" ")
        
        if not pmids:
            print()
            continue
        
        for i in range(0, len(pmids), 50):
            batch = pmids[i:i+50]
            new_only = [p for p in batch if p not in seen_pmids]
            
            if not new_only:
                continue
            
            time.sleep(RATE_LIMIT_DELAY)
            xml = efetch(new_only)
            articles = parse_abstracts_xml(xml)
            
            for art in articles:
                if art["pmid"] not in seen_pmids:
                    art["search_query"] = query
                    all_articles.append(art)
                    seen_pmids.add(art["pmid"])
            
            print(f"→{len(all_articles)}", end=" ", flush=True)
    
    except Exception as e:
        print(f"ERROR: {e}")
    
    print()

print(f"\n✅ Total unique articles: {len(all_articles)}")

if all_articles:
    df = pd.DataFrame(all_articles)
    df.to_json(PROCESSED_DIR / "pubmed_microbiome_articles.json", orient="records", indent=2)
    df.to_csv(PROCESSED_DIR / "pubmed_microbiome_articles.csv", index=False)
    print(f"\nSaved:")
    print(f"  - pubmed_microbiome_articles.json")
    print(f"  - pubmed_microbiome_articles.csv")
    
    print("\n📄 Top articles by year:")
    recent = df[df['year'].astype(str).str.isdigit()].sort_values('year', ascending=False).head(5)
    for _, r in recent.iterrows():
        print(f"  [{r['year']}] {r['title'][:80]}")
else:
    print("⚠️ No articles collected!")

print(f"\n✅ Phase 2 COMPLETE at {datetime.now().strftime('%H:%M:%S')}")