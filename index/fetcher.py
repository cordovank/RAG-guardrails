"""
Create a JSONL data file by fetching and cleaning web pages from a list of URLs.
This is designed for predefined domains (e.g., "Travel & Places" - Wikivoyage - NYC).
"""

import os, json
import sys, time, argparse
from pathlib import Path

try:
    import trafilatura
except ImportError:
    print("Please install trafilatura: conda install trafilatura")
    sys.exit(1)


root = "."
domains_dir = f"{root}/index/data/domains"
data_dir = f"{root}/index/data"
eval_dir = f"{root}/eval/fixtures"
os.makedirs(data_dir, exist_ok=True)
os.makedirs(eval_dir, exist_ok=True)
default_domain = "wikivoyage_nynj"

# def create_url_file(domain: str):
#     """
#     Create a text file for given predefined domains (e.g., "Travel & Places" - Wikivoyage - NYC) 
#     containing a list of URLs to be used for data extraction.
#     """

#     domains = {
#         # URL list (from https://en.wikivoyage.org/wiki/New_York_City)
#         "wikivoyage_nynj": [
#             # New York
#             "https://en.wikivoyage.org/wiki/New_York_City",
#             "https://en.wikivoyage.org/wiki/Manhattan",
#             "https://en.wikivoyage.org/wiki/Brooklyn",
#             "https://en.wikivoyage.org/wiki/Queens",
#             "https://en.wikivoyage.org/wiki/The_Bronx",
#             "https://en.wikivoyage.org/wiki/Staten_Island",
#             # New Jersey
#             "https://en.wikivoyage.org/wiki/Jersey_City",
#             "https://en.wikivoyage.org/wiki/Hoboken",
#             "https://en.wikivoyage.org/wiki/Jersey_Shore",
#             "https://en.wikivoyage.org/wiki/Fort_Lee",
#             "https://en.wikivoyage.org/wiki/Newark",
#             "https://en.wikivoyage.org/wiki/Princeton",
#             "https://en.wikivoyage.org/wiki/Paterson",
#         ]
#     }

#     # write URLs to file
#     urls = domains[domain]
#     with open(f"{data_dir}/{domain}_urls.txt", "w", encoding="utf-8") as f:
#         f.write("\n".join(urls))

def fetch_and_clean(url: str, domain: str):
    """Fetch and clean the content from a given URL using trafilatura."""

    html = trafilatura.fetch_url(url, no_ssl=True)
    if not html:
        return None
    text = trafilatura.extract(
        html,
        include_comments=False,
        include_formatting=True,
        target_language="en"
    )
    if not text:
        return None
    
    title = url.rstrip('/').split('/')[-1] or "index"
    id = f"{domain}_{title}"
    return {"id": id, "title": title, "url": url, "text": text}

def main():
    ap = argparse.ArgumentParser(description="Fetch and clean web pages from a list of URLs. " \
    "Use predefined domains using --predefined_domain or custom URL files using --custom_urls and --custom_qa.")
    ap.add_argument("--predefined_domain", required=False, default=default_domain,
                    help=f"Choose from predefined URL files (e.g., '{default_domain}').")
    ap.add_argument("--custom_urls", required=False, default="",
                    help="Path to the text file containing URLs (one per line).")
    ap.add_argument("--custom_qa", required=False, default="",
                    help="Path to the text file containing QA pairs (semi-colon separated).")
    args = ap.parse_args()

    """ If custom URLs are provided, use them; otherwise, use predefined domain files """
    if os.path.isfile(args.custom_urls):
        domain_name = Path(args.custom_urls).stem
        urls = args.custom_urls
        qa_file = args.custom_qa

        # Check if files exist
        if not os.path.isfile(urls) or not os.path.isfile(qa_file):
            print(f"Predefined URL or QA file not found for: {domain_name}")
            print(f"Please add file(s) to: {domains_dir}/")
            sys.exit(1)
    else:
        domain_name = args.predefined_domain
        urls = f"{domains_dir}/{domain_name}_urls.txt"
        qa_file = f"{domains_dir}/{domain_name}_qa.txt"

        # Check if files exist
        if not os.path.isfile(urls) or not os.path.isfile(qa_file):
            print(f"Predefined URL or QA file not found for: {domain_name}")
            print(f"Please add file(s) to: {domains_dir}/")
            sys.exit(1)

    print(f"-- The following files will be used: \n"
            f"   - {urls} \n"
            f"   - {qa_file}")
    
    out_data = Path(f"{data_dir}/{domain_name}_data.jsonl")
    out_qa = Path(f"{eval_dir}/{domain_name}_qa.jsonl")


    """ Fetch and clean each URL, then write to a JSONL file """
    n_ok = 0
    with open(urls, "r", encoding="utf-8") as f, out_data.open("w", encoding="utf-8") as w:
        for line in f:
            url = line.strip()
            if not url:
                continue
            doc = fetch_and_clean(url, domain=domain_name)
            if doc:
                w.write(json.dumps(doc, ensure_ascii=False) + "\n")
                n_ok += 1
            time.sleep(0.5)
    print(f"-- Wrote {n_ok} docs to: {out_data}")

    
    """ Load predefined QA pairs and write to a JSONL file """
    n_ok = 0
    with open(qa_file, "r", encoding="utf-8") as f, out_qa.open("w", encoding="utf-8") as w:
        for line in f:
            qa_line = line.strip().split(";")
            question = qa_line[0].strip()
            tail = ";".join(qa_line[1:]).strip()
            docs = [d.strip().strip(',') for d in tail.split(",") if d.strip()]
            qa = {"question": question, "doc_contains": docs}
            w.write(json.dumps(qa, ensure_ascii=False) + "\n")
            n_ok += 1
            time.sleep(0.5)
    print(f"-- Wrote {n_ok} QA pairs to: {out_qa}")
    
    
    """ Add to temp files cleanup """
    with open("./temp_files.txt", "a") as tf:
        tf.write(f"{out_data}\n")
        tf.write(f"{out_qa}\n")


if __name__ == "__main__":
    main()
