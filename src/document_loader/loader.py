#loader.py
import os
from typing import List, Dict
from ..utils.cleaning import clean_text
from ..utils.hashing import sha256_text


def load_documents(folder: str) -> List[Dict]:
    """Load all .txt files from folder and return list of metadata dicts.

    Each dict:
      - doc_id (filename without ext)
      - text (cleaned)
      - hash (sha256)
      - length (number of chars)
      - filename (full path)
    """
    docs = []
    for root, _, files in os.walk(folder):
        for fname in sorted(files):
            if not fname.lower().endswith('.txt'):
                continue
            path = os.path.join(root, fname)
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                raw = f.read()
            text = clean_text(raw)
            doc_id = os.path.splitext(fname)[0]
            docs.append({
                'doc_id': doc_id,
                'text': text,
                'hash': sha256_text(text),
                'length': len(text),
                'filename': path,
            })
    return docs


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--folder', required=True)
    args = p.parse_args()
    docs = load_documents(args.folder)
    print(f'Loaded {len(docs)} documents')