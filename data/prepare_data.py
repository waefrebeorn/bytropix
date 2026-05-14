"""
Convert OpenWebMath sample into a training format compatible with WubuNestGPT.

Usage: python prepare_data.py
"""

import gzip
import json
import os
import sys

INPUT = "/home/wubu/bytropix/data/openwebmath_sample.jsonl.gz"
OUTPUT = "/home/wubu/bytropix/data/train_data.bin"
STATS = "/home/wubu/bytropix/data/dataset_stats.json"

def main():
    docs = []
    total_chars = 0
    with gzip.open(INPUT, 'rt', encoding='utf-8', errors='replace') as f:
        for line in f:
            doc = json.loads(line)
            text = doc.get('text', '')
            if len(text) > 500:  # filter very short snippets
                docs.append(text)
                total_chars += len(text)

    # Save concatenated text
    all_text = '\n\n---DOC---\n\n'.join(docs)
    
    # Save as UTF-8 bytes
    with open(OUTPUT, 'wb') as f:
        f.write(all_text.encode('utf-8'))
    
    stats = {
        'documents': len(docs),
        'total_chars': total_chars,
        'file_size_mb': os.path.getsize(OUTPUT) / (1024 * 1024),
        'avg_doc_len': total_chars // max(len(docs), 1),
    }
    
    with open(STATS, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"  Documents: {stats['documents']:,}")
    print(f"  Total chars: {stats['total_chars']:,}")
    print(f"  File size: {stats['file_size_mb']:.1f} MB")
    print(f"  Saved to: {OUTPUT}")


if __name__ == '__main__':
    main()
