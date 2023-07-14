#!/usr/bin/env python3
import re
import os
import sys
import json
import gzip

def generate_text_corpus(supervisions, outfile, outwords):
    """Write transcriptions to a unified corpus file.

    Write transcriptions from one or more supervision file to 
    a unified transcription file for BPE training.

    """
    words = set([])
    with open(outfile, "w") as ofp:
        for supervision in supervisions:
            with gzip.open(supervision, 'rt') as ifp:
                for entry in ifp:
                    entry = json.loads(entry.strip())
                    text = entry.get('text', '')
                    if not text == text.strip():
                        print(entry)
                    if not re.match(r"^\s*$", text):
                        tokens = re.split(r"\s+", text)
                        for token in tokens:
                            words.add(token)
                        print(text, file=ofp)

    words = ["<eps>", "<unk>"] + list(words) + ["#0", "<s>", "</s>"]
    with open(outwords, "w") as ofp:
        for idx,word in enumerate(words):
            print(f"{word} {idx}", file=ofp)
    
    return

if __name__ == "__main__":
    import argparse

    example = f"{sys.argv[0]} --text text.jsonl.gz -o corpus.txt"
    parser = argparse.ArgumentParser(description=example)
    parser.add_argument("--text", "-t", help="Supervisions in jsonl.gz format",
                        action="append", required=True)
    parser.add_argument("--outfile", "-o", help="Output text corpus file.",
                        required=True)
    parser.add_argument("--outwords", "-w", help="Output words file.",
                        required=True)
    args = parser.parse_args()

    generate_text_corpus(args.text, args.outfile, args.outwords)
