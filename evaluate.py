import argparse
import json

from rag_core import RAGIndex

def run(index: RAGIndex, items: list[dict], k: int, hybrid: bool) -> tuple[float, float]:
    name_to_id = {v: k_ for k_, v in index.documents().items()}
    hits, rr = 0, 0.0
    for it in items:
        doc_ids = [name_to_id[it["pdf"]]] if it.get("pdf") in name_to_id else None
        results = index.search(it["question"], k=k, doc_ids=doc_ids, hybrid=hybrid)
        ranks = [r for r, c in enumerate(results, 1) if c.page in it["pages"]]
        if ranks:
            hits += 1
            rr += 1 / ranks[0]
        else:
            print(f"  MISS ({'hybrid' if hybrid else 'vector'}): {it['question']}")
    return hits / len(items), rr / len(items)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_file")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    items = json.load(open(args.eval_file))
    index = RAGIndex()
    for hybrid in (False, True):
        hit, mrr = run(index, items, args.k, hybrid)
        print(f"{'hybrid' if hybrid else 'vector':>7}: hit@{args.k} = {hit:.0%}   MRR = {mrr:.3f}")