"""Phase 0: one gene vocabulary for every panel.

Token ids start from CellWorld's 19,228-entry list (so its gene table can be loaded as-is) and new genes are appended. Every
panel gene is first mapped to its current HGNC-approved symbol (approved > previous > alias; an alias shared by several genes
is not used), then to a token: CellWorld's own entry if it names the same gene, else a new token.

usage: python -m bento_jepa.vocab HGNC_TSV CELLWORLD_GENE_VOCAB_JSON OUT_DIR PANEL_NAME=GENES_JSON [PANEL_NAME=GENES_JSON ...]
writes OUT_DIR/vocab.json and OUT_DIR/panel_<name>.json
"""
import sys, json, csv, collections, os


OVERRIDES = {"QARS": "QARS1"}   # HGNC lists QARS as a previous symbol of more than one gene; QARS -> QARS1 is the 2020 rename


def hgnc_index(path):
    """symbol -> approved symbol, using approved, previous and (unambiguous) alias symbols."""
    approved, prev, alias = {}, collections.defaultdict(set), collections.defaultdict(set)
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            s = r["symbol"]; approved[s.upper()] = s
            for p in filter(None, r.get("prev_symbol", "").strip('"').split("|")): prev[p.upper()].add(s)
            for a in filter(None, r.get("alias_symbol", "").strip('"').split("|")): alias[a.upper()].add(s)

    def resolve(g):
        u = g.upper()
        if g in OVERRIDES: return OVERRIDES[g], "override"
        if u in approved: return approved[u], "approved"
        if len(prev[u]) == 1: return next(iter(prev[u])), "previous"
        if len(alias[u]) == 1: return next(iter(alias[u])), "alias"
        return None, "ambiguous" if (prev[u] or alias[u]) else "unknown"
    return resolve


def main(hgnc, cw_vocab, out, panels):
    resolve = hgnc_index(hgnc); cw = json.load(open(cw_vocab)); tokens = [None] * len(cw)
    for g, i in cw.items(): tokens[i] = g
    assert all(t is not None for t in tokens) and tokens[0] == "<pad>"
    by_approved = {}
    for i, g in enumerate(tokens[1:], 1):
        a, _ = resolve(g); by_approved.setdefault(a or g, i)          # CellWorld's genes, keyed by approved symbol
    os.makedirs(out, exist_ok=True); report = {}
    for name, path in panels:
        genes = json.load(open(path)); rows = []
        for g in genes:
            a, how = resolve(g); key = a or g
            if key in by_approved: tid, src = by_approved[key], "cellworld"
            else: tokens.append(key); tid = by_approved[key] = len(tokens) - 1; src = "new"
            rows.append(dict(panel_symbol=g, approved=a, hgnc_match=how, token=tid, token_source=src))
        json.dump(rows, open(os.path.join(out, f"panel_{name}.json"), "w"), indent=0)
        c = collections.Counter((r["hgnc_match"], r["token_source"]) for r in rows)
        report[name] = dict(n=len(genes), by_match={f"{k[0]}/{k[1]}": v for k, v in c.items()},
                            renamed=[(r["panel_symbol"], r["approved"]) for r in rows if r["approved"] and r["approved"] != r["panel_symbol"]],
                            not_in_hgnc=[r["panel_symbol"] for r in rows if r["approved"] is None])
    json.dump(dict(tokens=tokens, n_cellworld=len(cw), source="CellWorld data/gene_vocab.json + HGNC complete set"), open(os.path.join(out, "vocab.json"), "w"))
    print(json.dumps(dict(vocab_size=len(tokens), appended=len(tokens) - len(cw), panels=report), indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], [a.split("=", 1) for a in sys.argv[4:]])
