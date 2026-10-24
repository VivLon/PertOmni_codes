"""Assignment of every CRISPRi target gene to one MSigDB GO:BP pathway (the "pathway" part of the CRISPRi text input).

Greedy exclusivity selection: repeatedly pick the pathway covering the most genes that no other remaining pathway
covers (ties: unique ratio, overlap, smaller pathway, name), claim its genes, and repeat; then pathways with fewer
than min_size genes are dissolved and their genes moved to the largest other pathway containing them.
"""

import re


def select_domain_pathways(gene_list, gmt, overlap_floor=5):
    """{"assignment": gene -> pathway, "selected_order": [pathway, ...]}; gmt: pathway -> genes."""
    genes = set(gene_list)
    pathways = {p: set(g) for p, g in gmt.items()}
    touching = {p: pathways[p] & genes for p in pathways if pathways[p] & genes}
    if not touching:
        best = min(pathways, key=lambda p: (len(pathways[p]), p)) if pathways else None
        return {"assignment": {g: best for g in genes}, "selected_order": []}

    def ranked(current, remaining):
        membership = {}
        for members in current.values():
            for g in members & remaining:
                membership[g] = membership.get(g, 0) + 1
        rows = []
        for name, members in current.items():
            overlap = members & remaining
            k = len(overlap)
            if k < overlap_floor:
                continue
            u = sum(membership.get(g, 0) == 1 for g in overlap)
            rows.append((u, u / k, k, -len(members), name, overlap))
        return sorted(rows, key=lambda r: r[:5], reverse=True)

    # stage 1: greedy selection by exclusivity
    remaining, selected_order, claimed = set(genes), [], {}
    current = {p: set(s) for p, s in touching.items()}
    while remaining:
        current = {p: s & remaining for p, s in current.items() if s & remaining}
        rows = ranked(current, remaining)
        if not rows:
            break
        name, overlap = rows[0][4], rows[0][5]
        selected_order.append(name)
        claimed[name] = set(overlap)
        remaining -= overlap

    # stage 2: every gene gets a pathway
    assignment = {g: next((p for p in selected_order if g in claimed[p]), None) for g in genes}
    overlap_with_list = {p: len(pathways[p] & genes) for p in pathways}

    def best_of(candidates):
        return max(candidates, key=lambda p: (overlap_with_list[p], -len(pathways[p]), p))

    selected = set(selected_order)
    for pool in (selected, [p for p in pathways if p not in selected]):
        for g in [g for g, p in assignment.items() if p is None]:
            candidates = [p for p in pool if g in pathways[p]]
            if candidates:
                assignment[g] = best_of(candidates)
    leftovers = [g for g, p in assignment.items() if p is None]
    if leftovers:
        fallback = best_of(pathways)
        for g in leftovers:
            assignment[g] = fallback
    return {"assignment": assignment, "selected_order": selected_order}


def reassign_min_size(res, gmt, min_size=10):
    """Dissolve pathways with fewer than min_size assigned genes and move their genes, one by one and
    deterministically, to the pathway containing the gene that would then be largest."""
    assignment = dict(res["assignment"])
    pathways = {p: set(g) for p, g in gmt.items()}
    genes = set(assignment)
    path2genes = {}
    for g, p in assignment.items():
        path2genes.setdefault(p, set()).add(g)
    overlap_with_list = {p: len(pathways.get(p, set()) & genes) for p in pathways}

    def small_paths():
        return [p for p, gs in path2genes.items() if len(gs) < min_size]

    if not small_paths():
        return res
    gene2paths = {}
    for p, members in pathways.items():
        for g in members:
            gene2paths.setdefault(g, set()).add(p)

    def move(g, src, dst):
        if src is not None:
            path2genes[src].discard(g)
        assignment[g] = dst
        path2genes.setdefault(dst, set()).add(g)

    for sp in sorted(small_paths()):
        for g in sorted(path2genes.get(sp, set())):
            candidates = [p for p in gene2paths.get(g, []) if p != sp] or [p for p in pathways if p != sp]
            best = max(
                candidates,
                key=lambda p: (
                    len(path2genes.get(p, set())) + 1,
                    overlap_with_list.get(p, 0),
                    -len(pathways.get(p, set())),
                    p,
                ),
            )
            move(g, assignment[g], best)

    still_small = small_paths()
    if still_small:
        destinations = sorted(
            (p for p in path2genes if p not in still_small),
            key=lambda p: (len(path2genes[p]), overlap_with_list.get(p, 0), -len(pathways.get(p, set())), p),
            reverse=True,
        )
        for sp in sorted(still_small):
            for g in sorted(path2genes.get(sp, set())):
                fallback = max(pathways, key=lambda p: (overlap_with_list.get(p, 0), -len(pathways.get(p, set())), p))
                move(g, sp, destinations[0] if destinations else fallback)

    selected_order = [p for p in res.get("selected_order", []) if path2genes.get(p)]
    selected_order += sorted(p for p in path2genes if p not in selected_order and path2genes[p])
    return {"assignment": assignment, "selected_order": selected_order}


def assign_pathways(target_genes, gmt, control="non-targeting", overlap_floor=5, min_size=10):
    """target gene -> GO:BP pathway; the control maps to itself."""
    genes = sorted(set(target_genes) - {control})
    assignment = reassign_min_size(select_domain_pathways(genes, gmt, overlap_floor), gmt, min_size)["assignment"]
    assignment[control] = control
    return assignment


def go_bp_gmt(dbver="2025.1.Hs"):
    """MSigDB C5 GO:BP gene sets {pathway: genes} (gseapy, downloads from MSigDB)."""
    from gseapy import Msigdb

    return Msigdb().get_gmt(category="c5.go.bp", dbver=dbver)


def go_bp_descriptions(db_path):
    """{pathway: brief description} of the C5:GO:BP gene sets in an MSigDB SQLite database (msigdb_v*.Hs.db)."""
    import sqlite3

    con = sqlite3.connect(db_path)
    rows = con.execute(
        "SELECT gs.standard_name, gsd.description_brief FROM gene_set AS gs "
        "JOIN gene_set_details AS gsd ON gsd.gene_set_id = gs.id WHERE gs.collection_name = 'C5:GO:BP'"
    ).fetchall()
    con.close()
    return {name: re.sub(r"\[[^\]]*\]", "", text).strip().replace("]", "") for name, text in rows}
