from typing import FrozenSet, List, Optional, Set

import networkx as nx

__all__ = ["close_nodes", "suc_is_close_by_other_way", "motif_H"]


def close_nodes(G: nx.DiGraph) -> Set[FrozenSet[int]]:
    """
    For a node [within an junction], if its successor node can [easily] return to this node, then the aggregation
    Generally, it is a "□"-shaped structure or a "△" structure in the junction.
    """
    # Construct a subgraph that does not contain [way that dose not connect to any junction]
    # Criterion: length > 100
    shortG = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        if data["length"] <= 100:
            shortG.add_edge(u, v, **data)
    # Traverse all nodes [within the junction] (both out-degree and in-degree ≥ 1)
    group = set()
    for node in shortG.nodes:
        if shortG.in_degree(node) == 0 or shortG.out_degree(node) == 0:
            continue
        # Check whether the successor node can go back to this node
        successors = list(shortG.successors(node))
        if len(successors) == 0:
            continue
        for suc in successors:
            try:
                path = nx.shortest_path(shortG, suc, node, weight="length")
                if len(path) <= 2:
                    # ====O=====O====
                    # Forget what you get from this two-way split, we won’t deal with it here.
                    continue
                # The distance between two paths should be less than X meters
                bad = False
                sum_length = 0
                for i in range(len(path) - 1):
                    this_length = G[path[i]][path[i + 1]]["length"]
                    if this_length > 60:
                        bad = True
                        break
                    sum_length += this_length
                if bad:
                    continue
                if sum_length > 100:
                    if len(path) > 4:
                        # Possibly multiple local short circuits
                        continue
                # Merge
                group.add(frozenset(path))
            except nx.NetworkXNoPath:
                continue
    return group


def suc_is_close_by_other_way(G: nx.DiGraph) -> Set[FrozenSet[int]]:
    r"""
    If there is another path (very short) that can reach my successor, merge it
    including but not limited to
                 |
                 v
    -->---C-->---A-->--
           \     v
            \    |
             v   |
              \  |
               \ |
                \|
                 B
                 v
                 |
    或
    --->---C-->---B-->--
            \    /
             \  /
              \/
               A
              | |
              v ^
              | |
    """
    # Construct a subgraph that does not contain [way that dose not connect to any junction]
    # Criterion: length > 100
    shortG = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        if data["length"] <= 150:
            shortG.add_edge(u, v, **data)
    # Check whether the candidate set meets the conditions
    # Condition: A successor node (B) of point C is a successor node of another successor node (A)

    def get_good_path(
        G: nx.DiGraph, n1: int, n2: int, max_length: float
    ) -> Optional[List[int]]:
        """
        Check if there is a short path from n1 to n2
        """
        try:
            path = nx.shortest_path(G, n1, n2, weight="length")
            # If the hop count of the path is ≤ 4 and the total length is less than 100 meters or twice the original connection length, it is considered good.
            if (
                len(path) <= 4
                and sum(
                    [
                        shortG[path[i]][path[i + 1]]["length"]
                        for i in range(len(path) - 1)
                    ]
                )
                <= max_length
            ):
                return path  # type: ignore
            else:
                return None
        except nx.NetworkXNoPath:
            return None

    group = set()
    for n in shortG.nodes:
        successors = list(shortG.successors(n))
        for suc in successors:
            length = G[n][suc]["length"]
            # remove edge of n->suc
            shortG.remove_edge(n, suc)
            max_length = max(100, 2 * length)
            # Check if there are other paths to reach suc
            forward = get_good_path(shortG, n, suc, max_length)
            if forward is not None:
                # Also ensure unidirectionality, i.e. the path from suc to n does not exist or is very long
                if get_good_path(shortG, suc, n, max_length) is None:
                    group.add(frozenset(forward))
            # Restore removed edges
            shortG.add_edge(n, suc, length=length)
    return group


def motif_H(G: nx.DiGraph) -> Set[FrozenSet[int]]:
    """
    Identify motif as below
         |    |
         v    ^
         |    |
    =====O====O=====
         |    |
         v    ^
         |    |
    或
         |    |
         v    ^
         |    |
    =====O====O
         |    |
         v    ^
         |    |
    """
    # Build candidate set, conditions:
    # 1. The length of one side is less than 30m
    # 2. Be each other’s predecessors and successors
    candidates = set()  # set[pair]
    for u, v, data in G.edges(data=True):
        if u == v:
            continue
        if data["length"] > 30:
            continue
        if G.has_edge(v, u):
            candidates.add(frozenset([u, v]))
    # Check whether the degrees of two points meet the conditions
    # The one with more: in-degree == out-degree == 3
    # The one with the less: in-degree == out-degree == 2 or 3
    motifs = set()
    for c in candidates:
        u, v = list(c)
        u_in_degree = G.in_degree(u)
        u_out_degree = G.out_degree(u)
        v_in_degree = G.in_degree(v)
        v_out_degree = G.out_degree(v)
        if u_in_degree != u_out_degree:
            continue
        u_degree = u_in_degree
        if v_in_degree != v_out_degree:
            continue
        v_degree = v_in_degree
        # u>=v
        if u_degree < v_degree:
            u, v = v, u
            u_degree, v_degree = v_degree, u_degree
        if not (u_degree == 3 and v_degree in (2, 3)):
            continue
        # The external out- and in-degrees should not be obtained by splitting two-way roads, otherwise the road network composed of two-way T-shaped junctions will be combined.
        # For the successor of u (excluding v), there can only be at most 1 successor whose successor is u
        u_sucs = set(G.successors(u)) - {v}
        if sum(u in set(G.successors(suc)) for suc in u_sucs) > u_degree - 2:
            continue
        # For the successor of v (except u), if the degree is 2, the successor that cannot have a successor is v. If the degree is 3, the successor that has at most 1 successor is v.
        v_sucs = set(G.successors(v)) - {u}
        if sum(v in set(G.successors(suc)) for suc in v_sucs) > v_degree - 2:
            continue
        motifs.add(frozenset([u, v]))
    return motifs
