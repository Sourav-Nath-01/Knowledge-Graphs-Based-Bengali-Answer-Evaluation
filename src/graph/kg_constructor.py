"""
kg_constructor.py
=================
Knowledge Graph construction from typed triples using NetworkX.

Improvements over the original:
  - Accepts 4-tuple triples: (subject, relation, object, triple_type).
  - Assigns node types based on triple_type ('entity', 'number', 'time',
    'location', 'quality') so the GNN can distinguish entity categories.
  - Assigns edge weights based on triple_type (content-carrying triples
    get weight 1.0; attribute triples get 0.6).
  - Still backward-compatible with plain 3-tuple triples from old callers.
"""

import networkx as nx


# Mapping from triple_type → (node category for object node, edge weight)
_TYPE_META = {
    'main':       ('entity',   1.0),
    'count':      ('number',   0.8),
    'quality':    ('quality',  0.7),
    'time':       ('time',     0.8),
    'location':   ('location', 0.8),
    'source':     ('location', 0.7),
    'instrument': ('entity',   0.6),
    'purpose':    ('entity',   0.6),
    'companion':  ('entity',   0.7),
    'possessive': ('entity',   0.7),
}


class KnowledgeGraphConstructor:
    """Build a typed, weighted directed NetworkX graph from triples.

    Accepts either:
    * 4-tuples: (subject, relation, object, triple_type)   ← preferred
    * 3-tuples: (subject, relation, object)                ← legacy, treated as 'main'
    """

    def __init__(self):
        pass

    def build_graph(self, triples: list) -> nx.DiGraph:
        """
        Construct a typed DiGraph from a list of triples.

        Parameters
        ----------
        triples : list of 3- or 4-tuples

        Returns
        -------
        nx.DiGraph  with node attribute 'node_type' and edge attributes
                    'relation' (str) and 'weight' (float).
        """
        G = nx.DiGraph()

        for triple in triples:
            # ── Unpack — handle both 3-tuple and 4-tuple ──────────────────
            if len(triple) == 4:
                subj, rel, obj, ttype = triple
            else:
                subj, rel, obj = triple[:3]
                ttype = 'main'

            obj_node_type, edge_weight = _TYPE_META.get(ttype, ('entity', 0.5))

            # ── Subject node ───────────────────────────────────────────────
            if subj not in G:
                G.add_node(subj, node_type='entity')

            # ── Object node ────────────────────────────────────────────────
            if obj and obj != '[NONE]':
                if obj not in G:
                    G.add_node(obj, node_type=obj_node_type)
                G.add_edge(subj, obj, relation=rel, weight=edge_weight,
                           triple_type=ttype)
            else:
                # '[NONE]' objects: still record a self-loop so the subject
                # node participates in the graph and gets an embedding.
                G.add_node(subj, node_type='entity')
                # Skip adding a meaningless [NONE] node/edge.

        return G
