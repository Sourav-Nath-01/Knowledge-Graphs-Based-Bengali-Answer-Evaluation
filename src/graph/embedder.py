"""
embedder.py
===========
LaBSE sentence embedder and BanglaBERT node embedder.

Offline-first loading: tries the local HuggingFace cache first, then
falls back to downloading if the cache misses.  Set the environment
variable HF_HUB_OFFLINE=1 to force cache-only mode system-wide.
"""

import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel


def _load_sentence_transformer(model_name: str, device: str) -> Optional[SentenceTransformer]:
    """
    Load a SentenceTransformer model.
    Tries local cache first (local_files_only=True) to avoid network calls.
    Falls back to a download attempt if the cache misses.
    """
    # 1) Try from local cache (fast, no network)
    try:
        model = SentenceTransformer(model_name, device=device, local_files_only=True)
        print(f'  [OK] {model_name} loaded from cache.')
        return model
    except Exception:
        pass

    # 2) Try downloading (needs internet)
    try:
        model = SentenceTransformer(model_name, device=device)
        print(f'  [OK] {model_name} downloaded and loaded.')
        return model
    except Exception as e:
        print(f'  [FAIL] {model_name}: {e}')
        return None


def _load_hf_model(model_name: str, device: torch.device) -> Tuple[Optional[object], Optional[object]]:
    """
    Load a HuggingFace tokenizer + model.
    Tries local cache first, then falls back to download.
    """
    for local_only in (True, False):
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, local_files_only=local_only)
            model = AutoModel.from_pretrained(
                model_name, local_files_only=local_only)
            model.to(device)
            model.eval()
            source = 'cache' if local_only else 'download'
            print(f'  [OK] {model_name} loaded from {source}.')
            return tokenizer, model
        except Exception as e:
            if local_only:
                continue          # try the download fallback
            print(f'  [FAIL] {model_name}: {e}')
            return None, None
    return None, None


class BanglaBERTEmbedder:
    """
    Dual-encoder:
    * LaBSE for sentence-level cosine similarity (used as baseline feature).
    * BanglaBERT for node-level embeddings used in the GNN.
    """

    def __init__(self,
                 sentence_model_name: str = 'sentence-transformers/LaBSE',
                 node_model_name:     str = 'csebuetnlp/banglabert'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sentence_model  = None
        self.node_tokenizer  = None
        self.node_model      = None

        # ── LaBSE ──
        print(f'Loading LaBSE: {sentence_model_name}...')
        self.sentence_model = _load_sentence_transformer(
            sentence_model_name, str(self.device))

        # ── BanglaBERT ──
        print(f'Loading BanglaBERT: {node_model_name}...')
        self.node_tokenizer, self.node_model = _load_hf_model(
            node_model_name, self.device)

    # ─── Node embedding ───────────────────────────────────────────────────────

    def get_embedding(self, text: str) -> torch.Tensor:
        """Return a mean-pooled BanglaBERT embedding for *text* (shape: [768])."""
        if self.node_model is None:
            return torch.zeros(768)
        inputs = self.node_tokenizer(text, return_tensors='pt', truncation=True,
                                     padding=True, max_length=128).to(self.device)
        with torch.no_grad():
            outputs = self.node_model(**inputs)
        token_embs  = outputs.last_hidden_state
        attn_mask   = inputs['attention_mask']
        mask_exp    = attn_mask.unsqueeze(-1).expand(token_embs.size()).float()
        sum_emb     = torch.sum(token_embs * mask_exp, 1)
        sum_mask    = torch.clamp(mask_exp.sum(1), min=1e-9)
        return (sum_emb / sum_mask).squeeze(0).cpu()

    def generate_node_embeddings(self, graph) -> dict:
        """Return a dict {node_label: {'emb': tensor, 'node_type': str}} for all nodes."""
        result = {}
        for node in graph.nodes():
            node_type = graph.nodes[node].get('node_type', 'entity')
            if node == '[NONE]':
                result[node] = {'emb': torch.zeros(768), 'node_type': node_type}
            else:
                result[node] = {'emb': self.get_embedding(node), 'node_type': node_type}
        return result

    # ─── Sentence similarity ──────────────────────────────────────────────────

    def cosine_similarity(self, text1: str, text2: str) -> float:
        """Return LaBSE cosine similarity between *text1* and *text2*."""
        if self.sentence_model is None:
            return 0.0
        embeddings = self.sentence_model.encode([text1, text2], convert_to_tensor=True)
        return F.cosine_similarity(embeddings[0].unsqueeze(0),
                                   embeddings[1].unsqueeze(0)).item()

    def word_cosine_similarity(self, word1: str, word2: str) -> float:
        """Return BanglaBERT cosine similarity between two word-level texts."""
        emb1 = self.get_embedding(word1).unsqueeze(0)
        emb2 = self.get_embedding(word2).unsqueeze(0)
        return F.cosine_similarity(emb1, emb2).item()

    def encode_sentences(self, sentences: list) -> list:
        """Encode a list of sentences with LaBSE and return numpy array."""
        if self.sentence_model is None:
            import numpy as np
            return np.zeros((len(sentences), 768))
        return self.sentence_model.encode(sentences, convert_to_numpy=True)
