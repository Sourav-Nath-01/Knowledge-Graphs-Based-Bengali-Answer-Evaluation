"""
pipeline.py
===========
Main feature extraction pipeline, GNN training, and GNN inference.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from src.models.siamese_gnn import nx_to_pyg_data
from src.utils.helpers import negation_mismatch, compute_rouge_l, sentence_coverage_score


def compute_entity_mismatch(ref_parsed, stu_parsed, ref_text: str, stu_text: str,
                             embedder_obj=None, ner_pipeline_obj=None) -> float:
    """
    Compute a numeric entity mismatch score in [0, 1].

    Combines NER-detected entities and dependency-parsed content words.
    Uses embedding cosine similarity to allow synonym matching.
    """
    ref_word_set = set(ref_text.replace('।', '').replace(',', '').split())
    stu_word_set = set(stu_text.replace('।', '').replace(',', '').split())
    if ref_word_set == stu_word_set:
        return 0.0

    ref_entities, stu_entities = set(), set()
    if ner_pipeline_obj is not None:
        try:
            ref_ner = ner_pipeline_obj(ref_text)
            stu_ner = ner_pipeline_obj(stu_text)
            ref_entities = {e['word'].replace('##', '') for e in ref_ner}
            stu_entities = {e['word'].replace('##', '') for e in stu_ner}
        except Exception:
            pass

    content_pos  = {'NOUN', 'PROPN', 'NUM', 'ADJ'}
    ref_content, stu_content = set(), set()
    for sent in ref_parsed:
        for word in sent:
            if word.get('upos', 'X') in content_pos and word['deprel'] != 'punct':
                ref_content.add(word.get('lemma') or word['text'])
    for sent in stu_parsed:
        for word in sent:
            if word.get('upos', 'X') in content_pos and word['deprel'] != 'punct':
                stu_content.add(word.get('lemma') or word['text'])

    ref_content |= ref_entities
    stu_content |= stu_entities

    if not ref_content:
        overlap = len(ref_word_set & stu_word_set) / max(len(ref_word_set), 1)
        return max(0.0, 1.0 - overlap)

    missing = ref_content - stu_content
    added   = stu_content - ref_content
    SYNONYM_THRESHOLD = 0.75
    truly_missing = set()
    for ref_word in missing:
        found = False
        if embedder_obj is not None and added:
            for stu_word in added:
                try:
                    if embedder_obj.word_cosine_similarity(ref_word, stu_word) >= SYNONYM_THRESHOLD:
                        found = True
                        break
                except Exception:
                    pass
        if not found:
            truly_missing.add(ref_word)

    missing_ratio        = len(truly_missing) / len(ref_content)
    substitution_penalty = 0.2 if truly_missing and added else 0.0
    return min(1.0, missing_ratio + substitution_penalty)


def create_graphs_and_features(df, text_processor, dep_parser, triple_extractor,
                                kg_constructor, embedder, validator,
                                coref_resolver=None, ner_pipeline_obj=None):
    """
    Extract all features for every row in *df*.

    Returns
    -------
    (baseline_sims, pyg_ref_data, pyg_stu_data,
     karak_penalties, entity_mismatches,
     coverage_scores, negation_flags, rouge_l_scores)
    """
    print(f'Processing {len(df)} samples...')
    baseline_sims, pyg_ref_data, pyg_stu_data = [], [], []
    karak_penalties, entity_mismatches         = [], []
    coverage_scores, negation_flags, rouge_l_scores = [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc='Feature extraction'):
        ref_text = str(row['reference_answer'])
        stu_text = str(row['student_answer'])
        q_text   = str(row.get('question', ''))
        stu_resolved = (coref_resolver.resolve(stu_text, q_text, ref_text)
                        if coref_resolver else stu_text)

        baseline_sims.append(embedder.cosine_similarity(ref_text, stu_resolved))

        ref_parsed = dep_parser.parse(text_processor.normalize(ref_text))
        stu_parsed = dep_parser.parse(text_processor.normalize(stu_resolved))
        ref_triples = triple_extractor.extract_triples(raw_text=ref_text)
        stu_triples = triple_extractor.extract_triples(raw_text=stu_resolved)
        ref_kg  = kg_constructor.build_graph(ref_triples)
        stu_kg  = kg_constructor.build_graph(stu_triples)
        ref_emb = embedder.generate_node_embeddings(ref_kg)
        stu_emb = embedder.generate_node_embeddings(stu_kg)
        pyg_ref_data.append(nx_to_pyg_data(ref_kg, ref_emb))
        pyg_stu_data.append(nx_to_pyg_data(stu_kg, stu_emb))

        penalty, _ = validator.validate(ref_parsed, stu_parsed,
                                        ref_raw=ref_text, stu_raw=stu_resolved)
        karak_penalties.append(penalty)
        entity_mismatches.append(compute_entity_mismatch(
            ref_parsed, stu_parsed, ref_text, stu_resolved,
            embedder_obj=embedder, ner_pipeline_obj=ner_pipeline_obj))
        coverage_scores.append(sentence_coverage_score(ref_text, stu_resolved, embedder))
        negation_flags.append(negation_mismatch(ref_text, stu_resolved))
        rouge_l_scores.append(compute_rouge_l(ref_text, stu_resolved))

    print('Feature extraction complete.')
    return (baseline_sims, pyg_ref_data, pyg_stu_data,
            karak_penalties, entity_mismatches,
            coverage_scores, negation_flags, rouge_l_scores)


def train_gnn(gnn_model, pyg_ref_data, pyg_stu_data, target_scores, epochs: int = 20):
    """
    Train *gnn_model* on (ref_graph, stu_graph) pairs supervised by *target_scores*.

    Parameters
    ----------
    gnn_model : SiameseGNN
    pyg_ref_data, pyg_stu_data : lists of PyG Data objects
    target_scores : array-like of float scores in [0, 100]
    epochs : int

    Returns
    -------
    trained gnn_model
    """
    print(f'\nTraining GAT-GNN for {epochs} epochs...')
    optimizer = optim.Adam(gnn_model.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    gnn_model.train()

    for ep in range(epochs):
        total_loss = 0.0
        for ref_g, stu_g, score in zip(pyg_ref_data, pyg_stu_data, target_scores):
            optimizer.zero_grad()
            pred_sim   = gnn_model(ref_g, stu_g)
            target_sim = torch.tensor([score / 100.0], dtype=torch.float32)
            loss = criterion(pred_sim, target_sim)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / max(len(target_scores), 1)
        lr = scheduler.get_last_lr()[0]
        print(f'  Epoch {ep+1}/{epochs} - Loss: {avg_loss:.4f} - LR: {lr:.6f}')

    return gnn_model


def get_gnn_similarities(gnn_model, pyg_ref_data, pyg_stu_data) -> list:
    """Return cosine similarities for all (ref, stu) graph pairs."""
    gnn_model.eval()
    with torch.no_grad():
        return [max(0.0, gnn_model(r, s).item()) for r, s in zip(pyg_ref_data, pyg_stu_data)]
