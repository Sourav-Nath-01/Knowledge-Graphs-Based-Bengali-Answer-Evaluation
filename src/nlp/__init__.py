"""src/nlp/__init__.py"""
from src.nlp.dependency_parser import BengaliDependencyParser
from src.nlp.coreference import BengaliCoreferenceResolver
from src.nlp.triple_extractor import TripleExtractor

__all__ = ["BengaliDependencyParser", "BengaliCoreferenceResolver", "TripleExtractor"]
