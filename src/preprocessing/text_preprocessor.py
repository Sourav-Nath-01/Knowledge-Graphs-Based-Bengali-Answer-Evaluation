"""
text_preprocessor.py
=====================
Bengali text normalisation using the Indic NLP Library.
"""

from indicnlp.normalize.indic_normalize import IndicNormalizerFactory


class TextPreprocessor:
    """Normalise and tokenise Bengali text."""

    STOPWORDS = {
        'কী', 'কেন', 'কোথায়', 'কখন', 'কিভাবে', 'যে', 'ও', 'এবং', 'আর',
        'কিন্তু', 'অথবা', 'নয়', 'না', 'নি', 'করি', 'করে', 'করা', 'হয়',
        'হয়েছে', 'ছিল', 'ছিলেন', 'হবে', 'হবেন', 'তো', 'যা', 'এই', 'সেই',
        'একটি', 'একটা', 'সব', 'কোনো', 'কোন', 'সকল', 'প্রতি',
    }

    def __init__(self):
        factory = IndicNormalizerFactory()
        self.normalizer = factory.get_normalizer('bn')

    def normalize(self, text: str) -> str:
        """Apply Indic normalisation to *text*."""
        if not isinstance(text, str):
            text = str(text)
        return self.normalizer.normalize(text)

    def remove_punctuation(self, text: str) -> str:
        """Remove common punctuation marks and collapse whitespace."""
        for p in ['।', ',', '.', '?', '!', ';', ':', '"', "'", '(', ')', '[', ']', '{', '}']:
            text = text.replace(p, ' ')
        return ' '.join(text.split())

    def process(self, text: str, remove_stops: bool = False) -> list:
        """
        Normalise *text*, strip punctuation, and split into tokens.

        .. warning::
            This method removes ALL punctuation including '।' before splitting,
            so sentence boundaries are **lost**.  Use it only on a single,
            already-isolated sentence.  For multi-sentence text, call
            ``sentence_split()`` first, then ``process()`` each sentence.

        Parameters
        ----------
        text : str
        remove_stops : bool
            If True, filter out stopwords.

        Returns
        -------
        list of str
        """
        text = self.normalize(text)
        text = self.remove_punctuation(text)
        tokens = text.split()
        if remove_stops:
            tokens = [t for t in tokens if t not in self.STOPWORDS]
        return tokens

    def sentence_split(self, text: str) -> list:
        """
        Split *text* on Bengali sentence boundaries (「।」 and newlines).

        Returns a list of normalised, non-empty sentence strings — with
        punctuation still intact so that downstream POS taggers can use it.

        Parameters
        ----------
        text : str   (may be multi-sentence)

        Returns
        -------
        list of str
        """
        import re
        normalized = self.normalize(text)
        parts = re.split(r'[।\n]+', normalized)
        return [p.strip() for p in parts if p.strip()]

