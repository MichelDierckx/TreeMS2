class IndexConfig:
    def __init__(self, similarity: float = 0.8):
        """
        Configuration for indexing.
        :param similarity: Minimum cosine similarity score for 2 spectra to be considered similar.
        """
        self.similarity = similarity
