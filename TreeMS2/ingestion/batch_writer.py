import multiprocessing
from collections import defaultdict, Counter
from typing import Dict, Union

from TreeMS2.ingestion.spectra_dataset.treems2_spectrum import TreeMS2Spectrum
from TreeMS2.ingestion.storage.vector_stores import VectorStores
from TreeMS2.ingestion.vectorization.spectra_vector_transformer import SpectraVectorTransformer


class BatchWriter:
    def __init__(self, buffer_size: int, vectorizer: SpectraVectorTransformer, vector_store_manager: VectorStores):
        self.buffer_size = buffer_size
        self.vectorizer = vectorizer
        self.vector_store_manager = vector_store_manager
        self.buffers = defaultdict(list)

    def add(self, store_name: str, spectrum: TreeMS2Spectrum):
        self.buffers[store_name].append(spectrum)
        if len(self.buffers[store_name]) >= self.buffer_size:
            self._flush_store(store_name)

    def _flush_store(self, store_name: str):
        buffer = self.buffers[store_name]
        vectors = self.vectorizer.vectorize([s.spectrum for s in buffer])
        dict_list = [{**s.to_dict(), "vector": v} for s, v in zip(buffer, vectors)]
        self.vector_store_manager.write(
            vector_store_name=store_name,
            entries_to_write=dict_list,
            use_incremental_compaction=True,
        )
        buffer.clear()

    def flush(self):
        for store_name in list(self.buffers.keys()):
            if self.buffers[store_name]:
                self._flush_store(store_name)

