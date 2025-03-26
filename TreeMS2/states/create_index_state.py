import os

from TreeMS2.index.ms2_index import MS2Index
from TreeMS2.states.context import Context
from TreeMS2.states.query_index_state import QueryIndexState
from TreeMS2.states.state import State
from TreeMS2.vector_store.vector_store import VectorStore


class CreateIndexState(State):
    MAX_VECTORS_IN_MEM = 1_000

    def __init__(self, context: Context, vector_store: VectorStore):
        super().__init__(context)
        # work directory
        self.work_dir: str = context.config.work_dir

        # create index
        self.low_dim: int = context.config.low_dim
        self.use_gpu: bool = context.config.use_gpu

        # data generated from reading/processing spectra
        self.vector_store: VectorStore = vector_store

    def run(self):
        if not self.context.config.overwrite:
            index = MS2Index.load(path=os.path.join(self.vector_store.directory, f"{self.vector_store.name}.index"),
                                  total_valid_spectra=self.context.groups.total_valid_spectra(), d=self.low_dim,
                                  use_gpu=self.use_gpu)
            if index is not None:
                self.context.replace_state(
                    state=QueryIndexState(context=self.context,
                                          vector_store=self.vector_store,
                                          index=index))
                return
        index = self._generate()
        self.context.replace_state(
            state=QueryIndexState(context=self.context, vector_store=self.vector_store,
                                  index=index))

    def _generate(self) -> MS2Index:
        # create an index
        index = MS2Index(total_valid_spectra=self.context.groups.total_valid_spectra(), d=self.low_dim,
                         use_gpu=self.use_gpu)
        # train the index
        index.train(vector_store=self.vector_store)
        # index the spectra for the groups
        index.add(vector_store=self.vector_store, batch_size=CreateIndexState.MAX_VECTORS_IN_MEM)
        # save the index to disk
        index.save_index(path=os.path.join(self.vector_store.directory, f"{self.vector_store.name}.index"))
        return index
