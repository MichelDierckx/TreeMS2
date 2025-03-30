import os

from TreeMS2.index.vector_store_index import VectorStoreIndex
from TreeMS2.states.context import Context
from TreeMS2.states.query_index_state import QueryIndexState
from TreeMS2.states.state import State
from TreeMS2.states.state_type import StateType
from TreeMS2.vector_store.vector_store import VectorStore


class CreateIndexState(State):
    STATE_TYPE = StateType.CREATE_INDEX
    MAX_VECTORS_IN_MEM = 1_000

    def __init__(self, context: Context, vector_store: VectorStore):
        super().__init__(context)
        # work directory
        self.work_dir: str = context.config.work_dir

        # create index
        self.use_gpu: bool = context.config.use_gpu

        # data generated from reading/processing spectra
        self.vector_store: VectorStore = vector_store

    def run(self):
        if not self.context.config.overwrite:
            index = VectorStoreIndex.load(
                path=os.path.join(self.vector_store.directory, f"{self.vector_store.name}.index"),
                vector_store=self.vector_store)
            if index is not None:
                self.context.replace_state(
                    state=QueryIndexState(context=self.context,
                                          index=index))
                return
        index = self._generate()
        self.context.replace_state(
            state=QueryIndexState(context=self.context,
                                  index=index))

    def _generate(self) -> VectorStoreIndex:
        # create an index
        index = VectorStoreIndex(vector_store=self.vector_store)
        # train the index
        index.train(use_gpu=self.use_gpu)
        # index the spectra for the groups
        index.add(batch_size=CreateIndexState.MAX_VECTORS_IN_MEM)
        # save the index to disk
        index.save_index(path=os.path.join(self.vector_store.directory, f"{self.vector_store.name}.index"))
        return index
