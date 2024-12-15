import os

from TreeMS2.groups.groups import Groups
from TreeMS2.index.ms2_index import MS2Index
from TreeMS2.states.context import Context
from TreeMS2.states.query_index_state import QueryIndexState
from TreeMS2.states.state import State
from TreeMS2.vector_store.vector_store import VectorStore

INDEX_FILE = "spectra.index"


class CreateIndexState(State):
    MAX_VECTORS_IN_MEM = 1_000

    def __init__(self, context: Context, groups: Groups, vector_store: VectorStore):
        super().__init__(context)
        # work directory
        self.work_dir: str = context.config.work_dir

        # create index
        self.low_dim: int = context.config.low_dim

        # data generated from reading/processing spectra
        self.groups: Groups = groups
        self.vector_store: VectorStore = vector_store

    def run(self, overwrite: bool):
        if overwrite or not self._is_output_generated():
            # generate the required output
            index = self._generate()
        else:
            # load the required output
            index = self._load()
        # move to the query index state
        self.context.replace_state(
            state=QueryIndexState(context=self.context, groups=self.groups, vector_store=self.vector_store,
                                  index=index))

    def _generate(self) -> MS2Index:
        # create an index
        index = MS2Index(total_valid_spectra=self.groups.total_valid_spectra(), d=self.low_dim)
        # train the index
        index.train(vector_store=self.vector_store)
        # index the spectra for the groups
        index.add(vector_store=self.vector_store, batch_size=CreateIndexState.MAX_VECTORS_IN_MEM)
        # save the index to disk
        index.save_index(path=os.path.join(self.work_dir, INDEX_FILE))
        return index

    def _load(self) -> MS2Index:
        # TODO: properly load index from disk
        index = MS2Index(total_valid_spectra=self.groups.total_valid_spectra(), d=self.low_dim)
        index.load_index(path=os.path.join(self.work_dir, INDEX_FILE))
        return index

    def _is_output_generated(self) -> bool:
        if not os.path.isfile(os.path.join(self.work_dir, INDEX_FILE)):
            return False
        return True
