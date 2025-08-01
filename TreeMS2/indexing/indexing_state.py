import os

from TreeMS2.config.logger_config import log_section_title, get_logger
from TreeMS2.indexing.vector_store_index import VectorStoreIndex
from TreeMS2.ingestion.storage.vector_store import VectorStore
from TreeMS2.search.vector_store_search_state import QueryIndexState
from TreeMS2.states.context import Context
from TreeMS2.states.state import State
from TreeMS2.states.state_type import StateType

logger = get_logger(__name__)


class IndexingState(State):
    STATE_TYPE = StateType.INDEXING_STATE

    def __init__(self, context: Context, vector_store: VectorStore):
        super().__init__(context)

        # create index
        self.use_gpu: bool = context.config.use_gpu
        self.batch_size: int = context.config.batch_size

        # data generated from reading/processing spectra
        self.vector_store: VectorStore = vector_store

    def run(self):
        log_section_title(
            logger=logger, title=f"[ Building Lance Index ({self.vector_store.name}) ]"
        )
        if not self.context.config.overwrite:
            index = VectorStoreIndex.load(
                path=os.path.join(
                    self.context.indexes_dir, f"{self.vector_store.name}.index"
                ),
                vector_store=self.vector_store,
            )
            if index is not None:
                logger.info(
                    f"Found existing index ('{os.path.join(self.context.indexes_dir, f"{self.vector_store.name}.index")}'). Skipping processing and loading index from disk."
                )
                self.context.replace_state(
                    state=QueryIndexState(context=self.context, index=index)
                )
                return
        index = self._generate()
        self.context.replace_state(
            state=QueryIndexState(context=self.context, index=index)
        )

    def _generate(self) -> VectorStoreIndex:
        # create an index
        index = VectorStoreIndex(vector_store=self.vector_store)
        logger.info(
            f"Created FAISS index with factory string '{index.factory_string}' for vector store '{self.vector_store.dataset_path}'."
        )
        # train the index
        index.train(use_gpu=self.use_gpu)
        # index the spectra for the groups
        index.add(batch_size=self.batch_size)
        # save the index to disk
        index.save_index(
            path=os.path.join(
                self.context.indexes_dir, f"{self.vector_store.name}.index"
            )
        )
        logger.info(
            f"Saved index to '{os.path.join(self.context.indexes_dir, f"{self.vector_store.name}.index")}'."
        )
        return index
