import multiprocessing
import threading
from contextlib import contextmanager

from TreeMS2.ingestion.storage.vector_store import VectorStore


class LockManager:
    """Manages temporary multiprocessing locks for a group of vector stores."""

    def __init__(self, vector_stores: list[VectorStore]):
        self.vector_stores = vector_stores
        self.manager = None

    @contextmanager
    def use_multiprocessing_locks(self):
        """Assign Manager().Lock() to each vector store for multiprocessing phase."""
        self.manager = multiprocessing.Manager()
        locks = [self.manager.Lock() for _ in self.vector_stores]

        # Inject manager locks
        for vs, lock in zip(self.vector_stores, locks):
            vs.set_lock(lock)

        try:
            yield  # Run parallel processing
        finally:
            # Restore threading locks
            for vs in self.vector_stores:
                vs.set_lock(threading.Lock())

            # Shut down manager and clean resources
            self.manager.shutdown()
            self.manager = None
