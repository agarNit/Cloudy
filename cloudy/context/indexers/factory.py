from cloudy.config import config
from cloudy.context.vector_store_health import qdrant_reachable
from cloudy.observability.logger import get_logger


logger = get_logger(__name__)


def _effective_provider() -> str:
   provider = config["vector_store"]["provider"]
   if provider == "qdrant" and not qdrant_reachable():
       return "chroma"
   return provider


def get_indexer():
   mode = config["rag"]["mode"]
   provider = _effective_provider()


   if mode == "hybrid" and provider == "qdrant":
       from .hybrid_qdrant import index_codebase
   elif provider == "qdrant":
       from .semantic_qdrant import index_codebase
   else:
       from .semantic_chroma import index_codebase


   return index_codebase


def get_index_inspector():
   """Return the right show_index function based on rag mode and vector_store in config."""
   mode = config["rag"]["mode"]
   provider = _effective_provider()


   if mode == "hybrid" and provider == "qdrant":
       from .hybrid_qdrant import show_index
   elif provider == "qdrant":
       from .semantic_qdrant import show_index
   else:
       from .semantic_chroma import show_index


   return show_index
