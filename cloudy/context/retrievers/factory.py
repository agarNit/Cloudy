from cloudy.config import config
from cloudy.context.vector_store_health import qdrant_reachable
from cloudy.observability.logger import get_logger


logger = get_logger(__name__)




def get_retriever():
   mode = config["rag"]["mode"]
   provider = config["vector_store"]["provider"]
   if provider == "qdrant" and not qdrant_reachable():
       provider = "chroma"


   if mode == "hybrid" and provider == "qdrant":
       from .hybrid_qdrant import retrieve
   elif provider == "qdrant":
       from .semantic_qdrant import retrieve
   else:
       from .semantic_chroma import retrieve


   return retrieve
