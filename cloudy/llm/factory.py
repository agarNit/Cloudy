import threading
from functools import lru_cache

from cloudy.config import config
from cloudy.observability.logger import get_logger

logger = get_logger(__name__)

# The sentence-transformers/torch stack underneath HuggingFaceEmbeddings isn't safe
# under concurrent calls from multiple threads — verified with a real, reproducible
# segfault (exit 139, ~2/3 runs) when the agent made parallel search_codebase calls,
# each dispatched to its own thread by LangChain's sync-tool execution. Every actual
# embedding call (not just construction) must go through this lock.
EMBED_LOCK = threading.Lock()

@lru_cache(maxsize=1)
def get_llm():
    provider = config["llm"]["provider"]
    model = config["llm"]["model"]
    logger.info(f"Using LLM provider: {provider}, model: {model}")

    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model=model)

@lru_cache(maxsize=1)
def get_embedder():
    """Cached — HuggingFaceEmbeddings loads real model weights and can touch the
    network on construction. Previously this was rebuilt on every single
    retrieve() call; under plan mode's parallel search_codebase calls, two
    concurrent from-scratch loads were the actual cause of multi-minute hangs
    that looked like a plan-mode bug but weren't.
    """
    provider = config["embeddings"]["provider"]
    model = config["embeddings"]["model"]
    logger.info(f"Using embedding provider: {provider}, model: {model}")

    # PyTorch's own intra-op thread pool has known crash reports when entered from
    # a non-main thread while other torch work is in flight (macOS/Accelerate
    # backend especially) — confirmed via an actual crash report: the faulting
    # thread's native stack was entirely inside libtorch_cpu.dylib. Forcing
    # single-threaded tensor ops removes that risk; EMBED_LOCK above still
    # serializes Python-level access on top of this, for defense in depth.
    import torch
    torch.set_num_threads(1)

    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=model)