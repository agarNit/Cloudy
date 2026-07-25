from cloudy.config import config
from cloudy.observability.logger import get_logger

logger = get_logger(__name__)

def get_llm():
    provider = config["llm"]["provider"]
    model = config["llm"]["model"]
    logger.info(f"Using LLM provider: {provider}, model: {model}")

    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model=model)

def get_embedder():
    provider = config["embeddings"]["provider"]
    model = config["embeddings"]["model"]
    logger.info(f"Using embedding provider: {provider}, model: {model}")

    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=model)