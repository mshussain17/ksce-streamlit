from langchain_core.callbacks.manager import Callbacks
from langchain_community.vectorstores import FAISS
from langchain_core.documents import BaseDocumentCompressor, Document
from pydantic import Extra, root_validator
from langchain_core.utils import get_from_dict_or_env
from typing import Any, Dict, List, Optional, Sequence, Union
from langchain.retrievers import ContextualCompressionRetriever
from langchain_openai import OpenAIEmbeddings
from copy import deepcopy
import cohere
import os
from dotenv import load_dotenv
load_dotenv()

cohere_api = os.getenv("COHERE_API_KEY")

embeddings=OpenAIEmbeddings(model="text-embedding-3-large")
db=FAISS.load_local("ksce_local",embeddings,allow_dangerous_deserialization=True)
retriever=db.as_retriever(search_kwargs={"k":40})

# cohere_api="nOYMEU3Hx5z7JB7rcKyqkpnzKUQglufMGDf2ilzX"

class CohereRerank(BaseDocumentCompressor):
    """Document compressor that uses `Cohere Rerank API`."""

    client: Any = None
    """Cohere client to use for compressing documents."""
    top_n: Optional[int] = 3
    """Number of documents to return."""
    model: str = "rerank-english-v2.0"
    """Model to use for reranking."""
    cohere_api_key: Optional[str] = None
    """Cohere API key. Must be specified directly or via environment variable
        COHERE_API_KEY."""
    user_agent: str = "langchain"
    """Identifier for the application making the request."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        if not values.get("client"):
            cohere_api_key = get_from_dict_or_env(
                values, "cohere_api_key", "COHERE_API_KEY"
            )
            client_name = values.get("user_agent", "langchain")
            values["client"] = cohere.Client(cohere_api_key, client_name=client_name)
        return values

    def rerank(
        self,
        documents: Sequence[Union[str, Document, dict]],
        query: str,
        *,
        model: Optional[str] = None,
        top_n: Optional[int] = -1,
        max_chunks_per_doc: Optional[int] = None,
    ) -> List[Dict[str, Any]]:

        if len(documents) == 0:  # to avoid empty api call
            return []
        docs = [
            doc.page_content if isinstance(doc, Document) else doc for doc in documents
        ]
        model = model or self.model
        top_n = top_n if (top_n is None or top_n > 0) else self.top_n
        results = self.client.rerank(
            query=query,
            documents=docs,
            model=model,
            top_n=top_n,
            max_chunks_per_doc=max_chunks_per_doc,
        )
        result_dicts = []
        for res in results.results:
            result_dicts.append(
                {"index": res.index, "relevance_score": res.relevance_score}
            )
        return result_dicts

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:

        compressed = []
        for res in self.rerank(documents, query):
            doc = documents[res["index"]]
            doc_copy = Document(page_content=doc.page_content, metadata=deepcopy(doc.metadata))
            doc_copy.metadata["relevance_score"] = res["relevance_score"]
            compressed.append(doc_copy)
        return compressed

cohere_rerank = CohereRerank(cohere_api_key=cohere_api,model="rerank-multilingual-v3.0",top_n=3)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=cohere_rerank,
    base_retriever=retriever
)