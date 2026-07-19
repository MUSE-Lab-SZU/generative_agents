"""generative_agents.storage.index"""

import os
import time
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.core.schema import TextNode
from llama_index import core as index_core
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings

from modules import utils
from modules.model.llm_model import (
    format_call_error_details,
    resolve_ollama_timeout_seconds,
    safe_exception_message_for_log,
)


STORAGE_RETRY_MAX = 3
STORAGE_RETRY_SLEEP_SECONDS = 5


class LlamaIndex:
    def __init__(self, embedding_config, path=None):
        self._config = {"max_nodes": 0}
        self._embedding_log_context = {
            "provider": str(embedding_config.get("provider", "") or ""),
            "model": str(embedding_config.get("model", "") or ""),
            "base_url": str(embedding_config.get("base_url", "") or ""),
        }
        if embedding_config["provider"] == "hugging_face":
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding

            embed_model = HuggingFaceEmbedding(model_name=embedding_config["model"])
        elif embedding_config["provider"] == "ollama":
            from llama_index.embeddings.ollama import OllamaEmbedding

            embed_model = OllamaEmbedding(
                model_name=embedding_config["model"],
                base_url=embedding_config["base_url"],
                ollama_additional_kwargs={"mirostat": 0},
                client_kwargs={
                    "timeout": resolve_ollama_timeout_seconds(embedding_config),
                },
            )
        elif embedding_config["provider"] == "openai":
            from llama_index.embeddings.openai import OpenAIEmbedding

            embed_model = OpenAIEmbedding(
                model_name=embedding_config["model"],
                api_base=embedding_config["base_url"],
                api_key=embedding_config["api_key"],
            )
        else:
            raise NotImplementedError(
                "embedding provider {} is not supported".format(embedding_config["provider"])
            )

        Settings.embed_model = embed_model
        Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=64)
        Settings.num_output = 1024
        Settings.context_window = 4096
        if path and os.path.exists(path):
            self._index = index_core.load_index_from_storage(
                index_core.StorageContext.from_defaults(persist_dir=path),
                show_progress=True,
            )
            self._config = utils.load_dict(os.path.join(path, "index_config.json"))
        else:
            self._index = index_core.VectorStoreIndex([], show_progress=True)
        self._path = path

    def _run_with_retry(self, op_name, callback, error_formatter):
        last_error = None
        for retry_count in range(1, STORAGE_RETRY_MAX + 1):
            started_at = time.monotonic()
            try:
                return callback()
            except Exception as e:
                last_error = e
                context = getattr(self, "_embedding_log_context", {}) or {}
                details = format_call_error_details(
                    e,
                    caller="embedding.{}".format(op_name),
                    stage="request",
                    provider=context.get("provider", ""),
                    model=context.get("model", ""),
                    base_url=context.get("base_url", ""),
                    attempt=retry_count,
                    total_attempts=STORAGE_RETRY_MAX,
                    retrying=retry_count < STORAGE_RETRY_MAX,
                    elapsed_ms=(time.monotonic() - started_at) * 1000,
                )
                print(
                    "{} | [EMBEDDING_CALL_ERROR] {}".format(
                        error_formatter(
                            retry_count,
                            safe_exception_message_for_log(e),
                        ),
                        details,
                    ),
                    flush=True,
                )
                if retry_count >= STORAGE_RETRY_MAX:
                    raise
                time.sleep(STORAGE_RETRY_SLEEP_SECONDS)
        raise last_error

    def add_node(
        self,
        text,
        metadata=None,
        exclude_llm_keys=None,
        exclude_embedding_keys=None,
        id=None,
    ):
        metadata = metadata or {}
        exclude_llm_keys = exclude_llm_keys or list(metadata.keys())
        exclude_embedding_keys = exclude_embedding_keys or list(metadata.keys())
        auto_id = id is None
        node_id = id or "node_" + str(self._config["max_nodes"])

        def _insert_node():
            node = TextNode(
                text=text,
                id_=node_id,
                metadata=metadata,
                excluded_llm_metadata_keys=exclude_llm_keys,
                excluded_embed_metadata_keys=exclude_embedding_keys,
            )
            self._index.insert_nodes([node])
            if auto_id:
                self._config["max_nodes"] += 1
            return node

        return self._run_with_retry(
            "add_node",
            _insert_node,
            lambda retry_count, error: (
                "[LlamaIndex.add_node][retry={}] node_id={} text_len={} metadata_keys={} error={}"
                .format(
                    retry_count,
                    node_id,
                    len(text or ""),
                    list(metadata.keys()),
                    error,
                )
            ),
        )

    def has_node(self, node_id):
        return node_id in self._index.docstore.docs

    def find_node(self, node_id):
        return self._index.docstore.docs[node_id]

    def get_nodes(self, filter=None):
        def _check(node):
            if not filter:
                return True
            return filter(node)

        return [n for n in self._index.docstore.docs.values() if _check(n)]

    def remove_nodes(self, node_ids, delete_from_docstore=True):
        self._index.delete_nodes(node_ids, delete_from_docstore=delete_from_docstore)

    def cleanup(self):
        now, remove_ids = utils.get_timer().get_date(), []
        for node_id, node in self._index.docstore.docs.items():
            create = utils.to_date(node.metadata["create"])
            expire = utils.to_date(node.metadata["expire"])
            if create > now or expire < now:
                remove_ids.append(node_id)
        self.remove_nodes(remove_ids)
        return remove_ids

    def retrieve(
        self,
        text,
        similarity_top_k=5,
        filters=None,
        node_ids=None,
        retriever_creator=None,
    ):
        started_at = time.monotonic()
        try:
            retriever_creator = retriever_creator or VectorIndexRetriever
            return retriever_creator(
                self._index,
                similarity_top_k=similarity_top_k,
                filters=filters,
                node_ids=node_ids,
            ).retrieve(text)
        except Exception as e:
            context = getattr(self, "_embedding_log_context", {}) or {}
            details = format_call_error_details(
                e,
                caller="embedding.retrieve",
                stage="request",
                provider=context.get("provider", ""),
                model=context.get("model", ""),
                base_url=context.get("base_url", ""),
                attempt=1,
                total_attempts=1,
                retrying=False,
                elapsed_ms=(time.monotonic() - started_at) * 1000,
            )
            print(
                "LlamaIndex.retrieve() caused an error: {} | [EMBEDDING_CALL_ERROR] {}".format(
                    safe_exception_message_for_log(e),
                    details,
                ),
                flush=True,
            )
            return []

    def query(
        self,
        text,
        similarity_top_k=5,
        text_qa_template=None,
        refine_template=None,
        filters=None,
        query_creator=None,
    ):
        kwargs = {
            "similarity_top_k": similarity_top_k,
            "text_qa_template": text_qa_template,
            "refine_template": refine_template,
            "filters": filters,
        }

        def _run_query():
            if query_creator:
                query_engine = query_creator(retriever=self._index.as_retriever(**kwargs))
            else:
                query_engine = self._index.as_query_engine(**kwargs)
            return query_engine.query(text)

        return self._run_with_retry(
            "query",
            _run_query,
            lambda retry_count, error: "LlamaIndex.query()[retry={}] text_len={} error={}".format(
                retry_count,
                len(text or ""),
                error,
            ),
        )

    def save(self, path=None):
        path = path or self._path
        self._index.storage_context.persist(path)
        utils.save_dict(self._config, os.path.join(path, "index_config.json"))

    @property
    def nodes_num(self):
        return len(self._index.docstore.docs)
