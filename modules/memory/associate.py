"""generative_agents.memory.associate"""

import datetime
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever

from modules.storage.index import LlamaIndex
from modules import utils
from .event import Event


class Concept:
    def __init__(
        self,
        describe,
        node_id,
        node_type,
        subject,
        predicate,
        object,
        address,
        poignancy,
        create=None,
        expire=None,
        access=None,
        **metadata,
    ):
        self.node_id = node_id
        self.node_type = node_type
        self.event = Event(
            subject, predicate, object, describe=describe, address=address.split(":")
        )
        self.poignancy = poignancy
        self.create = utils.to_date(create) if create else utils.get_timer().get_date()
        if expire:
            self.expire = utils.to_date(expire)
        else:
            self.expire = self.create + datetime.timedelta(days=30)
        self.access = utils.to_date(access) if access else self.create
        self.metadata = dict(metadata or {})

    def abstract(self):
        return {
            "{}(P.{})".format(self.node_type, self.poignancy): str(self.event),
            "duration": "{} ~ {} (access: {})".format(
                self.create.strftime("%Y%m%d-%H:%M"),
                self.expire.strftime("%Y%m%d-%H:%M"),
                self.access.strftime("%Y%m%d-%H:%M"),
            ),
        }

    def __str__(self):
        return utils.dump_dict(self.abstract())

    @property
    def describe(self):
        return self.event.get_describe()

    @classmethod
    def from_node(cls, node):
        return cls(node.text, node.id_, **(node.metadata or {}))

    @classmethod
    def from_event(cls, node_id, node_type, event, poignancy):
        return cls(
            event.get_describe(),
            node_id,
            node_type,
            event.subject,
            event.predicate,
            event.object,
            ":".join(event.address),
            poignancy,
        )


class AssociateRetriever(BaseRetriever):
    def __init__(self, config, *args, **kwargs) -> None:
        self._config = config
        self._vector_retriever = VectorIndexRetriever(*args, **kwargs)
        super().__init__()

    def _retrieve(self, query_bundle):
        """Retrieve nodes given query."""

        nodes = self._vector_retriever.retrieve(query_bundle)
        if not nodes:
            return []
        profile = self._config.get("profile", {}) or {}

        def _safe_float(value, default):
            if isinstance(value, bool):
                return float(default)
            try:
                return float(value)
            except Exception:
                return float(default)

        recency_mul = _safe_float(profile.get("recency_weight_multiplier", 1.0), 1.0)
        relevance_mul = _safe_float(profile.get("relevance_weight_multiplier", 1.0), 1.0)
        importance_mul = _safe_float(profile.get("importance_weight_multiplier", 1.0), 1.0)

        role_bonus_cfg = profile.get("role_bonus", {}) or {}
        role_bonus_enabled = bool(role_bonus_cfg.get("enabled", False))
        role_bonus_val = _safe_float(role_bonus_cfg.get("bonus", 0.0), 0.0)
        role_bonus_doctor = str(role_bonus_cfg.get("doctor", "") or "")
        role_bonus_patients = set(role_bonus_cfg.get("patients", []) or [])

        def _node_role_bonus(node):
            if not role_bonus_enabled or role_bonus_val == 0:
                return 0.0
            metadata = getattr(node, "metadata", {}) or {}
            subject = str(metadata.get("subject", "") or "")
            obj = str(metadata.get("object", "") or "")
            if not role_bonus_doctor or not role_bonus_patients:
                return 0.0
            doctor_hit = (subject == role_bonus_doctor) or (obj == role_bonus_doctor)
            patient_hit = (subject in role_bonus_patients) or (obj in role_bonus_patients)
            if doctor_hit and patient_hit:
                return role_bonus_val
            return 0.0

        nodes = sorted(
            nodes, key=lambda n: utils.to_date(n.metadata["access"]), reverse=True
        )
        # get scores
        fac = self._config["recency_decay"]
        recency_scores = self._normalize(
            [fac**i for i in range(1, len(nodes) + 1)],
            self._config["recency_weight"] * recency_mul,
        )
        relevance_scores = self._normalize(
            [n.score for n in nodes],
            self._config["relevance_weight"] * relevance_mul,
        )
        importance_scores = self._normalize(
            [
                _safe_float((getattr(n, "metadata", {}) or {}).get("poignancy", 0), 0)
                for n in nodes
            ],
            self._config["importance_weight"] * importance_mul,
        )
        final_scores = {
            n.id_: r1 + r2 + i + _node_role_bonus(n)
            for n, r1, r2, i in zip(
                nodes, recency_scores, relevance_scores, importance_scores
            )
        }
        # re-rank nodes
        nodes = sorted(nodes, key=lambda n: final_scores[n.id_], reverse=True)
        retrieve_max = self._config.get("retrieve_max", 30)
        if isinstance(retrieve_max, bool):
            retrieve_max = 30
        try:
            retrieve_max = int(retrieve_max)
        except Exception:
            retrieve_max = 30
        if retrieve_max != -1:
            retrieve_max = max(1, retrieve_max)
            nodes = nodes[: retrieve_max]
        access_ts = utils.get_timer().get_date("%Y%m%d-%H:%M:%S")
        update_ok = True
        on_access_update = self._config.get("on_access_update")
        if callable(on_access_update):
            try:
                update_ok = bool(on_access_update([n.id_ for n in nodes], access_ts))
            except Exception:
                update_ok = False
        if update_ok:
            for n in nodes:
                n.metadata["access"] = access_ts
        return nodes

    def _normalize(self, data, factor=1, t_min=0, t_max=1):
        min_val, max_val = min(data), max(data)
        diff = max_val - min_val
        if diff == 0:
            return [(t_max - t_min) * factor / 2 for _ in data]
        return [(d - min_val) * (t_max - t_min) * factor / diff + t_min for d in data]


class Associate:
    def __init__(
        self,
        path,
        embedding,
        retention=8,
        max_memory=-1,
        max_importance=10,
        recency_decay=0.995,
        recency_weight=0.5,
        relevance_weight=3,
        importance_weight=2,
        memory=None,
        chat_retrieve=None,
        logger=None,
        external_write_hook=None,
        recent_dedup_limit=None,
    ):
        self._index = LlamaIndex(embedding, path)
        self.logger = logger
        self._external_write_hook = external_write_hook
        base_memory = {"event": [], "thought": [], "chat": []}
        if isinstance(memory, dict):
            for key in base_memory.keys():
                value = memory.get(key, [])
                if isinstance(value, list):
                    base_memory[key] = value
        self.memory = base_memory
        self.cleanup_index()
        self._prune_missing_memory_refs()
        self.retention = retention
        self.max_memory = max_memory
        self.max_importance = max_importance
        self.recent_dedup_limit = self._resolve_recent_dedup_limit(
            recent_dedup_limit,
            default=self.retention,
        )
        self._retrieve_config = {
            "recency_decay": recency_decay,
            "recency_weight": recency_weight,
            "relevance_weight": relevance_weight,
            "importance_weight": importance_weight,
        }
        if isinstance(chat_retrieve, dict):
            self._chat_retrieve_config = dict(chat_retrieve)
        else:
            self._chat_retrieve_config = {}
        self.chat_retrieve_mode = self._resolve_chat_retrieve_mode(
            self._chat_retrieve_config.get("mode", "direct")
        )
        self.chat_similarity_top_k = self._normalize_limit(
            self._chat_retrieve_config.get("similarity_top_k", 5),
            default=5,
            allow_unlimited=True,
            minimum=1,
        )

    def abstract(self):
        des = {"nodes": self._index.nodes_num}
        for t in ["event", "chat", "thought"]:
            des[t] = [c.describe for c in self._retrieve_nodes(t, limit=-1)]
        return des

    def __str__(self):
        return utils.dump_dict(self.abstract())

    def cleanup_index(self):
        node_ids = self._index.cleanup()
        self.memory = {
            n_type: [n for n in nodes if n not in node_ids]
            for n_type, nodes in self.memory.items()
        }
        self._prune_missing_memory_refs()

    def _prune_missing_memory_refs(self, node_type=None):
        if node_type is None:
            types = ["event", "chat", "thought"]
        else:
            types = [node_type]
        removed = {}
        for n_type in types:
            nodes = list(self.memory.get(n_type, []) or [])
            kept = []
            dropped = []
            for node_id in nodes:
                if self._index.has_node(node_id):
                    kept.append(node_id)
                else:
                    dropped.append(node_id)
            self.memory[n_type] = kept
            if dropped:
                removed[n_type] = dropped
        return removed

    def _valid_memory_ids(self, node_type):
        self._prune_missing_memory_refs(node_type=node_type)
        return self.memory.get(node_type, [])

    def _safe_node_filling(self, filling):
        if not isinstance(filling, dict):
            return {}
        allowed = {
            "forced",
            "meeting_id",
            "expire_days",
            "retrieval_scope",
        }
        safe = {}
        for key in allowed:
            if key not in filling:
                continue
            value = filling.get(key)
            if value is None:
                continue
            if isinstance(value, bool):
                safe[key] = value
            elif isinstance(value, int) and not isinstance(value, bool):
                safe[key] = value
            elif isinstance(value, float):
                safe[key] = value
            else:
                text = str(value).strip()
                if text:
                    safe[key] = text
        return safe

    def add_node(
        self,
        node_type,
        event,
        poignancy,
        create=None,
        expire=None,
        filling=None,
    ):
        create = create or utils.get_timer().get_date()
        expire = expire or (create + datetime.timedelta(days=30))
        metadata = {
            "node_type": node_type,
            "subject": event.subject,
            "predicate": event.predicate,
            "object": event.object,
            "address": ":".join(event.address),
            "poignancy": poignancy,
            "create": create.strftime("%Y%m%d-%H:%M:%S"),
            "expire": expire.strftime("%Y%m%d-%H:%M:%S"),
            "access": create.strftime("%Y%m%d-%H:%M:%S"),
        }
        metadata.update(self._safe_node_filling(filling))
        node = self._index.add_node(event.get_describe(), metadata)
        memory = self._valid_memory_ids(node_type)
        memory.insert(0, node.id_)
        if len(memory) > self.max_memory > 0:
            overflow = memory[self.max_memory:]
            if overflow:
                self._index.remove_nodes(overflow)
            self.memory[node_type] = memory[: self.max_memory]
        if callable(self._external_write_hook):
            try:
                self._external_write_hook(
                    {
                        "node_id": node.id_,
                        "node_type": node_type,
                        "event": event,
                        "create": create,
                        "expire": expire,
                    }
                )
            except Exception as exc:
                if self.logger:
                    self.logger.warning(
                        "[EXT_MEMORY_INGEST_FAIL] node_id={} node_type={} error={}".format(
                            node.id_,
                            node_type,
                            exc,
                        )
                    )
        return self.to_concept(node)

    def to_concept(self, node):
        return Concept.from_node(node)

    def _log_access_sync(self, level, message):
        logger = getattr(self, "logger", None)
        if not logger:
            return
        fn = getattr(logger, level, None)
        if not callable(fn):
            fn = getattr(logger, "info", None)
        if callable(fn):
            fn(message)

    def _touch_access_in_store(self, node_ids, access_ts):
        if not isinstance(access_ts, str) or not access_ts:
            self._log_access_sync("warning", "[ACCESS_SYNC_FAIL] reason=invalid_access_ts")
            return False
        ordered_ids, seen = [], set()
        for node_id in node_ids or []:
            if not isinstance(node_id, str) or not node_id:
                continue
            if node_id in seen:
                continue
            seen.add(node_id)
            ordered_ids.append(node_id)
        if not ordered_ids:
            self._log_access_sync("warning", "[ACCESS_SYNC_FAIL] reason=empty_node_ids")
            return False

        try:
            index_obj = getattr(self._index, "_index", None)
            docstore = getattr(index_obj, "docstore", None)
            docstore_docs = getattr(docstore, "docs", None) if docstore is not None else None
        except Exception as e:
            self._log_access_sync(
                "warning",
                "[ACCESS_SYNC_FAIL] reason=docstore_resolve_error detail={}".format(e),
            )
            return False
        if not isinstance(docstore_docs, dict):
            self._log_access_sync("warning", "[ACCESS_SYNC_FAIL] reason=docstore_unavailable")
            return False

        doc_targets = []
        for node_id in ordered_ids:
            node = docstore_docs.get(node_id)
            if node is None:
                self._log_access_sync(
                    "warning",
                    "[ACCESS_SYNC_FAIL] reason=docstore_node_not_found node_id={}".format(node_id),
                )
                return False
            metadata = getattr(node, "metadata", None)
            if not isinstance(metadata, dict):
                self._log_access_sync(
                    "warning",
                    "[ACCESS_SYNC_FAIL] reason=docstore_metadata_invalid node_id={}".format(node_id),
                )
                return False
            doc_targets.append((node_id, node, metadata.get("access")))

        try:
            vector_store = getattr(getattr(self._index, "_index", None), "vector_store", None)
            metadata_dict = None
            if vector_store is not None:
                data_obj = getattr(vector_store, "data", None)
                if data_obj is None:
                    data_obj = getattr(vector_store, "_data", None)
                if data_obj is not None:
                    metadata_dict = getattr(data_obj, "metadata_dict", None)
                if metadata_dict is None:
                    metadata_dict = getattr(vector_store, "metadata_dict", None)
        except Exception as e:
            self._log_access_sync(
                "warning",
                "[ACCESS_SYNC_FAIL] reason=vector_store_resolve_error detail={}".format(e),
            )
            return False
        if not isinstance(metadata_dict, dict):
            self._log_access_sync("warning", "[ACCESS_SYNC_FAIL] reason=vector_store_metadata_unavailable")
            return False

        vector_targets = []
        for node_id in ordered_ids:
            v_meta = metadata_dict.get(node_id)
            if not isinstance(v_meta, dict):
                self._log_access_sync(
                    "warning",
                    "[ACCESS_SYNC_FAIL] reason=vector_store_node_missing node_id={}".format(node_id),
                )
                return False
            vector_targets.append((v_meta, v_meta.get("access")))

        try:
            for v_meta, _old in vector_targets:
                v_meta["access"] = access_ts
        except Exception as e:
            self._log_access_sync(
                "warning",
                "[ACCESS_SYNC_FAIL] reason=vector_apply_failed detail={}".format(e),
            )
            return False

        try:
            for _node_id, node, _old in doc_targets:
                metadata = getattr(node, "metadata", None)
                if not isinstance(metadata, dict):
                    raise TypeError("docstore metadata is not dict")
                metadata["access"] = access_ts
        except Exception as e:
            self._log_access_sync(
                "warning",
                "[ACCESS_SYNC_PARTIAL] reason=docstore_apply_failed_no_rollback node_count={} access_ts={} detail={}".format(
                    len(ordered_ids),
                    access_ts,
                    e,
                ),
            )
            return True

        mismatch_ids = []
        for node_id, _node, _old in doc_targets:
            verify_node = docstore_docs.get(node_id)
            verify_meta = getattr(verify_node, "metadata", None)
            if not isinstance(verify_meta, dict):
                mismatch_ids.append(node_id)
                continue
            if verify_meta.get("access") != access_ts:
                mismatch_ids.append(node_id)

        if mismatch_ids:
            sample_ids = ",".join(mismatch_ids[:5])
            self._log_access_sync(
                "warning",
                "[ACCESS_SYNC_PARTIAL] reason=docstore_verify_mismatch node_count={} mismatch_count={} access_ts={} sample_node_ids={}".format(
                    len(ordered_ids),
                    len(mismatch_ids),
                    access_ts,
                    sample_ids,
                ),
            )
            return True

        self._log_access_sync(
            "info",
            "[ACCESS_SYNC_SUCCESS] node_count={} access_ts={}".format(
                len(ordered_ids),
                access_ts,
            ),
        )
        return True

    def find_concept(self, node_id):
        self._prune_missing_memory_refs()
        return self.to_concept(self._index.find_node(node_id))

    def _normalize_limit(
        self, value, default, allow_unlimited=False, minimum=1
    ):
        if isinstance(value, bool):
            value = default
        try:
            val = int(value)
        except Exception:
            val = default
        if allow_unlimited and val == -1:
            return -1
        if val < minimum:
            return default
        return val

    def _resolve_recent_dedup_limit(self, value, default):
        if isinstance(value, bool):
            value = default
        try:
            val = int(value)
        except Exception:
            val = default
        if val == 0:
            return 0
        return self._normalize_limit(
            val,
            default=default,
            allow_unlimited=True,
            minimum=1,
        )

    def _resolve_chat_retrieve_mode(self, value):
        mode = str(value or "direct").strip().lower()
        if mode not in {"direct", "semantic"}:
            return "direct"
        return mode

    def _retrieve_nodes(
        self, node_type, text=None, limit=None, similarity_top_k=None, node_ids=None
    ):
        node_ids = list(node_ids) if node_ids is not None else self._valid_memory_ids(node_type)
        final_limit = self._normalize_limit(
            self.retention if limit is None else limit,
            default=self.retention,
            allow_unlimited=True,
            minimum=1,
        )
        if text:
            filters = MetadataFilters(
                filters=[ExactMatchFilter(key="node_type", value=node_type)]
            )
            retrieve_kwargs = {
                "filters": filters,
                "node_ids": node_ids,
            }
            if similarity_top_k is not None:
                retrieve_kwargs["similarity_top_k"] = similarity_top_k
            nodes = self._index.retrieve(text, **retrieve_kwargs)
        else:
            nodes = [self._index.find_node(n) for n in node_ids]
        if final_limit == -1:
            selected = nodes
        else:
            selected = nodes[:final_limit]
        return [self.to_concept(n) for n in selected]

    def retrieve_events(self, text=None, limit=None):
        return self._retrieve_nodes("event", text=text, limit=limit)

    def retrieve_thoughts(self, text=None, limit=None):
        return self._retrieve_nodes("thought", text=text, limit=limit)

    def _chat_matches_name(self, concept, name=None):
        if not name:
            return True
        return concept.event.subject == name or concept.event.object == name

    def _is_forced_chat(self, concept):
        value = (getattr(concept, "metadata", {}) or {}).get("forced", False)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y"}
        return bool(value)

    def _sort_chat_concepts(self, concepts, prefer_forced=False):
        return sorted(
            concepts,
            key=lambda c: (
                1 if prefer_forced and self._is_forced_chat(c) else 0,
                c.create,
            ),
            reverse=True,
        )

    def _limit_chat_concepts(self, concepts, limit=None):
        final_limit = self._normalize_limit(
            self.retention if limit is None else limit,
            default=self.retention,
            allow_unlimited=True,
            minimum=1,
        )
        if final_limit == -1:
            return concepts
        return concepts[:final_limit]

    def _chat_candidate_concepts(self, name=None):
        selected = []
        for node_id in self._valid_memory_ids("chat"):
            concept = self.find_concept(node_id)
            if not self._chat_matches_name(concept, name=name):
                continue
            selected.append(concept)
        return selected

    def _retrieve_chats_direct(self, name=None, limit=None, prefer_forced=False):
        selected = []
        for concept in self._chat_candidate_concepts(name=name):
            selected.append(concept)
        selected = self._sort_chat_concepts(
            selected,
            prefer_forced=prefer_forced,
        )
        return self._limit_chat_concepts(selected, limit=limit)

    def retrieve_chats(
        self,
        name=None,
        limit=None,
        mode=None,
        query=None,
        prefer_forced=False,
        force_direct=False,
    ):
        mode = self._resolve_chat_retrieve_mode(
            mode if mode is not None else self.chat_retrieve_mode
        )
        if force_direct or mode == "direct":
            return self._retrieve_chats_direct(
                name=name,
                limit=limit,
                prefer_forced=prefer_forced,
            )
        candidates = self._chat_candidate_concepts(name=name)
        if not candidates:
            return []
        text = str(query or "").strip()
        if not text:
            text = ("对话 " + name) if name else None
        if not text:
            candidates = self._sort_chat_concepts(
                candidates,
                prefer_forced=prefer_forced,
            )
            return self._limit_chat_concepts(candidates, limit=limit)
        candidate_ids = [c.node_id for c in candidates]
        similarity_top_k = self.chat_similarity_top_k
        if similarity_top_k == -1:
            similarity_top_k = max(1, len(candidate_ids))
        retrieved = self._retrieve_nodes(
            "chat",
            text=text,
            limit=-1,
            similarity_top_k=similarity_top_k if text else None,
            node_ids=candidate_ids,
        )
        if prefer_forced:
            seen = {c.node_id for c in retrieved}
            for concept in candidates:
                if concept.node_id in seen or not self._is_forced_chat(concept):
                    continue
                retrieved.append(concept)
                seen.add(concept.node_id)
        retrieved = self._sort_chat_concepts(
            retrieved,
            prefer_forced=prefer_forced,
        )
        return self._limit_chat_concepts(retrieved, limit=limit)

    def retrieve_focus(self, focus, retrieve_max=30, reduce_all=True, retrieval_profile=None):
        def _create_retriever(*args, **kwargs):
            retrieve_cfg = dict(self._retrieve_config)
            retrieve_cfg["retrieve_max"] = retrieve_max
            retrieve_cfg["profile"] = retrieval_profile or {}
            retrieve_cfg["on_access_update"] = self._touch_access_in_store
            return AssociateRetriever(retrieve_cfg, *args, **kwargs)

        retrieved = {}
        node_ids = self.memory["event"] + self.memory["thought"]
        for text in focus:
            nodes = self._index.retrieve(
                text,
                similarity_top_k=len(node_ids),
                node_ids=node_ids,
                retriever_creator=_create_retriever,
            )
            if reduce_all:
                retrieved.update({n.id_: n for n in nodes})
            else:
                retrieved[text] = nodes
        if reduce_all:
            return [self.to_concept(v) for v in retrieved.values()]
        return {
            text: [self.to_concept(n) for n in nodes]
            for text, nodes, in retrieved.items()
        }

    def get_relation(self, node):
        return {
            "node": node,
            "events": self.retrieve_events(node.describe),
            "thoughts": self.retrieve_thoughts(node.describe),
        }

    def to_dict(self):
        self._index.save()
        return {
            "memory": self.memory,
            "recency_decay": self._retrieve_config.get("recency_decay", 0.995),
            "recency_weight": self._retrieve_config.get("recency_weight", 0.5),
            "relevance_weight": self._retrieve_config.get("relevance_weight", 3),
            "importance_weight": self._retrieve_config.get("importance_weight", 2),
            "recent_dedup_limit": self.recent_dedup_limit,
        }

    @property
    def index(self):
        return self._index
