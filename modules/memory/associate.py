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
        return cls(node.text, node.id_, **node.metadata)

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
        for n in nodes:
            n.metadata["access"] = utils.get_timer().get_date("%Y%m%d-%H:%M:%S")
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
    ):
        self._index = LlamaIndex(embedding, path)
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
        node = self._index.add_node(event.get_describe(), metadata)
        memory = self._valid_memory_ids(node_type)
        memory.insert(0, node.id_)
        if len(memory) > self.max_memory > 0:
            overflow = memory[self.max_memory:]
            if overflow:
                self._index.remove_nodes(overflow)
            self.memory[node_type] = memory[: self.max_memory]
        return self.to_concept(node)

    def to_concept(self, node):
        return Concept.from_node(node)

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

    def _resolve_chat_retrieve_mode(self, value):
        mode = str(value or "direct").strip().lower()
        if mode not in {"direct", "semantic"}:
            return "direct"
        return mode

    def _retrieve_nodes(
        self, node_type, text=None, limit=None, similarity_top_k=None
    ):
        node_ids = self._valid_memory_ids(node_type)
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

    def _retrieve_chats_direct(self, name=None, limit=None):
        selected = []
        for node_id in self._valid_memory_ids("chat"):
            concept = self.find_concept(node_id)
            if name:
                if (
                    concept.event.subject != name
                    and concept.event.object != name
                ):
                    continue
            selected.append(concept)
        final_limit = self._normalize_limit(
            self.retention if limit is None else limit,
            default=self.retention,
            allow_unlimited=True,
            minimum=1,
        )
        if final_limit == -1:
            return selected
        return selected[:final_limit]

    def retrieve_chats(self, name=None, limit=None, mode=None):
        mode = self._resolve_chat_retrieve_mode(
            mode if mode is not None else self.chat_retrieve_mode
        )
        if mode == "direct":
            return self._retrieve_chats_direct(name=name, limit=limit)
        text = ("对话 " + name) if name else None
        similarity_top_k = self.chat_similarity_top_k
        if similarity_top_k == -1:
            similarity_top_k = max(1, len(self.memory["chat"]))
        return self._retrieve_nodes(
            "chat",
            text=text,
            limit=limit,
            similarity_top_k=similarity_top_k if text else None,
        )

    def retrieve_focus(self, focus, retrieve_max=30, reduce_all=True, retrieval_profile=None):
        def _create_retriever(*args, **kwargs):
            retrieve_cfg = dict(self._retrieve_config)
            retrieve_cfg["retrieve_max"] = retrieve_max
            retrieve_cfg["profile"] = retrieval_profile or {}
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
        return {"memory": self.memory}

    @property
    def index(self):
        return self._index
