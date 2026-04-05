"""Intervention manager with low-intrusion hooks.

阶段 1 + 阶段 2 + 阶段 3：
- 保持低侵入：通过 hook 接入，不重写主认知流程
- 支持周期规则触发 meeting lock（weekly / interval_days / interval_minutes / interval_steps）
- 会话后自动解锁并提取医嘱，写入患者 forced_tasks
- make_schedule 阶段合并到日程（hard plan）
"""

from __future__ import annotations

import datetime
import copy
import json
import os
import traceback
from string import Template
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

from modules import depression_runtime_manager as drm
from modules.intervention_consult_record import (
    ConsultRecordError,
    ConsultRecordPromptError,
    ConsultRecordValidationError,
    build_dedup_key,
    build_full_record,
    build_pair_key,
    flatten_record_for_injection,
    project_whitelist,
    to_conversation_text,
    validate_full_record,
    validate_soap_only,
)
from modules.model import create_llm_model
from modules.session_prompt_injection_manager import SessionPromptInjectionManager


@dataclass
class InterventionLock:
    enabled: bool = False
    mode: str = "meeting"
    priority: str = "hard"
    target_agent: str = ""
    meeting_id: str = ""
    expire_at: str = ""
    trigger_step: int = -1
    deferred_by_sleep: bool = False
    defer_count: int = 0


class InterventionManager:
    """Runtime intervention orchestrator."""

    def __init__(self, config: Dict[str, Any], logger: Optional[Any] = None):
        self.config = config
        self.logger = logger
        self._agents_ref: Dict[str, Any] = {}

        intervention_cfg = config.get("intervention", {}) or {}
        self.enabled = bool(intervention_cfg.get("enabled", False))
        self.doctor = intervention_cfg.get("doctor", "")
        self.patients = intervention_cfg.get("patients", []) or []
        self.meeting_rules = intervention_cfg.get("meeting_rules", []) or []
        self.meeting_queue_cfg = intervention_cfg.get("meeting_queue", {}) or {}
        self.meeting_queue_enabled = bool(self.meeting_queue_cfg.get("enabled", False))
        self.meeting_queue_max_per_doctor = self._safe_int(
            self.meeting_queue_cfg.get("max_queue_per_doctor", 128), 128
        )
        if self.meeting_queue_max_per_doctor <= 0:
            self.meeting_queue_max_per_doctor = 128
        self.order_extract = intervention_cfg.get("order_extract", {}) or {}
        self.session_prompt_injection_cfg = intervention_cfg.get("session_prompt_injection", {}) or {}
        self.stop_rule_scheduling_on_session_completed = bool(
            self.session_prompt_injection_cfg.get("stop_rule_scheduling_on_session_completed", False)
        )
        self.chat_controls_cfg = intervention_cfg.get("chat_controls", {}) or {}
        self.forced_llm_cfg = intervention_cfg.get("forced_llm", {}) or {}
        self.consult_record_cfg = intervention_cfg.get("consult_record", {}) or {}
        self._consult_record_llm = None
        self._consult_record_llm_key = ""

        # 全局运行时状态（随 config 落盘）
        self.state = config.setdefault(
            "intervention_state",
            {
                "active_meetings": {},
                "rule_last_trigger": {},
                "doctor_meeting_queues": {},
                "doctor_current_meeting": {},
                "meeting_dedup": {},
                "meeting_seq": 0,
            },
        )
        self._ensure_state_schema()
        self.session_prompt_injection = SessionPromptInjectionManager(
            config=self.config,
            state=self.state,
            logger=self.logger,
        )

        self._log(
            "InterventionManager initialized: enabled={}, doctor={}, patients={}, rules={}".format(
                self.enabled,
                self.doctor,
                len(self.patients),
                len(self.meeting_rules),
            )
        )
        self._log_highlight(
            "MEETING_QUEUE_INIT enabled={} max_queue_per_doctor={}".format(
                self.meeting_queue_enabled,
                self.meeting_queue_max_per_doctor,
            )
        )

    def on_step_start(self, game: Any, now: Any) -> None:
        """每个 step 开始时调用：匹配规则并下发锁定。"""
        step_no = int(self.config.get("step", 0) or 0) + 1
        self._log_highlight(
            "TICK step={} now={} enabled={} rules={}".format(
                step_no, self._fmt_dt(now), self.enabled, len(self.meeting_rules)
            )
        )
        if not self.enabled:
            return

        self._agents_ref = getattr(game, "agents", {}) if isinstance(getattr(game, "agents", {}), dict) else {}
        self._refresh_meeting_queue_cfg()
        self._ensure_state_schema()
        self._log_queue_event(
            event="queue_switch_hit",
            doctor=self.doctor,
            meeting_id="",
            pair_key="",
            step_phase="",
            reason="meeting_queue_enabled" if self.meeting_queue_enabled else "meeting_queue_disabled",
        )

        now_step = int(self.config.get("step", 0) or 0) + 1
        update_cfg = (self.config.get("intervention", {}) or {}).get("depression_update", {}) or {}
        for agent in game.agents.values():
            profile = drm.ensure_profile(getattr(agent, "depression_profile", {}))
            profile = self._ensure_depression_runtime_initialized(
                agent=agent,
                profile=profile,
                now_step=now_step,
                update_cfg=update_cfg,
            )
            drm.decay_short_term(profile, now_step, update_cfg)
            changed, next_profile = drm.try_rebase_case_config(profile, now_step, update_cfg)
            agent.depression_profile = next_profile
            if changed:
                self._log_highlight(
                    "[DEPR][REBASE] agent={} step={} applied=true".format(agent.name, now_step)
                )

        if self.meeting_queue_enabled:
            self._recover_queue_state(game, now)

        self._cleanup_expired_locks(game, now)
        if self.stop_rule_scheduling_on_session_completed:
            purged = self._purge_completed_pairs_meetings(
                now=now,
                reason="session_completed_step_start_purge",
            )
            if purged > 0:
                self._log_highlight(
                    "SESSION_COMPLETED_PURGE step={} purged_meetings={}".format(step_no, purged)
                )

        for idx, rule in enumerate(self.meeting_rules):
            rule_id = rule.get("rule_id", f"rule_{idx}")
            if not rule.get("enabled", True):
                self._log_highlight("RULE_SKIPPED rule_id={} reason=disabled".format(rule_id))
                continue
            blocked, blocked_patient = self._is_rule_blocked_by_session_completed(rule)
            if blocked:
                self._log_highlight(
                    "RULE_SKIPPED rule_id={} reason=session_completed_stop_schedule patient={}".format(
                        rule_id,
                        blocked_patient,
                    )
                )
                continue
            if self._should_trigger(rule_id, rule, now):
                self._trigger_meeting(game, rule_id, rule, now)

        if self.meeting_queue_enabled:
            self._promote_all_doctors(game, now)

    def before_agent_think(self, agent: Any, agents: Dict[str, Any], now: Any) -> None:
        """Agent think 前调用：兜底状态，并在 lock 期间强制目标指向。"""
        if not self.enabled:
            return
        if isinstance(agents, dict):
            self._agents_ref = agents

        self._ensure_agent_state(agent)
        lock = agent.status["intervention"]["lock"]
        if not lock.get("enabled"):
            return

        is_awake = self._agent_is_awake(agent)
        if not is_awake:
            lock["deferred_by_sleep"] = True
            lock["defer_count"] = int(lock.get("defer_count", 0) or 0) + 1
            self._log_highlight(
                "LOCK_DEFERRED_SLEEP agent={} meeting_id={} defer_count={} expire_at={}".format(
                    agent.name,
                    lock.get("meeting_id", ""),
                    lock.get("defer_count", 0),
                    lock.get("expire_at", ""),
                )
            )

        if self._is_lock_expired(lock, now):
            if bool(lock.get("deferred_by_sleep", False)) and not is_awake:
                defer_minutes = self._lock_defer_window_minutes()
                next_expire = now + datetime.timedelta(minutes=defer_minutes)
                lock["expire_at"] = self._fmt_dt(next_expire)
                self._log_highlight(
                    "LOCK_EXPIRE_EXTENDED_SLEEP agent={} meeting_id={} defer_minutes={} next_expire_at={}".format(
                        agent.name,
                        lock.get("meeting_id", ""),
                        defer_minutes,
                        lock.get("expire_at", ""),
                    )
                )
                return
            self._log_highlight("LOCK_EXPIRED agent={} now={}".format(agent.name, self._fmt_dt(now)))
            self._clear_lock(agent)
            return

        if is_awake and bool(lock.get("deferred_by_sleep", False)):
            self._log_highlight(
                "LOCK_RESUMED_AWAKE agent={} meeting_id={} defer_count={}".format(
                    agent.name,
                    lock.get("meeting_id", ""),
                    lock.get("defer_count", 0),
                )
            )
            lock["deferred_by_sleep"] = False

        target = lock.get("target_agent", "")
        if target and target in agents and getattr(agent, "action", None):
            # 低侵入：复用 Agent.find_path 的 <persona, target> 逻辑
            agent.action.event.address = ["<persona>", target]
            self._log_highlight(
                "BEFORE_THINK agent={} rewrite_address=<persona,{}> meeting_id={}".format(
                    agent.name,
                    target,
                    lock.get("meeting_id", ""),
                )
            )

    def after_chat(self, speaker: Any, other: Any, chats: Any, summary: str, start_time: Any) -> None:
        """会话后回调：若命中锁定关系，清理 lock 并抽取医嘱。"""
        if not self.enabled:
            return

        self._ensure_agent_state(speaker)
        self._ensure_agent_state(other)

        lock_s = speaker.status["intervention"]["lock"]
        lock_o = other.status["intervention"]["lock"]
        meeting_id_s = lock_s.get("meeting_id", "")
        meeting_id_o = lock_o.get("meeting_id", "")
        closed_meeting_id = ""

        # 双方 meeting_id 一致时认为会面闭环达成
        if meeting_id_s and meeting_id_s == meeting_id_o:
            closed_meeting_id = str(meeting_id_s or "")
            meeting_snapshot = self.state.get("active_meetings", {}).get(meeting_id_s, {}) or {}
            meeting_doctor = str(meeting_snapshot.get("doctor", "") or "")
            meeting_patient = str(meeting_snapshot.get("patient", "") or "")
            pair_key = "{}<->{}".format(meeting_doctor or getattr(speaker, "name", ""), meeting_patient or getattr(other, "name", ""))
            self._clear_lock(speaker)
            self._clear_lock(other)
            if self.meeting_queue_enabled:
                self._finalize_meeting(meeting_id_s, start_time, reason="chat_finished")
                self._promote_next_for_doctor_by_pair(speaker, other, start_time)
            else:
                self.state.get("active_meetings", {}).pop(meeting_id_s, None)
            speaker.status["intervention"]["last_intervention_at"] = self._fmt_dt(start_time)
            other.status["intervention"]["last_intervention_at"] = self._fmt_dt(start_time)
            self._log_highlight(
                "MEETING_FINISHED meeting_id={} pair={}<->{}".format(
                    meeting_id_s,
                    getattr(speaker, "name", ""),
                    getattr(other, "name", ""),
                )
            )
            self._log_queue_event(
                event="meeting_finished",
                doctor=meeting_doctor,
                meeting_id=meeting_id_s,
                pair_key=pair_key,
                step_phase=meeting_snapshot.get("step_phase", ""),
                reason="chat_finished",
            )

        # 医患会话后尝试抽取医嘱并写入患者 forced_tasks
        self._log_highlight(
            "ORDER_EXTRACT_START speaker={} other={} chats={}".format(
                getattr(speaker, "name", ""),
                getattr(other, "name", ""),
                len(chats or []),
            )
        )
        self._extract_and_queue_orders(speaker, other, chats)

        if self.session_prompt_injection and self.enabled:
            doctor_for_session, patient_for_session = self._resolve_doctor_patient_pair(speaker, other)
            if doctor_for_session and patient_for_session:
                lock_s_meeting_id = str(lock_s.get("meeting_id", "") or "")
                lock_o_meeting_id = str(lock_o.get("meeting_id", "") or "")
                meeting_id_for_session = lock_s_meeting_id or lock_o_meeting_id
                if meeting_id_for_session:
                    audit = self.session_prompt_injection.on_chat_finished(
                        doctor_name=doctor_for_session.name,
                        patient_name=patient_for_session.name,
                        chats=chats or [],
                        meeting_id=meeting_id_for_session,
                        now_str=self._fmt_dt(start_time),
                    )
                    self._log_highlight(
                        "SESSION_PROMPT_AFTER_CHAT pair={}<->{} action={} current_session={} completed={}".format(
                            doctor_for_session.name,
                            patient_for_session.name,
                            audit.get("action", ""),
                            audit.get("current_session", ""),
                            bool(audit.get("completed", False)),
                        )
                    )
                    if self.stop_rule_scheduling_on_session_completed and bool(audit.get("completed", False)):
                        purged = self._purge_patient_meetings(
                            patient_name=patient_for_session.name,
                            now=start_time,
                            reason="session_completed_purge",
                        )
                        self._log_highlight(
                            "SESSION_COMPLETED_RULE_SHUTDOWN pair={}<->{} purged_meetings={}".format(
                                doctor_for_session.name,
                                patient_for_session.name,
                                purged,
                            )
                        )

        self._handle_consult_record_after_chat(
            speaker=speaker,
            other=other,
            chats=chats,
            summary=summary,
            start_time=start_time,
            meeting_id=closed_meeting_id,
        )

        doctor, patient = self._resolve_doctor_patient_pair(speaker, other)
        if not doctor or not patient:
            return

        now_step = int(self.config.get("step", 0) or 0)
        update_cfg = (self.config.get("intervention", {}) or {}).get("depression_update", {}) or {}
        update_cfg = copy.deepcopy(update_cfg)
        update_cfg["llm_patch_required_on_doctor_chat"] = bool(
            update_cfg.get("llm_patch_required_on_doctor_chat", True)
        )
        depr_summary = self._build_depr_summary_with_consult_fallback(
            speaker=speaker,
            other=other,
            chat_summary=summary or "",
            meeting_id=closed_meeting_id,
        )
        before = copy.deepcopy(getattr(patient, "depression_profile", {}))
        result = drm.evaluate_post_chat_update(
            patient_agent=patient,
            chats=chats or [],
            summary=depr_summary,
            now_step=now_step,
            now_time=self._fmt_dt(start_time),
            update_cfg=update_cfg,
        )
        after = copy.deepcopy(getattr(patient, "depression_profile", {}))
        audit = drm.diff_runtime(before, after)

        self._log_highlight(
            "[DEPR][EVAL] patient={} step={} throttled={} reason={}".format(
                patient.name,
                now_step,
                bool(result.get("throttled", False)),
                result.get("reason", ""),
            )
        )
        cfg_adjustments = result.get("cfg_adjustments", {}) if isinstance(result, dict) else {}
        if cfg_adjustments:
            self._log_highlight(
                "[DEPR][CFG] patient={} normalized={}".format(
                    patient.name,
                    cfg_adjustments,
                )
            )

        self._log_highlight(
            "[DEPR][EVENT] patient={} wording={} long_term_triggered={} chat_count={}".format(
                patient.name,
                result.get("current_event_wording", ""),
                bool(result.get("long_term_triggered", False)),
                int(result.get("doctor_chat_count", 0) or 0),
            )
        )
        if result.get("throttled"):
            self._log_highlight(
                "[DEPR][THROTTLE] patient={} step={} reason={}".format(
                    patient.name,
                    now_step,
                    result.get("reason", ""),
                )
            )
        self._log_highlight(
            "[DEPR][DIFF] patient={} changed_paths={}".format(
                patient.name,
                audit.get("changed_paths", []),
            )
        )
        self._log_highlight(
            "[DEPR][LAYER] patient={} short_term_changed={} long_term_changed={} llm_patch_attempted={} llm_patch_used={}".format(
                patient.name,
                audit.get("short_term_changed", False),
                audit.get("long_term_changed", False),
                bool(result.get("llm_patch_attempted", False)),
                bool(result.get("llm_patch_used", False)),
            )
        )

    def get_doctor_session_prompt_injection(self, speaker: Any, other: Any, forced: bool) -> str:
        if not self.enabled:
            return ""
        if not bool(forced):
            return ""

        doctor, patient = self._resolve_doctor_patient_pair(speaker, other)
        if not doctor or not patient:
            return ""
        if getattr(speaker, "name", "") != doctor.name:
            return ""

        self._ensure_agent_state(doctor)
        lock = doctor.status.get("intervention", {}).get("lock", {})
        meeting_id = ""
        if isinstance(lock, dict):
            meeting_id = str(lock.get("meeting_id", "") or "")

        base_block = ""
        if self.session_prompt_injection:
            base_block = self.session_prompt_injection.get_doctor_injection_block(
                doctor_name=doctor.name,
                patient_name=patient.name,
                forced=bool(forced),
                meeting_id=meeting_id,
            )

        consult_block = ""
        if self._consult_record_enabled():
            pair_key = build_pair_key(doctor.name, patient.name)
            record = self._get_latest_consult_record_by_pair(pair_key)
            if isinstance(record, dict) and record:
                self._log_consult(
                    "INJECTION start pair_key={} meeting_id={}".format(
                        pair_key,
                        meeting_id or "<empty>",
                    )
                )
                try:
                    prompt_tpl = self._load_prompt_txt_or_raise(self._consult_prompt_injection_file())
                    prompt_text = self._render_prompt_template(
                        prompt_tpl,
                        flatten_record_for_injection(record),
                    )
                    consult_block = (
                        "<DOCTOR_CONSULT_RECORD_INJECTION>\n"
                        + prompt_text
                        + "\n</DOCTOR_CONSULT_RECORD_INJECTION>"
                    )
                    self._log_consult(
                        "INJECTION ready pair_key={} record_id={} chars={}".format(
                            pair_key,
                            str(record.get("record_id", "") or ""),
                            len(consult_block),
                        )
                    )
                except ConsultRecordError as err:
                    self._log_consult(
                        "INJECTION skipped pair_key={} reason={} retryable={}".format(
                            pair_key,
                            err.reason,
                            bool(err.retryable),
                        )
                    )

        return self._join_prompt_blocks(base_block, consult_block)

    def get_forced_chat_advance_marker(self, speaker: Any, other: Any, forced: bool = False) -> str:
        if not self.enabled:
            return ""
        if not bool(forced):
            return ""

        doctor, patient = self._resolve_doctor_patient_pair(speaker, other)
        if not doctor or not patient:
            return ""

        cfg = self.session_prompt_injection_cfg if isinstance(self.session_prompt_injection_cfg, dict) else {}
        marker = str(cfg.get("advance_marker", "[SESSION_END]") or "[SESSION_END]").strip()
        return marker

    def is_forced_chat_advance_marker_hit(
        self,
        speaker: Any,
        other: Any,
        text: str,
        forced: bool = False,
    ) -> bool:
        marker = self.get_forced_chat_advance_marker(speaker, other, forced=forced)
        if not marker:
            return False

        doctor, patient = self._resolve_doctor_patient_pair(speaker, other)
        if not doctor or not patient:
            return False

        speaker_name = str(getattr(speaker, "name", "") or "")
        doctor_name = str(getattr(doctor, "name", "") or "")
        if speaker_name != doctor_name:
            return False

        return marker in str(text or "")

    def get_chat_control_policy(self, initiator: Any, responder: Any, forced: bool) -> Dict[str, Any]:
        fallback = self._default_chat_control_policy()
        initiator_name = getattr(initiator, "name", "")
        responder_name = getattr(responder, "name", "")

        if not bool(forced):
            fallback["source"] = "normal_chat"
            self._log_highlight(
                "[CHAT_CTRL_POLICY] initiator={} responder={} forced={} source={} enabled={} forced_only={} notes={}".format(
                    initiator_name,
                    responder_name,
                    bool(forced),
                    fallback.get("source", "fallback"),
                    bool(fallback.get("enabled", False)),
                    bool(fallback.get("forced_only", True)),
                    "-",
                )
            )
            return fallback

        intervention_cfg = self.config.get("intervention", {}) or {}
        chat_controls = intervention_cfg.get("chat_controls", {}) or {}
        if not isinstance(chat_controls, dict) or not chat_controls:
            fallback["source"] = "fallback"
            fallback["normalize_notes"] = ["chat_controls_missing"]
            self._log_highlight(
                "[CHAT_CTRL_FALLBACK] initiator={} responder={} reason=chat_controls_missing".format(
                    initiator_name,
                    responder_name,
                )
            )
            self._log_highlight(
                "[CHAT_CTRL_POLICY] initiator={} responder={} forced={} source={} enabled={} forced_only={} notes={}".format(
                    initiator_name,
                    responder_name,
                    bool(forced),
                    fallback.get("source", "fallback"),
                    bool(fallback.get("enabled", False)),
                    bool(fallback.get("forced_only", True)),
                    "|".join(fallback.get("normalize_notes", []) or []) or "-",
                )
            )
            return fallback

        global_enabled = self._safe_bool(chat_controls.get("enabled", False), False)
        if not global_enabled:
            fallback["source"] = "chat_controls_disabled"
            fallback["normalize_notes"] = ["global_disabled"]
            self._log_highlight(
                "[CHAT_CTRL_POLICY] initiator={} responder={} forced={} source={} enabled={} forced_only={} notes={}".format(
                    initiator_name,
                    responder_name,
                    bool(forced),
                    fallback.get("source", "fallback"),
                    bool(fallback.get("enabled", False)),
                    bool(fallback.get("forced_only", True)),
                    "|".join(fallback.get("normalize_notes", []) or []) or "-",
                )
            )
            return fallback

        raw_defaults = chat_controls.get("defaults", {}) or {}
        raw_overrides = chat_controls.get("agent_overrides", {}) or {}
        raw_policy = {}
        source = "fallback"

        if isinstance(raw_defaults, dict):
            raw_policy.update(raw_defaults)
            source = "defaults"
        else:
            self._log_highlight("[CHAT_CTRL_CFG_INVALID] field=defaults reason=not_dict use_empty=true")

        if isinstance(raw_overrides, dict):
            override = raw_overrides.get(initiator_name)
            if isinstance(override, dict):
                raw_policy.update(override)
                source = "agent_override"
            elif override is not None:
                self._log_highlight(
                    "[CHAT_CTRL_CFG_INVALID] field=agent_overrides.{} reason=not_dict use_defaults=true".format(
                        initiator_name
                    )
                )
        else:
            self._log_highlight("[CHAT_CTRL_CFG_INVALID] field=agent_overrides reason=not_dict use_defaults=true")

        if not raw_policy:
            fallback["source"] = "fallback"
            fallback["normalize_notes"] = ["empty_policy"]
            self._log_highlight(
                "[CHAT_CTRL_FALLBACK] initiator={} responder={} reason=empty_policy".format(
                    initiator_name,
                    responder_name,
                )
            )
            self._log_highlight(
                "[CHAT_CTRL_POLICY] initiator={} responder={} forced={} source={} enabled={} forced_only={} notes={}".format(
                    initiator_name,
                    responder_name,
                    bool(forced),
                    fallback.get("source", "fallback"),
                    bool(fallback.get("enabled", False)),
                    bool(fallback.get("forced_only", True)),
                    "|".join(fallback.get("normalize_notes", []) or []) or "-",
                )
            )
            return fallback

        policy = self._normalize_chat_control_policy(raw_policy, source)
        if bool(policy.get("forced_only", True)) and (not bool(forced)):
            policy["enabled"] = False
            policy.setdefault("normalize_notes", []).append("forced_only_guard")

        self._log_highlight(
            "[CHAT_CTRL_POLICY] initiator={} responder={} forced={} source={} enabled={} forced_only={} notes={}".format(
                initiator_name,
                responder_name,
                bool(forced),
                policy.get("source", source),
                bool(policy.get("enabled", False)),
                bool(policy.get("forced_only", True)),
                "|".join(policy.get("normalize_notes", []) or []) or "-",
            )
        )
        return policy

    def get_memory_retrieval_profile(self, speaker: Any, other: Any, forced: bool = False) -> Dict[str, Any]:
        speaker_name = getattr(speaker, "name", "")
        other_name = getattr(other, "name", "")
        disabled = {
            "enabled": False,
            "profile": {},
            "scope": "",
            "reason": "disabled",
        }
        if not self.enabled:
            disabled["reason"] = "intervention_disabled"
            return disabled
        intervention_cfg = self.config.get("intervention", {}) or {}
        memory_policy = intervention_cfg.get("memory_policy", {}) or {}
        if not isinstance(memory_policy, dict) or not memory_policy:
            disabled["reason"] = "memory_policy_missing"
            return disabled
        if not self._safe_bool(memory_policy.get("enabled", False), False):
            disabled["reason"] = "memory_policy_disabled"
            return disabled

        scope = str(memory_policy.get("scope", "forced_chat_only") or "forced_chat_only").strip()
        if scope == "forced_chat_only" and (not bool(forced)):
            disabled["scope"] = scope
            disabled["reason"] = "scope_guard_non_forced"
            return disabled

        doctor, patient = self._resolve_doctor_patient_pair(speaker, other)
        if not doctor or not patient:
            disabled["scope"] = scope
            disabled["reason"] = "not_doctor_patient_pair"
            return disabled

        profile = memory_policy.get("retrieve_profile", {}) or {}
        if not isinstance(profile, dict):
            disabled["scope"] = scope
            disabled["reason"] = "retrieve_profile_invalid"
            return disabled

        result = {
            "enabled": True,
            "scope": scope,
            "reason": "matched",
            "profile": profile,
        }
        self._log_highlight(
            "[MEMORY_RETRIEVE_PROFILE] speaker={} other={} forced={} enabled={} scope={} reason={}".format(
                speaker_name,
                other_name,
                bool(forced),
                bool(result.get("enabled", False)),
                result.get("scope", ""),
                result.get("reason", ""),
            )
        )
        return result

    def get_forced_chat_expire_days(self, speaker: Any, other: Any, forced: bool = False) -> Optional[int]:
        speaker_name = getattr(speaker, "name", "")
        other_name = getattr(other, "name", "")
        if not self.enabled:
            return None
        intervention_cfg = self.config.get("intervention", {}) or {}
        memory_policy = intervention_cfg.get("memory_policy", {}) or {}
        if not isinstance(memory_policy, dict) or not memory_policy:
            return None
        if not self._safe_bool(memory_policy.get("enabled", False), False):
            return None
        scope = str(memory_policy.get("scope", "forced_chat_only") or "forced_chat_only").strip()
        if scope == "forced_chat_only" and (not bool(forced)):
            return None
        doctor, patient = self._resolve_doctor_patient_pair(speaker, other)
        if not doctor or not patient:
            return None

        raw_days = memory_policy.get("forced_chat_expire_days", 30)
        try:
            expire_days = int(raw_days)
        except Exception:
            expire_days = 30

        if expire_days == -1:
            self._log_highlight(
                "[FORCED_CHAT_EXPIRE_POLICY] speaker={} other={} forced={} expire_days=-1(long_keep)".format(
                    speaker_name,
                    other_name,
                    bool(forced),
                )
            )
            return -1

        if expire_days <= 0:
            self._log_highlight(
                "[FORCED_CHAT_EXPIRE_POLICY] speaker={} other={} forced={} raw_days={} fallback=30".format(
                    speaker_name,
                    other_name,
                    bool(forced),
                    raw_days,
                )
            )
            return 30

        self._log_highlight(
            "[FORCED_CHAT_EXPIRE_POLICY] speaker={} other={} forced={} expire_days={}".format(
                speaker_name,
                other_name,
                bool(forced),
                expire_days,
            )
        )
        return expire_days

    def get_forced_llm_runtime_config(self) -> Optional[Dict[str, Any]]:
        intervention_cfg = self.config.get("intervention", {}) or {}
        forced_llm = intervention_cfg.get("forced_llm", {}) or {}
        if not isinstance(forced_llm, dict) or not forced_llm:
            self._log_highlight("[FORCED_LLM] config_missing_or_invalid")
            return None

        enabled = self._safe_bool(forced_llm.get("enabled", False), False)
        if not enabled:
            self._log_highlight("[FORCED_LLM] disabled_by_config")
            return None

        provider = str(forced_llm.get("provider", "") or "").strip()
        model = str(forced_llm.get("model", "") or "").strip()
        base_url = str(forced_llm.get("base_url", "") or "").strip()
        api_key_env = str(forced_llm.get("api_key_env", "DEEPSEEK_API_KEY") or "DEEPSEEK_API_KEY").strip() or "DEEPSEEK_API_KEY"
        api_key = str(os.getenv(api_key_env, "") or "").strip()

        if (not provider) or (not model) or (not base_url):
            self._log_highlight(
                "[FORCED_LLM] required_fields_missing provider={} model={} base_url_present={}".format(
                    bool(provider),
                    bool(model),
                    bool(base_url),
                )
            )
            return None

        if not api_key:
            self._log_highlight(
                "[FORCED_LLM] api_key_missing env_var={}".format(api_key_env)
            )
            return None

        runtime_cfg = {
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "api_key": api_key,
            "retry": self._safe_int(forced_llm.get("retry", 2), 2),
        }

        try:
            runtime_cfg["temperature"] = float(forced_llm.get("temperature", 0.5))
        except Exception:
            runtime_cfg["temperature"] = 0.5

        return runtime_cfg

    def get_decide_chat_terminate_runtime_policy(self) -> Dict[str, Any]:
        default_retry = 2
        default_force_forced_llm = True
        policy = {
            "retry": default_retry,
            "force_forced_llm": default_force_forced_llm,
            "source": "defaults",
            "notes": [],
        }

        intervention_cfg = self.config.get("intervention", {}) or {}
        forced_llm = intervention_cfg.get("forced_llm", {}) or {}
        if not isinstance(forced_llm, dict):
            policy["notes"].append("forced_llm_not_dict")
            self._log_highlight(
                "[DECIDE_CHAT_TERMINATE_POLICY] source=defaults reason=forced_llm_not_dict retry={} force_forced_llm={}".format(
                    policy["retry"],
                    policy["force_forced_llm"],
                )
            )
            return policy

        keys_present = (
            "decide_chat_terminate_retry" in forced_llm
            or "decide_chat_terminate_force_forced_llm" in forced_llm
        )
        if keys_present:
            policy["source"] = "forced_llm_config"

        raw_retry = forced_llm.get("decide_chat_terminate_retry", default_retry)
        retry = self._safe_int(raw_retry, default_retry)
        if retry < 1:
            policy["notes"].append("retry_lt_1_fallback")
            retry = default_retry
        policy["retry"] = retry

        raw_force = forced_llm.get(
            "decide_chat_terminate_force_forced_llm",
            default_force_forced_llm,
        )
        policy["force_forced_llm"] = self._safe_bool(
            raw_force,
            default_force_forced_llm,
        )

        self._log_highlight(
            "[DECIDE_CHAT_TERMINATE_POLICY] source={} retry={} force_forced_llm={} notes={}".format(
                policy.get("source", "defaults"),
                policy.get("retry", default_retry),
                policy.get("force_forced_llm", default_force_forced_llm),
                "|".join(policy.get("notes", [])) or "none",
            )
        )
        return policy

    def get_depr_short_term_runtime_policy(self) -> Dict[str, Any]:
        default_retry = 2
        default_force_forced_llm = True
        policy = {
            "retry": default_retry,
            "force_forced_llm": default_force_forced_llm,
            "source": "defaults",
            "notes": [],
        }

        intervention_cfg = self.config.get("intervention", {}) or {}
        forced_llm = intervention_cfg.get("forced_llm", {}) or {}
        if not isinstance(forced_llm, dict):
            policy["notes"].append("forced_llm_not_dict")
            self._log_highlight(
                "[DEPR_SHORT_TERM_POLICY] source=defaults reason=forced_llm_not_dict retry={} force_forced_llm={}".format(
                    policy["retry"],
                    policy["force_forced_llm"],
                )
            )
            return policy

        keys_present = (
            "depr_short_term_retry" in forced_llm
            or "depr_short_term_force_forced_llm" in forced_llm
        )
        if keys_present:
            policy["source"] = "forced_llm_config"

        raw_retry = forced_llm.get("depr_short_term_retry", default_retry)
        retry = self._safe_int(raw_retry, default_retry)
        if retry < 1:
            policy["notes"].append("retry_lt_1_fallback")
            retry = default_retry
        policy["retry"] = retry

        raw_force = forced_llm.get(
            "depr_short_term_force_forced_llm",
            default_force_forced_llm,
        )
        policy["force_forced_llm"] = self._safe_bool(
            raw_force,
            default_force_forced_llm,
        )

        self._log_highlight(
            "[DEPR_SHORT_TERM_POLICY] source={} retry={} force_forced_llm={} notes={}".format(
                policy.get("source", "defaults"),
                policy.get("retry", default_retry),
                policy.get("force_forced_llm", default_force_forced_llm),
                "|".join(policy.get("notes", [])) or "none",
            )
        )
        return policy

    def get_depr_current_event_runtime_policy(self) -> Dict[str, Any]:
        default_retry = 2
        default_force_forced_llm = False
        policy = {
            "retry": default_retry,
            "force_forced_llm": default_force_forced_llm,
            "source": "defaults",
            "notes": [],
        }

        intervention_cfg = self.config.get("intervention", {}) or {}
        forced_llm = intervention_cfg.get("forced_llm", {}) or {}
        if not isinstance(forced_llm, dict):
            policy["notes"].append("forced_llm_not_dict")
            self._log_highlight(
                "[DEPR_CURRENT_EVENT_POLICY] source=defaults reason=forced_llm_not_dict retry={} force_forced_llm={}".format(
                    policy["retry"],
                    policy["force_forced_llm"],
                )
            )
            return policy

        keys_present = (
            "depr_current_event_retry" in forced_llm
            or "depr_current_event_force_forced_llm" in forced_llm
        )
        if keys_present:
            policy["source"] = "forced_llm_config"

        raw_retry = forced_llm.get("depr_current_event_retry", default_retry)
        retry = self._safe_int(raw_retry, default_retry)
        if retry < 1:
            policy["notes"].append("retry_lt_1_fallback")
            retry = default_retry
        policy["retry"] = retry

        raw_force = forced_llm.get(
            "depr_current_event_force_forced_llm",
            default_force_forced_llm,
        )
        policy["force_forced_llm"] = self._safe_bool(
            raw_force,
            default_force_forced_llm,
        )

        self._log_highlight(
            "[DEPR_CURRENT_EVENT_POLICY] source={} retry={} force_forced_llm={} notes={}".format(
                policy.get("source", "defaults"),
                policy.get("retry", default_retry),
                policy.get("force_forced_llm", default_force_forced_llm),
                "|".join(policy.get("notes", [])) or "none",
            )
        )
        return policy

    def should_route_forced_llm(self, speaker: Any, other: Any, forced: bool = False) -> bool:
        speaker_name = getattr(speaker, "name", "")
        other_name = getattr(other, "name", "")
        if (not self.enabled) or (not bool(forced)):
            return False

        doctor, patient = self._resolve_doctor_patient_pair(speaker, other)
        if not doctor or not patient:
            self._log_highlight(
                "[FORCED_LLM_ROUTE] speaker={} other={} result=false reason=not_doctor_patient_pair".format(
                    speaker_name,
                    other_name,
                )
            )
            return False

        intervention_cfg = self.config.get("intervention", {}) or {}
        forced_llm = intervention_cfg.get("forced_llm", {}) or {}
        enabled = self._safe_bool(forced_llm.get("enabled", False), False)
        if not enabled:
            self._log_highlight(
                "[FORCED_LLM_ROUTE] speaker={} other={} result=false reason=forced_llm_disabled".format(
                    speaker_name,
                    other_name,
                )
            )
            return False

        self._ensure_agent_state(speaker)
        self._ensure_agent_state(other)
        lock_s = speaker.status.get("intervention", {}).get("lock", {})
        lock_o = other.status.get("intervention", {}).get("lock", {})
        lock_s_enabled = bool(lock_s.get("enabled", False)) if isinstance(lock_s, dict) else False
        lock_o_enabled = bool(lock_o.get("enabled", False)) if isinstance(lock_o, dict) else False
        meeting_id_s = str(lock_s.get("meeting_id", "") or "") if isinstance(lock_s, dict) else ""
        meeting_id_o = str(lock_o.get("meeting_id", "") or "") if isinstance(lock_o, dict) else ""

        if (not lock_s_enabled) or (not lock_o_enabled) or (not meeting_id_s) or (meeting_id_s != meeting_id_o):
            self._log_highlight(
                "[FORCED_LLM_ROUTE] speaker={} other={} result=false reason=lock_guard_failed lock_s_enabled={} lock_o_enabled={} meeting_id_s={} meeting_id_o={}".format(
                    speaker_name,
                    other_name,
                    lock_s_enabled,
                    lock_o_enabled,
                    meeting_id_s or "<empty>",
                    meeting_id_o or "<empty>",
                )
            )
            return False

        api_key_env = str(forced_llm.get("api_key_env", "DEEPSEEK_API_KEY") or "DEEPSEEK_API_KEY").strip() or "DEEPSEEK_API_KEY"
        if not str(os.getenv(api_key_env, "") or "").strip():
            self._log_highlight(
                "[FORCED_LLM_ROUTE] speaker={} other={} result=false reason=api_key_missing env_var={}".format(
                    speaker_name,
                    other_name,
                    api_key_env,
                )
            )
            return False

        self._log_highlight(
            "[FORCED_LLM_ROUTE] speaker={} other={} result=true meeting_id={}".format(
                speaker_name,
                other_name,
                meeting_id_s,
            )
        )
        return True

    def _default_chat_control_policy(self) -> Dict[str, Any]:
        return {
            "enabled": False,
            "forced_only": True,
            "forced_chat_iter": -1,
            "repeat_break_threshold": 2,
            "repeat_detection_enabled": True,
            "forced_chat_min_turns": -1,
            "forced_chat_max_turns": -1,
            "terminate_detection_tail_window_enabled": False,
            "terminate_detection_tail_window_turns": 2,
            "source": "fallback",
            "normalize_notes": [],
        }

    def _normalize_chat_control_policy(self, raw: Dict[str, Any], source: str) -> Dict[str, Any]:
        defaults = self._default_chat_control_policy()
        policy = dict(defaults)
        notes = []

        if isinstance(raw, dict):
            policy.update(raw)
        else:
            notes.append("raw_policy_not_dict")
            self._log_highlight("[CHAT_CTRL_CFG_INVALID] field=policy reason=not_dict use_defaults=true")

        normalized = {
            "enabled": self._safe_bool(policy.get("enabled", defaults["enabled"]), defaults["enabled"]),
            "forced_only": self._safe_bool(policy.get("forced_only", defaults["forced_only"]), defaults["forced_only"]),
            "forced_chat_iter": self._normalize_limit_field(
                "forced_chat_iter", policy.get("forced_chat_iter", defaults["forced_chat_iter"]), defaults["forced_chat_iter"], notes
            ),
            "repeat_break_threshold": self._safe_int(
                policy.get("repeat_break_threshold", defaults["repeat_break_threshold"]),
                defaults["repeat_break_threshold"],
            ),
            "repeat_detection_enabled": self._safe_bool(
                policy.get("repeat_detection_enabled", defaults["repeat_detection_enabled"]),
                defaults["repeat_detection_enabled"],
            ),
            "forced_chat_min_turns": self._normalize_limit_field(
                "forced_chat_min_turns",
                policy.get("forced_chat_min_turns", defaults["forced_chat_min_turns"]),
                defaults["forced_chat_min_turns"],
                notes,
            ),
            "forced_chat_max_turns": self._normalize_limit_field(
                "forced_chat_max_turns",
                policy.get("forced_chat_max_turns", defaults["forced_chat_max_turns"]),
                defaults["forced_chat_max_turns"],
                notes,
            ),
            "terminate_detection_tail_window_enabled": self._safe_bool(
                policy.get(
                    "terminate_detection_tail_window_enabled",
                    defaults["terminate_detection_tail_window_enabled"],
                ),
                defaults["terminate_detection_tail_window_enabled"],
            ),
            "terminate_detection_tail_window_turns": self._safe_int(
                policy.get(
                    "terminate_detection_tail_window_turns",
                    defaults["terminate_detection_tail_window_turns"],
                ),
                defaults["terminate_detection_tail_window_turns"],
            ),
            "source": source,
        }

        if normalized["repeat_break_threshold"] < 1:
            notes.append("repeat_break_threshold_invalid")
            self._log_highlight(
                "[CHAT_CTRL_CFG_INVALID] field=repeat_break_threshold reason=lt_1 use_default={}".format(
                    defaults["repeat_break_threshold"]
                )
            )
            normalized["repeat_break_threshold"] = defaults["repeat_break_threshold"]

        if normalized["terminate_detection_tail_window_turns"] < 1:
            notes.append("terminate_detection_tail_window_turns_invalid")
            self._log_highlight(
                "[CHAT_CTRL_CFG_INVALID] field=terminate_detection_tail_window_turns reason=lt_1 use_default={}".format(
                    defaults["terminate_detection_tail_window_turns"]
                )
            )
            normalized["terminate_detection_tail_window_turns"] = defaults["terminate_detection_tail_window_turns"]

        min_turn = normalized["forced_chat_min_turns"]
        max_turn = normalized["forced_chat_max_turns"]
        if max_turn != -1 and min_turn != -1 and max_turn < min_turn:
            notes.append("forced_chat_max_turns_lt_min_turns_fixup")
            self._log_highlight(
                "[CHAT_CTRL_CFG_INVALID] field=forced_chat_max_turns reason=lt_min_turns auto_fix={}".format(
                    min_turn
                )
            )
            normalized["forced_chat_max_turns"] = min_turn

        normalized["normalize_notes"] = notes
        return normalized

    def _normalize_limit_field(self, name: str, value: Any, default: int, notes: Optional[list] = None) -> int:
        if notes is None:
            notes = []
        if isinstance(value, bool):
            notes.append("{}_bool_invalid".format(name))
            self._log_highlight(
                "[CHAT_CTRL_CFG_INVALID] field={} reason=bool_not_allowed use_default={}".format(
                    name,
                    default,
                )
            )
            return default
        if value == 0:
            notes.append("{}_deprecated_zero_to_minus_one".format(name))
            self._log_highlight(
                "[CHAT_CTRL_CFG_DEPRECATED_VALUE] field={} old=0 mapped=-1".format(name)
            )
            return -1

        ivalue = self._safe_int(value, default)
        if ivalue < -1:
            notes.append("{}_lt_minus_one_invalid".format(name))
            self._log_highlight(
                "[CHAT_CTRL_CFG_INVALID] field={} reason=lt_-1 use_default={}".format(
                    name,
                    default,
                )
            )
            return default
        return ivalue

    def _safe_bool(self, value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("1", "true", "yes", "y", "on"):
                return True
            if lowered in ("0", "false", "no", "n", "off"):
                return False
        return bool(default)

    def _safe_int(self, value: Any, default: int) -> int:
        if isinstance(value, bool):
            return int(default)
        try:
            return int(value)
        except Exception:
            return int(default)

    def _ensure_depression_runtime_initialized(
        self,
        agent: Any,
        profile: Dict[str, Any],
        now_step: int,
        update_cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        """幂等初始化 depression runtime，确保 topic 首次固化。"""
        ensured = drm.ensure_profile(profile)
        initialized = drm.initialize_current_event_if_needed(
            patient_agent=agent,
            profile=ensured,
            now_step=int(now_step),
            update_cfg=update_cfg or {},
        )
        return initialized

    def apply_forced_tasks(self, agent: Any, now: Any) -> None:
        """将 pending 医嘱任务并入当天日程。"""
        if not self.enabled:
            return
        self._ensure_agent_state(agent)

        intervention = agent.status["intervention"]
        forced_tasks = intervention.get("forced_tasks", [])
        if not forced_tasks:
            return

        now_dt = now if isinstance(now, datetime.datetime) else self._parse_dt(str(now))
        if not now_dt:
            now_dt = datetime.datetime.now()

        changed = False
        for task in forced_tasks:
            if not isinstance(task, dict):
                continue
            if task.get("state") != "pending":
                continue

            start_dt = self._task_start_datetime(task)
            if not start_dt:
                continue
            # 仅注入当天任务；历史任务或未来任务留待后续 step
            if start_dt.date() != now_dt.date():
                continue

            start_minute = start_dt.hour * 60 + start_dt.minute
            duration = int(task.get("duration", 30) or 30)
            describe = task.get("describe", "") or "执行医生建议"
            source = task.get("source", "doctor_order")
            inserted = agent.schedule.insert_hard_plan(
                describe=describe,
                start=start_minute,
                duration=duration,
                source=source,
                meta={"task_id": task.get("task_id", "")},
            )
            if inserted:
                task["state"] = "scheduled"
                changed = True
                self._log_highlight(
                    "TASK_INJECTED agent={} task_id={} start={} duration={}".format(
                        agent.name,
                        task.get("task_id", ""),
                        start_minute,
                        duration,
                    )
                )

        if changed:
            intervention["last_intervention_at"] = self._fmt_dt(now_dt)

    def _extract_and_queue_orders(self, speaker: Any, other: Any, chats: Any) -> None:
        if not self.order_extract.get("enabled", True):
            return

        doctor, patient = self._resolve_doctor_patient_pair(speaker, other)
        if not doctor or not patient:
            return

        conversation = "\n".join(["{}: {}".format(n, t) for n, t in chats or []])
        if not conversation.strip():
            return

        result = doctor.completion(
            "extract_doctor_order",
            doctor.name,
            patient.name,
            self._fmt_dt(self._now()),
            conversation,
        )
        tasks = []
        if isinstance(result, dict):
            tasks = result.get("tasks", [])
        if not isinstance(tasks, list):
            return

        min_conf = float(self.order_extract.get("min_confidence", 0.6) or 0.6)
        max_tasks = int(self.order_extract.get("max_tasks_per_chat", 3) or 3)

        self._ensure_agent_state(patient)
        queue = patient.status["intervention"]["forced_tasks"]

        accepted = 0
        for task in tasks:
            if accepted >= max_tasks:
                break
            if not isinstance(task, dict):
                continue
            confidence = float(task.get("confidence", 0.0) or 0.0)
            if confidence < min_conf:
                continue
            if not task.get("date") or not task.get("time") or not task.get("describe"):
                continue

            task_id = "order_{}_{}_{}".format(
                self._fmt_dt(self._now()).replace(":", "").replace("-", ""),
                patient.name,
                accepted + 1,
            )
            queue.append(
                {
                    "task_id": task_id,
                    "source": "doctor_order",
                    "doctor": doctor.name,
                    "patient": patient.name,
                    "describe": str(task.get("describe", "")).strip(),
                    "date": str(task.get("date", "")).strip(),
                    "time": str(task.get("time", "")).strip(),
                    "duration": int(task.get("duration", 30) or 30),
                    "address_hint": str(task.get("address_hint", "")).strip(),
                    "must_do": bool(task.get("must_do", True)),
                    "confidence": confidence,
                    "state": "pending",
                }
            )
            accepted += 1

        if accepted > 0:
            self._log(
                "queued doctor orders: doctor={}, patient={}, count={}".format(
                    doctor.name,
                    patient.name,
                    accepted,
                )
            )

    def _resolve_doctor_patient_pair(self, a: Any, b: Any):
        doctor_name = self.doctor
        patient_candidates = set(self.patients)

        if doctor_name and a.name == doctor_name and (not patient_candidates or b.name in patient_candidates):
            return a, b
        if doctor_name and b.name == doctor_name and (not patient_candidates or a.name in patient_candidates):
            return b, a
        return None, None

    def _is_session_completed_for_pair(self, doctor_name: str, patient_name: str) -> bool:
        if not self.stop_rule_scheduling_on_session_completed:
            return False
        doctor = str(doctor_name or "").strip()
        patient = str(patient_name or "").strip()
        if (not doctor) or (not patient):
            return False
        if not self.session_prompt_injection or (not bool(getattr(self.session_prompt_injection, "enabled", False))):
            return False
        try:
            state = self.session_prompt_injection.resolve_current_session(doctor, patient)
        except Exception:
            return False
        if not isinstance(state, dict):
            return False
        return bool(state.get("completed", False))

    def _is_rule_blocked_by_session_completed(self, rule: Dict[str, Any]):
        if not self.stop_rule_scheduling_on_session_completed:
            return False, ""
        doctor = str(rule.get("doctor") or self.doctor or "").strip()
        patients = self._build_rule_patients(rule)
        if (not doctor) or (not patients):
            return False, ""
        for patient_name in patients:
            patient = str(patient_name or "").strip()
            if not patient:
                continue
            if self._is_session_completed_for_pair(doctor, patient):
                return True, patient
        return False, ""

    def _purge_patient_meetings(self, patient_name: str, now: Any, reason: str) -> int:
        patient = str(patient_name or "").strip()
        if not patient:
            return 0
        self._ensure_state_schema()
        active = self.state.setdefault("active_meetings", {})
        purged = 0
        for meeting_id, meeting in list(active.items()):
            if not isinstance(meeting, dict):
                continue
            meeting_patient = str(meeting.get("patient", "") or "").strip()
            if meeting_patient != patient:
                continue
            self._finalize_meeting(meeting_id, now, reason=reason)
            purged += 1
        return purged

    def _purge_completed_pairs_meetings(self, now: Any, reason: str) -> int:
        if not self.stop_rule_scheduling_on_session_completed:
            return 0
        session_prompt_state = self.state.get("session_prompt_state", {})
        if not isinstance(session_prompt_state, dict):
            return 0
        pairs = session_prompt_state.get("pairs", {})
        if not isinstance(pairs, dict):
            return 0

        completed_patients = []
        seen = set()
        for pair_key, pair_state in pairs.items():
            if not isinstance(pair_state, dict):
                continue
            if not bool(pair_state.get("completed", False)):
                continue
            raw_key = str(pair_key or "")
            parts = raw_key.split("::", 1)
            if len(parts) != 2:
                continue
            patient = str(parts[1] or "").strip()
            if (not patient) or (patient in seen):
                continue
            seen.add(patient)
            completed_patients.append(patient)

        purged_total = 0
        for patient in completed_patients:
            purged_total += self._purge_patient_meetings(patient, now, reason=reason)
        return purged_total

    def _refresh_meeting_queue_cfg(self) -> None:
        intervention_cfg = self.config.get("intervention", {}) or {}
        self.meeting_queue_cfg = intervention_cfg.get("meeting_queue", {}) or {}
        self.meeting_queue_enabled = bool(self.meeting_queue_cfg.get("enabled", False))
        self.meeting_queue_max_per_doctor = self._safe_int(
            self.meeting_queue_cfg.get("max_queue_per_doctor", 128), 128
        )
        if self.meeting_queue_max_per_doctor <= 0:
            self.meeting_queue_max_per_doctor = 128

    def _ensure_state_schema(self) -> None:
        if not isinstance(self.state, dict):
            self.state = {}
            self.config["intervention_state"] = self.state
        if not isinstance(self.state.get("active_meetings"), dict):
            self.state["active_meetings"] = {}
        if not isinstance(self.state.get("rule_last_trigger"), dict):
            self.state["rule_last_trigger"] = {}
        if not isinstance(self.state.get("doctor_meeting_queues"), dict):
            self.state["doctor_meeting_queues"] = {}
        if not isinstance(self.state.get("doctor_current_meeting"), dict):
            self.state["doctor_current_meeting"] = {}
        if not isinstance(self.state.get("meeting_dedup"), dict):
            self.state["meeting_dedup"] = {}
        meeting_seq = self.state.get("meeting_seq", 0)
        try:
            meeting_seq = int(meeting_seq)
        except Exception:
            meeting_seq = 0
        if meeting_seq < 0:
            meeting_seq = 0
        self.state["meeting_seq"] = meeting_seq
        self._ensure_consult_record_state_schema()

    def _ensure_consult_record_state_schema(self) -> None:
        state = self.state.setdefault("consult_record_state", {})
        if not isinstance(state, dict):
            state = {}
            self.state["consult_record_state"] = state
        if not isinstance(state.get("records_by_id"), dict):
            state["records_by_id"] = {}
        if not isinstance(state.get("meeting_to_record"), dict):
            state["meeting_to_record"] = {}
        if not isinstance(state.get("latest_record_by_pair"), dict):
            state["latest_record_by_pair"] = {}
        if not isinstance(state.get("dedup_keys"), dict):
            state["dedup_keys"] = {}
        if not isinstance(state.get("write_audit"), list):
            state["write_audit"] = []

    def _build_rule_patients(self, rule: Dict[str, Any]) -> List[str]:
        patient = str(rule.get("patient", "") or "").strip()
        if patient:
            return [patient]
        patients = rule.get("patients", []) or []
        if isinstance(patients, list):
            deduped = []
            seen = set()
            for p in patients:
                name = str(p or "").strip()
                if not name or name in seen:
                    continue
                seen.add(name)
                deduped.append(name)
            if deduped:
                return deduped
        if self.patients:
            fallback = str(self.patients[0] or "").strip()
            if fallback:
                return [fallback]
        return []

    def _build_slot_key(self, rule: Dict[str, Any], now: datetime.datetime) -> str:
        rule_type = str(rule.get("type", "") or "").strip().lower()
        if rule_type == "interval_steps":
            next_step = int(self.config.get("step", 0) or 0) + 1
            return "step:{}".format(next_step)
        if rule_type == "interval_minutes":
            minutes = max(1, int(rule.get("interval_minutes", 1) or 1))
            bucket = int(now.timestamp() // (minutes * 60))
            return "minute_bucket:{}".format(bucket)
        if rule_type == "interval_days":
            return "day:{}".format(now.strftime("%Y-%m-%d"))
        if rule_type == "weekly":
            return "week:{}-{}".format(now.strftime("%Y"), now.strftime("%W"))
        return "time:{}".format(self._fmt_dt(now))

    def _build_meeting_dedup_key(self, rule_id: str, doctor: str, patient: str, slot_key: str) -> str:
        return "{}|{}|{}|{}".format(rule_id, doctor, patient, slot_key)

    def _next_meeting_id(self, rule_id: str, now: datetime.datetime) -> str:
        self._ensure_state_schema()
        seq = int(self.state.get("meeting_seq", 0) or 0) + 1
        self.state["meeting_seq"] = seq
        return "m_{}_{}_{}".format(rule_id, self._fmt_dt(now).replace(":", ""), seq)

    def _enqueue_meeting_from_rule(
        self,
        game: Any,
        rule_id: str,
        rule: Dict[str, Any],
        now: datetime.datetime,
    ) -> Optional[str]:
        self._ensure_state_schema()
        doctor = str(rule.get("doctor") or self.doctor or "").strip()
        patients = self._build_rule_patients(rule)
        if not doctor or not patients:
            self._log_queue_event(
                event="enqueue_fail",
                doctor=doctor,
                meeting_id="",
                pair_key="",
                step_phase=rule.get("step_phase", ""),
                reason="doctor_or_patient_missing",
            )
            return None

        agents = getattr(game, "agents", {}) if isinstance(getattr(game, "agents", {}), dict) else {}
        self._agents_ref = agents or self._agents_ref
        if doctor not in agents:
            self._log_queue_event(
                event="enqueue_fail",
                doctor=doctor,
                meeting_id="",
                pair_key="",
                step_phase=rule.get("step_phase", ""),
                reason="doctor_not_found",
            )
            return None

        active = self.state.setdefault("active_meetings", {})
        queues = self.state.setdefault("doctor_meeting_queues", {})
        dedup_map = self.state.setdefault("meeting_dedup", {})
        queue = queues.setdefault(doctor, [])
        slot_key = self._build_slot_key(rule, now)

        enqueued_meeting_id = None
        accepted = 0
        for patient in patients:
            if patient not in agents:
                self._log_queue_event(
                    event="enqueue_fail",
                    doctor=doctor,
                    meeting_id="",
                    pair_key="{}<->{}".format(doctor, patient),
                    step_phase=rule.get("step_phase", ""),
                    reason="patient_not_found",
                )
                continue

            dedup_key = self._build_meeting_dedup_key(rule_id, doctor, patient, slot_key)
            existed = str(dedup_map.get(dedup_key, "") or "")
            if existed and existed in active:
                self._log_queue_event(
                    event="enqueue_skip",
                    doctor=doctor,
                    meeting_id=existed,
                    pair_key="{}<->{}".format(doctor, patient),
                    step_phase=rule.get("step_phase", ""),
                    reason="dedup_hit",
                )
                continue

            if len(queue) >= self.meeting_queue_max_per_doctor:
                self._log_queue_event(
                    event="enqueue_fail",
                    doctor=doctor,
                    meeting_id="",
                    pair_key="{}<->{}".format(doctor, patient),
                    step_phase=rule.get("step_phase", ""),
                    reason="queue_full",
                )
                continue

            duration = self._safe_int(rule.get("duration", 30), 30)
            if duration <= 0:
                duration = 30
            meeting_id = self._next_meeting_id(rule_id, now)
            active[meeting_id] = {
                "meeting_id": meeting_id,
                "rule_id": rule_id,
                "doctor": doctor,
                "patient": patient,
                "priority": "hard",
                "status": "queued",
                "duration_minutes": duration,
                "enqueue_at": self._fmt_dt(now),
                "activated_at": "",
                "finished_at": "",
                "expire_at": "",
                "start_at": self._fmt_dt(now),
                "dedup_key": dedup_key,
                "step_phase": rule.get("step_phase", ""),
            }
            queue.append(meeting_id)
            dedup_map[dedup_key] = meeting_id
            accepted += 1
            enqueued_meeting_id = meeting_id
            self._log_queue_event(
                event="enqueue_ok",
                doctor=doctor,
                meeting_id=meeting_id,
                pair_key="{}<->{}".format(doctor, patient),
                step_phase=rule.get("step_phase", ""),
                reason="queued",
            )

        if accepted > 0:
            self.state.setdefault("rule_last_trigger", {})[rule_id] = self._fmt_dt(now)
        return enqueued_meeting_id

    def _promote_all_doctors(self, game: Any, now: datetime.datetime) -> None:
        self._ensure_state_schema()
        doctors = set(self.state.get("doctor_meeting_queues", {}).keys())
        doctors.update(self.state.get("doctor_current_meeting", {}).keys())
        for doctor_name in sorted(doctors):
            self._promote_next_for_doctor(game, doctor_name, now)

    def _promote_next_for_doctor(self, game: Any, doctor_name: str, now: datetime.datetime) -> bool:
        self._ensure_state_schema()
        if not doctor_name:
            return False
        agents = getattr(game, "agents", {}) if isinstance(getattr(game, "agents", {}), dict) else self._agents_ref
        if not isinstance(agents, dict):
            agents = {}
        self._agents_ref = agents or self._agents_ref
        if doctor_name not in agents:
            self._log_queue_event(
                event="queue_promote_skip",
                doctor=doctor_name,
                meeting_id="",
                pair_key="",
                step_phase="",
                reason="doctor_not_found",
            )
            return False

        doctor_agent = agents[doctor_name]
        self._ensure_agent_state(doctor_agent)
        doctor_lock = doctor_agent.status["intervention"]["lock"]
        current_meeting_id = str(doctor_lock.get("meeting_id", "") or "")
        if bool(doctor_lock.get("enabled", False)) and (not self._is_lock_expired(doctor_lock, now)):
            self.state.setdefault("doctor_current_meeting", {})[doctor_name] = current_meeting_id
            self._log_queue_event(
                event="queue_promote_skip",
                doctor=doctor_name,
                meeting_id=current_meeting_id,
                pair_key="",
                step_phase="",
                reason="doctor_busy",
            )
            return False

        queue = self.state.setdefault("doctor_meeting_queues", {}).setdefault(doctor_name, [])
        while queue:
            meeting_id = str(queue.pop(0) or "")
            meeting = self.state.setdefault("active_meetings", {}).get(meeting_id)
            if not isinstance(meeting, dict):
                self._log_queue_event(
                    event="queue_promote_skip",
                    doctor=doctor_name,
                    meeting_id=meeting_id,
                    pair_key="",
                    step_phase="",
                    reason="meeting_missing",
                )
                continue
            if str(meeting.get("doctor", "") or "") != doctor_name:
                target_doctor = str(meeting.get("doctor", "") or "")
                if target_doctor:
                    self.state.setdefault("doctor_meeting_queues", {}).setdefault(target_doctor, []).append(meeting_id)
                self._log_queue_event(
                    event="exception_recover",
                    doctor=doctor_name,
                    meeting_id=meeting_id,
                    pair_key="{}<->{}".format(meeting.get("doctor", ""), meeting.get("patient", "")),
                    step_phase=meeting.get("step_phase", ""),
                    reason="doctor_mismatch_requeue",
                )
                continue
            if self._activate_meeting(game, meeting_id, now):
                self._log_queue_event(
                    event="queue_promote_ok",
                    doctor=doctor_name,
                    meeting_id=meeting_id,
                    pair_key="{}<->{}".format(meeting.get("doctor", ""), meeting.get("patient", "")),
                    step_phase=meeting.get("step_phase", ""),
                    reason="activated",
                )
                return True
        self.state.setdefault("doctor_current_meeting", {}).pop(doctor_name, None)
        return False

    def _promote_next_for_doctor_by_pair(self, speaker: Any, other: Any, now: datetime.datetime) -> bool:
        doctor, _ = self._resolve_doctor_patient_pair(speaker, other)
        if not doctor:
            return False
        class _GameProxy:
            def __init__(self, agents):
                self.agents = agents
        return self._promote_next_for_doctor(_GameProxy(self._agents_ref), doctor.name, now)

    def _activate_meeting(self, game: Any, meeting_id: str, now: datetime.datetime) -> bool:
        self._ensure_state_schema()
        active = self.state.setdefault("active_meetings", {})
        meeting = active.get(meeting_id)
        if not isinstance(meeting, dict):
            self._log_queue_event(
                event="lock_fail",
                doctor="",
                meeting_id=meeting_id,
                pair_key="",
                step_phase="",
                reason="meeting_not_found",
            )
            return False

        doctor = str(meeting.get("doctor", "") or "")
        patient = str(meeting.get("patient", "") or "")
        agents = getattr(game, "agents", {}) if isinstance(getattr(game, "agents", {}), dict) else self._agents_ref
        if (not doctor) or (not patient) or doctor not in agents or patient not in agents:
            self._log_queue_event(
                event="lock_fail",
                doctor=doctor,
                meeting_id=meeting_id,
                pair_key="{}<->{}".format(doctor, patient),
                step_phase=meeting.get("step_phase", ""),
                reason="doctor_or_patient_not_found",
            )
            self._finalize_meeting(meeting_id, now, reason="missing_agent")
            return False

        duration = self._safe_int(meeting.get("duration_minutes", 30), 30)
        if duration <= 0:
            duration = 30
        expire_at = now + datetime.timedelta(minutes=duration)
        meeting["status"] = "active"
        meeting["activated_at"] = self._fmt_dt(now)
        meeting["expire_at"] = self._fmt_dt(expire_at)
        self.state.setdefault("doctor_current_meeting", {})[doctor] = meeting_id

        self._set_lock(agents[doctor], patient, meeting_id, expire_at)
        self._set_lock(agents[patient], doctor, meeting_id, expire_at)
        self._log_queue_event(
            event="lock_ok",
            doctor=doctor,
            meeting_id=meeting_id,
            pair_key="{}<->{}".format(doctor, patient),
            step_phase=meeting.get("step_phase", ""),
            reason="lock_activated",
        )
        return True

    def _remove_meeting_from_doctor_queue(self, doctor: str, meeting_id: str) -> None:
        queue = self.state.setdefault("doctor_meeting_queues", {}).setdefault(doctor, [])
        filtered = [m for m in queue if str(m or "") != str(meeting_id or "")]
        self.state.setdefault("doctor_meeting_queues", {})[doctor] = filtered

    def _finalize_meeting(self, meeting_id: str, finished_at: Any, reason: str) -> None:
        self._ensure_state_schema()
        active = self.state.setdefault("active_meetings", {})
        meeting = active.get(meeting_id)
        if not isinstance(meeting, dict):
            return
        doctor = str(meeting.get("doctor", "") or "")
        patient = str(meeting.get("patient", "") or "")
        meeting["status"] = "expired" if reason == "expired" else "finished"
        meeting["finished_at"] = self._fmt_dt(finished_at)

        current = self.state.setdefault("doctor_current_meeting", {}).get(doctor, "")
        if str(current or "") == str(meeting_id or ""):
            self.state.setdefault("doctor_current_meeting", {}).pop(doctor, None)
        self._remove_meeting_from_doctor_queue(doctor, meeting_id)

        dedup_key = str(meeting.get("dedup_key", "") or "")
        if dedup_key:
            dedup_map = self.state.setdefault("meeting_dedup", {})
            if str(dedup_map.get(dedup_key, "") or "") == str(meeting_id or ""):
                dedup_map.pop(dedup_key, None)

        for agent_name in (doctor, patient):
            agent = self._agents_ref.get(agent_name)
            if not agent:
                continue
            self._ensure_agent_state(agent)
            lock = agent.status.get("intervention", {}).get("lock", {})
            if not isinstance(lock, dict):
                continue
            if str(lock.get("meeting_id", "") or "") == str(meeting_id or ""):
                self._clear_lock(agent)

        active.pop(meeting_id, None)
        self._log_queue_event(
            event="meeting_finalize",
            doctor=doctor,
            meeting_id=meeting_id,
            pair_key="{}<->{}".format(doctor, patient),
            step_phase=meeting.get("step_phase", ""),
            reason=reason,
        )

    def _recover_queue_state(self, game: Any, now: datetime.datetime) -> None:
        self._ensure_state_schema()
        agents = getattr(game, "agents", {}) if isinstance(getattr(game, "agents", {}), dict) else {}
        self._agents_ref = agents or self._agents_ref

        active = self.state.setdefault("active_meetings", {})
        queues = self.state.setdefault("doctor_meeting_queues", {})
        current_map = self.state.setdefault("doctor_current_meeting", {})
        dedup_map = self.state.setdefault("meeting_dedup", {})

        for doctor, queue in list(queues.items()):
            if not isinstance(queue, list):
                queues[doctor] = []
                continue
            repaired = []
            for meeting_id in queue:
                mid = str(meeting_id or "")
                if mid in active:
                    repaired.append(mid)
                else:
                    self._log_queue_event(
                        event="exception_recover",
                        doctor=doctor,
                        meeting_id=mid,
                        pair_key="",
                        step_phase="",
                        reason="queue_orphan_removed",
                    )
            queues[doctor] = repaired

        for doctor, meeting_id in list(current_map.items()):
            mid = str(meeting_id or "")
            if mid not in active:
                current_map.pop(doctor, None)
                self._log_queue_event(
                    event="exception_recover",
                    doctor=doctor,
                    meeting_id=mid,
                    pair_key="",
                    step_phase="",
                    reason="current_missing_removed",
                )

        for meeting_id, meeting in list(active.items()):
            if not isinstance(meeting, dict):
                active.pop(meeting_id, None)
                continue
            doctor = str(meeting.get("doctor", "") or "")
            patient = str(meeting.get("patient", "") or "")
            status = str(meeting.get("status", "queued") or "queued")
            if (not doctor) or (not patient) or doctor not in self._agents_ref or patient not in self._agents_ref:
                self._finalize_meeting(meeting_id, now, reason="recover_missing_agent")
                continue
            if status == "active":
                doctor_agent = self._agents_ref.get(doctor)
                patient_agent = self._agents_ref.get(patient)
                self._ensure_agent_state(doctor_agent)
                self._ensure_agent_state(patient_agent)
                d_lock = doctor_agent.status.get("intervention", {}).get("lock", {})
                p_lock = patient_agent.status.get("intervention", {}).get("lock", {})
                d_ok = bool(d_lock.get("enabled", False)) and str(d_lock.get("meeting_id", "") or "") == str(meeting_id)
                p_ok = bool(p_lock.get("enabled", False)) and str(p_lock.get("meeting_id", "") or "") == str(meeting_id)
                if d_ok and p_ok:
                    current_map[doctor] = meeting_id
                    self._remove_meeting_from_doctor_queue(doctor, meeting_id)
                else:
                    meeting["status"] = "queued"
                    meeting["activated_at"] = ""
                    meeting["expire_at"] = ""
                    queue = queues.setdefault(doctor, [])
                    if meeting_id not in queue:
                        queue.insert(0, meeting_id)
                    current_map.pop(doctor, None)
                    self._log_queue_event(
                        event="exception_recover",
                        doctor=doctor,
                        meeting_id=meeting_id,
                        pair_key="{}<->{}".format(doctor, patient),
                        step_phase=meeting.get("step_phase", ""),
                        reason="active_without_lock_requeue",
                    )
            else:
                meeting["status"] = "queued"
                queue = queues.setdefault(doctor, [])
                if meeting_id not in queue:
                    queue.append(meeting_id)

        for dedup_key, meeting_id in list(dedup_map.items()):
            if str(meeting_id or "") not in active:
                dedup_map.pop(dedup_key, None)

    def _trigger_meeting(self, game: Any, rule_id: str, rule: Dict[str, Any], now: datetime.datetime) -> None:
        if self.meeting_queue_enabled:
            self._enqueue_meeting_from_rule(game, rule_id, rule, now)
            return

        doctor = rule.get("doctor") or self.doctor
        patient = rule.get("patient")
        if not patient:
            patients = rule.get("patients") or self.patients
            if patients:
                patient = patients[0]

        if not doctor or not patient:
            return
        if doctor not in game.agents or patient not in game.agents:
            return

        duration = int(rule.get("duration", 30) or 30)
        expire_at = now + datetime.timedelta(minutes=duration)
        meeting_id = f"{rule_id}:{self._fmt_dt(now)}"

        self.state.setdefault("active_meetings", {})[meeting_id] = {
            "meeting_id": meeting_id,
            "rule_id": rule_id,
            "doctor": doctor,
            "patient": patient,
            "start_at": self._fmt_dt(now),
            "expire_at": self._fmt_dt(expire_at),
            "priority": "hard",
        }
        self.state.setdefault("rule_last_trigger", {})[rule_id] = self._fmt_dt(now)

        self._set_lock(game.agents[doctor], patient, meeting_id, expire_at)
        self._set_lock(game.agents[patient], doctor, meeting_id, expire_at)

        self._log_highlight(
            "TRIGGER rule={} meeting={} doctor={} patient={} expire_at={}".format(
                rule_id, meeting_id, doctor, patient, self._fmt_dt(expire_at)
            )
        )
    def _set_lock(self, agent: Any, target: str, meeting_id: str, expire_at: datetime.datetime) -> None:
        self._ensure_agent_state(agent)
        lock = agent.status["intervention"]["lock"]
        trigger_step = int(self.config.get("step", 0) or 0) + 1
        lock.update(
            {
                "enabled": True,
                "mode": "meeting",
                "priority": "hard",
                "target_agent": target,
                "meeting_id": meeting_id,
                "expire_at": self._fmt_dt(expire_at),
                "trigger_step": trigger_step,
                "deferred_by_sleep": False,
                "defer_count": 0,
            }
        )
        self._log_highlight(
            "LOCK_SET agent={} target={} meeting_id={} trigger_step={} expire_at={}".format(
                agent.name,
                target,
                meeting_id,
                trigger_step,
                self._fmt_dt(expire_at),
            )
        )

    def _clear_lock(self, agent: Any) -> None:
        self._ensure_agent_state(agent)
        agent.status["intervention"]["lock"] = InterventionLock().__dict__.copy()

    def _cleanup_expired_locks(self, game: Any, now: datetime.datetime) -> None:
        for agent in game.agents.values():
            self._ensure_agent_state(agent)
            lock = agent.status["intervention"]["lock"]
            if lock.get("enabled") and self._is_lock_expired(lock, now):
                if bool(lock.get("deferred_by_sleep", False)) and (not self._agent_is_awake(agent)):
                    defer_minutes = self._lock_defer_window_minutes()
                    next_expire = now + datetime.timedelta(minutes=defer_minutes)
                    lock["expire_at"] = self._fmt_dt(next_expire)
                    self._log_highlight(
                        "LOCK_EXPIRE_EXTENDED_SLEEP agent={} meeting_id={} defer_minutes={} next_expire_at={}".format(
                            agent.name,
                            lock.get("meeting_id", ""),
                            defer_minutes,
                            lock.get("expire_at", ""),
                        )
                    )
                    continue
                self._log_highlight(
                    "FORCED_MISSED_EXPIRED agent={} meeting_id={} trigger_step={} now={}".format(
                        agent.name,
                        lock.get("meeting_id", ""),
                        int(lock.get("trigger_step", -1) or -1),
                        self._fmt_dt(now),
                    )
                )
                if self.meeting_queue_enabled:
                    meeting_id = str(lock.get("meeting_id", "") or "")
                    if meeting_id:
                        self._log_queue_event(
                            event="lock_fail",
                            doctor=getattr(agent, "name", ""),
                            meeting_id=meeting_id,
                            pair_key="",
                            step_phase="",
                            reason="lock_expired_cleanup",
                        )
                self._clear_lock(agent)

        active = self.state.setdefault("active_meetings", {})
        expired = []
        for meeting_id, meeting in active.items():
            expire_at = self._parse_dt(meeting.get("expire_at", ""))
            if expire_at and now >= expire_at:
                if self._meeting_deferred_by_sleep(game, meeting_id):
                    defer_minutes = self._lock_defer_window_minutes()
                    next_expire = now + datetime.timedelta(minutes=defer_minutes)
                    meeting["expire_at"] = self._fmt_dt(next_expire)
                    self._log_highlight(
                        "MEETING_EXPIRE_EXTENDED_SLEEP meeting_id={} defer_minutes={} next_expire_at={}".format(
                            meeting_id,
                            defer_minutes,
                            meeting.get("expire_at", ""),
                        )
                    )
                    continue
                expired.append(meeting_id)
        for meeting_id in expired:
            if self.meeting_queue_enabled:
                meeting = active.get(meeting_id, {}) or {}
                doctor = str(meeting.get("doctor", "") or "")
                self._finalize_meeting(meeting_id, now, reason="expired")
                self._log_queue_event(
                    event="exception_recover",
                    doctor=doctor,
                    meeting_id=meeting_id,
                    pair_key="{}<->{}".format(meeting.get("doctor", ""), meeting.get("patient", "")),
                    step_phase=meeting.get("step_phase", ""),
                    reason="expired_finalize",
                )
                if doctor:
                    self._promote_next_for_doctor(game, doctor, now)
            else:
                active.pop(meeting_id, None)

    def _should_trigger(self, rule_id: str, rule: Dict[str, Any], now: datetime.datetime) -> bool:
        rule_type = (rule.get("type") or "").strip().lower()
        if not rule_type:
            return False

        if rule_type == "interval_steps":
            interval_steps = int(rule.get("interval_steps", 0) or 0)
            if interval_steps <= 0:
                self._log_highlight("RULE_CHECK rule_id={} type=interval_steps invalid_interval".format(rule_id))
                return False
            # on_step_start 在步开始触发，这里用“下一步编号”参与判定
            next_step = int(self.config.get("step", 0) or 0) + 1
            step_phase = self._safe_int(rule.get("step_phase", 0), 0)
            normalized_phase = step_phase % interval_steps
            if next_step <= 0 or next_step % interval_steps != normalized_phase:
                self._log_highlight(
                    "RULE_CHECK rule_id={} type=interval_steps step={} interval={} step_phase={} normalized_phase={} result=False".format(
                        rule_id,
                        next_step,
                        interval_steps,
                        step_phase,
                        normalized_phase,
                    )
                )
                return False
            last = self.state.get("rule_last_trigger", {}).get(rule_id, "")
            result = (not last or last != self._fmt_dt(now))
            self._log_highlight(
                "RULE_CHECK rule_id={} type=interval_steps step={} interval={} step_phase={} normalized_phase={} last={} result={}".format(
                    rule_id,
                    next_step,
                    interval_steps,
                    step_phase,
                    normalized_phase,
                    last or "<none>",
                    result,
                )
            )
            return result

        if rule_type == "interval_minutes":
            minutes = int(rule.get("interval_minutes", 0) or 0)
            if minutes <= 0:
                self._log_highlight("RULE_CHECK rule_id={} type=interval_minutes invalid_interval".format(rule_id))
                return False
            last = self._parse_dt(self.state.get("rule_last_trigger", {}).get(rule_id, ""))
            if not last:
                self._log_highlight(
                    "RULE_CHECK rule_id={} type=interval_minutes last=<none> result=True".format(rule_id)
                )
                return True
            result = (now - last).total_seconds() >= minutes * 60
            self._log_highlight(
                "RULE_CHECK rule_id={} type=interval_minutes now={} last={} interval={} result={}".format(
                    rule_id,
                    self._fmt_dt(now),
                    self._fmt_dt(last),
                    minutes,
                    result,
                )
            )
            return result

        if rule_type == "interval_days":
            days = int(rule.get("interval_days", 0) or 0)
            if days <= 0:
                self._log_highlight("RULE_CHECK rule_id={} type=interval_days invalid_interval".format(rule_id))
                return False
            time_ok = self._time_match(rule, now)
            if not time_ok:
                self._log_highlight(
                    "RULE_CHECK rule_id={} type=interval_days now={} time={} result=False".format(
                        rule_id,
                        self._fmt_dt(now),
                        rule.get("time", ""),
                    )
                )
                return False
            last = self._parse_dt(self.state.get("rule_last_trigger", {}).get(rule_id, ""))
            if not last:
                self._log_highlight(
                    "RULE_CHECK rule_id={} type=interval_days last=<none> result=True".format(rule_id)
                )
                return True
            result = (now.date() - last.date()).days >= days
            self._log_highlight(
                "RULE_CHECK rule_id={} type=interval_days last={} interval={} result={}".format(
                    rule_id,
                    self._fmt_dt(last),
                    days,
                    result,
                )
            )
            return result

        if rule_type == "weekly":
            time_ok = self._time_match(rule, now)
            if not time_ok:
                self._log_highlight(
                    "RULE_CHECK rule_id={} type=weekly now={} time={} result=False".format(
                        rule_id,
                        self._fmt_dt(now),
                        rule.get("time", ""),
                    )
                )
                return False
            rule_weekday = int(rule.get("weekday", -1))
            # 兼容两种写法：0-6(周一=0) 或 1-7(周一=1)
            if 1 <= rule_weekday <= 7:
                rule_weekday -= 1
            if now.weekday() != rule_weekday:
                self._log_highlight(
                    "RULE_CHECK rule_id={} type=weekly now_weekday={} rule_weekday={} result=False".format(
                        rule_id, now.weekday(), rule_weekday
                    )
                )
                return False
            last = self._parse_dt(self.state.get("rule_last_trigger", {}).get(rule_id, ""))
            result = (not last or last.date() != now.date())
            self._log_highlight(
                "RULE_CHECK rule_id={} type=weekly last={} result={}".format(
                    rule_id,
                    self._fmt_dt(last) if last else "<none>",
                    result,
                )
            )
            return result

        self._log_highlight("RULE_CHECK rule_id={} type={} result=False unsupported_type".format(rule_id, rule_type))
        return False

    def _time_match(self, rule: Dict[str, Any], now: datetime.datetime) -> bool:
        t = (rule.get("time") or "").strip()
        if not t:
            return True
        try:
            hh, mm = [int(x) for x in t.split(":", 1)]
        except Exception:
            return False
        return now.hour == hh and now.minute == mm

    def _is_lock_expired(self, lock: Dict[str, Any], now: datetime.datetime) -> bool:
        expire_at = self._parse_dt(lock.get("expire_at", ""))
        return bool(expire_at and now >= expire_at)

    def _task_start_datetime(self, task: Dict[str, Any]) -> Optional[datetime.datetime]:
        date = str(task.get("date", "")).strip()
        time = str(task.get("time", "")).strip()
        if not date or not time:
            return None
        try:
            return datetime.datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
        except ValueError:
            return None

    def _ensure_agent_state(self, agent: Any) -> None:
        status = getattr(agent, "status", None)
        if not isinstance(status, dict):
            return
        intervention = status.setdefault("intervention", {})
        lock = intervention.setdefault("lock", InterventionLock().__dict__.copy())
        if isinstance(lock, dict):
            lock.setdefault("trigger_step", -1)
            lock.setdefault("deferred_by_sleep", False)
            lock.setdefault("defer_count", 0)
        intervention.setdefault("forced_tasks", [])
        intervention.setdefault("last_intervention_at", "")

    def _agent_is_awake(self, agent: Any) -> bool:
        checker = getattr(agent, "is_awake", None)
        if callable(checker):
            try:
                return bool(checker())
            except Exception:
                return True
        return True

    def _lock_defer_window_minutes(self) -> int:
        intervention_cfg = self.config.get("intervention", {}) or {}
        minutes = int(intervention_cfg.get("lock_defer_window_minutes", 60) or 60)
        return max(1, minutes)

    def _meeting_deferred_by_sleep(self, game: Any, meeting_id: str) -> bool:
        for agent in game.agents.values():
            self._ensure_agent_state(agent)
            lock = agent.status.get("intervention", {}).get("lock", {})
            if not isinstance(lock, dict):
                continue
            if not bool(lock.get("enabled", False)):
                continue
            if str(lock.get("meeting_id", "") or "") != str(meeting_id or ""):
                continue
            if bool(lock.get("deferred_by_sleep", False)) and (not self._agent_is_awake(agent)):
                return True
        return False

    def _parse_dt(self, value: str) -> Optional[datetime.datetime]:
        if not value:
            return None
        for fmt in ("%Y%m%d-%H:%M:%S", "%Y%m%d-%H:%M"):
            try:
                return datetime.datetime.strptime(value, fmt)
            except ValueError:
                continue
        return None

    def _fmt_dt(self, dt: Any) -> str:
        if isinstance(dt, datetime.datetime):
            return dt.strftime("%Y%m%d-%H:%M:%S")
        return str(dt)

    def _fmt_iso8601_with_tz(self, dt: Any) -> str:
        parsed: Optional[datetime.datetime] = None
        if isinstance(dt, datetime.datetime):
            parsed = dt
        elif isinstance(dt, str):
            parsed = self._parse_dt(dt)
        if parsed is None:
            parsed = datetime.datetime.now().astimezone()
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=datetime.datetime.now().astimezone().tzinfo)
        return parsed.isoformat(timespec="seconds")

    def _now(self):
        return datetime.datetime.now()

    def _consult_record_enabled(self) -> bool:
        intervention_cfg = self.config.get("intervention", {}) or {}
        consult_cfg = intervention_cfg.get("consult_record", {}) or {}
        return bool(self.enabled and self._safe_bool(consult_cfg.get("enabled", False), False))

    def _consult_prompt_generate_file(self) -> str:
        intervention_cfg = self.config.get("intervention", {}) or {}
        consult_cfg = intervention_cfg.get("consult_record", {}) or {}
        return str(
            consult_cfg.get(
                "prompt_generate_file",
                "data/prompts/intervention/consult_record_generate.txt",
            )
            or ""
        ).strip()

    def _consult_prompt_injection_file(self) -> str:
        intervention_cfg = self.config.get("intervention", {}) or {}
        consult_cfg = intervention_cfg.get("consult_record", {}) or {}
        return str(
            consult_cfg.get(
                "prompt_injection_file",
                "data/prompts/intervention/doctor_consult_record_injection.txt",
            )
            or ""
        ).strip()

    def _consult_validation_cfg(self) -> Dict[str, Any]:
        intervention_cfg = self.config.get("intervention", {}) or {}
        consult_cfg = intervention_cfg.get("consult_record", {}) or {}
        validation = consult_cfg.get("validation", {}) or {}
        if not isinstance(validation, dict):
            validation = {}
        return validation

    def _resolve_consult_soap_text_max_len(self) -> int:
        value = self._safe_int(self._consult_validation_cfg().get("soap_text_max_len", 120), 120)
        return max(1, value)

    def _consult_audit_max_entries(self) -> int:
        intervention_cfg = self.config.get("intervention", {}) or {}
        consult_cfg = intervention_cfg.get("consult_record", {}) or {}
        audit_cfg = consult_cfg.get("audit", {}) or {}
        if not isinstance(audit_cfg, dict):
            return 2000
        value = self._safe_int(audit_cfg.get("max_entries", 2000), 2000)
        if value <= 0:
            return 2000
        return value

    def _load_prompt_txt_or_raise(self, path: str) -> str:
        file_path = str(path or "").strip()
        if not file_path:
            raise ConsultRecordPromptError(reason="prompt_file_not_found")
        if not os.path.exists(file_path):
            raise ConsultRecordPromptError(reason="prompt_file_not_found")
        try:
            with open(file_path, "r", encoding="utf-8") as fp:
                content = fp.read()
        except (UnicodeDecodeError, OSError):
            raise ConsultRecordPromptError(reason="prompt_file_read_failed")
        if not str(content or "").strip():
            raise ConsultRecordPromptError(reason="prompt_file_empty")
        return content

    def _render_prompt_template(self, template_text: str, data: Dict[str, Any]) -> str:
        variables = data if isinstance(data, dict) else {}
        try:
            return Template(str(template_text or "")).safe_substitute(variables)
        except Exception:
            return str(template_text or "")

    def _json_loads_loose(self, text: str) -> Optional[Dict[str, Any]]:
        raw = str(text or "").strip()
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass
        fenced = raw
        if "```" in raw:
            parts = raw.split("```")
            if len(parts) >= 3:
                fenced = parts[1]
                if fenced.lstrip().startswith("json"):
                    fenced = fenced.lstrip()[4:].strip()
                try:
                    parsed = json.loads(fenced)
                    return parsed if isinstance(parsed, dict) else None
                except Exception:
                    pass
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(raw[start : end + 1])
                return parsed if isinstance(parsed, dict) else None
            except Exception:
                return None
        return None

    def _call_forced_llm_json(self, prompt_text: str, retry: int) -> Dict[str, Any]:
        runtime_cfg = self.get_forced_llm_runtime_config()
        if not runtime_cfg:
            raise ConsultRecordError(reason="forced_llm_unavailable", retryable=True)

        retry_count = max(1, int(retry or runtime_cfg.get("retry", 2) or 2))
        cache_key = "{}/{}/{}".format(
            runtime_cfg.get("provider", ""),
            runtime_cfg.get("model", ""),
            runtime_cfg.get("base_url", ""),
        )
        if self._consult_record_llm is None or self._consult_record_llm_key != cache_key:
            self._consult_record_llm = create_llm_model(runtime_cfg)
            self._consult_record_llm_key = cache_key

        self._log_consult(
            "LLM_CALL start provider={} model={} retry={}".format(
                runtime_cfg.get("provider", ""),
                runtime_cfg.get("model", ""),
                retry_count,
            )
        )
        payload = self._consult_record_llm.completion(
            prompt_text,
            retry=retry_count,
            callback=self._json_loads_loose,
            failsafe=None,
            caller="consult_record_forced_llm",
            temperature=float(runtime_cfg.get("temperature", 0.5) or 0.5),
        )
        if not isinstance(payload, dict):
            self._log_consult("LLM_CALL end success=false reason=json_parse_failed")
            raise ConsultRecordError(reason="json_parse_failed", retryable=False)
        self._log_consult("LLM_CALL end success=true")
        return payload

    def _append_consult_audit(
        self,
        pair_key: str,
        meeting_id: str,
        record_id: str,
        status: str,
        reason: str,
        message: str,
        retryable: bool,
    ) -> None:
        self._ensure_consult_record_state_schema()
        state = self.state.setdefault("consult_record_state", {})
        audit = state.setdefault("write_audit", [])
        if not isinstance(audit, list):
            audit = []
            state["write_audit"] = audit
        audit.append(
            {
                "ts": self._fmt_iso8601_with_tz(self._now()),
                "pair_key": str(pair_key or ""),
                "meeting_id": str(meeting_id or ""),
                "record_id": str(record_id or ""),
                "status": str(status or ""),
                "reason": str(reason or ""),
                "message": str(message or ""),
                "retryable": bool(retryable),
            }
        )
        max_entries = self._consult_audit_max_entries()
        if len(audit) > max_entries:
            state["write_audit"] = audit[-max_entries:]

    def _write_consult_record(self, record: Dict[str, Any], pair_key: str, summary: str) -> Dict[str, Any]:
        self._ensure_consult_record_state_schema()
        state = self.state.setdefault("consult_record_state", {})
        records_by_id = state.setdefault("records_by_id", {})
        meeting_to_record = state.setdefault("meeting_to_record", {})
        latest_record_by_pair = state.setdefault("latest_record_by_pair", {})
        dedup_keys = state.setdefault("dedup_keys", {})

        meeting_id = str(record.get("meeting_id", "") or "")
        record_id = str(record.get("record_id", "") or "")
        if meeting_id in meeting_to_record:
            return {
                "status": "skipped",
                "reason": "idempotent_hit_meeting",
                "record_id": str(meeting_to_record.get(meeting_id, "") or ""),
            }

        dedup_key = build_dedup_key(meeting_id, summary)
        if dedup_key in dedup_keys:
            return {
                "status": "skipped",
                "reason": "dedup_key_hit",
                "record_id": str(dedup_keys.get(dedup_key, "") or ""),
            }

        records_by_id[record_id] = record
        meeting_to_record[meeting_id] = record_id
        latest_record_by_pair[pair_key] = record_id
        dedup_keys[dedup_key] = record_id
        return {
            "status": "success",
            "reason": "write_success",
            "record_id": record_id,
            "dedup_key": dedup_key,
        }

    def _get_latest_consult_record_by_pair(self, pair_key: str) -> Optional[Dict[str, Any]]:
        self._ensure_consult_record_state_schema()
        state = self.state.setdefault("consult_record_state", {})
        latest_map = state.setdefault("latest_record_by_pair", {})
        records = state.setdefault("records_by_id", {})
        record_id = str(latest_map.get(pair_key, "") or "")
        if not record_id:
            return None
        record = records.get(record_id)
        if isinstance(record, dict):
            return record
        return None

    def _build_depr_summary_with_consult_fallback(
        self,
        speaker: Any,
        other: Any,
        chat_summary: str,
        meeting_id: str,
    ) -> str:
        fallback_summary = str(chat_summary or "")
        doctor, patient = self._resolve_doctor_patient_pair(speaker, other)
        if not doctor or not patient:
            return fallback_summary

        doctor_name = str(getattr(doctor, "name", "") or "")
        patient_name = str(getattr(patient, "name", "") or "")
        pair_key = build_pair_key(doctor_name, patient_name)
        step = int(self.config.get("step", 0) or 0)

        try:
            if not self._consult_record_enabled():
                raise RuntimeError("consult_record_disabled")

            record = self._get_latest_consult_record_by_pair(pair_key)
            if not isinstance(record, dict) or not record:
                raise RuntimeError("latest_consult_record_missing")

            flat = flatten_record_for_injection(record)
            if not isinstance(flat, dict):
                raise RuntimeError("flatten_record_invalid")

            evidence_parts: List[str] = []
            current_session = str(record.get("current_session", "") or "").strip()
            record_id = str(record.get("record_id", "") or "").strip()
            if record_id:
                evidence_parts.append(f"record_id={record_id}")
            if current_session:
                evidence_parts.append(f"current_session={current_session}")

            for key in [
                "s_expr",
                "s_focus",
                "s_impact",
                "o_obs",
                "o_source",
                "o_resp",
                "a_formulation",
                "a_outcome",
                "a_understanding",
            ]:
                value = str(flat.get(key, "") or "").strip()
                if value:
                    evidence_parts.append(f"{key}={value}")

            consult_summary = "\n".join(evidence_parts).strip()
            if not consult_summary:
                raise RuntimeError("flatten_record_empty")
            return consult_summary
        except Exception as err:
            self._log_consult(
                "ERROR trigger=InterventionManager.after_chat->depr_short_term_summary "
                "doctor={} patient={} meeting_id={} pair_key={} step={} "
                "fallback=fallback_to_chat_summary reason={} traceback={}".format(
                    doctor_name,
                    patient_name,
                    str(meeting_id or ""),
                    pair_key,
                    step,
                    str(err),
                    traceback.format_exc().replace("\n", "\\n"),
                )
            )
            return fallback_summary

    def _next_consult_record_id(self) -> str:
        self._ensure_consult_record_state_schema()
        state = self.state.setdefault("consult_record_state", {})
        records = state.setdefault("records_by_id", {})
        seq = len(records) + 1
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return "rec_{}_{}".format(ts, str(seq).zfill(3))

    def _join_prompt_blocks(self, base_block: str, ext_block: str) -> str:
        base = str(base_block or "").strip()
        ext = str(ext_block or "").strip()
        if base and ext:
            return base + "\n\n" + ext
        return base or ext

    def _log_consult(self, message: str) -> None:
        self._log("[CONSULT_RECORD] {}".format(message))

    def _handle_consult_record_after_chat(
        self,
        speaker: Any,
        other: Any,
        chats: Any,
        summary: str,
        start_time: Any,
        meeting_id: str,
    ) -> None:
        doctor, patient = self._resolve_doctor_patient_pair(speaker, other)
        doctor_name = getattr(doctor, "name", "") if doctor else ""
        patient_name = getattr(patient, "name", "") if patient else ""
        pair_key = build_pair_key(doctor_name, patient_name) if doctor and patient else ""
        conversation = to_conversation_text(chats)

        self._log_consult(
            "AFTER_CHAT trigger speaker={} other={} meeting_id={} chats={} summary_chars={} conversation_chars={}".format(
                getattr(speaker, "name", ""),
                getattr(other, "name", ""),
                str(meeting_id or "<empty>"),
                len(chats or []),
                len(str(summary or "")),
                len(conversation),
            )
        )

        if not self._consult_record_enabled():
            self._log_consult("PRECHECK skip reason=consult_record_disabled")
            return
        if not doctor or not patient:
            self._append_consult_audit(
                pair_key="",
                meeting_id=str(meeting_id or ""),
                record_id="",
                status="failed",
                reason="pair_mismatch",
                message="doctor_patient_pair_not_matched",
                retryable=False,
            )
            self._log_consult("PRECHECK fail reason=pair_mismatch")
            return
        if not str(meeting_id or "").strip():
            self._append_consult_audit(
                pair_key=pair_key,
                meeting_id="",
                record_id="",
                status="failed",
                reason="meeting_id_mismatch",
                message="meeting_id_missing_or_not_closed",
                retryable=False,
            )
            self._log_consult("PRECHECK fail reason=meeting_id_mismatch")
            return
        if not conversation:
            self._append_consult_audit(
                pair_key=pair_key,
                meeting_id=str(meeting_id),
                record_id="",
                status="failed",
                reason="required_field_empty",
                message="conversation_empty",
                retryable=False,
            )
            self._log_consult("PRECHECK fail reason=required_field_empty detail=conversation_empty")
            return

        try:
            retry = 0
            intervention_cfg = self.config.get("intervention", {}) or {}
            consult_cfg = intervention_cfg.get("consult_record", {}) or {}
            llm_retry = self._safe_int(consult_cfg.get("llm_retry", -1), -1)
            if llm_retry > 0:
                retry = llm_retry
            else:
                retry = self._safe_int((intervention_cfg.get("forced_llm", {}) or {}).get("retry", 2), 2)

            self._log_consult("PROMPT load file={}".format(self._consult_prompt_generate_file()))
            prompt_tpl = self._load_prompt_txt_or_raise(self._consult_prompt_generate_file())
            prompt_text = self._render_prompt_template(
                prompt_tpl,
                {
                    "doctor": doctor_name,
                    "patient": patient_name,
                    "conversation": conversation,
                },
            )
            self._log_consult("PROMPT render done chars={}".format(len(prompt_text)))

            llm_raw = self._call_forced_llm_json(prompt_text, retry=retry)
            projected, dropped = project_whitelist(llm_raw)
            self._log_consult(
                "PARSE project done dropped_count={} dropped_paths={}".format(
                    len(dropped),
                    "|".join(dropped) if dropped else "-",
                )
            )
            soap_text_max_len = self._resolve_consult_soap_text_max_len()
            validated_soap = validate_soap_only(projected, soap_text_max_len=soap_text_max_len)
            self._log_consult("VALIDATE soap success pre_max_len={}".format(soap_text_max_len))

            current_session = ""
            if self.session_prompt_injection:
                current_session = str(
                    self.session_prompt_injection.resolve_current_session(doctor_name, patient_name).get("current_session", "")
                    or ""
                )
            if not current_session:
                current_session = "session_unknown"

            full_record = build_full_record(
                record_id=self._next_consult_record_id(),
                meeting_id=str(meeting_id or ""),
                current_session=current_session,
                session_started_at=self._fmt_iso8601_with_tz(start_time),
                doctor=doctor_name,
                patient=patient_name,
                soap=validated_soap.get("soap", {}),
            )
            full_record = validate_full_record(
                full_record,
                expected_meeting_id=str(meeting_id or ""),
                expected_doctor=doctor_name,
                expected_patient=patient_name,
                soap_text_max_len=soap_text_max_len,
            )
            self._log_consult("VALIDATE full_record success final_max_len={}".format(soap_text_max_len))

            write_result = self._write_consult_record(full_record, pair_key=pair_key, summary=summary or "")
            self._log_consult(
                "WRITE result status={} reason={} record_id={} meeting_id={} pair_key={}".format(
                    write_result.get("status", ""),
                    write_result.get("reason", ""),
                    write_result.get("record_id", ""),
                    str(meeting_id or ""),
                    pair_key,
                )
            )
            self._append_consult_audit(
                pair_key=pair_key,
                meeting_id=str(meeting_id or ""),
                record_id=str(write_result.get("record_id", "") or ""),
                status=str(write_result.get("status", "") or ""),
                reason=str(write_result.get("reason", "") or ""),
                message="ok",
                retryable=False,
            )
            if dropped:
                self._append_consult_audit(
                    pair_key=pair_key,
                    meeting_id=str(meeting_id or ""),
                    record_id=str(write_result.get("record_id", "") or ""),
                    status="warning",
                    reason="extra_fields_dropped",
                    message="|".join(dropped),
                    retryable=False,
                )
        except ConsultRecordError as err:
            self._append_consult_audit(
                pair_key=pair_key,
                meeting_id=str(meeting_id or ""),
                record_id="",
                status="failed",
                reason=str(err.reason or "unknown_error"),
                message=str(err.message or ""),
                retryable=bool(err.retryable),
            )
            self._log_consult(
                "ERROR reason={} retryable={} message={}".format(
                    str(err.reason or "unknown_error"),
                    bool(err.retryable),
                    str(err.message or ""),
                )
            )
        except Exception as err:
            self._append_consult_audit(
                pair_key=pair_key,
                meeting_id=str(meeting_id or ""),
                record_id="",
                status="failed",
                reason="unknown_error",
                message=str(err),
                retryable=False,
            )
            self._log_consult("ERROR reason=unknown_error message={}".format(str(err)))

    def _log(self, message: str) -> None:
        if self.logger:
            self.logger.info(f"[Intervention] {message}")

    def _log_highlight(self, message: str) -> None:
        text = f"========== [INTERVENTION] {message} =========="
        if self.logger:
            self.logger.info(text)

    def _log_queue_event(
        self,
        event: str,
        doctor: str,
        meeting_id: str,
        pair_key: str,
        step_phase: Any,
        reason: str,
    ) -> None:
        self._log_highlight(
            "MEETING_QUEUE event={} doctor={} meeting_id={} pair_key={} step_phase={} reason={}".format(
                event,
                doctor or "",
                meeting_id or "",
                pair_key or "",
                str(step_phase if step_phase is not None else ""),
                reason or "",
            )
        )
