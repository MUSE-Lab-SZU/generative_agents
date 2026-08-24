"""Intervention manager with low-intrusion hooks.

阶段 1 + 阶段 2 + 阶段 3：
- 保持低侵入：通过 hook 接入，不重写主认知流程
- 支持周期规则触发 meeting lock（weekly / interval_days / interval_minutes / interval_steps）
- 会话后自动解锁并提取医嘱，可交给 Environment Model 生成任务结果记忆
"""

from __future__ import annotations

import datetime
import copy
import json
import os
import traceback
from string import Template
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, List

from modules import utils
from modules.complaint_graph_trace import (
    capture_runtime_chat_trace,
    capture_runtime_reflection_trace,
    unavailable_trace,
    update_session_chat_summary,
    update_session_reflection_summary,
)
from modules.cbt_minimal_controller import (
    MINIMAL_STAGE_ADAPTER_VERSION,
    MINIMAL_PROGRESS_DEFAULT,
    PATIENT_STATE_DEFAULT,
    PROGRESSIVE_D_CONTROLLER_VERSION,
    PROGRESSIVE_D_PATIENT_STATE_DEFAULT,
    STAGE_ADAPTER_ACTIONS,
    compact_patient_state_text,
    compact_progressive_d_patient_state_text,
    compact_prior_progress_text,
    compact_progressive_d_subgoal_progress_text,
    compact_route_text,
    copy_soft_step_back_default,
    eligible_soft_step_back_targets,
    generate_session_end,
    initialize_subgoal_progress_state,
    load_prompt_template,
    load_stage_subgoal_map,
    load_strategy_map,
    load_term_glossary,
    merge_progressive_d_batch_progress,
    normalize_minimal_judge_output,
    normalize_minimal_session_eval,
    normalize_patient_state,
    normalize_progressive_d_control_eval,
    normalize_progressive_d_patient_state,
    progressive_d_completion_audit,
    recent_dialogue,
    route_strategies,
    safe_judge_fallback,
    split_utterances,
    tagged_dialogue,
)
from modules.memory_injection_manager import MemoryInjectionManager
from modules.intervention_consult_record import (
    ConsultRecordError,
    ConsultRecordPromptError,
    ConsultRecordValidationError,
    build_consult_history_record,
    build_dedup_key,
    build_full_record,
    build_pair_key,
    flatten_record_for_injection,
    project_whitelist,
    to_conversation_text,
    validate_consult_history_record,
    validate_full_record,
    validate_soap_only,
)
from modules.model import create_llm_model
from modules.resident_chat_scheduler import ResidentChatScheduler
from modules.session_prompt_injection_manager import SessionPromptInjectionManager
from modules.storage.index import LlamaIndex


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
        self.resident_chat_scheduler_cfg = intervention_cfg.get("resident_chat_scheduler", {}) or {}
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
        self.environment_model_cfg = intervention_cfg.get("environment_model", {}) or {}
        self.consult_record_cfg = intervention_cfg.get("consult_record", {}) or {}
        self.consult_history_cfg = intervention_cfg.get("consult_history", {}) or {}
        self.reflection_policy_cfg = (
            ((config.get("agent", {}) or {}).get("think", {}) or {}).get("reflection_policy", {}) or {}
        )
        self._consult_record_llm = None
        self._consult_record_llm_key = ""
        self._environment_llm = None
        self._environment_llm_key = ""
        self._think_llm = None
        self._think_llm_key = ""
        self._cbt_strategy_map_cache: Optional[Dict[str, Any]] = None
        self._cbt_strategy_map_cache_path = ""
        self._cbt_term_glossary_cache: Optional[Dict[str, Any]] = None
        self._cbt_term_glossary_cache_path = ""
        self._cbt_stage_subgoals_cache: Optional[Dict[str, Any]] = None
        self._cbt_stage_subgoals_cache_path = ""
        self._consult_history_indexes: Dict[str, Any] = {}

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
        self.memory_injection = MemoryInjectionManager(
            config=self.config,
            state=self.state,
            logger=self.logger,
        )
        self.session_prompt_injection = SessionPromptInjectionManager(
            config=self.config,
            state=self.state,
            logger=self.logger,
        )
        self.resident_chat_scheduler = ResidentChatScheduler(
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
        self._reset_step_flags()
        game_agents = getattr(game, "agents", {}) if isinstance(getattr(game, "agents", {}), dict) else {}
        if self.memory_injection:
            try:
                self.memory_injection.apply_rules_once(
                    agents=game_agents,
                    now=now,
                    step_no=step_no,
                )
            except Exception as exc:
                self._log_highlight(
                    "MEMORY_INJECTION_APPLY_ERROR step={} detail={}".format(
                        step_no,
                        str(exc),
                    )
                )
        if not self.enabled:
            return

        self._agents_ref = game_agents
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
        self._log_highlight(
            "[DEPR][STEP] step={} legacy_update_chain=skipped reason=dynamic_runtime_authoritative".format(
                now_step
            )
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

        scheduler = getattr(self, "resident_chat_scheduler", None)
        if scheduler:
            try:
                for payload in scheduler.collect_triggered_meetings(
                    now=now,
                    agents=game_agents,
                    should_trigger=self._should_trigger,
                ):
                    payload_rule_id = str(payload.get("rule_id", "") or "resident_chat")
                    self._trigger_meeting(game, payload_rule_id, payload, now)
            except Exception as exc:
                self._log_highlight(
                    "RESIDENT_CHAT_SCHEDULER_ERROR step={} detail={}".format(
                        step_no,
                        str(exc),
                    )
                )

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

    def after_chat(
        self,
        speaker: Any,
        other: Any,
        chats: Any,
        summary: str,
        start_time: Any,
    ) -> None:
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
        closed_meeting_kind = ""
        closed_meeting_snapshot = {}
        closed_doctor_name = ""
        closed_patient_name = ""
        closed_pair_key = ""
        doctor_for_session = None
        patient_for_session = None

        # 双方 meeting_id 一致时认为会面闭环达成
        if meeting_id_s and meeting_id_s == meeting_id_o:
            closed_meeting_id = str(meeting_id_s or "")
            meeting_snapshot = self.state.get("active_meetings", {}).get(meeting_id_s, {}) or {}
            closed_meeting_snapshot = meeting_snapshot if isinstance(meeting_snapshot, dict) else {}
            closed_doctor_name = str(closed_meeting_snapshot.get("doctor", "") or "")
            closed_patient_name = str(closed_meeting_snapshot.get("patient", "") or "")
            closed_meeting_kind = str(closed_meeting_snapshot.get("meeting_kind", "doctor_consult") or "doctor_consult")
            closed_pair_key = "{}<->{}".format(
                closed_doctor_name or getattr(speaker, "name", ""),
                closed_patient_name or getattr(other, "name", ""),
            )
            if str(getattr(speaker, "name", "") or "") == closed_doctor_name and str(getattr(other, "name", "") or "") == closed_patient_name:
                doctor_for_session, patient_for_session = speaker, other
            elif str(getattr(other, "name", "") or "") == closed_doctor_name and str(getattr(speaker, "name", "") or "") == closed_patient_name:
                doctor_for_session, patient_for_session = other, speaker
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
                doctor=closed_doctor_name,
                meeting_id=meeting_id_s,
                pair_key=closed_pair_key,
                step_phase=closed_meeting_snapshot.get("step_phase", ""),
                reason="chat_finished",
            )

        skip_consult_artifacts = closed_meeting_kind == "resident_chat"
        if skip_consult_artifacts and closed_meeting_id:
            self._mark_resident_chat_completed(closed_patient_name)
            self.mark_step_flag("forced_consult_happened", True)
        elif closed_meeting_id:
            self._mark_completed_meeting(
                doctor_name=closed_doctor_name,
                patient_name=closed_patient_name,
                meeting_id=closed_meeting_id,
                meeting_kind=closed_meeting_kind or "doctor_consult",
                finished_at=start_time,
            )
            self.mark_step_flag("forced_consult_happened", True)
        if skip_consult_artifacts:
            self._log_highlight(
                "RESIDENT_CHAT_AFTER_CHAT_SKIP_CONSULT meeting_id={} speaker={} other={}".format(
                    closed_meeting_id,
                    getattr(speaker, "name", ""),
                    getattr(other, "name", ""),
                )
            )
        else:
            if closed_meeting_id and closed_meeting_kind == "doctor_consult":
                # 正式医患会诊后抽取医生作业，并交给环境模型生成任务结果记忆
                self._log_highlight(
                    "ORDER_EXTRACT_START speaker={} other={} chats={}".format(
                        getattr(speaker, "name", ""),
                        getattr(other, "name", ""),
                        len(chats or []),
                    )
                )
                self._extract_and_queue_orders(
                    speaker=speaker,
                    other=other,
                    chats=chats,
                    meeting_id=closed_meeting_id,
                    meeting_kind=closed_meeting_kind,
                    start_time=start_time,
                )

            self._handle_consult_history_after_chat(
                speaker=speaker,
                other=other,
                chats=chats,
                summary=summary,
                start_time=start_time,
                meeting_id=closed_meeting_id,
            )

            self._handle_consult_record_after_chat(
                speaker=speaker,
                other=other,
                chats=chats,
                summary=summary,
                start_time=start_time,
                meeting_id=closed_meeting_id,
            )

        if self.session_prompt_injection and self.enabled:
            if not (doctor_for_session and patient_for_session):
                doctor_for_session, patient_for_session = self._resolve_doctor_patient_pair(speaker, other)
            if doctor_for_session and patient_for_session:
                meeting_id_for_session = str(closed_meeting_id or meeting_id_s or meeting_id_o or "")
                if meeting_id_for_session:
                    session_eval_end: Optional[bool] = None
                    post_treatment_followup_active = self._is_post_treatment_followup_active_for_pair(
                        doctor_for_session.name,
                        patient_for_session.name,
                    )
                    skip_post_treatment_session_eval = bool(
                        post_treatment_followup_active
                        and self._should_skip_session_eval_for_post_treatment_followup()
                    )
                    if skip_post_treatment_session_eval:
                        current_session = self._resolve_session_prompt_current_session(
                            doctor_for_session.name,
                            patient_for_session.name,
                        )
                        audit = {
                            "applied": True,
                            "pair_key": build_pair_key(doctor_for_session.name, patient_for_session.name),
                            "session_before": current_session,
                            "action": "post_treatment_followup",
                            "current_session": current_session,
                            "completed": True,
                            "post_treatment_followup": True,
                            "session_eval_skipped": True,
                        }
                        self._log_highlight(
                            "SESSION_PROMPT_POST_TREATMENT_FOLLOWUP meeting_id={} pair={}<->{} session_eval_skipped=true".format(
                                meeting_id_for_session,
                                doctor_for_session.name,
                                patient_for_session.name,
                            )
                        )
                    else:
                        pair_key_for_session = build_pair_key(
                            doctor_for_session.name,
                            patient_for_session.name,
                        )
                        minimal_duplicate = self._is_minimal_meeting_processed(
                            pair_key_for_session,
                            meeting_id_for_session,
                        )
                        if minimal_duplicate:
                            current_session = self._resolve_session_prompt_current_session(
                                doctor_for_session.name,
                                patient_for_session.name,
                            )
                            audit = {
                                "applied": True,
                                "pair_key": pair_key_for_session,
                                "session_before": current_session,
                                "action": "skip_duplicate_meeting",
                                "current_session": current_session,
                                "completed": bool(
                                    self.session_prompt_injection.resolve_current_session(
                                        doctor_for_session.name,
                                        patient_for_session.name,
                                    ).get("completed", False)
                                ),
                            }
                        else:
                            minimal_mode = self.get_cbt_controller_mode() == "minimal"
                            progressive_d_mode = bool(
                                minimal_mode
                                and self._is_progressive_d_configured()
                            )
                            session_eval_payload = None
                            if progressive_d_mode:
                                self._log_highlight(
                                    "[PROGRESSIVE_D_SESSION_EVAL_SKIPPED] meeting_id={}".format(
                                        meeting_id_for_session
                                    )
                                )
                            else:
                                try:
                                    session_eval_payload = self.evaluate_session_after_chat(
                                        doctor=doctor_for_session,
                                        patient=patient_for_session,
                                        chats=chats or [],
                                        meeting_id=meeting_id_for_session,
                                    )
                                except Exception as exc:
                                    if not minimal_mode:
                                        raise
                                    self._log_highlight(
                                        "[MINIMAL_SESSION_EVAL_FAIL_CLOSED] meeting_id={} error={}".format(
                                            meeting_id_for_session,
                                            str(exc),
                                        )
                                    )
                            subgoal_record = None
                            if progressive_d_mode:
                                try:
                                    subgoal_record = self.evaluate_progressive_d_control_after_chat(
                                        doctor=doctor_for_session,
                                        patient=patient_for_session,
                                        chats=chats or [],
                                        meeting_id=meeting_id_for_session,
                                    )
                                except Exception as exc:
                                    self._log_highlight(
                                        "[PROGRESSIVE_D_CONTROL_ISOLATED_ERROR] meeting_id={} error={}".format(
                                            meeting_id_for_session,
                                            str(exc),
                                        )
                                    )
                            if progressive_d_mode:
                                current_session = self._resolve_session_prompt_current_session(
                                    doctor_for_session.name,
                                    patient_for_session.name,
                                )
                                expected_version = str(
                                    self.get_progressive_d_runtime_policy().get(
                                        "controller_version",
                                        MINIMAL_STAGE_ADAPTER_VERSION,
                                    )
                                    or MINIMAL_STAGE_ADAPTER_VERSION
                                )
                                latest_record = (
                                    subgoal_record
                                    if isinstance(subgoal_record, dict)
                                    else {}
                                )
                                record_fresh = bool(
                                    str(
                                        latest_record.get(
                                            "meeting_id",
                                            "",
                                        )
                                        or ""
                                    )
                                    == meeting_id_for_session
                                    and str(
                                        latest_record.get(
                                            "legacy_session",
                                            "",
                                        )
                                        or ""
                                    )
                                    == current_session
                                    and str(
                                        latest_record.get(
                                            "cbt_controller_version",
                                            "",
                                        )
                                        or ""
                                    )
                                    == expected_version
                                )
                                transition_adapter = (
                                    latest_record.get(
                                        "transition_adapter",
                                        {},
                                    )
                                    if record_fresh
                                    else {}
                                )
                                if not isinstance(
                                    transition_adapter,
                                    dict,
                                ):
                                    transition_adapter = {}
                                if not transition_adapter:
                                    transition_adapter = {
                                        "recommended_action": "",
                                        "effective_action": "hold",
                                        "session_end": False,
                                        "reason": (
                                            "stage_adapter_missing"
                                            if record_fresh
                                            else "stage_adapter_record_stale"
                                        ),
                                    }
                                session_eval_end = bool(
                                    transition_adapter.get(
                                        "session_end",
                                        False,
                                    )
                                )
                                self.append_dialog_judge_trace_eval(
                                    meeting_id=meeting_id_for_session,
                                    pair_key=pair_key_for_session,
                                    step_time=self._resolve_trace_step_time(),
                                    normalized_eval_payload={
                                        "control_source": (
                                            "batch_control_adapter"
                                            if self._is_progressive_d_v3_configured()
                                            else "subgoal_adapter"
                                        ),
                                        "session_end": session_eval_end,
                                        "stage_transition_adapter": copy.deepcopy(
                                            transition_adapter
                                        ),
                                    },
                                )
                            elif isinstance(session_eval_payload, dict):
                                current_session = self._resolve_session_prompt_current_session(
                                    doctor_for_session.name,
                                    patient_for_session.name,
                                )
                                reason_text = str(session_eval_payload.get("reason", "") or "")
                                self._store_session_eval_reason(
                                    pair_key=pair_key_for_session,
                                    current_session=current_session,
                                    meeting_id=meeting_id_for_session,
                                    reason=reason_text,
                                )
                                if minimal_mode:
                                    self._store_minimal_session_eval_result(
                                        pair_key=pair_key_for_session,
                                        meeting_id=meeting_id_for_session,
                                        payload=session_eval_payload,
                                    )
                                self.append_dialog_judge_trace_eval(
                                    meeting_id=meeting_id_for_session,
                                    pair_key=pair_key_for_session,
                                    step_time=self._resolve_trace_step_time(),
                                    normalized_eval_payload=session_eval_payload,
                                )
                                session_eval_end = bool(
                                    session_eval_payload.get("session_end", False)
                                )
                            if minimal_mode and session_eval_end is None:
                                session_eval_end = False
                            audit = self.session_prompt_injection.on_chat_finished(
                                doctor_name=doctor_for_session.name,
                                patient_name=patient_for_session.name,
                                chats=chats or [],
                                meeting_id=meeting_id_for_session,
                                now_str=self._fmt_dt(start_time),
                                session_eval_end=session_eval_end,
                            )
                            if minimal_mode:
                                self._mark_controller_meeting_processed(
                                    pair_key_for_session,
                                    meeting_id_for_session,
                                )
                            if (
                                minimal_mode and progressive_d_mode
                            ):
                                try:
                                    self._progressive_d_progress_state_for_pair(
                                        pair_key_for_session,
                                        str(audit.get("current_session", "") or ""),
                                    )
                                except Exception as exc:
                                    self._log_highlight(
                                        "[SUBGOAL_PROGRESS_SYNC_ERROR] meeting_id={} error={}".format(
                                            meeting_id_for_session,
                                            str(exc),
                                        )
                                    )
                    if session_eval_end is not None:
                        self._log_highlight(
                            "[SESSION_PROMPT_EVAL_END] meeting_id={} session_eval_end={} action={}".format(
                                meeting_id_for_session,
                                bool(session_eval_end),
                                audit.get("action", ""),
                            )
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
                    try:
                        action = str((audit or {}).get("action", "") or "").strip()
                        bridge = getattr(patient_for_session, "external_memory_bridge", None)
                        session_before = str((audit or {}).get("session_before", "") or "").strip()
                        if bridge:
                            session_chat_node_id = self._resolve_latest_chat_node_id(patient_for_session)
                            bridge_summary = bridge.apply_session_chat_memory_level_adjustment(
                                session_before=session_before,
                                session_chat_node_id=session_chat_node_id,
                                trace={
                                    "meeting_id": meeting_id_for_session,
                                    "pair_key": build_pair_key(
                                        doctor_for_session.name,
                                        patient_for_session.name,
                                    ),
                                    "action": action,
                                    "all_session_chat_enabled": (
                                        "all_session_chat" in getattr(bridge, "updata_to_l2_apply_scenes", [])
                                    ),
                                },
                            )
                            self._log_highlight(
                                "SESSION_CHAT_LEVEL_ADJUST pair={}<->{} summary={}".format(
                                    doctor_for_session.name,
                                    patient_for_session.name,
                                    bridge_summary,
                                )
                            )
                    except Exception as exc:
                        self._log_highlight(
                            "SESSION_CHAT_LEVEL_ADJUST_ERROR pair={}<->{} detail={}".format(
                                doctor_for_session.name,
                                patient_for_session.name,
                                str(exc),
                            )
                        )
                    if self.memory_injection:
                        try:
                            agents_map = dict(self._agents_ref) if isinstance(self._agents_ref, dict) else {}
                            speaker_name = str(getattr(speaker, "name", "") or "")
                            other_name = str(getattr(other, "name", "") or "")
                            if speaker_name and (speaker_name not in agents_map):
                                agents_map[speaker_name] = speaker
                            if other_name and (other_name not in agents_map):
                                agents_map[other_name] = other
                            self.memory_injection.apply_rules_on_session_completed(
                                agents=agents_map,
                                doctor_name=doctor_for_session.name,
                                patient_name=patient_for_session.name,
                                audit=audit if isinstance(audit, dict) else {},
                                meeting_id=meeting_id_for_session,
                                now=start_time,
                                step_no=int(self.config.get("step", 0) or 0),
                            )
                        except Exception as exc:
                            self._log_highlight(
                                "MEMORY_INJECTION_SESSION_APPLY_ERROR detail={}".format(str(exc))
                            )
                    if str((audit or {}).get("action", "") or "").strip() == "completed_stop_injection":
                        self.mark_step_flag("treatment_completed_this_step", True)
                    if (
                        self.stop_rule_scheduling_on_session_completed
                        and bool(audit.get("completed", False))
                        and not self._is_post_treatment_followup_active_for_pair(
                            doctor_for_session.name,
                            patient_for_session.name,
                        )
                    ):
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

        self._handle_conditional_reflection_after_chat(
            speaker=speaker,
            other=other,
            chats=chats or [],
            summary=str(summary or ""),
            start_time=start_time,
            meeting_id=closed_meeting_id,
            meeting_kind=closed_meeting_kind or "doctor_consult",
            doctor=doctor_for_session,
            patient=patient_for_session,
            closed_doctor_name=closed_doctor_name,
            closed_patient_name=closed_patient_name,
        )

        doctor, patient = self._resolve_doctor_patient_pair(speaker, other)
        if not doctor or not patient:
            return
        if closed_meeting_id and closed_meeting_kind != "resident_chat":
            self.append_dialog_judge_trace_reflection(
                doctor=doctor,
                patient=patient,
                meeting_id=closed_meeting_id,
            )

        now_step = int(self.config.get("step", 0) or 0)
        runtime_snapshot = self._get_patient_depression_runtime_snapshot(patient)
        self._log_highlight(
            "[DEPR][EVAL] patient={} step={} skipped=true reason=legacy_update_chain_removed".format(
                patient.name,
                now_step,
            )
        )
        self._log_highlight(
            "[DEPR][EVENT] patient={} wording={}".format(
                patient.name,
                runtime_snapshot.get("current_event_wording", ""),
            )
        )
        self._log_highlight(
            "[DEPR][LAYER] patient={} short_term_changed={} long_term_changed={}".format(
                patient.name,
                bool(runtime_snapshot.get("short_term_changed", False)),
                bool(runtime_snapshot.get("long_term_changed", False)),
            )
        )

    def _handle_conditional_reflection_after_chat(
        self,
        speaker: Any,
        other: Any,
        chats: Any,
        summary: str,
        start_time: Any,
        meeting_id: str,
        meeting_kind: str,
        doctor: Any = None,
        patient: Any = None,
        closed_doctor_name: str = "",
        closed_patient_name: str = "",
    ) -> None:
        if not str(meeting_id or "").strip():
            return
        cfg = self._reflection_conditional_cfg()
        if not self._safe_bool(cfg.get("enabled", True), True):
            self._log_highlight(
                "REFLECTION_AFTER_CHAT_SKIP meeting_id={} reason=conditional_disabled".format(
                    str(meeting_id or "")
                )
            )
            return
        rules = cfg.get("rules", [])
        if not isinstance(rules, list):
            rules = []
        if not rules:
            rules = [self._default_after_chat_reflection_rule()]

        if not (doctor and patient):
            resolved_doctor, resolved_patient = self._resolve_doctor_patient_pair(speaker, other)
            doctor = doctor or resolved_doctor
            patient = patient or resolved_patient

        agents_map = self._reflection_agents_map(speaker, other, doctor, patient)
        transcript = to_conversation_text(chats or [])
        matched = 0
        for idx, rule in enumerate(rules):
            if not isinstance(rule, dict):
                continue
            if not self._safe_bool(rule.get("enabled", True), True):
                continue
            if str(rule.get("event", "after_chat") or "after_chat").strip() != "after_chat":
                continue
            if not self._reflection_rule_matches_meeting_kind(rule, meeting_kind):
                continue
            targets = self._resolve_reflection_rule_targets(
                rule=rule,
                speaker=speaker,
                other=other,
                doctor=doctor,
                patient=patient,
                agents_map=agents_map,
            )
            if not targets:
                self._log_highlight(
                    "REFLECTION_AFTER_CHAT_SKIP meeting_id={} rule_index={} reason=target_empty".format(
                        str(meeting_id or ""),
                        idx,
                    )
                )
                continue
            matched += 1
            once_per_meeting = self._safe_bool(rule.get("once_per_meeting", True), True)
            for target_role, target in targets:
                if target is None or not hasattr(target, "trigger_reflection"):
                    continue
                target_name = str(getattr(target, "name", "") or "")
                if once_per_meeting and hasattr(target, "has_reflected_for_meeting"):
                    try:
                        if target.has_reflected_for_meeting(meeting_id):
                            self._log_highlight(
                                "REFLECTION_AFTER_CHAT_SKIP meeting_id={} target={} reason=already_reflected".format(
                                    str(meeting_id or ""),
                                    target_name,
                                )
                            )
                            continue
                    except Exception:
                        pass
                chat_node_id = ""
                try:
                    chat_node_id = self._resolve_latest_chat_node_id(target)
                except Exception:
                    chat_node_id = ""
                context = {
                    "event": "after_chat",
                    "meeting_kind": str(meeting_kind or ""),
                    "meeting_id": str(meeting_id or ""),
                    "target_role": str(target_role or ""),
                    "doctor": str(getattr(doctor, "name", "") or closed_doctor_name or ""),
                    "patient": str(getattr(patient, "name", "") or closed_patient_name or ""),
                    "speaker": str(getattr(speaker, "name", "") or ""),
                    "other": str(getattr(other, "name", "") or ""),
                    "chat_summary": str(summary or ""),
                    "chat_transcript": transcript,
                    "chat_node_id": str(chat_node_id or ""),
                    "step": int(self.config.get("step", 0) or 0),
                    "started_at": self._fmt_dt(start_time),
                }
                try:
                    result = target.trigger_reflection(
                        trigger_source="after_chat",
                        trigger_context=context,
                    )
                    self._log_highlight(
                        "REFLECTION_AFTER_CHAT_DONE meeting_id={} target={} role={} triggered={} reason={}".format(
                            str(meeting_id or ""),
                            target_name,
                            str(target_role or ""),
                            bool((result or {}).get("triggered", False)) if isinstance(result, dict) else bool(result),
                            str((result or {}).get("reason", "") if isinstance(result, dict) else ""),
                        )
                    )
                except Exception as exc:
                    self._log_highlight(
                        "REFLECTION_AFTER_CHAT_ERROR meeting_id={} target={} detail={}".format(
                            str(meeting_id or ""),
                            target_name,
                            str(exc),
                        )
                    )
        if matched <= 0:
            self._log_highlight(
                "REFLECTION_AFTER_CHAT_SKIP meeting_id={} meeting_kind={} reason=no_rule_matched".format(
                    str(meeting_id or ""),
                    str(meeting_kind or ""),
                )
            )

    def _reflection_conditional_cfg(self) -> Dict[str, Any]:
        policy = self.reflection_policy_cfg if isinstance(self.reflection_policy_cfg, dict) else {}
        conditional = policy.get("conditional", {}) if isinstance(policy.get("conditional", {}), dict) else {}
        if not conditional:
            conditional = {
                "enabled": True,
                "rules": [self._default_after_chat_reflection_rule()],
            }
        return conditional

    @staticmethod
    def _default_after_chat_reflection_rule() -> Dict[str, Any]:
        return {
            "event": "after_chat",
            "meeting_kind": "doctor_consult",
            "target_role": "patient",
            "once_per_meeting": True,
        }

    def _reflection_rule_matches_meeting_kind(self, rule: Dict[str, Any], meeting_kind: str) -> bool:
        expected = rule.get("meeting_kind", "doctor_consult")
        actual = str(meeting_kind or "doctor_consult").strip() or "doctor_consult"
        if isinstance(expected, str):
            text = expected.strip()
            return text in {"", "*", "all"} or text == actual
        if isinstance(expected, (list, tuple, set)):
            normalized = {str(item or "").strip() for item in expected}
            return "*" in normalized or "all" in normalized or actual in normalized
        return False

    def _reflection_agents_map(self, *agents: Any) -> Dict[str, Any]:
        agents_map = dict(self._agents_ref) if isinstance(self._agents_ref, dict) else {}
        for agent in agents:
            name = str(getattr(agent, "name", "") or "")
            if name and name not in agents_map:
                agents_map[name] = agent
        return agents_map

    def _resolve_reflection_rule_targets(
        self,
        rule: Dict[str, Any],
        speaker: Any,
        other: Any,
        doctor: Any,
        patient: Any,
        agents_map: Dict[str, Any],
    ) -> List[Any]:
        target_agents = rule.get("target_agents", [])
        if isinstance(target_agents, str):
            target_names = [target_agents]
        elif isinstance(target_agents, (list, tuple, set)):
            target_names = list(target_agents)
        else:
            target_names = []
        if target_names:
            results = []
            seen = set()
            for raw_name in target_names:
                name = str(raw_name or "").strip()
                if not name or name in seen:
                    continue
                seen.add(name)
                target = agents_map.get(name)
                if target is not None:
                    results.append(("target_agent", target))
            return results

        role = str(rule.get("target_role", "patient") or "patient").strip()
        candidates = []
        if role == "patient":
            candidates = [("patient", patient)]
        elif role == "doctor":
            candidates = [("doctor", doctor)]
        elif role == "speaker":
            candidates = [("speaker", speaker)]
        elif role == "other":
            candidates = [("other", other)]
        elif role in {"both", "pair", "doctor_patient"}:
            candidates = [("doctor", doctor), ("patient", patient)]
        else:
            candidates = [(role, agents_map.get(role))]

        results = []
        seen = set()
        for target_role, target in candidates:
            name = str(getattr(target, "name", "") or "")
            if not name or name in seen:
                continue
            seen.add(name)
            results.append((target_role, target))
        return results

    def _get_patient_depression_runtime_snapshot(self, patient: Any) -> Dict[str, Any]:
        snapshot = {
            "current_event_wording": "",
            "emotion_label": "",
            "emotion_style": "",
            "emotion_intensity": 0.0,
            "last_throttle_reason": "",
            "short_term_changed": False,
            "long_term_changed": False,
            "changed_paths": [],
        }
        engine = getattr(patient, "depression_dynamic", None)
        if engine is None or not callable(getattr(engine, "get_current_state_info", None)):
            return snapshot
        try:
            info = engine.get_current_state_info()
        except Exception:
            return snapshot
        if not isinstance(info, dict):
            return snapshot

        stage = info.get("current_stage", {}) if isinstance(info.get("current_stage", {}), dict) else {}
        emotion = info.get("emotion", {}) if isinstance(info.get("emotion", {}), dict) else {}
        wording = ""
        for key in ("summary", "label", "core_belief", "id"):
            text = str(stage.get(key, "") or "").strip()
            if text:
                wording = text
                break
        snapshot["current_event_wording"] = wording
        snapshot["emotion_label"] = str(emotion.get("label", "") or "")
        snapshot["emotion_style"] = str(emotion.get("style", "") or "")
        try:
            snapshot["emotion_intensity"] = float(emotion.get("intensity", 0.0) or 0.0)
        except Exception:
            snapshot["emotion_intensity"] = 0.0
        return snapshot

    def get_session_prompt_for_judge(self, speaker: Any, other: Any, forced: bool) -> str:
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

        if self.session_prompt_injection:
            return self.session_prompt_injection.get_doctor_injection_block(
                doctor_name=doctor.name,
                patient_name=patient.name,
                forced=bool(forced),
                meeting_id=meeting_id,
            )
        return ""

    def get_doctor_consult_record_injection(self, speaker: Any, other: Any, forced: bool) -> str:
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

        return consult_block

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

    def get_cbt_controller_mode(self) -> str:
        intervention_cfg = self.config.get("intervention", {}) or {}
        mode = str(intervention_cfg.get("cbt_controller_mode", "legacy") or "legacy").strip().lower()
        return mode if mode in ("legacy", "minimal") else "legacy"

    def _minimal_progress_state(self, pair_key: str) -> Dict[str, Any]:
        self._ensure_session_eval_state_schema()
        state = self.state.setdefault("session_eval_state", {})
        pairs = state.setdefault("minimal_by_pair", {})
        item = pairs.setdefault(str(pair_key or ""), copy.deepcopy(MINIMAL_PROGRESS_DEFAULT))
        if not isinstance(item, dict):
            item = copy.deepcopy(MINIMAL_PROGRESS_DEFAULT)
            pairs[str(pair_key or "")] = item
        for key, value in MINIMAL_PROGRESS_DEFAULT.items():
            if key not in item:
                item[key] = copy.deepcopy(value)
        if not isinstance(item.get("latest_patient_state"), dict):
            item["latest_patient_state"] = {}
        item["tracker_meeting_id"] = str(item.get("tracker_meeting_id", "") or "")
        if not isinstance(item.get("high_risk_seen"), bool):
            item["high_risk_seen"] = False
        item.pop("last_patient_evidence", None)
        return item

    def _minimal_state_for_meeting(
        self,
        pair_key: str,
        meeting_id: str,
    ) -> Dict[str, Any]:
        progress_state = self._minimal_progress_state(pair_key)
        target = str(meeting_id or "").strip()
        current = str(progress_state.get("tracker_meeting_id", "") or "").strip()
        if not target or current != target:
            progress_state["latest_patient_state"] = copy.deepcopy(PATIENT_STATE_DEFAULT)
            progress_state["tracker_meeting_id"] = target
            progress_state["high_risk_seen"] = False
        elif not progress_state.get("latest_patient_state"):
            progress_state["latest_patient_state"] = copy.deepcopy(PATIENT_STATE_DEFAULT)
        return progress_state

    def _progressive_d_tracker_state(self, pair_key: str) -> Dict[str, Any]:
        self._ensure_progressive_d_state_schema()
        state = self.state.setdefault("progressive_d_tracker_state", {})
        pairs = state.setdefault("by_pair", {})
        pair = str(pair_key or "")
        item = pairs.setdefault(
            pair,
            {
                "latest_patient_state": copy.deepcopy(
                    PROGRESSIVE_D_PATIENT_STATE_DEFAULT
                ),
                "tracker_meeting_id": "",
                "high_risk_seen": False,
            },
        )
        if not isinstance(item, dict):
            item = {
                "latest_patient_state": copy.deepcopy(
                    PROGRESSIVE_D_PATIENT_STATE_DEFAULT
                ),
                "tracker_meeting_id": "",
                "high_risk_seen": False,
            }
            pairs[pair] = item
        latest = item.get("latest_patient_state")
        if not isinstance(latest, dict):
            item["latest_patient_state"] = copy.deepcopy(
                PROGRESSIVE_D_PATIENT_STATE_DEFAULT
            )
        else:
            item["latest_patient_state"] = normalize_progressive_d_patient_state(
                latest
            )
        item["tracker_meeting_id"] = str(
            item.get("tracker_meeting_id", "") or ""
        )
        item["high_risk_seen"] = bool(item.get("high_risk_seen", False))
        item.pop("tracker_valid", None)
        return item

    def _progressive_d_state_for_meeting(
        self,
        pair_key: str,
        meeting_id: str,
    ) -> Dict[str, Any]:
        state = self._progressive_d_tracker_state(pair_key)
        target = str(meeting_id or "").strip()
        current = str(state.get("tracker_meeting_id", "") or "").strip()
        if target and current != target:
            state["tracker_meeting_id"] = target
            state["high_risk_seen"] = False
        return state

    def _progressive_d_risk_check_state(
        self,
        pair_key: str,
    ) -> Dict[str, Any]:
        self._ensure_progressive_d_state_schema()
        state = self.state.setdefault("risk_check_state_by_pair", {})
        pair = str(pair_key or "")
        item = state.setdefault(
            pair,
            {
                "last_risk_level": "none",
                "episode": 0,
                "check_pending": False,
                "checked": False,
                "awaiting_patient_response": False,
            },
        )
        if not isinstance(item, dict):
            item = {}
            state[pair] = item
        item["last_risk_level"] = str(
            item.get("last_risk_level", "none") or "none"
        )
        try:
            item["episode"] = max(0, int(item.get("episode", 0) or 0))
        except (TypeError, ValueError):
            item["episode"] = 0
        for field in (
            "check_pending",
            "checked",
            "awaiting_patient_response",
        ):
            item[field] = bool(item.get(field, False))
        return item

    def _update_progressive_d_risk_check_after_patient(
        self,
        pair_key: str,
        risk_level: str,
    ) -> Dict[str, Any]:
        item = self._progressive_d_risk_check_state(pair_key)
        current = str(risk_level or "none").strip().lower()
        previous = str(item.get("last_risk_level", "none") or "none")
        if current == "none":
            item.update(
                {
                    "last_risk_level": "none",
                    "check_pending": False,
                    "checked": False,
                    "awaiting_patient_response": False,
                }
            )
            return item
        if current == "high":
            if previous != "high":
                item["episode"] = int(item.get("episode", 0) or 0) + 1
            item.update(
                {
                    "last_risk_level": "high",
                    "check_pending": True,
                    "checked": False,
                    "awaiting_patient_response": False,
                }
            )
            return item
        if previous == "none":
            item["episode"] = int(item.get("episode", 0) or 0) + 1
            item["check_pending"] = True
            item["checked"] = False
        if bool(item.get("awaiting_patient_response", False)):
            item["awaiting_patient_response"] = False
            item["check_pending"] = False
            item["checked"] = True
        item["last_risk_level"] = "possible"
        return item

    def _mark_progressive_d_safety_check_requested(
        self,
        pair_key: str,
    ) -> None:
        item = self._progressive_d_risk_check_state(pair_key)
        item["awaiting_patient_response"] = True
        item["check_pending"] = False

    def _is_minimal_meeting_processed(self, pair_key: str, meeting_id: str) -> bool:
        if self.get_cbt_controller_mode() != "minimal":
            return False
        return self._is_controller_meeting_processed(
            pair_key,
            meeting_id,
        )

    def _minimal_state_tracker_policy(self) -> Dict[str, Any]:
        intervention_cfg = self.config.get("intervention", {}) or {}
        raw = intervention_cfg.get("state_tracker", {}) or {}
        if self._is_progressive_d_v3_configured():
            progressive_d_policy = self.get_progressive_d_runtime_policy()
            return {
                "enabled": bool(progressive_d_policy.get("enabled", False)),
                "prompt_file": str(
                    progressive_d_policy.get("state_tracker_prompt_file", "")
                    or "data/prompts/intervention/state_tracker_progressive_d.txt"
                ).strip(),
                "retry": int(progressive_d_policy.get("state_tracker_retry", 2) or 2),
                "recent_utterances": int(
                    progressive_d_policy.get("state_tracker_recent_utterances", 8)
                    or 8
                ),
            }
        default_prompt = (
            "data/prompts/intervention/state_tracker_progressive_d.txt"
            if self._is_progressive_d_v3_configured()
            else "data/prompts/intervention/state_tracker.txt"
        )
        configured_prompt_key = (
            "progressive_d_prompt_file"
            if self._is_progressive_d_v3_configured()
            else "prompt_file"
        )
        return {
            "enabled": self._safe_bool(raw.get("enabled", True), True),
            "prompt_file": str(
                raw.get(configured_prompt_key, default_prompt)
                or default_prompt
            ).strip(),
            "retry": max(1, self._safe_int(raw.get("retry", 2), 2)),
            "recent_utterances": max(1, self._safe_int(raw.get("recent_utterances", 8), 8)),
        }

    def _minimal_strategy_map(self) -> Dict[str, Any]:
        intervention_cfg = self.config.get("intervention", {}) or {}
        router_cfg = intervention_cfg.get("strategy_router", {}) or {}
        path = str(
            router_cfg.get("strategy_map_file", "data/intervention/cbt_strategy_map.json")
            or "data/intervention/cbt_strategy_map.json"
        ).strip()
        if self._cbt_strategy_map_cache is None or self._cbt_strategy_map_cache_path != path:
            self._cbt_strategy_map_cache = load_strategy_map(path)
            self._cbt_strategy_map_cache_path = path
        return self._cbt_strategy_map_cache

    def get_progressive_d_runtime_policy(self) -> Dict[str, Any]:
        """Return the native D policy, with a strict fallback for old D configs."""

        intervention_cfg = self.config.get("intervention", {}) or {}
        raw = intervention_cfg.get("progressive_d")
        native = isinstance(raw, dict)
        if native:
            state_tracker = raw.get("state_tracker", {}) or {}
            judge = raw.get("judge", {}) or {}
            control_eval = raw.get("control_eval", {}) or {}
            transitions = raw.get("transitions", {}) or {}
            if not isinstance(state_tracker, dict):
                state_tracker = {}
            if not isinstance(judge, dict):
                judge = {}
            if not isinstance(control_eval, dict):
                control_eval = {}
            if not isinstance(transitions, dict):
                transitions = {}
            enabled = bool(
                self.get_cbt_controller_mode() == "minimal"
                and self._safe_bool(raw.get("enabled", False), False)
            )
            route = str(
                control_eval.get("route", "forced_llm") or "forced_llm"
            ).strip().lower()
            if route not in {"forced_llm", "think_llm"}:
                route = "forced_llm"
            raw_actions = transitions.get(
                "allowed_actions",
                ["stay", "side_step", "soft_step_back", "hold"],
            )
            allowed_actions = []
            if isinstance(raw_actions, list):
                for value in raw_actions:
                    action = str(value or "").strip().lower()
                    if action in STAGE_ADAPTER_ACTIONS and action not in allowed_actions:
                        allowed_actions.append(action)
            raw_predecessors = transitions.get("soft_step_back_predecessors", {}) or {}
            predecessors = (
                copy.deepcopy(raw_predecessors)
                if isinstance(raw_predecessors, dict)
                else {}
            )
            return {
                "enabled": enabled,
                "source": "progressive_d",
                "controller_version": str(
                    raw.get("controller_version", "") or ""
                ).strip(),
                "state_tracker_prompt_file": str(
                    state_tracker.get(
                        "prompt_file",
                        "data/prompts/intervention/state_tracker_progressive_d.txt",
                    )
                    or "data/prompts/intervention/state_tracker_progressive_d.txt"
                ).strip(),
                "state_tracker_retry": max(
                    1,
                    self._safe_int(state_tracker.get("retry", 2), 2),
                ),
                "state_tracker_recent_utterances": max(
                    1,
                    self._safe_int(state_tracker.get("recent_utterances", 8), 8),
                ),
                "judge_prompt_file": str(
                    judge.get(
                        "prompt_file",
                        "data/prompts/intervention/dialog_judge_progressive_d.txt",
                    )
                    or "data/prompts/intervention/dialog_judge_progressive_d.txt"
                ).strip(),
                "judge_retry": max(1, self._safe_int(judge.get("retry", 2), 2)),
                "control_route": route,
                "control_prompt_file": str(
                    control_eval.get(
                        "prompt_file",
                        "data/prompts/intervention/control_eval_progressive_d.txt",
                    )
                    or "data/prompts/intervention/control_eval_progressive_d.txt"
                ).strip(),
                "stage_subgoals_file": str(
                    control_eval.get(
                        "stage_subgoals_file",
                        "data/intervention/cbt_stage_subgoals.json",
                    )
                    or "data/intervention/cbt_stage_subgoals.json"
                ).strip(),
                "control_retry": max(
                    1,
                    self._safe_int(control_eval.get("retry", 2), 2),
                ),
                "allowed_actions": allowed_actions,
                "soft_step_back_predecessors": predecessors,
            }

        shadow = intervention_cfg.get("subgoal_shadow", {}) or {}
        progress = intervention_cfg.get("subgoal_progress_for_judge", {}) or {}
        limited = intervention_cfg.get("limited_subgoal_transitions", {}) or {}
        adapter = intervention_cfg.get("legacy_stage_transition_adapter", {}) or {}
        if not isinstance(shadow, dict):
            shadow = {}
        if not isinstance(progress, dict):
            progress = {}
        if not isinstance(limited, dict):
            limited = {}
        if not isinstance(adapter, dict):
            adapter = {}
        version = str(intervention_cfg.get("cbt_controller_version", "") or "").strip()
        adapter_version = str(adapter.get("controller_version", "") or "").strip()
        enabled = bool(
            self.get_cbt_controller_mode() == "minimal"
            and self._safe_bool(intervention_cfg.get("subgoal_shadow_enabled", False), False)
            and self._safe_bool(progress.get("enabled", False), False)
            and self._safe_bool(limited.get("enabled", False), False)
            and self._safe_bool(adapter.get("enabled", False), False)
            and version == PROGRESSIVE_D_CONTROLLER_VERSION
            and adapter_version == PROGRESSIVE_D_CONTROLLER_VERSION
        )
        return {
            "enabled": enabled,
            "source": "legacy_progressive_d_config",
            "controller_version": adapter_version,
            "state_tracker_prompt_file": str(
                (intervention_cfg.get("state_tracker", {}) or {}).get(
                    "progressive_d_prompt_file",
                    "data/prompts/intervention/state_tracker_progressive_d.txt",
                )
                or "data/prompts/intervention/state_tracker_progressive_d.txt"
            ).strip(),
            "state_tracker_retry": max(
                1,
                self._safe_int(
                    (intervention_cfg.get("state_tracker", {}) or {}).get("retry", 2),
                    2,
                ),
            ),
            "state_tracker_recent_utterances": max(
                1,
                self._safe_int(
                    (intervention_cfg.get("state_tracker", {}) or {}).get(
                        "recent_utterances",
                        8,
                    ),
                    8,
                ),
            ),
            "judge_prompt_file": str(
                (intervention_cfg.get("dialog_judge", {}) or {}).get(
                    "progressive_d_prompt_file",
                    "data/prompts/intervention/dialog_judge_progressive_d.txt",
                )
                or "data/prompts/intervention/dialog_judge_progressive_d.txt"
            ).strip(),
            "judge_retry": max(
                1,
                self._safe_int(
                    (intervention_cfg.get("dialog_judge", {}) or {}).get("retry", 2),
                    2,
                ),
            ),
            "control_route": str(shadow.get("route", "forced_llm") or "forced_llm").strip().lower(),
            "control_prompt_file": str(
                shadow.get(
                    "progressive_d_prompt_file",
                    "data/prompts/intervention/control_eval_progressive_d.txt",
                )
                or "data/prompts/intervention/control_eval_progressive_d.txt"
            ).strip(),
            "stage_subgoals_file": str(
                shadow.get(
                    "stage_subgoals_file",
                    "data/intervention/cbt_stage_subgoals.json",
                )
                or "data/intervention/cbt_stage_subgoals.json"
            ).strip(),
            "control_retry": max(1, self._safe_int(shadow.get("retry", 2), 2)),
            "allowed_actions": copy.deepcopy(
                adapter.get("allowed_actions", list(STAGE_ADAPTER_ACTIONS))
            ),
            "soft_step_back_predecessors": copy.deepcopy(
                adapter.get("soft_step_back_predecessors", {}) or {}
            ),
        }

    def _minimal_term_glossary(self) -> Dict[str, Any]:
        intervention_cfg = self.config.get("intervention", {}) or {}
        router_cfg = intervention_cfg.get("strategy_router", {}) or {}
        path = str(
            router_cfg.get(
                "term_glossary_file",
                "data/intervention/cbt_term_glossary.json",
            )
            or "data/intervention/cbt_term_glossary.json"
        ).strip()
        if (
            self._cbt_term_glossary_cache is None
            or self._cbt_term_glossary_cache_path != path
        ):
            self._cbt_term_glossary_cache = load_term_glossary(
                path,
                self._minimal_strategy_map(),
            )
            self._cbt_term_glossary_cache_path = path
        return self._cbt_term_glossary_cache


    def _is_progressive_d_configured(self) -> bool:
        return bool(self.get_progressive_d_runtime_policy().get("enabled", False))

    def _is_progressive_d_v3_configured(self) -> bool:
        policy = self.get_progressive_d_runtime_policy()
        intervention_cfg = self.config.get("intervention", {}) or {}
        configured_version = str(intervention_cfg.get("cbt_controller_version", "") or "").strip()
        policy_version = str(policy.get("controller_version", "") or "").strip()
        return bool(
            policy.get("enabled", False)
            and configured_version == PROGRESSIVE_D_CONTROLLER_VERSION
            and policy_version == PROGRESSIVE_D_CONTROLLER_VERSION
        )

    def _minimal_stage_subgoals(self, path: str) -> Dict[str, Any]:
        path_text = str(path or "").strip()
        if (
            self._cbt_stage_subgoals_cache is None
            or self._cbt_stage_subgoals_cache_path != path_text
        ):
            self._cbt_stage_subgoals_cache = load_stage_subgoal_map(path_text)
            self._cbt_stage_subgoals_cache_path = path_text
        return self._cbt_stage_subgoals_cache


    def _progressive_d_progress_state_for_pair(
        self,
        pair_key: str,
        current_session: str,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        policy = self.get_progressive_d_runtime_policy()
        if not bool(policy.get("enabled", False)):
            raise ValueError("Progressive D is disabled")
        stage_map = self._minimal_stage_subgoals(
            str(policy.get("stage_subgoals_file", "") or "")
        )
        stage_item = stage_map.get(str(current_session or ""))
        if not isinstance(stage_item, dict):
            raise ValueError(
                f"no static subgoal mapping for fixed outline: {current_session}"
            )
        self._ensure_subgoal_progress_state_schema()
        pairs = self.state.setdefault("subgoal_progress_by_pair", {})
        pair_key_text = str(pair_key or "")
        current = initialize_subgoal_progress_state(
            str(current_session or ""),
            stage_item,
            pairs.get(pair_key_text),
            preserve_active=True,
            include_side_step=True,
            include_stage_adapter=True,
            include_evidence_turn_id=False,
            include_batch_control=True,
            controller_version=str(
                policy.get("controller_version", PROGRESSIVE_D_CONTROLLER_VERSION)
                or PROGRESSIVE_D_CONTROLLER_VERSION
            ),
        )
        soft_step_back = current.get("soft_step_back", {})
        if isinstance(soft_step_back, dict) and soft_step_back.get("active", False):
            eligible_targets = eligible_soft_step_back_targets(
                stage_map=stage_map,
                legacy_order=list(
                    self.session_prompt_injection.injection_cfg.get("order", [])
                    or []
                ),
                current_session=str(current_session or ""),
                current_macro_stage=str(stage_item.get("macro_stage", "") or ""),
                active_subgoal_id=str(
                    soft_step_back.get("return_to_subgoal_id", "") or ""
                ),
                configured_predecessors=policy.get(
                    "soft_step_back_predecessors",
                    {},
                ),
            )
            if str(soft_step_back.get("target_subgoal_id", "") or "") not in eligible_targets:
                current["soft_step_back"] = copy_soft_step_back_default()
        pairs[pair_key_text] = current
        return current, stage_item


    def _apply_progressive_d_control_result(
        self,
        pair_key: str,
        meeting_id: str,
        legacy_session: str,
        payload: Dict[str, Any],
        evaluator_valid: bool = True,
    ) -> Dict[str, Any]:
        """Merge one D batch and derive the calibrated graded completion audit."""

        result: Dict[str, Any] = {
            "recommended_action": str(
                payload.get("recommended_action", "") or ""
            ),
            "effective_action": "hold",
            "session_end": False,
            "reason": "",
            "session_prompt_complete": False,
            "completion_reason": "not_ready",
            "completion_score": 0.0,
            "completion_threshold": 0.0,
            "meetings_in_current_prompt": 0,
        }
        tracker = self._progressive_d_tracker_state(pair_key)
        meeting_id_text = str(meeting_id or "").strip()
        if (
            not meeting_id_text
            or str(tracker.get("tracker_meeting_id", "") or "")
            != meeting_id_text
        ):
            result["reason"] = "meeting_state_stale"
            return result
        intervention_cfg = self.config.get("intervention", {}) or {}
        stage_policy = self.get_progressive_d_runtime_policy()
        expected_version = str(
            intervention_cfg.get("cbt_controller_version", "") or ""
        )
        actual_version = str(
            stage_policy.get("controller_version", "") or ""
        )
        if (
            expected_version != PROGRESSIVE_D_CONTROLLER_VERSION
            or actual_version != expected_version
        ):
            result["reason"] = "controller_version_mismatch"
            return result

        progress_state, stage_item = self._progressive_d_progress_state_for_pair(
            pair_key,
            legacy_session,
        )
        processed = progress_state.get("meeting_ids_in_current_prompt", [])
        if isinstance(processed, list) and meeting_id_text in processed:
            result["reason"] = "duplicate_meeting"
            return result
        candidate = copy.deepcopy(progress_state)
        if evaluator_valid:
            try:
                merge_progressive_d_batch_progress(
                    candidate,
                    stage_item,
                    payload.get("subgoals", []),
                    meeting_id_text,
                )
            except Exception as exc:
                result["reason"] = "batch_merge_invalid:{}".format(str(exc))
                return result
        else:
            meeting_ids = candidate.setdefault(
                "meeting_ids_in_current_prompt",
                [],
            )
            if not isinstance(meeting_ids, list):
                meeting_ids = []
                candidate["meeting_ids_in_current_prompt"] = meeting_ids
            meeting_ids.append(meeting_id_text)
            candidate["meetings_in_current_prompt"] = len(meeting_ids)

        audit = progressive_d_completion_audit(
            candidate,
            high_risk_seen=bool(tracker.get("high_risk_seen", False)),
        )
        result.update(copy.deepcopy(audit))
        prompt_state = (
            self.session_prompt_injection.resolve_current_session(
                *str(pair_key or "").split("::", 1)
            )
            if "::" in str(pair_key or "")
            else {}
        )
        order = list(
            self.session_prompt_injection.injection_cfg.get("order", [])
            or []
        )
        current = str(
            prompt_state.get("current_session", legacy_session) or ""
        )
        try:
            current_index = int(prompt_state.get("current_index", 0) or 0)
        except (TypeError, ValueError):
            current_index = -1
        if (
            current != str(legacy_session or "")
            or current not in order
            or current_index < 0
            or current_index >= len(order)
            or order[current_index] != current
        ):
            result["reason"] = "legacy_state_invalid"
            return result

        action = str(payload.get("recommended_action", "") or "")
        if not evaluator_valid:
            result["effective_action"] = "hold"
            result["reason"] = "control_evaluator_invalid"
            result["session_prompt_complete"] = False
            result["completion_reason"] = "blocked_by_hold"
        elif action == "hold":
            result["effective_action"] = "hold"
            result["reason"] = "recommended_hold"
            result["session_prompt_complete"] = False
            result["completion_reason"] = "blocked_by_hold"
        elif action == "side_step":
            side_step = candidate.get("side_step")
            if not isinstance(side_step, dict):
                side_step = {}
                candidate["side_step"] = side_step
            if side_step.get("active", False):
                result["reason"] = "side_step_already_active"
            elif not str(payload.get("reason", "") or "").strip():
                result["reason"] = "side_step_reason_empty"
            else:
                side_step.update(
                    {
                        "active": True,
                        "reason": str(payload.get("reason", "") or "").strip(),
                        "return_to_subgoal_id": str(
                            candidate.get("active_subgoal_id", "") or ""
                        ),
                        "started_meeting_id": meeting_id_text,
                    }
                )
                result["effective_action"] = "side_step"
                result["reason"] = "side_step_activated"
            result["session_prompt_complete"] = False
            result["completion_reason"] = (
                "blocked_by_side_step"
                if result["effective_action"] == "side_step"
                else "blocked_by_hold"
            )
        elif action == "soft_step_back":
            target = str(payload.get("target_subgoal_id", "") or "")
            stage_map = self._minimal_stage_subgoals(
                str(
                    self.get_progressive_d_runtime_policy().get(
                        "stage_subgoals_file",
                        "",
                    )
                    or ""
                )
            )
            eligible = eligible_soft_step_back_targets(
                stage_map=stage_map,
                legacy_order=order,
                current_session=current,
                current_macro_stage=str(
                    stage_item.get("macro_stage", "") or ""
                ),
                active_subgoal_id=str(
                    candidate.get("active_subgoal_id", "") or ""
                ),
                configured_predecessors=stage_policy.get(
                    "soft_step_back_predecessors",
                    {},
                ),
            )
            if target not in eligible:
                result["reason"] = "soft_step_back_target_not_allowed"
            else:
                candidate["soft_step_back"] = {
                    "active": True,
                    "target_subgoal_id": target,
                    "return_to_subgoal_id": str(
                        candidate.get("active_subgoal_id", "") or ""
                    ),
                    "reason": str(payload.get("reason", "") or "").strip(),
                }
                result["effective_action"] = "soft_step_back"
                result["reason"] = "soft_step_back_activated"
            result["session_prompt_complete"] = False
            result["completion_reason"] = (
                "blocked_by_soft_step_back"
                if result["effective_action"] == "soft_step_back"
                else "blocked_by_hold"
            )
        elif bool(audit.get("session_prompt_complete", False)):
            result["effective_action"] = "advance_stage"
            result["session_end"] = True
            result["reason"] = str(
                audit.get("completion_reason", "") or "score_threshold"
            )
        elif action == "stay":
            result["effective_action"] = "stay"
            result["reason"] = str(
                audit.get("completion_reason", "") or "not_ready"
            )
        else:
            result["effective_action"] = "hold"
            result["reason"] = "recommended_hold"
            result["session_prompt_complete"] = False
            result["completion_reason"] = "blocked_by_hold"

        self.state.setdefault("subgoal_progress_by_pair", {})[
            str(pair_key or "")
        ] = candidate
        return result

    def _compact_subgoal_progress_for_judge(
        self,
        pair_key: str,
        current_session: str,
    ) -> str:
        progress_state, stage_item = self._progressive_d_progress_state_for_pair(
            pair_key,
            current_session,
        )
        glossary = self._minimal_term_glossary()
        return compact_progressive_d_subgoal_progress_text(
            stage_item,
            progress_state,
            glossary.get("macro_stages", {}),
        )

    def _compact_prior_control_progress_for_judge(
        self,
        pair_key: str,
        current_session: str,
        progress_state: Mapping[str, Any],
    ) -> str:
        """Expose only the final controller decision to Progressive D Judge."""

        if not self._is_progressive_d_configured():
            return compact_prior_progress_text(progress_state)

        state = self.state.get(
            (
                "progressive_d_control_state"
                if self._is_progressive_d_v3_configured()
                else "subgoal_shadow_state"
            ),
            {},
        )
        latest_by_pair = (
            state.get("latest_by_pair", {})
            if isinstance(state, Mapping)
            else {}
        )
        record = (
            latest_by_pair.get(str(pair_key or ""), {})
            if isinstance(latest_by_pair, Mapping)
            else {}
        )
        if not isinstance(record, Mapping):
            record = {}
        adapter = record.get("transition_adapter", {})
        if not isinstance(adapter, Mapping):
            adapter = {}
        current_text = str(current_session or "").strip() or "未知"
        if not record or not adapter:
            return (
                "尚无上次会后最终控制结论；"
                "当前固定会谈提纲={}，继续围绕当前提纲推进。"
            ).format(current_text)

        expected_version = str(
            self.get_progressive_d_runtime_policy().get(
                "controller_version",
                MINIMAL_STAGE_ADAPTER_VERSION,
            )
            or MINIMAL_STAGE_ADAPTER_VERSION
        ).strip()
        record_version = str(
            record.get("cbt_controller_version", "") or ""
        ).strip()
        if not record_version or record_version != expected_version:
            return (
                "上次会后控制结论版本不匹配，暂缓采用；"
                "当前固定会谈提纲={}，继续围绕当前提纲推进。"
            ).format(current_text)

        effective_action = str(
            adapter.get("effective_action", "") or ""
        ).strip()
        reason = str(adapter.get("reason", "") or "").strip()
        previous_session = str(
            record.get("legacy_session", "") or ""
        ).strip()
        action_labels = self._minimal_term_glossary().get("actions", {})
        action_label = (
            str(action_labels.get(effective_action, "") or "").strip()
            if isinstance(action_labels, Mapping)
            else ""
        )
        if not effective_action or effective_action == "hold":
            return (
                "上次最终控制结论=暂缓并继续当前固定会谈提纲；"
                "当前固定会谈提纲={}；"
                "控制器说明={}"
            ).format(current_text, reason or "尚无可执行的会后推进结论")
        if effective_action == "advance_stage":
            return (
                "上次最终控制结论=当前固定会谈提纲已完成，并严格相邻推进；"
                "上一固定会谈提纲={}；当前固定会谈提纲={}；"
                "控制器说明={}"
            ).format(
                previous_session or "未知",
                current_text,
                reason or "全部推进条件已通过",
            )
        return (
            "上次最终控制结论={}；"
            "当前固定会谈提纲={}，继续围绕当前提纲推进；"
            "控制器说明={}"
        ).format(
            action_label or effective_action,
            current_text,
            reason or "无",
        )

    def update_patient_state_after_utterance(
        self,
        speaker: Any,
        other: Any,
        chats: Any,
        patient_utterance: str,
        forced: bool,
        turn_no: int,
    ) -> Dict[str, Any]:
        progressive_d_v3 = self._is_progressive_d_v3_configured()
        fallback = copy.deepcopy(
            PROGRESSIVE_D_PATIENT_STATE_DEFAULT
            if progressive_d_v3
            else PATIENT_STATE_DEFAULT
        )
        if self.get_cbt_controller_mode() != "minimal" or not self.enabled or not bool(forced):
            return fallback
        doctor, patient = self._resolve_doctor_patient_pair(speaker, other)
        if not doctor or not patient:
            return fallback
        if str(getattr(speaker, "name", "") or "") != str(getattr(patient, "name", "") or ""):
            return fallback
        meeting_id, pair_key, _ = self._resolve_trace_session_context(doctor, patient)
        progress_state = (
            self._progressive_d_state_for_meeting(pair_key, meeting_id)
            if progressive_d_v3
            else self._minimal_state_for_meeting(pair_key, meeting_id)
        )
        policy = self._minimal_state_tracker_policy()
        if not bool(policy.get("enabled", True)):
            if not progressive_d_v3:
                progress_state["latest_patient_state"] = copy.deepcopy(fallback)
            return fallback

        patient_utterances, _ = split_utterances(chats or [], patient.name)
        prompt_text = ""
        raw: Any = {}
        error = ""
        try:
            if progressive_d_v3:
                patient_state_text = self._get_progressive_d_patient_state_text(
                    patient
                )
                if not patient_state_text:
                    raise ValueError("Progressive D patient internal state is empty")
                prompt_text = load_prompt_template(
                    str(policy.get("prompt_file", "") or ""),
                    {"PATIENT_INTERNAL_STATE": patient_state_text},
                    required_placeholders=("PATIENT_INTERNAL_STATE",),
                )
            else:
                prompt_text = load_prompt_template(
                    str(policy.get("prompt_file", "") or ""),
                    {
                        "CURRENT_PATIENT_REPLY": str(patient_utterance or ""),
                        "RECENT_DIALOGUE": recent_dialogue(
                            chats or [],
                            int(policy.get("recent_utterances", 8) or 8),
                        ),
                    },
                    required_placeholders=("CURRENT_PATIENT_REPLY", "RECENT_DIALOGUE"),
                )
            raw = self._call_think_llm_json(
                prompt_text=prompt_text,
                retry=int(policy.get("retry", 2) or 2),
                doctor_agent=doctor,
                caller="cbt_state_tracker",
            )
            normalized = (
                normalize_progressive_d_patient_state(raw)
                if progressive_d_v3
                else normalize_patient_state(raw, patient_utterances)
            )
        except Exception as exc:
            prior = progress_state.get("latest_patient_state", {})
            normalized = (
                normalize_progressive_d_patient_state(prior)
                if progressive_d_v3 and isinstance(prior, dict)
                else fallback
            )
            error = str(exc)

        progress_state["latest_patient_state"] = copy.deepcopy(normalized)
        if progressive_d_v3:
            self._update_progressive_d_risk_check_after_patient(
                pair_key,
                str(normalized.get("risk_level", "none") or "none"),
            )
        if str(normalized.get("risk_level", "") or "") == "high":
            progress_state["high_risk_seen"] = True
        self.append_forced_prompt_trace_record(
            speaker=doctor,
            other=patient,
            role="state_tracker_llm",
            prompt_text=prompt_text,
            output={
                "raw_output": (
                    normalize_progressive_d_patient_state(raw)
                    if progressive_d_v3
                    else copy.deepcopy(raw)
                ),
                "normalized_output": copy.deepcopy(normalized),
            },
            turn_no=int(turn_no or -1),
            meta={
                "source": (
                    "progressive_d_state_tracker"
                    if progressive_d_v3 and not error
                    else (
                        "progressive_d_state_tracker_error"
                        if progressive_d_v3
                        else (
                            "state_tracker"
                            if not error
                            else "state_tracker_error"
                        )
                    )
                ),
                "route": "think_llm",
                "caller": "cbt_state_tracker",
                "error": error,
            },
        )
        return normalized

    def evaluate_progressive_d_control_after_chat(
        self,
        doctor: Any,
        patient: Any,
        chats: Any,
        meeting_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Batch-evaluate all current subgoals for Progressive D v3."""

        if not self._is_progressive_d_v3_configured():
            return None
        policy = self.get_progressive_d_runtime_policy()
        if not bool(policy.get("enabled", False)):
            return None
        pair_key = build_pair_key(
            str(getattr(doctor, "name", "") or ""),
            str(getattr(patient, "name", "") or ""),
        )
        current_session = self._resolve_session_prompt_current_session(
            str(getattr(doctor, "name", "") or ""),
            str(getattr(patient, "name", "") or ""),
        )
        prompt_text = ""
        raw: Any = {}
        normalized: Optional[Dict[str, Any]] = None
        error = ""
        stage_item: Dict[str, Any] = {}
        macro_stage = ""
        soft_targets: List[str] = []
        try:
            stage_map = self._minimal_stage_subgoals(
                str(policy.get("stage_subgoals_file", "") or "")
            )
            configured = stage_map.get(current_session)
            if not isinstance(configured, dict):
                raise ValueError(
                    f"no static subgoal mapping for fixed outline: {current_session}"
                )
            stage_item = configured
            macro_stage = str(stage_item.get("macro_stage", "") or "")
            progress_state, _ = self._progressive_d_progress_state_for_pair(
                pair_key,
                current_session,
            )
            stage_policy = self.get_progressive_d_runtime_policy()
            soft_targets = eligible_soft_step_back_targets(
                stage_map=stage_map,
                legacy_order=list(
                    self.session_prompt_injection.injection_cfg.get(
                        "order",
                        [],
                    )
                    or []
                ),
                current_session=current_session,
                current_macro_stage=macro_stage,
                active_subgoal_id=str(
                    progress_state.get("active_subgoal_id", "") or ""
                ),
                configured_predecessors=stage_policy.get(
                    "soft_step_back_predecessors",
                    {},
                ),
            )
            configured_subgoals = [
                {
                    "子目标ID": str(item.get("id", "") or ""),
                    "目标": str(item.get("goal", "") or ""),
                    "完成线索": str(
                        item.get("completion_hint", "") or ""
                    ),
                }
                for item in stage_item.get("subgoals", [])
                if isinstance(item, Mapping)
            ]
            prompt_text = load_prompt_template(
                str(policy.get("control_prompt_file", "") or ""),
                {
                    "DOCTOR": str(getattr(doctor, "name", "") or ""),
                    "PATIENT": str(getattr(patient, "name", "") or ""),
                    "FIXED_OUTLINE_ID": current_session,
                    "MACRO_STAGE": macro_stage,
                    "SUBGOAL_CONFIG": json.dumps(
                        configured_subgoals,
                        ensure_ascii=False,
                        indent=2,
                    ),
                    "PRIOR_PROGRESS": compact_progressive_d_subgoal_progress_text(
                        stage_item,
                        progress_state,
                        self._minimal_term_glossary().get(
                            "macro_stages",
                            {},
                        ),
                    ),
                    "MEETING_NUMBER": str(
                        int(
                            progress_state.get(
                                "meetings_in_current_prompt",
                                0,
                            )
                            or 0
                        )
                        + 1
                    ),
                    "CURRENT_SESSION_PROMPT": str(
                        self.get_session_prompt_for_judge(
                            speaker=doctor,
                            other=patient,
                            forced=True,
                        )
                        or ""
                    ),
                    "DIALOGUE_HISTORY": to_conversation_text(chats or []),
                    "SOFT_STEP_BACK_TARGETS": (
                        "、".join(soft_targets) or "无"
                    ),
                },
                required_placeholders=(
                    "DOCTOR",
                    "PATIENT",
                    "FIXED_OUTLINE_ID",
                    "MACRO_STAGE",
                    "SUBGOAL_CONFIG",
                    "PRIOR_PROGRESS",
                    "MEETING_NUMBER",
                    "CURRENT_SESSION_PROMPT",
                    "DIALOGUE_HISTORY",
                    "SOFT_STEP_BACK_TARGETS",
                ),
            )
            if str(policy.get("control_route", "forced_llm") or "") == "think_llm":
                raw = self._call_think_llm_json(
                    prompt_text=prompt_text,
                    retry=int(policy.get("control_retry", 2) or 2),
                    doctor_agent=doctor,
                    caller="cbt_progressive_d_control_eval",
                )
            else:
                raw = self._call_forced_llm_json(
                    prompt_text,
                    retry=int(policy.get("control_retry", 2) or 2),
                    caller="cbt_progressive_d_control_eval",
                )
            normalized = normalize_progressive_d_control_eval(
                raw,
                configured_subgoal_ids=[
                    str(item.get("id", "") or "")
                    for item in stage_item.get("subgoals", [])
                    if isinstance(item, Mapping)
                ],
                allowed_soft_targets=soft_targets,
            )
        except Exception as exc:
            error = str(exc)

        adapter = self._apply_progressive_d_control_result(
            pair_key=pair_key,
            meeting_id=str(meeting_id or ""),
            legacy_session=current_session,
            payload=normalized or {},
            evaluator_valid=isinstance(normalized, dict) and not bool(error),
        )
        record = {
            "meeting_id": str(meeting_id or ""),
            "legacy_session": current_session,
            "macro_stage": macro_stage,
            "subgoals": copy.deepcopy(
                (normalized or {}).get("subgoals", [])
            ),
            "recommended_action": str(
                (normalized or {}).get("recommended_action", "") or ""
            ),
            "target_subgoal_id": str(
                (normalized or {}).get("target_subgoal_id", "") or ""
            ),
            "reason": str((normalized or {}).get("reason", "") or ""),
            "valid": isinstance(normalized, dict) and not bool(error),
            "error": error,
            "transition_adapter": copy.deepcopy(adapter),
            "cbt_controller_version": PROGRESSIVE_D_CONTROLLER_VERSION,
            "updated_at": self._fmt_dt(self._now()),
        }
        self._ensure_progressive_d_state_schema()
        control = self.state.setdefault("progressive_d_control_state", {})
        control.setdefault("latest_by_pair", {})[pair_key] = copy.deepcopy(
            record
        )
        history = control.setdefault("history_by_pair", {}).setdefault(
            pair_key,
            [],
        )
        if isinstance(history, list):
            history.append(copy.deepcopy(record))
        self.append_forced_prompt_trace_record(
            speaker=doctor,
            other=patient,
            role="control_eval_llm",
            prompt_text=prompt_text,
            output={
                "normalized_output": copy.deepcopy(normalized),
                "legacy_session": current_session,
                "macro_stage": macro_stage,
                "transition_adapter": copy.deepcopy(adapter),
            },
            turn_no=-1,
            meeting_id=str(meeting_id or ""),
            pair_key=pair_key,
            meta={
                "source": (
                    "progressive_d_control_eval"
                    if not error
                    else "progressive_d_control_eval_error"
                ),
                "route": str(policy.get("control_route", "") or ""),
                "error": error,
                "controller_version": PROGRESSIVE_D_CONTROLLER_VERSION,
            },
        )
        return copy.deepcopy(record)


    def build_doctor_reply_guidance(self, judge: Any) -> str:
        if not isinstance(judge, dict):
            return ""
        root_strategy_context = str(
            judge.get("_root_complaint_strategy_context", "") or ""
        ).strip()
        if self.get_cbt_controller_mode() != "minimal":
            guidance = str(judge.get("advice", "") or "").strip()
            if root_strategy_context:
                guidance = "{}\n{}".format(guidance, root_strategy_context).strip()
            return guidance
        glossary = self._minimal_term_glossary()
        primary = self._doctor_guidance_term(
            str(judge.get("primary_strategy", "") or ""),
            glossary.get("primary_strategies", {}),
        )
        micro = self._doctor_guidance_term(
            str(judge.get("micro_skill", "") or ""),
            glossary.get("micro_skills", {}),
        )
        guidance = (
            "主要策略：{}\n微技能：{}\n本轮目标：{}"
        ).format(
            primary,
            micro,
            str(judge.get("turn_goal", "") or ""),
        ).strip()
        if self._is_progressive_d_v3_configured():
            internal = str(
                judge.get("_patient_internal_state", "") or ""
            ).strip()
            if internal:
                guidance += (
                    "\n已知患者材料：{}\n回应边界：围绕一个明确材料缺口推进；"
                    "不要重复泛问感受、身体感觉或已经核对过的安全问题。"
                ).format(internal)
        if root_strategy_context:
            guidance += "\n" + root_strategy_context
        return guidance

    @staticmethod
    def _doctor_guidance_term(term_id: str, terms: Any) -> str:
        item = terms.get(term_id, {}) if isinstance(terms, Mapping) else {}
        if not isinstance(item, Mapping):
            return str(term_id or "")
        label = str(item.get("label", "") or "").strip()
        instruction = str(item.get("instruction", "") or "").strip()
        if label and instruction:
            return "{}。执行要点：{}".format(label, instruction)
        return label or str(term_id or "")

    def get_dialog_judge_runtime_policy(self) -> Dict[str, Any]:
        default_policy = {
            "enabled": False,
            "prompt_file": "data/prompts/intervention/dialog_judge.txt",
            "legacy_prompt_file": "data/prompts/intervention/dialog_judge_legacy.txt",
            "progressive_d_prompt_file": "data/prompts/intervention/dialog_judge_progressive_d.txt",
            "retry": 2,
            "force_forced_llm": True,
            "patient_state_prompt_file": "data/prompts/intervention/dialog_judge_patient_state_summary.txt",
            "patient_state_summary_retry": 2,
            "source": "defaults",
        }
        if self.get_cbt_controller_mode() == "legacy":
            default_policy["prompt_file"] = default_policy["legacy_prompt_file"]
        intervention_cfg = self.config.get("intervention", {}) or {}
        raw_cfg = intervention_cfg.get("dialog_judge", {}) or {}
        if not isinstance(raw_cfg, dict):
            self._log_highlight(
                "[DIALOG_JUDGE_POLICY] source=defaults enabled={} retry={} force_forced_llm={} prompt_file={} patient_state_prompt_file={} patient_state_summary_retry={}".format(
                    bool(default_policy["enabled"]),
                    int(default_policy["retry"]),
                    bool(default_policy["force_forced_llm"]),
                    default_policy["prompt_file"],
                    default_policy["patient_state_prompt_file"],
                    int(default_policy["patient_state_summary_retry"]),
                )
            )
            return default_policy

        policy = dict(default_policy)
        policy["source"] = "dialog_judge_config"
        policy["enabled"] = self._safe_bool(raw_cfg.get("enabled", False), False)
        prompt_file = str(raw_cfg.get("prompt_file", default_policy["prompt_file"]) or "").strip()
        policy["prompt_file"] = prompt_file or default_policy["prompt_file"]
        configured_legacy_prompt_file = str(raw_cfg.get("legacy_prompt_file", "") or "").strip()
        legacy_prompt_file = configured_legacy_prompt_file or default_policy["legacy_prompt_file"]
        policy["legacy_prompt_file"] = legacy_prompt_file or default_policy["legacy_prompt_file"]
        if self.get_cbt_controller_mode() == "legacy":
            policy["prompt_file"] = (
                policy["legacy_prompt_file"]
                if configured_legacy_prompt_file
                else policy["prompt_file"]
            )
        retry = self._safe_int(raw_cfg.get("retry", default_policy["retry"]), default_policy["retry"])
        policy["retry"] = retry if retry >= 1 else default_policy["retry"]
        policy["force_forced_llm"] = self._safe_bool(
            raw_cfg.get("force_forced_llm", default_policy["force_forced_llm"]),
            default_policy["force_forced_llm"],
        )
        patient_state_prompt_file = str(
            raw_cfg.get("patient_state_prompt_file", default_policy["patient_state_prompt_file"]) or ""
        ).strip()
        policy["patient_state_prompt_file"] = patient_state_prompt_file or default_policy["patient_state_prompt_file"]
        patient_state_summary_retry = self._safe_int(
            raw_cfg.get("patient_state_summary_retry", default_policy["patient_state_summary_retry"]),
            default_policy["patient_state_summary_retry"],
        )
        policy["patient_state_summary_retry"] = (
            patient_state_summary_retry if patient_state_summary_retry >= 1 else default_policy["patient_state_summary_retry"]
        )
        if self._is_progressive_d_v3_configured():
            progressive_d_policy = self.get_progressive_d_runtime_policy()
            policy["subgoal_progress_enabled"] = bool(
                progressive_d_policy.get("enabled", False)
            )
            policy["prompt_file"] = str(
                progressive_d_policy.get(
                    "judge_prompt_file",
                    default_policy["progressive_d_prompt_file"],
                )
                or default_policy["progressive_d_prompt_file"]
            )
            policy["retry"] = int(
                progressive_d_policy.get("judge_retry", policy["retry"])
                or policy["retry"]
            )
        else:
            policy["subgoal_progress_enabled"] = False
        self._log_highlight(
            "[DIALOG_JUDGE_POLICY] source={} enabled={} retry={} force_forced_llm={} prompt_file={} patient_state_prompt_file={} patient_state_summary_retry={}".format(
                policy["source"],
                bool(policy["enabled"]),
                int(policy["retry"]),
                bool(policy["force_forced_llm"]),
                policy["prompt_file"],
                policy["patient_state_prompt_file"],
                int(policy["patient_state_summary_retry"]),
            )
        )
        return policy

    def get_session_eval_runtime_policy(self) -> Dict[str, Any]:
        default_policy = {
            "enabled": False,
            "route": "forced_llm",
            "prompt_file": "data/prompts/intervention/session_eval.txt",
            "legacy_prompt_file": "data/prompts/intervention/session_eval_legacy.txt",
            "retry": 2,
            "history_recent_n": 3,
            "source": "defaults",
        }
        if self.get_cbt_controller_mode() == "legacy":
            default_policy["prompt_file"] = default_policy["legacy_prompt_file"]
        intervention_cfg = self.config.get("intervention", {}) or {}
        raw_cfg = intervention_cfg.get("session_eval", {}) or {}
        if not isinstance(raw_cfg, dict):
            self._log_highlight(
                "[SESSION_EVAL_POLICY] source=defaults enabled={} route={} retry={} prompt_file={}".format(
                    bool(default_policy["enabled"]),
                    default_policy["route"],
                    int(default_policy["retry"]),
                    default_policy["prompt_file"],
                )
            )
            return default_policy

        policy = dict(default_policy)
        policy["source"] = "session_eval_config"
        policy["enabled"] = self._safe_bool(raw_cfg.get("enabled", False), False)
        route = str(raw_cfg.get("route", default_policy["route"]) or default_policy["route"]).strip().lower()
        if route not in ("forced_llm", "think_llm"):
            route = default_policy["route"]
        policy["route"] = route
        prompt_file = str(raw_cfg.get("prompt_file", default_policy["prompt_file"]) or "").strip()
        policy["prompt_file"] = prompt_file or default_policy["prompt_file"]
        configured_legacy_prompt_file = str(raw_cfg.get("legacy_prompt_file", "") or "").strip()
        legacy_prompt_file = configured_legacy_prompt_file or default_policy["legacy_prompt_file"]
        policy["legacy_prompt_file"] = legacy_prompt_file or default_policy["legacy_prompt_file"]
        if self.get_cbt_controller_mode() == "legacy":
            policy["prompt_file"] = (
                policy["legacy_prompt_file"]
                if configured_legacy_prompt_file
                else policy["prompt_file"]
            )
        retry = self._safe_int(raw_cfg.get("retry", default_policy["retry"]), default_policy["retry"])
        policy["retry"] = retry if retry >= 1 else default_policy["retry"]
        history_recent_n = self._safe_int(
            raw_cfg.get("history_recent_n", default_policy["history_recent_n"]),
            default_policy["history_recent_n"],
        )
        policy["history_recent_n"] = history_recent_n if history_recent_n >= 0 else default_policy["history_recent_n"]
        self._log_highlight(
            "[SESSION_EVAL_POLICY] source={} enabled={} route={} retry={} prompt_file={} history_recent_n={}".format(
                policy["source"],
                bool(policy["enabled"]),
                policy["route"],
                int(policy["retry"]),
                policy["prompt_file"],
                int(policy["history_recent_n"]),
            )
        )
        return policy

    def get_consult_history_runtime_policy(self) -> Dict[str, Any]:
        default_policy = {
            "enabled": False,
            "retrieve_top_k": 3,
            "gate_prompt_file": "data/prompts/intervention/consult_history_gate.txt",
            "gate_retry": 2,
            "summary_route": "forced_llm",
            "summary_prompt_file": "data/prompts/intervention/consult_history_summary.txt",
            "summary_retry": 2,
            "source": "defaults",
        }
        raw_cfg = self.consult_history_cfg if isinstance(self.consult_history_cfg, dict) else {}
        if not isinstance(raw_cfg, dict):
            return default_policy
        policy = dict(default_policy)
        policy["source"] = "consult_history_config"
        policy["enabled"] = self._safe_bool(raw_cfg.get("enabled", False), False)
        retrieve_top_k = self._safe_int(raw_cfg.get("retrieve_top_k", default_policy["retrieve_top_k"]), default_policy["retrieve_top_k"])
        policy["retrieve_top_k"] = retrieve_top_k if retrieve_top_k >= 1 else default_policy["retrieve_top_k"]
        gate_prompt_file = str(raw_cfg.get("gate_prompt_file", default_policy["gate_prompt_file"]) or "").strip()
        policy["gate_prompt_file"] = gate_prompt_file or default_policy["gate_prompt_file"]
        gate_retry = self._safe_int(raw_cfg.get("gate_retry", default_policy["gate_retry"]), default_policy["gate_retry"])
        policy["gate_retry"] = gate_retry if gate_retry >= 1 else default_policy["gate_retry"]
        summary_route = str(raw_cfg.get("summary_route", default_policy["summary_route"]) or default_policy["summary_route"]).strip().lower()
        if summary_route not in ("forced_llm", "think_llm"):
            summary_route = default_policy["summary_route"]
        policy["summary_route"] = summary_route
        summary_prompt_file = str(raw_cfg.get("summary_prompt_file", default_policy["summary_prompt_file"]) or "").strip()
        policy["summary_prompt_file"] = summary_prompt_file or default_policy["summary_prompt_file"]
        summary_retry = self._safe_int(raw_cfg.get("summary_retry", default_policy["summary_retry"]), default_policy["summary_retry"])
        policy["summary_retry"] = summary_retry if summary_retry >= 1 else default_policy["summary_retry"]
        self._log_highlight(
            "[CONSULT_HISTORY_POLICY] source={} enabled={} top_k={} gate_retry={} summary_route={} summary_retry={} gate_prompt_file={} summary_prompt_file={}".format(
                policy["source"],
                bool(policy["enabled"]),
                int(policy["retrieve_top_k"]),
                int(policy["gate_retry"]),
                policy["summary_route"],
                int(policy["summary_retry"]),
                policy["gate_prompt_file"],
                policy["summary_prompt_file"],
            )
        )
        return policy

    def evaluate_session_after_chat(
        self,
        doctor: Any,
        patient: Any,
        chats: Any,
        meeting_id: str,
    ) -> Optional[Dict[str, Any]]:
        policy = self.get_session_eval_runtime_policy()
        if not bool(policy.get("enabled", False)):
            return None
        if self.get_cbt_controller_mode() == "minimal":
            return self._evaluate_minimal_session_after_chat(
                doctor=doctor,
                patient=patient,
                chats=chats,
                meeting_id=meeting_id,
                policy=policy,
            )

        pair_key = build_pair_key(getattr(doctor, "name", ""), getattr(patient, "name", ""))
        current_session = ""
        if self.session_prompt_injection:
            try:
                session_state = self.session_prompt_injection.resolve_current_session(
                    str(getattr(doctor, "name", "") or ""),
                    str(getattr(patient, "name", "") or ""),
                )
                if isinstance(session_state, dict):
                    current_session = str(session_state.get("current_session", "") or "")
            except Exception:
                current_session = ""
        history_items = self._collect_session_eval_history_for_eval(
            pair_key=pair_key,
            current_session=current_session,
            history_recent_n=int(policy.get("history_recent_n", 3) or 3),
        )
        history_text = self._format_session_eval_history_reasons_for_prompt(history_items)
        session_usage_log = self._build_session_usage_log(
            pair_key=pair_key,
            current_session=current_session,
        )
        fallback_levels = 0
        for item in history_items:
            sid = str((item or {}).get("session_id", "") or "")
            if current_session and sid and sid != current_session:
                fallback_levels += 1
        self._log_highlight(
            "[SESSION_EVAL_HISTORY_BUILD] pair_key={} current_session={} picked={} requested={} fallback_levels={} text_len={}".format(
                str(pair_key or ""),
                str(current_session or ""),
                len(history_items),
                int(policy.get("history_recent_n", 3) or 3),
                int(fallback_levels),
                len(str(history_text or "")),
            )
        )
        conversation_text = to_conversation_text(chats or [])
        session_prompt_text = self.get_session_prompt_for_judge(
            speaker=doctor,
            other=patient,
            forced=True,
        )
        self._log_highlight(
            "[SESSION_EVAL_CALL] meeting_id={} pair_key={} route={} retry={} prompt_file={}".format(
                str(meeting_id or ""),
                str(pair_key or ""),
                policy.get("route", "forced_llm"),
                int(policy.get("retry", 2) or 2),
                policy.get("prompt_file", ""),
            )
        )
        prompt_text = ""
        try:
            prompt_tpl = self._load_prompt_txt_or_raise(str(policy.get("prompt_file", "") or ""))
            prompt_text = self._render_prompt_template(
                prompt_tpl,
                {
                    "doctor": str(getattr(doctor, "name", "") or ""),
                    "patient": str(getattr(patient, "name", "") or ""),
                    "session_prompt": str(session_prompt_text or ""),
                    "session_eval_history_reasons": str(history_text or ""),
                    "session_usage_log": str(session_usage_log or ""),
                    "conversation": conversation_text,
                },
            )
            route = str(policy.get("route", "forced_llm") or "forced_llm").strip().lower()
            if route == "think_llm":
                raw = self._call_think_llm_json(
                    prompt_text=prompt_text,
                    retry=int(policy.get("retry", 2) or 2),
                    doctor_agent=doctor,
                )
            else:
                raw = self._call_forced_llm_json(
                    prompt_text=prompt_text,
                    retry=int(policy.get("retry", 2) or 2),
                    caller="session_eval",
                )
            normalized = self._normalize_session_eval_output(raw)
            self._log_highlight(
                "[SESSION_EVAL_NORM] meeting_id={} efficacy_score={} session_end={} reason_len={}".format(
                    str(meeting_id or ""),
                    normalized.get("efficacy_score", 0),
                    bool(normalized.get("session_end", False)),
                    len(str(normalized.get("reason", "") or "")),
                )
            )
            self.append_forced_prompt_trace_record(
                speaker=doctor,
                other=patient,
                role="session_eval_llm",
                prompt_text=prompt_text,
                output=normalized,
                turn_no=-1,
                meeting_id=str(meeting_id or ""),
                pair_key=str(pair_key or ""),
                meta={
                    "source": "session_eval",
                    "route": str(policy.get("route", "forced_llm") or ""),
                    "retry": int(policy.get("retry", 2) or 2),
                },
            )
            return normalized
        except Exception as exc:
            fallback = self._normalize_session_eval_output({})
            fallback["reason"] = "session_eval_error:{}".format(str(exc))
            self._log_highlight(
                "[SESSION_EVAL_NORM] meeting_id={} efficacy_score={} session_end={} reason_len={}".format(
                    str(meeting_id or ""),
                    fallback.get("efficacy_score", 0),
                    bool(fallback.get("session_end", False)),
                    len(str(fallback.get("reason", "") or "")),
                )
            )
            self.append_forced_prompt_trace_record(
                speaker=doctor,
                other=patient,
                role="session_eval_llm",
                prompt_text=prompt_text,
                output=fallback,
                turn_no=-1,
                meeting_id=str(meeting_id or ""),
                pair_key=str(pair_key or ""),
                meta={
                    "source": "session_eval_error",
                    "route": str(policy.get("route", "forced_llm") or ""),
                    "retry": int(policy.get("retry", 2) or 2),
                    "error": str(exc),
                },
            )
            return fallback

    def _evaluate_minimal_session_after_chat(
        self,
        doctor: Any,
        patient: Any,
        chats: Any,
        meeting_id: str,
        policy: Dict[str, Any],
    ) -> Dict[str, Any]:
        pair_key = build_pair_key(getattr(doctor, "name", ""), getattr(patient, "name", ""))
        progress_state = self._minimal_state_for_meeting(pair_key, meeting_id)
        current_session_prompt = self.get_session_prompt_for_judge(
            speaker=doctor,
            other=patient,
            forced=True,
        )
        prompt_text = ""
        raw: Any = {}
        error = ""
        dialogue_text, _ = tagged_dialogue(
            chats or [],
            str(getattr(patient, "name", "") or ""),
        )
        try:
            prompt_text = load_prompt_template(
                str(policy.get("prompt_file", "") or ""),
                {
                    "DOCTOR": str(getattr(doctor, "name", "") or ""),
                    "PATIENT": str(getattr(patient, "name", "") or ""),
                    "CURRENT_SESSION_PROMPT": str(current_session_prompt or ""),
                    "PRIOR_PROGRESS": compact_prior_progress_text(progress_state),
                    "DIALOGUE_HISTORY": dialogue_text,
                },
                required_placeholders=(
                    "DOCTOR",
                    "PATIENT",
                    "CURRENT_SESSION_PROMPT",
                    "PRIOR_PROGRESS",
                    "DIALOGUE_HISTORY",
                ),
            )
            if str(policy.get("route", "forced_llm") or "forced_llm").strip().lower() == "think_llm":
                raw = self._call_think_llm_json(
                    prompt_text=prompt_text,
                    retry=int(policy.get("retry", 2) or 2),
                    doctor_agent=doctor,
                    caller="cbt_minimal_session_eval",
                )
            else:
                raw = self._call_forced_llm_json(
                    prompt_text=prompt_text,
                    retry=int(policy.get("retry", 2) or 2),
                    caller="cbt_minimal_session_eval",
                )
            normalized = normalize_minimal_session_eval(raw)
        except Exception as exc:
            error = str(exc)
            normalized = normalize_minimal_session_eval({})
            normalized["reason"] = "session_eval_error:{}".format(error)

        latest_patient_state = progress_state.get("latest_patient_state", {})
        risk_level = (
            latest_patient_state.get("risk_level", "possible")
            if isinstance(latest_patient_state, dict)
            else "possible"
        )
        result = copy.deepcopy(normalized)
        adapter = generate_session_end(
            session_eval=normalized,
            chats=chats or [],
            patient_name=str(getattr(patient, "name", "") or ""),
            risk_level=str(risk_level or "possible"),
            high_risk_seen=bool(
                progress_state.get("high_risk_seen", False)
            ),
        )
        result.update(copy.deepcopy(adapter))
        prompt_adapter_output = {
            "session_end_adapter": copy.deepcopy(adapter)
        }
        self.append_forced_prompt_trace_record(
            speaker=doctor,
            other=patient,
            role="session_eval_llm",
            prompt_text=prompt_text,
            output={
                "raw_output": copy.deepcopy(raw),
                "normalized_output": copy.deepcopy(normalized),
                **prompt_adapter_output,
            },
            turn_no=-1,
            meeting_id=str(meeting_id or ""),
            pair_key=str(pair_key or ""),
            meta={
                "source": "minimal_session_eval" if not error else "minimal_session_eval_error",
                "route": str(policy.get("route", "forced_llm") or ""),
                "error": error,
                "controller_version": "",
            },
        )
        return result

    def judge_forced_dialog_before_doctor_speak(
        self,
        speaker: Any,
        other: Any,
        chats: Any,
        forced: bool,
        turn_no: int,
        session_prompt_text: str = "",
    ) -> Dict[str, Any]:
        default_output = {
            "valid": False,
            "terminate": False,
            "advice": "",
        }
        if not self.enabled:
            return default_output

        policy = self.get_dialog_judge_runtime_policy()
        if not bool(policy.get("enabled", False)):
            return default_output
        if not bool(forced):
            return default_output

        doctor, patient = self._resolve_doctor_patient_pair(speaker, other)
        if not doctor or not patient:
            return default_output
        if str(getattr(speaker, "name", "") or "") != str(getattr(doctor, "name", "") or ""):
            return default_output

        pair_key = build_pair_key(doctor.name, patient.name)
        current_session = ""
        if self.session_prompt_injection:
            try:
                session_state = self.session_prompt_injection.resolve_current_session(
                    doctor.name,
                    patient.name,
                )
                if isinstance(session_state, dict):
                    current_session = str(session_state.get("current_session", "") or "")
            except Exception:
                current_session = ""
        prev_session_eval_reason = self._get_session_eval_reason_for_judge(
            pair_key=pair_key,
            current_session=current_session,
        )
        if self.get_cbt_controller_mode() == "minimal":
            return self._judge_forced_dialog_minimal(
                doctor=doctor,
                patient=patient,
                chats=chats,
                turn_no=turn_no,
                session_prompt_text=session_prompt_text,
                pair_key=pair_key,
                current_session=current_session,
                policy=policy,
            )
        if not bool(policy.get("force_forced_llm", True)):
            self._log_highlight(
                "[DIALOG_JUDGE_FALLBACK] turn={} reason=force_forced_llm_disabled".format(
                    int(turn_no or 0)
                )
            )
            return default_output

        prompt_text = ""
        try:
            self._log_highlight(
                "[DIALOG_JUDGE_CALL] turn={} retry={} prompt_file={}".format(
                    int(turn_no or 0),
                    int(policy.get("retry", 2) or 2),
                    policy.get("prompt_file", ""),
                )
            )
            prompt_tpl = self._load_prompt_txt_or_raise(str(policy.get("prompt_file", "") or ""))
            patient_state_text = self._build_patient_dynamic_state_text(
                patient_agent=patient,
                doctor_agent=doctor,
                policy=policy,
            )
            root_strategy_context = self._build_root_complaint_strategy_context(patient)
            if root_strategy_context:
                patient_state_text = "{}\n\n{}".format(
                    str(patient_state_text or "").strip(),
                    root_strategy_context,
                ).strip()
            prompt_text = self._render_prompt_template(
                prompt_tpl,
                {
                    "patient_state": patient_state_text,
                    "conversation": to_conversation_text(chats or []),
                    "session_prompt": str(session_prompt_text or ""),
                    "prev_session_eval_reason": str(prev_session_eval_reason or ""),
                },
            )
            raw = self._call_forced_llm_json(
                prompt_text,
                retry=int(policy.get("retry", 2) or 2),
                caller="dialog_judge",
            )
            normalized = self._normalize_dialog_judge_output(raw)
            normalized["valid"] = True
            if root_strategy_context:
                normalized["_root_complaint_strategy_context"] = root_strategy_context
            self._log_highlight(
                "[DIALOG_JUDGE_NORM] turn={} terminate={}".format(
                    int(turn_no or 0),
                    bool(normalized.get("terminate", False)),
                )
            )
            self.append_forced_prompt_trace_record(
                speaker=speaker,
                other=other,
                role="judge_llm",
                prompt_text=prompt_text,
                output=normalized,
                turn_no=int(turn_no or -1),
                meta={
                    "source": "dialog_judge",
                    "retry": int(policy.get("retry", 2) or 2),
                },
            )
            return normalized
        except Exception as exc:
            self._log_highlight(
                "[DIALOG_JUDGE_FALLBACK] turn={} reason={}".format(
                    int(turn_no or 0),
                    str(exc),
                )
            )
            self.append_forced_prompt_trace_record(
                speaker=speaker,
                other=other,
                role="judge_llm",
                prompt_text=prompt_text,
                output={
                    "valid": False,
                    "terminate": False,
                    "advice": "",
                    "error": str(exc),
                },
                turn_no=int(turn_no or -1),
                meta={
                    "source": "dialog_judge_error",
                    "retry": int(policy.get("retry", 2) or 2),
                },
            )
            return default_output

    def _judge_forced_dialog_minimal(
        self,
        doctor: Any,
        patient: Any,
        chats: Any,
        turn_no: int,
        session_prompt_text: str,
        pair_key: str,
        current_session: str,
        policy: Dict[str, Any],
    ) -> Dict[str, Any]:
        meeting_id, _, _ = self._resolve_trace_session_context(doctor, patient)
        progressive_d_v3 = self._is_progressive_d_v3_configured()
        progress_state = (
            self._progressive_d_state_for_meeting(pair_key, meeting_id)
            if progressive_d_v3
            else self._minimal_state_for_meeting(pair_key, meeting_id)
        )
        patient_state = progress_state.get("latest_patient_state", {})
        if not isinstance(patient_state, dict) or not patient_state:
            patient_state = copy.deepcopy(
                PROGRESSIVE_D_PATIENT_STATE_DEFAULT
                if progressive_d_v3
                else PATIENT_STATE_DEFAULT
            )
        prompt_text = ""
        raw: Any = {}
        error = ""
        side_step_active = False
        route_session = current_session
        if bool(policy.get("subgoal_progress_enabled", False)):
            try:
                subgoal_state, _ = self._progressive_d_progress_state_for_pair(
                    pair_key,
                    current_session,
                )
                side_step = subgoal_state.get("side_step", {})
                side_step_active = bool(
                    isinstance(side_step, dict)
                    and side_step.get("active", False)
                )
                soft_step_back = subgoal_state.get(
                    "soft_step_back",
                    {},
                )
                if (
                    isinstance(soft_step_back, dict)
                    and soft_step_back.get("active", False)
                    and not side_step_active
                ):
                    target_id = str(
                        soft_step_back.get("target_subgoal_id", "") or ""
                    )
                    stage_map = self._minimal_stage_subgoals(
                        str(
                            self.get_progressive_d_runtime_policy().get(
                                "stage_subgoals_file",
                                "",
                            )
                            or ""
                        )
                    )
                    for session_id, stage_item in stage_map.items():
                        configured = (
                            stage_item.get("subgoals", [])
                            if isinstance(stage_item, dict)
                            else []
                        )
                        if any(
                            isinstance(item, dict)
                            and str(item.get("id", "") or "") == target_id
                            for item in configured
                        ):
                            route_session = str(session_id or current_session)
                            break
            except Exception as exc:
                error = str(exc)
        try:
            routed_patient_state = copy.deepcopy(patient_state)
            monitored_possible = False
            if progressive_d_v3:
                risk_check = self._progressive_d_risk_check_state(pair_key)
                if bool(progress_state.get("high_risk_seen", False)):
                    routed_patient_state["risk_level"] = "high"
                else:
                    monitored_possible = bool(
                        str(
                            patient_state.get("risk_level", "") or ""
                        ) == "possible"
                        and risk_check.get("checked", False)
                        and not risk_check.get("check_pending", False)
                    )
                    if monitored_possible:
                        routed_patient_state["risk_level"] = "none"
            route = route_strategies(
                current_session=route_session,
                patient_state=routed_patient_state,
                strategy_map=self._minimal_strategy_map(),
                side_step_active=side_step_active,
            )
            if monitored_possible:
                route["applied_rule"] = "risk_possible_monitored"
                route["explanation"] = (
                    "本次可能风险已经完成一次安全核对，继续当前 CBT 主线并保持监测。"
                )
            if route_session != current_session:
                route["applied_rule"] = "soft_step_back_target"
                route["explanation"] = (
                    "soft step-back temporarily reuses configured predecessor "
                    f"strategies from {route_session}"
                )
        except Exception as exc:
            error = str(exc)
            route = {
                "allowed_primary_strategies": ["support_stabilize"],
                "allowed_micro_skills": ["validation", "focused_question"],
                "applied_rule": "config_error_fallback",
                "explanation": error,
            }
        latest_patient_reply = self._extract_latest_patient_utterance(
            chats or [], patient.name
        )
        root_strategy_context = self._build_root_complaint_strategy_context(patient)
        subgoal_progress_text = ""
        if bool(policy.get("subgoal_progress_enabled", False)):
            try:
                subgoal_progress_text = self._compact_subgoal_progress_for_judge(
                    pair_key,
                    current_session,
                )
            except Exception as exc:
                error = "{}; {}".format(error, str(exc)).strip("; ")
                subgoal_progress_text = ""
        try:
            required_placeholders = [
                "CURRENT_SESSION_PROMPT",
                "DIALOGUE_HISTORY",
                "PATIENT_STATE",
                "CANDIDATE_STRATEGIES",
            ]
            if progressive_d_v3:
                required_placeholders.append("TURN_NO")
            else:
                required_placeholders.extend(
                    ["LATEST_PATIENT_REPLY", "PRIOR_PROGRESS"]
                )
            if bool(policy.get("subgoal_progress_enabled", False)):
                required_placeholders.append("SUBGOAL_PROGRESS")
            if progressive_d_v3:
                required_placeholders.append("PATIENT_INTERNAL_STATE")
            prompt_text = load_prompt_template(
                str(policy.get("prompt_file", "") or ""),
                {
                    "CURRENT_SESSION_PROMPT": str(session_prompt_text or ""),
                    "SUBGOAL_PROGRESS": str(subgoal_progress_text or ""),
                    "LATEST_PATIENT_REPLY": str(latest_patient_reply or ""),
                    "DIALOGUE_HISTORY": (
                        to_conversation_text(chats or [])
                        if progressive_d_v3
                        else recent_dialogue(chats or [], 8)
                    ),
                    "PRIOR_PROGRESS": (
                        ""
                        if progressive_d_v3
                        else self._compact_prior_control_progress_for_judge(
                            pair_key,
                            current_session,
                            progress_state,
                        )
                    ),
                    "PATIENT_STATE": "{}{}".format(
                        (
                            compact_progressive_d_patient_state_text(patient_state)
                            if progressive_d_v3
                            else compact_patient_state_text(patient_state)
                        ),
                        (
                            "\n" + root_strategy_context
                            if root_strategy_context and not progressive_d_v3
                            else ""
                        ),
                    ),
                    "PATIENT_INTERNAL_STATE": (
                        "{}{}".format(
                            self._compact_progressive_d_patient_internal_state(patient),
                            "\n" + root_strategy_context if root_strategy_context else "",
                        )
                        if progressive_d_v3
                        else ""
                    ),
                    "CANDIDATE_STRATEGIES": compact_route_text(
                        route,
                        self._minimal_term_glossary(),
                    ),
                    "TURN_NO": str(int(turn_no or 1)),
                },
                required_placeholders=required_placeholders,
            )
            raw = self._call_forced_llm_json(
                prompt_text,
                retry=int(policy.get("retry", 2) or 2),
                caller="cbt_minimal_dialog_judge",
            )
            normalized = normalize_minimal_judge_output(raw, route)
        except Exception as exc:
            error = "{}; {}".format(error, str(exc)).strip("; ")
            normalized = safe_judge_fallback(
                route.get("allowed_primary_strategies", []),
                route.get("allowed_micro_skills", []),
            )
        output = copy.deepcopy(normalized)
        output["valid"] = True
        if root_strategy_context:
            output["_root_complaint_strategy_context"] = root_strategy_context
        if progressive_d_v3:
            output["_patient_internal_state"] = (
                self._compact_progressive_d_patient_internal_state(patient)
            )
            if str(normalized.get("primary_strategy", "") or "") == "safety_check":
                self._mark_progressive_d_safety_check_requested(pair_key)
        self.append_forced_prompt_trace_record(
            speaker=doctor,
            other=patient,
            role="judge_llm",
            prompt_text=prompt_text,
            output={
                "raw_output": copy.deepcopy(raw),
                "normalized_output": copy.deepcopy(normalized),
                "router": copy.deepcopy(route),
            },
            turn_no=int(turn_no or -1),
            meta={
                "source": "minimal_dialog_judge" if not error else "minimal_dialog_judge_fallback",
                "error": error,
            },
        )
        return output

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

    def _get_root_complaint_anchor(self, patient_agent: Any) -> str:
        """Read the checkpointed case anchor without consulting current_stage."""
        engine = getattr(patient_agent, "depression_dynamic", None)
        if engine is None or not callable(getattr(engine, "get_current_state_info", None)):
            return ""
        try:
            info = engine.get_current_state_info()
        except Exception:
            return ""
        if not isinstance(info, dict):
            return ""
        return str(info.get("root_complaint_anchor", "") or "").strip()

    def _build_root_complaint_strategy_context(self, patient_agent: Any) -> str:
        anchor = self._get_root_complaint_anchor(patient_agent)
        if not anchor:
            return ""
        return (
            "<长期病例背景>\n"
            "核心主诉锚点：{}\n"
            "策略边界：允许近期事件和当前主诉按实际进展自然演变，不因锚点阻止新主题、"
            "阶段推进、好转或反复；仅在长期治疗方向明显偏离时用它维持可解释联系，"
            "不要求每轮回扣或提及。\n"
            "</长期病例背景>"
        ).format(anchor)

    def _get_patient_dynamic_state_raw_text(self, patient_agent: Any) -> str:
        intervention_state = getattr(patient_agent, "status", {}).get("intervention", {})
        cached = intervention_state.get("last_generate_chat_prompt", {}) if isinstance(intervention_state, dict) else {}
        prompt_text = str((cached or {}).get("prompt_text", "") or "")
        patient_name = str(getattr(patient_agent, "name", "") or "")
        if not prompt_text:
            self._log_highlight(
                "[DIALOG_JUDGE_PATIENT_STATE_EMPTY] level=warning patient={} reason=cache_missing".format(
                    patient_name
                )
            )
            return ""

        legacy_marker = "=== 当前任务指令层 ==="
        legacy_idx = prompt_text.find(legacy_marker)
        if legacy_idx > 0:
            return str(prompt_text[:legacy_idx].strip() or "")

        start_idx = -1
        for marker in (
            "=== 基础人格层 ===",
            "=== 当前主诉节点层 ===",
            "=== 会话上下文层 ===",
        ):
            idx = prompt_text.find(marker)
            if idx >= 0 and (start_idx < 0 or idx < start_idx):
                start_idx = idx

        if start_idx < 0:
            self._log_highlight(
                "[DIALOG_JUDGE_PATIENT_STATE_EMPTY] level=warning patient={} reason=state_boundary_not_found".format(
                    patient_name
                )
            )
            return ""

        end_idx = -1
        end_markers = []
        if patient_name:
            end_markers.append("以下是 {} 的记忆：".format(patient_name))
        end_markers.extend([
            "的记忆：",
            "当前位置：",
            "背景：",
            "<对话记录>",
        ])
        for marker in end_markers:
            idx = prompt_text.find(marker, start_idx)
            if idx > start_idx and (end_idx < 0 or idx < end_idx):
                end_idx = idx

        if end_idx > start_idx:
            return str(prompt_text[start_idx:end_idx].strip() or "")
        return str(prompt_text[start_idx:].strip() or "")

    def _get_progressive_d_patient_state_text(
        self,
        patient_agent: Any,
    ) -> str:
        """Select only the three patient-internal sections used by D."""

        raw = self._get_patient_dynamic_state_raw_text(patient_agent)
        if not raw:
            return ""
        wanted = {
            "=== 当前主诉节点层 ===",
            "=== 会话上下文层 ===",
            "=== 瞬时情绪与表达指导层 ===",
        }
        lines: List[str] = []
        active = False
        for line in raw.splitlines():
            stripped = line.strip()
            if stripped.startswith("===") and stripped.endswith("==="):
                active = stripped in wanted
            if active:
                lines.append(line)
        return "\n".join(lines).strip()

    def _compact_progressive_d_patient_internal_state(
        self,
        patient_agent: Any,
    ) -> str:
        """Project auditable labelled facts without another LLM call."""

        raw = self._get_progressive_d_patient_state_text(patient_agent)
        if not raw:
            return "患者内部状态暂不可用；请只根据当前固定会谈提纲与已有对话推进。"
        labels = (
            "节点概述：",
            "核心信念：",
            "当前最容易围绕这些问题组织表达：",
            "这轮说话姿态：",
            "当前瞬时情绪标签：",
            "情绪强度：",
            "暴露程度：",
            "防御水平：",
        )
        selected: List[str] = []
        for line in raw.splitlines():
            text = line.strip().lstrip("- ").strip()
            if any(text.startswith(label) for label in labels):
                selected.append(text)
        if not selected:
            return "患者内部状态已有记录，但缺少可压缩的结构化标签。"
        return "；".join(selected)

    def _strip_text_code_fence(self, text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        if raw.startswith("```") and raw.endswith("```"):
            parts = raw.split("```")
            if len(parts) >= 3:
                body = str(parts[1] or "").strip()
                if "\n" in body:
                    first_line, rest = body.split("\n", 1)
                    lang = first_line.strip()
                    if lang and all(ch.isalnum() or ch in ("_", "-", "+") for ch in lang):
                        body = rest.strip()
                return str(body.strip() or "")
        return raw

    def _build_patient_dynamic_state_text(
        self,
        patient_agent: Any,
        doctor_agent: Any,
        policy: Optional[Dict[str, Any]] = None,
    ) -> str:
        raw_state = self._get_patient_dynamic_state_raw_text(patient_agent)
        if not raw_state:
            return ""
        # This summarizer is also reused by non-CBT environment tasks. Keep the
        # root anchor out of the shared result; judge call sites add it back
        # explicitly as strategy context.
        raw_state = self._strip_root_complaint_anchor_layer(raw_state)

        runtime_policy = policy if isinstance(policy, dict) else self.get_dialog_judge_runtime_policy()
        prompt_file = str(runtime_policy.get("patient_state_prompt_file", "") or "").strip()
        retry = self._safe_int(runtime_policy.get("patient_state_summary_retry", 2), 2)
        patient_name = str(getattr(patient_agent, "name", "") or "")
        doctor_name = str(getattr(doctor_agent, "name", "") or "")

        try:
            prompt_tpl = self._load_prompt_txt_or_raise(prompt_file)
            prompt_text = self._render_prompt_template(
                prompt_tpl,
                {
                    "patient": patient_name,
                    "doctor": doctor_name,
                    "patient_state": raw_state,
                },
            )
            summary = self._call_think_llm_text(
                prompt_text=prompt_text,
                retry=retry,
                doctor_agent=doctor_agent,
                caller="dialog_judge_patient_state_summary",
            )
            cleaned = self._strip_text_code_fence(summary)
            if cleaned:
                self._log_highlight(
                    "[DIALOG_JUDGE_PATIENT_STATE_SUMMARY] patient={} doctor={} chars={} route=think_llm".format(
                        patient_name,
                        doctor_name,
                        len(cleaned),
                    )
                )
                return cleaned
            self._log_highlight(
                "[DIALOG_JUDGE_PATIENT_STATE_SUMMARY] patient={} doctor={} chars=0 route=fallback_raw reason=empty_summary".format(
                    patient_name,
                    doctor_name,
                )
            )
        except Exception as exc:
            self._log_highlight(
                "[DIALOG_JUDGE_PATIENT_STATE_SUMMARY] patient={} doctor={} route=fallback_raw reason={}".format(
                    patient_name,
                    doctor_name,
                    str(exc),
                )
            )
        return raw_state

    @staticmethod
    def _strip_root_complaint_anchor_layer(raw_state: str) -> str:
        text = str(raw_state or "")
        start_marker = "=== 长期病例背景 ==="
        end_marker = "=== 当前主诉节点层 ==="
        start_idx = text.find(start_marker)
        if start_idx < 0:
            return text
        end_idx = text.find(end_marker, start_idx + len(start_marker))
        if end_idx < 0:
            return text[:start_idx].rstrip()
        return "{}{}".format(text[:start_idx], text[end_idx:]).strip()

    def _normalize_dialog_judge_output(self, payload: Any) -> Dict[str, Any]:
        raw = payload if isinstance(payload, dict) else {}
        normalized = {
            "terminate": False,
            "advice": "",
        }

        def _parse_bool_strict(value: Any) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered == "true":
                    return True
                if lowered == "false":
                    return False
            return False

        normalized["terminate"] = _parse_bool_strict(raw.get("terminate"))
        advice = raw.get("advice")
        normalized["advice"] = advice if isinstance(advice, str) else ""
        return normalized

    def _resolve_think_llm_runtime(self, doctor_agent: Any) -> Any:
        llm_cfg = {}
        if doctor_agent is not None:
            think_cfg = getattr(doctor_agent, "think_config", {}) or {}
            if isinstance(think_cfg, dict):
                llm_cfg = think_cfg.get("llm", {}) or {}
        if not isinstance(llm_cfg, dict) or not llm_cfg:
            raise ConsultRecordError(reason="think_llm_config_missing", retryable=False)

        cache_key = "{}/{}/{}".format(
            llm_cfg.get("provider", ""),
            llm_cfg.get("model", ""),
            llm_cfg.get("base_url", ""),
        )
        if self._think_llm is None or self._think_llm_key != cache_key:
            self._think_llm = create_llm_model(llm_cfg)
            self._think_llm_key = cache_key
        return llm_cfg, self._think_llm

    def _call_think_llm_text(self, prompt_text: str, retry: int, doctor_agent: Any, caller: str) -> str:
        llm_cfg, llm = self._resolve_think_llm_runtime(doctor_agent)
        retry_count = max(1, int(retry or llm_cfg.get("retry", 2) or 2))
        payload = llm.completion(
            prompt_text,
            retry=retry_count,
            failsafe="",
            caller=str(caller or "think_llm_text"),
        )
        text = str(payload or "").strip()
        if not text:
            raise ConsultRecordError(reason="think_llm_empty_text", retryable=False)
        return text

    def _call_think_llm_json(
        self,
        prompt_text: str,
        retry: int,
        doctor_agent: Any,
        caller: str = "session_eval_think_llm",
    ) -> Dict[str, Any]:
        _, llm = self._resolve_think_llm_runtime(doctor_agent)
        retry_count = max(1, int(retry or 2))
        payload = llm.completion(
            prompt_text,
            retry=retry_count,
            callback=self._json_loads_loose,
            failsafe=None,
            caller=str(caller or "session_eval_think_llm"),
        )
        if not isinstance(payload, dict):
            raise ConsultRecordError(reason="json_parse_failed", retryable=False)
        return payload

    def _normalize_session_eval_output(self, payload: Any) -> Dict[str, Any]:
        raw = payload if isinstance(payload, dict) else {}
        normalized = {
            "efficacy_score": 0,
            "session_end": False,
            "reason": "",
        }

        def _parse_bool_strict(value: Any) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered == "true":
                    return True
                if lowered == "false":
                    return False
            return False

        score = raw.get("efficacy_score")
        if isinstance(score, bool):
            normalized["efficacy_score"] = 0
        elif isinstance(score, (int, float)):
            normalized["efficacy_score"] = score
        elif isinstance(score, str):
            try:
                normalized["efficacy_score"] = float(score.strip())
            except Exception:
                normalized["efficacy_score"] = 0
        normalized["session_end"] = _parse_bool_strict(raw.get("session_end"))
        reason = raw.get("reason")
        normalized["reason"] = reason if isinstance(reason, str) else ""
        return normalized

    def _store_session_eval_reason(
        self,
        pair_key: str,
        current_session: str,
        meeting_id: str,
        reason: str,
    ) -> None:
        self._ensure_session_eval_state_schema()
        state = self.state.setdefault("session_eval_state", {})
        latest = state.setdefault("latest_reason_by_pair", {})
        history_by_pair = state.setdefault("history_by_pair", {})

        pair_key_text = str(pair_key or "")
        current_session_text = str(current_session or "")
        reason_text = str(reason or "")
        now_text = self._fmt_dt(self._now())

        latest[pair_key_text] = {
            "reason": reason_text,
            "current_session": current_session_text,
            "updated_at": now_text,
        }

        history = history_by_pair.setdefault(pair_key_text, [])
        if not isinstance(history, list):
            history = []
            history_by_pair[pair_key_text] = history

        round_no = 0
        for item in history:
            if not isinstance(item, dict):
                continue
            if str(item.get("session_id", "") or "") == current_session_text:
                round_no += 1

        history.append(
            {
                "session_id": current_session_text,
                "session_round": int(round_no + 1),
                "meeting_id": str(meeting_id or ""),
                "reason": reason_text,
                "updated_at": now_text,
            }
        )

        self._log_highlight(
            "[SESSION_EVAL_STORE] pair_key={} current_session={} session_round={} reason_len={}".format(
                pair_key_text,
                current_session_text,
                int(round_no + 1),
                len(reason_text),
            )
        )

    def _store_minimal_session_eval_result(
        self,
        pair_key: str,
        meeting_id: str,
        payload: Dict[str, Any],
    ) -> None:
        progress_state = self._minimal_progress_state(pair_key)
        progress_state["last_progress"] = str(payload.get("progress", "none") or "none")
        progress_state["last_evidence_turn_id"] = str(
            payload.get("evidence_turn_id", "") or ""
        )
        progress_state["last_blocking_factor"] = str(
            payload.get("blocking_factor", "") or ""
        )
        progress_state["last_reason"] = str(payload.get("reason", "") or "")
        progress_state["meeting_id"] = str(meeting_id or "")

        state = self.state.setdefault("session_eval_state", {})
        latest = state.setdefault("latest_reason_by_pair", {}).get(str(pair_key or ""))
        if isinstance(latest, dict):
            latest["session_end"] = bool(payload.get("session_end", False))
            latest["progress"] = progress_state["last_progress"]
            latest["evidence_turn_id"] = progress_state[
                "last_evidence_turn_id"
            ]
            latest["blocking_factor"] = progress_state["last_blocking_factor"]
            if payload.get("cbt_controller_version"):
                latest["cbt_controller_version"] = str(
                    payload.get("cbt_controller_version", "") or ""
                )
            if isinstance(payload.get("stage_transition_adapter"), dict):
                latest["stage_transition_adapter"] = copy.deepcopy(
                    payload.get("stage_transition_adapter", {})
                )
        history = state.setdefault("history_by_pair", {}).get(str(pair_key or ""), [])
        if isinstance(history, list) and history:
            last_item = history[-1]
            if (
                isinstance(last_item, dict)
                and str(last_item.get("meeting_id", "") or "") == str(meeting_id or "")
            ):
                last_item.update(
                    {
                        "session_end": bool(payload.get("session_end", False)),
                        "progress": progress_state["last_progress"],
                        "evidence_turn_id": progress_state[
                            "last_evidence_turn_id"
                        ],
                        "blocking_factor": progress_state["last_blocking_factor"],
                    }
                )
                if payload.get("cbt_controller_version"):
                    last_item["cbt_controller_version"] = str(
                        payload.get("cbt_controller_version", "") or ""
                    )
                if isinstance(
                    payload.get("stage_transition_adapter"),
                    dict,
                ):
                    last_item["stage_transition_adapter"] = copy.deepcopy(
                        payload.get("stage_transition_adapter", {})
                    )


    def _get_session_eval_reason_for_judge(self, pair_key: str, current_session: str) -> str:
        self._ensure_session_eval_state_schema()
        current_session_text = str(current_session or "").strip() or "unknown"

        def _format_fresh_stage_state(first_dialog: bool) -> str:
            status = "当前阶段第一次对话" if first_dialog else "当前阶段刚开始"
            return (
                "当前阶段：{}\n"
                "阶段状态：{}；当前阶段已完成步骤：无。\n"
                "说明：上一阶段若已有评估结论，仅表示上一阶段已完成或已退出，"
                "不可视为当前阶段未完成步骤；请以本阶段会话Prompt作为当前阶段目标。"
            ).format(current_session_text, status)

        state = self.state.setdefault("session_eval_state", {})
        latest = state.setdefault("latest_reason_by_pair", {})
        item = latest.get(str(pair_key or ""))
        if not isinstance(item, dict):
            return _format_fresh_stage_state(first_dialog=True)

        item_session = str(item.get("current_session", "") or "").strip()
        reason_text = str(item.get("reason", "") or "").strip()
        if item_session and item_session == current_session_text and reason_text:
            return (
                "当前阶段：{}\n"
                "阶段状态：当前阶段延续中。\n"
                "当前阶段历史评估结论：\n"
                "{}"
            ).format(current_session_text, reason_text)

        return _format_fresh_stage_state(first_dialog=False)

    def _get_session_prompt_order(self) -> List[str]:
        intervention_cfg = self.config.get("intervention", {}) or {}
        injection_cfg = intervention_cfg.get("session_prompt_injection", {}) or {}
        raw_order = injection_cfg.get("order", []) or []
        if not isinstance(raw_order, list):
            return []
        result = []
        seen = set()
        for item in raw_order:
            session_id = str(item or "").strip()
            if (not session_id) or (session_id in seen):
                continue
            seen.add(session_id)
            result.append(session_id)
        return result

    def _collect_session_eval_history_for_eval(
        self,
        pair_key: str,
        current_session: str,
        history_recent_n: int,
    ) -> List[Dict[str, Any]]:
        self._ensure_session_eval_state_schema()
        if int(history_recent_n or 0) <= 0:
            return []
        state = self.state.setdefault("session_eval_state", {})
        history_by_pair = state.setdefault("history_by_pair", {})
        items = history_by_pair.get(str(pair_key or ""))
        if not isinstance(items, list) or (not items):
            return []

        normalized = []
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            reason_text = str(item.get("reason", "") or "").strip()
            if not reason_text:
                continue
            normalized.append(
                {
                    "session_id": str(item.get("session_id", "") or ""),
                    "session_round": self._safe_int(item.get("session_round", 0), 0),
                    "meeting_id": str(item.get("meeting_id", "") or ""),
                    "reason": reason_text,
                    "updated_at": str(item.get("updated_at", "") or ""),
                    "_idx": idx,
                }
            )
        if not normalized:
            return []

        remaining = int(history_recent_n or 0)
        picked = []
        picked_idx = set()
        current_session_text = str(current_session or "")

        def _pick_session(session_id: str):
            nonlocal remaining
            if remaining <= 0:
                return
            session_id_text = str(session_id or "")
            if not session_id_text:
                return
            for entry in reversed(normalized):
                if remaining <= 0:
                    break
                entry_idx = int(entry.get("_idx", -1))
                if entry_idx in picked_idx:
                    continue
                if str(entry.get("session_id", "") or "") != session_id_text:
                    continue
                picked_idx.add(entry_idx)
                picked.append(entry)
                remaining -= 1

        if current_session_text:
            _pick_session(current_session_text)

        if remaining > 0:
            order = self._get_session_prompt_order()
            if current_session_text and current_session_text in order:
                current_index = order.index(current_session_text)
                for idx in range(current_index - 1, -1, -1):
                    if remaining <= 0:
                        break
                    _pick_session(order[idx])
            if remaining > 0:
                for entry in reversed(normalized):
                    if remaining <= 0:
                        break
                    entry_idx = int(entry.get("_idx", -1))
                    if entry_idx in picked_idx:
                        continue
                    picked_idx.add(entry_idx)
                    picked.append(entry)
                    remaining -= 1

        picked.sort(key=lambda x: int(x.get("_idx", -1)))
        output = []
        for entry in picked:
            session_id = str(entry.get("session_id", "") or "")
            tag = "当前阶段" if (current_session_text and session_id == current_session_text) else "历史阶段，仅供背景"
            output.append(
                {
                    "session_id": session_id,
                    "session_round": self._safe_int(entry.get("session_round", 0), 0),
                    "meeting_id": str(entry.get("meeting_id", "") or ""),
                    "reason": str(entry.get("reason", "") or ""),
                    "updated_at": str(entry.get("updated_at", "") or ""),
                    "tag": tag,
                }
            )
        return output

    def _format_session_eval_history_reasons_for_prompt(self, items: List[Dict[str, Any]]) -> str:
        if not isinstance(items, list) or (not items):
            return "- （无历史评估结论）"
        lines = []
        for item in items:
            if not isinstance(item, dict):
                continue
            reason_text = str(item.get("reason", "") or "").strip()
            if not reason_text:
                continue
            session_id = str(item.get("session_id", "") or "").strip() or "unknown"
            round_no = self._safe_int(item.get("session_round", 0), 0)
            round_text = str(round_no) if round_no > 0 else "?"
            meeting_id = str(item.get("meeting_id", "") or "").strip()
            tag = str(item.get("tag", "") or "").strip() or "历史阶段，仅供背景"
            line = "- [阶段={}][第{}次会后评估]".format(session_id, round_text)
            if meeting_id:
                line += "[meeting_id={}]".format(meeting_id)
            line += "[标记={}] {}".format(tag, reason_text)
            lines.append(line)
        if not lines:
            return "- （无历史评估结论）"
        return "\n".join(lines)

    def _build_session_usage_log(
        self,
        pair_key: str,
        current_session: str,
    ) -> str:
        self._ensure_session_eval_state_schema()
        state = self.state.setdefault("session_eval_state", {})
        history_by_pair = state.setdefault("history_by_pair", {})
        items = history_by_pair.get(str(pair_key or ""))
        if not isinstance(items, list):
            items = []

        current_session_text = str(current_session or "").strip()
        order = self._get_session_prompt_order()
        current_index = order.index(current_session_text) if (current_session_text and current_session_text in order) else -1
        prior_sessions = []
        if current_index > 0:
            prior_sessions = order[:current_index]

        current_entries = []
        last_other_session = ""
        for item in items:
            if not isinstance(item, dict):
                continue
            session_id = str(item.get("session_id", "") or "").strip()
            if session_id == current_session_text and current_session_text:
                current_entries.append(item)
            elif session_id:
                last_other_session = session_id

        reason_texts = []
        for entry in current_entries[-3:]:
            reason = str(entry.get("reason", "") or "").strip()
            if reason:
                reason_texts.append(reason)

        normalized_text = " ".join(reason_texts)
        low_yield_hits = 0
        for keyword in ["暂不能结束当前session", "未完成", "推进", "答不上来", "不知道", "想不出来", "收益较低", "低收益", "卡住"]:
            if keyword and keyword in normalized_text:
                low_yield_hits += 1
        repeated_low_yield = bool(len(current_entries) >= 3 and low_yield_hits >= 2)

        lines = [
            "- 当前session: {}".format(current_session_text or "<unknown>"),
            "- 当前session累计会后评估次数: {}".format(len(current_entries)),
            "- 当前session在固定顺序中的位置: {} / {}".format((current_index + 1) if current_index >= 0 else "?", len(order)),
            "- 已经历过的更早session: {}".format(" -> ".join(prior_sessions) if prior_sessions else "（无）"),
            "- 最近一次非当前session: {}".format(last_other_session or "（无）"),
            "- 是否疑似连续低收益停留: {}".format("是" if repeated_low_yield else "否"),
        ]

        if reason_texts:
            lines.append("- 当前session最近几次评估阻塞摘要:")
            for idx, reason in enumerate(reason_texts, start=max(1, len(current_entries) - len(reason_texts) + 1)):
                lines.append("  - 第{}次: {}".format(idx, reason))
        else:
            lines.append("- 当前session最近几次评估阻塞摘要: （无）")

        return "\n".join(lines)

    def append_dialog_judge_trace_record(
        self,
        speaker: Any,
        other: Any,
        chats: Any,
        doctor_turn_judge_cache: Dict[str, Any],
        doctor_utterance: str,
    ) -> None:
        self._ensure_dialog_judge_trace_state_schema()
        doctor, patient = self._resolve_doctor_patient_pair(speaker, other)
        if not doctor or not patient:
            return
        meeting_id, pair_key, step_time = self._resolve_trace_session_context(doctor, patient)
        state = self.state.setdefault("dialog_judge_trace_state", {})
        sessions = state.setdefault("sessions", [])
        session_item = self._ensure_trace_session(sessions, meeting_id, pair_key, step_time)
        turns = session_item.setdefault("turns", [])
        patient_text = self._extract_latest_patient_utterance(chats or [], patient.name)
        try:
            complaint_graph_trace = capture_runtime_chat_trace(
                patient,
                str(patient_text or ""),
            )
        except Exception as exc:
            complaint_graph_trace = unavailable_trace(
                "runtime_capture_error:{}".format(str(exc)),
                {"mode": "runtime_capture"},
            )
        turn_item = {
            "patient": str(patient_text or ""),
            "judge": {
                "doctor_turn_judge_cache": copy.deepcopy(
                    doctor_turn_judge_cache if isinstance(doctor_turn_judge_cache, dict) else {}
                )
            },
            "doctor": str(doctor_utterance or ""),
            "complaint_graph": complaint_graph_trace,
        }
        turns.append(turn_item)
        update_session_chat_summary(
            session_item,
            complaint_graph_trace,
        )
        self._log_highlight(
            "[DIALOG_JUDGE_COMPLAINT_GRAPH] meeting_id={} turn_no={} status={} action={} changed={} reason={}".format(
                str(meeting_id or ""),
                (doctor_turn_judge_cache or {}).get("turn_no", -1),
                str(complaint_graph_trace.get("status", "") or ""),
                str(
                    (complaint_graph_trace.get("transition", {}) or {}).get(
                        "action",
                        "",
                    )
                    or ""
                ),
                bool(complaint_graph_trace.get("changed", False)),
                str(complaint_graph_trace.get("reason", "") or ""),
            )
        )
        self._log_highlight(
            "[DIALOG_JUDGE_TRACE_APPEND] meeting_id={} pair_key={} valid={} terminate={} advice_len={} patient_len={} doctor_len={}".format(
                str(meeting_id or ""),
                str(pair_key or ""),
                bool((doctor_turn_judge_cache or {}).get("valid", False)),
                bool((doctor_turn_judge_cache or {}).get("terminate", False)),
                len(str((doctor_turn_judge_cache or {}).get("advice", "") or "")),
                len(str(patient_text or "")),
                len(str(doctor_utterance or "")),
            )
        )

    def append_dialog_judge_trace_reflection(
        self,
        doctor: Any,
        patient: Any,
        meeting_id: str,
    ) -> None:
        self._ensure_dialog_judge_trace_state_schema()
        pair_key = build_pair_key(
            str(getattr(doctor, "name", "") or ""),
            str(getattr(patient, "name", "") or ""),
        )
        try:
            complaint_graph_trace = capture_runtime_reflection_trace(
                patient,
                str(meeting_id or ""),
            )
        except Exception as exc:
            complaint_graph_trace = unavailable_trace(
                "runtime_capture_error:{}".format(str(exc)),
                {"mode": "runtime_capture"},
            )
        state = self.state.setdefault("dialog_judge_trace_state", {})
        sessions = state.setdefault("sessions", [])
        session_item = self._ensure_trace_session(
            sessions,
            str(meeting_id or ""),
            pair_key,
            self._resolve_trace_step_time(),
        )
        update_session_reflection_summary(
            session_item,
            complaint_graph_trace,
        )
        self._log_highlight(
            "[DIALOG_JUDGE_COMPLAINT_GRAPH_REFLECTION] meeting_id={} status={} action={} changed={} reason={}".format(
                str(meeting_id or ""),
                str(complaint_graph_trace.get("status", "") or ""),
                str(
                    (complaint_graph_trace.get("transition", {}) or {}).get(
                        "action",
                        "",
                    )
                    or ""
                ),
                bool(complaint_graph_trace.get("changed", False)),
                str(complaint_graph_trace.get("reason", "") or ""),
            )
        )

    def append_dialog_judge_trace_eval(
        self,
        meeting_id: str,
        pair_key: str,
        step_time: str,
        normalized_eval_payload: Dict[str, Any],
    ) -> None:
        self._ensure_dialog_judge_trace_state_schema()
        state = self.state.setdefault("dialog_judge_trace_state", {})
        sessions = state.setdefault("sessions", [])
        session_item = self._ensure_trace_session(
            sessions,
            str(meeting_id or ""),
            str(pair_key or ""),
            str(step_time or self._resolve_trace_step_time()),
        )
        session_item["session_eval"] = copy.deepcopy(
            normalized_eval_payload if isinstance(normalized_eval_payload, dict) else {}
        )
        self._log_highlight(
            "[DIALOG_JUDGE_TRACE_EVAL_APPEND] meeting_id={} pair_key={} session_end={} efficacy_score={} reason_len={}".format(
                str(meeting_id or ""),
                str(pair_key or ""),
                bool((normalized_eval_payload or {}).get("session_end", False)),
                (normalized_eval_payload or {}).get("efficacy_score", 0),
                len(str((normalized_eval_payload or {}).get("reason", "") or "")),
            )
        )

    def export_dialog_judge_trace_payload(self) -> Dict[str, Any]:
        self._ensure_dialog_judge_trace_state_schema()
        state = self.state.setdefault("dialog_judge_trace_state", {})
        sessions = state.get("sessions", [])
        if not isinstance(sessions, list):
            sessions = []
        return {"sessions": copy.deepcopy(sessions)}

    def append_forced_prompt_trace_record(
        self,
        speaker: Any,
        other: Any,
        role: str,
        prompt_text: str,
        output: Any,
        turn_no: int = -1,
        meta: Optional[Dict[str, Any]] = None,
        meeting_id: str = "",
        pair_key: str = "",
    ) -> None:
        self._ensure_forced_prompt_trace_state_schema()

        resolved_role = str(role or "").strip().lower()
        doctor = None
        patient = None
        if speaker is not None and other is not None:
            doctor, patient = self._resolve_doctor_patient_pair(speaker, other)

        speaker_name = str(getattr(speaker, "name", "") or "")
        doctor_name = str(getattr(doctor, "name", "") or "")
        patient_name = str(getattr(patient, "name", "") or "")
        if not resolved_role:
            if speaker_name and speaker_name == doctor_name:
                resolved_role = "doctor"
            elif speaker_name and speaker_name == patient_name:
                resolved_role = "patient"
            else:
                resolved_role = "unknown"

        trace_meeting_id = str(meeting_id or "").strip()
        trace_pair_key = str(pair_key or "").strip()
        step_time = self._resolve_trace_step_time()
        if doctor and patient:
            resolved_meeting_id, resolved_pair_key, resolved_step_time = self._resolve_trace_session_context(
                doctor,
                patient,
            )
            if not trace_meeting_id:
                trace_meeting_id = str(resolved_meeting_id or "")
            if not trace_pair_key:
                trace_pair_key = str(resolved_pair_key or "")
            if resolved_step_time:
                step_time = str(resolved_step_time or step_time)

        state = self.state.setdefault("forced_prompt_trace_state", {})
        sessions = state.setdefault("sessions", [])
        session_item = self._ensure_trace_session(
            sessions,
            str(trace_meeting_id or ""),
            str(trace_pair_key or ""),
            str(step_time or self._resolve_trace_step_time()),
        )
        records = session_item.setdefault("records", [])
        if not isinstance(records, list):
            records = []
            session_item["records"] = records

        output_payload = self._normalize_forced_prompt_trace_output(output)
        output_preview = self._forced_prompt_trace_output_preview(output_payload)
        records.append(
            {
                "seq": len(records) + 1,
                "role": resolved_role,
                "turn_no": self._safe_int(turn_no, -1),
                "prompt_text": str(prompt_text or ""),
                "output": output_payload,
                "meta": copy.deepcopy(meta if isinstance(meta, dict) else {}),
                "ts": self._fmt_iso8601_with_tz(self._now()),
            }
        )
        self._log_highlight(
            "[FORCED_PROMPT_TRACE_APPEND] meeting_id={} pair_key={} role={} turn_no={} prompt_len={} output_len={}".format(
                str(trace_meeting_id or ""),
                str(trace_pair_key or ""),
                resolved_role,
                self._safe_int(turn_no, -1),
                len(str(prompt_text or "")),
                len(output_preview),
            )
        )

    def export_forced_prompt_trace_payload(self) -> Dict[str, Any]:
        self._ensure_forced_prompt_trace_state_schema()
        state = self.state.setdefault("forced_prompt_trace_state", {})
        sessions = state.get("sessions", [])
        if not isinstance(sessions, list):
            sessions = []
        return {"sessions": copy.deepcopy(sessions)}

    def render_forced_prompt_trace_markdown(self, session_item: Dict[str, Any]) -> str:
        session = session_item if isinstance(session_item, dict) else {}
        meeting = session.get("meeting", {}) if isinstance(session.get("meeting", {}), dict) else {}
        records = session.get("records", []) if isinstance(session.get("records", []), list) else []

        meeting_id = str(meeting.get("meeting_id", "") or "").strip() or "meeting_unknown"
        pair_key = str(meeting.get("pair_key", "") or "").strip()
        step_time = str(meeting.get("step_time", "") or "").strip()

        counts = {
            "patient": 0,
            "doctor": 0,
            "consult_history": 0,
            "judge_llm": 0,
            "state_tracker_llm": 0,
            "session_eval_llm": 0,
            "subgoal_shadow_eval_llm": 0,
            "control_eval_llm": 0,
            "order_extract_llm": 0,
            "environment_model_llm": 0,
            "unknown": 0,
        }
        for record in records:
            if not isinstance(record, dict):
                continue
            role_text = str(record.get("role", "") or "").strip().lower()
            if role_text in counts:
                counts[role_text] += 1
            else:
                counts["unknown"] += 1

        lines = [
            "# 强制干预对话 Prompt 全链路追踪",
            "",
            "- meeting_id: `{}`".format(meeting_id),
            "- pair_key: `{}`".format(pair_key),
            "- step_time: `{}`".format(step_time),
            "- total_records: `{}`".format(len(records)),
            "",
        ]

        for idx, record in enumerate(records, start=1):
            if not isinstance(record, dict):
                continue
            role_text = str(record.get("role", "") or "").strip().lower() or "unknown"
            turn_no = self._safe_int(record.get("turn_no", -1), -1)
            prompt_text = str(record.get("prompt_text", "") or "")
            output_payload = record.get("output", "")
            output_text = self._forced_prompt_trace_output_preview(output_payload)
            if isinstance(output_payload, (dict, list)):
                output_text = json.dumps(output_payload, ensure_ascii=False, indent=2)
            meta_payload = record.get("meta", {})
            meta_text = "{}"
            if isinstance(meta_payload, dict):
                meta_text = json.dumps(meta_payload, ensure_ascii=False, indent=2)

            lines.extend(
                [
                    "## Record {}".format(idx),
                    "",
                    "- role: `{}`".format(role_text),
                    "- turn_no: `{}`".format(turn_no),
                    "- ts: `{}`".format(str(record.get("ts", "") or "")),
                    "",
                    "### Prompt",
                    "```text",
                    prompt_text,
                    "```",
                    "",
                    "### Output",
                    "```text",
                    output_text,
                    "```",
                    "",
                    "### Meta",
                    "```json",
                    meta_text,
                    "```",
                    "",
                ]
            )

        lines.extend(
            [
                "## Summary",
                "",
                "- patient_count: `{}`".format(counts["patient"]),
                "- doctor_count: `{}`".format(counts["doctor"]),
                "- consult_history_count: `{}`".format(counts["consult_history"]),
                "- judge_count: `{}`".format(counts["judge_llm"]),
                "- state_tracker_count: `{}`".format(counts["state_tracker_llm"]),
                "- session_eval_count: `{}`".format(counts["session_eval_llm"]),
                "- subgoal_eval_count: `{}`".format(
                    counts["subgoal_shadow_eval_llm"]
                ),
                "- control_eval_count: `{}`".format(
                    (
                        counts["control_eval_llm"]
                        if counts["control_eval_llm"]
                        else (
                            counts["session_eval_llm"]
                            + counts["subgoal_shadow_eval_llm"]
                        )
                    )
                ),
                "- order_extract_count: `{}`".format(counts["order_extract_llm"]),
                "- environment_model_count: `{}`".format(counts["environment_model_llm"]),
                "- unknown_count: `{}`".format(counts["unknown"]),
                "- total_count: `{}`".format(len(records)),
                "",
            ]
        )
        return "\n".join(lines)

    def _normalize_forced_prompt_trace_output(self, output: Any) -> Any:
        if isinstance(output, (dict, list)):
            return copy.deepcopy(output)
        if output is None:
            return ""
        return str(output)

    def _forced_prompt_trace_output_preview(self, payload: Any) -> str:
        if isinstance(payload, (dict, list)):
            try:
                return json.dumps(payload, ensure_ascii=False)
            except Exception:
                return str(payload)
        return str(payload or "")

    def _resolve_trace_session_context(self, doctor: Any, patient: Any):
        meeting_id = ""
        pair_key = build_pair_key(str(getattr(doctor, "name", "") or ""), str(getattr(patient, "name", "") or ""))
        self._ensure_agent_state(doctor)
        self._ensure_agent_state(patient)
        lock_d = doctor.status.get("intervention", {}).get("lock", {})
        lock_p = patient.status.get("intervention", {}).get("lock", {})
        if isinstance(lock_d, dict):
            meeting_id = str(lock_d.get("meeting_id", "") or "")
        if (not meeting_id) and isinstance(lock_p, dict):
            meeting_id = str(lock_p.get("meeting_id", "") or "")
        return str(meeting_id or ""), str(pair_key or ""), self._resolve_trace_step_time()

    def _ensure_trace_session(self, sessions: list, meeting_id: str, pair_key: str, step_time: str) -> Dict[str, Any]:
        for item in sessions:
            if not isinstance(item, dict):
                continue
            meeting = item.get("meeting", {}) or {}
            if (
                str(meeting.get("meeting_id", "") or "") == str(meeting_id or "")
                and str(meeting.get("pair_key", "") or "") == str(pair_key or "")
            ):
                if not str(meeting.get("step_time", "") or ""):
                    meeting["step_time"] = str(step_time or "")
                    item["meeting"] = meeting
                return item

        session_item = {
            "meeting": {
                "meeting_id": str(meeting_id or ""),
                "pair_key": str(pair_key or ""),
                "step_time": str(step_time or ""),
            },
            "turns": [],
        }
        sessions.append(session_item)
        return session_item

    def _resolve_trace_step_time(self) -> str:
        raw = self.config.get("time", "")
        if isinstance(raw, str):
            text = str(raw or "").strip()
            if text:
                return text
        if isinstance(raw, dict):
            start = str(raw.get("start", "") or "").strip()
            if start:
                return start
        return self._fmt_dt(self._now())

    def _extract_latest_patient_utterance(self, chats: Any, patient_name: str) -> str:
        items = chats if isinstance(chats, list) else []
        for item in reversed(items):
            if not isinstance(item, (tuple, list)) or len(item) < 2:
                continue
            speaker = str(item[0] or "")
            text = str(item[1] or "")
            if speaker == str(patient_name or ""):
                return text
        return ""

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

    def _extract_and_queue_orders(
        self,
        speaker: Any,
        other: Any,
        chats: Any,
        meeting_id: str = "",
        meeting_kind: str = "doctor_consult",
        start_time: Any = None,
    ) -> None:
        meeting_id_text = str(meeting_id or "").strip()
        meeting_kind_text = str(meeting_kind or "").strip()
        if not meeting_id_text or meeting_kind_text != "doctor_consult":
            return
        if not self.order_extract.get("enabled", True):
            return
        env_policy = self.get_environment_model_runtime_policy()
        if not bool(env_policy.get("enabled", False)):
            self._log_highlight("ORDER_EXTRACT_SKIP reason=environment_model_disabled")
            return

        doctor, patient = self._resolve_doctor_patient_pair(speaker, other)
        if not doctor or not patient:
            return

        conversation = "\n".join(["{}: {}".format(n, t) for n, t in chats or []])
        if not conversation.strip():
            return

        if isinstance(start_time, datetime.datetime):
            order_now = start_time
        else:
            order_now = self._parse_dt(str(start_time or ""))
        if order_now is None:
            order_now = utils.get_timer().get_date()
        order_now_text = self._fmt_dt(order_now)

        result = doctor.completion(
            "extract_doctor_order",
            doctor.name,
            patient.name,
            order_now_text,
            conversation,
            prompt_file=str(self.order_extract.get("prompt_file", "") or "").strip(),
            _forced_prompt_trace={
                "enabled": True,
                "other": patient,
                "role": "order_extract_llm",
                "source": "order_extract",
                "meeting_id": meeting_id_text,
                "pair_key": build_pair_key(doctor.name, patient.name),
                "meta": {
                    "stage": "extract_doctor_order",
                    "prompt_file": str(self.order_extract.get("prompt_file", "") or "").strip(),
                    "max_tasks_per_chat": max(1, self._safe_int(self.order_extract.get("max_tasks_per_chat", 3), 3)),
                },
            },
        )
        tasks = []
        if isinstance(result, dict):
            tasks = result.get("tasks", [])
        if not isinstance(tasks, list):
            return

        max_tasks = int(self.order_extract.get("max_tasks_per_chat", 3) or 3)
        accepted_tasks = []
        for task in tasks:
            if len(accepted_tasks) >= max_tasks:
                break
            if not isinstance(task, dict):
                continue
            if not task.get("date") or not task.get("describe"):
                continue

            task_id = "order_{}_{}_{}".format(
                order_now_text.replace(":", "").replace("-", ""),
                patient.name,
                len(accepted_tasks) + 1,
            )
            accepted_tasks.append(
                {
                    "task_id": task_id,
                    "source": "doctor_order",
                    "doctor": doctor.name,
                    "patient": patient.name,
                    "describe": str(task.get("describe", "")).strip(),
                    "date": str(task.get("date", "")).strip(),
                }
            )

        if not accepted_tasks:
            self._log_highlight(
                "ORDER_EXTRACT_EMPTY doctor={} patient={}".format(
                    doctor.name,
                    patient.name,
                )
            )
            return

        self._log(
            "extracted doctor orders: doctor={}, patient={}, count={}".format(
                doctor.name,
                patient.name,
                len(accepted_tasks),
            )
        )
        self._apply_environment_task_outcomes(
            doctor=doctor,
            patient=patient,
            tasks=accepted_tasks,
            meeting_id=meeting_id_text,
            meeting_kind=meeting_kind_text,
            start_time=order_now,
            env_policy=env_policy,
        )

    def get_environment_model_runtime_policy(self) -> Dict[str, Any]:
        raw_cfg = self.environment_model_cfg if isinstance(self.environment_model_cfg, dict) else {}
        default_policy = {
            "enabled": False,
            "route": "forced_llm",
            "prompt_file": "data/prompts/intervention/environment_task_outcome.txt",
            "retry": 2,
            "poignancy": 5,
            "expire_days": 30,
            "max_agents_per_task": 3,
            "reflect_after_injection": True,
            "source": "defaults",
            "llm": {},
        }
        if not isinstance(raw_cfg, dict):
            return default_policy
        policy = dict(default_policy)
        policy["source"] = "environment_model_config"
        policy["enabled"] = self._safe_bool(raw_cfg.get("enabled", False), False)
        route = str(raw_cfg.get("route", default_policy["route"]) or default_policy["route"]).strip().lower()
        if route not in {"forced_llm", "think_llm", "custom_llm"}:
            route = default_policy["route"]
        policy["route"] = route
        prompt_file = str(raw_cfg.get("prompt_file", default_policy["prompt_file"]) or "").strip()
        policy["prompt_file"] = prompt_file or default_policy["prompt_file"]
        retry = self._safe_int(raw_cfg.get("retry", default_policy["retry"]), default_policy["retry"])
        policy["retry"] = retry if retry >= 1 else default_policy["retry"]
        policy["poignancy"] = max(1, self._safe_int(raw_cfg.get("poignancy", default_policy["poignancy"]), default_policy["poignancy"]))
        policy["expire_days"] = max(1, self._safe_int(raw_cfg.get("expire_days", default_policy["expire_days"]), default_policy["expire_days"]))
        policy["max_agents_per_task"] = max(
            0,
            self._safe_int(
                raw_cfg.get("max_agents_per_task", default_policy["max_agents_per_task"]),
                default_policy["max_agents_per_task"],
            ),
        )
        policy["reflect_after_injection"] = self._safe_bool(
            raw_cfg.get("reflect_after_injection", default_policy["reflect_after_injection"]),
            default_policy["reflect_after_injection"],
        )
        llm_cfg = raw_cfg.get("llm", {})
        policy["llm"] = copy.deepcopy(llm_cfg) if isinstance(llm_cfg, dict) else {}
        return policy

    def _build_environment_task_batch_id(self, patient: Any, meeting_id: str) -> str:
        patient_name = str(getattr(patient, "name", "") or "patient").strip() or "patient"
        meeting = str(meeting_id or "").strip()
        stamp = self._fmt_dt(self._now()).replace(":", "").replace("-", "").replace(" ", "_")
        if meeting:
            return "envtask_{}_{}".format(meeting, patient_name)
        return "envtask_{}_{}".format(stamp, patient_name)

    def _build_environment_task_prompt(
        self,
        doctor: Any,
        patient: Any,
        tasks: List[Dict[str, Any]],
        task_batch_id: str,
        agents_map: Dict[str, Any],
        env_policy: Dict[str, Any],
    ) -> str:
        try:
            prompt_tpl = self._load_prompt_txt_or_raise(str(env_policy.get("prompt_file", "") or ""))
        except Exception as exc:
            self._log_highlight(
                "ENV_TASK_PROMPT_LOAD_ERROR task_batch_id={} prompt_file={} detail={}".format(
                    task_batch_id,
                    str(env_policy.get("prompt_file", "") or ""),
                    str(exc),
                )
            )
            return ""
        known_agents = sorted(
            [
                str(name or "").strip()
                for name in (agents_map or {}).keys()
                if str(name or "").strip()
            ]
        )
        task_prompt_items = []
        for task in tasks or []:
            if not isinstance(task, dict):
                continue
            task_prompt_items.append(
                {
                    "describe": str(task.get("describe", "") or ""),
                    "date": str(task.get("date", "") or ""),
                }
            )
        task_json = json.dumps(task_prompt_items, ensure_ascii=False, indent=2)
        patient_state_summary = self._build_patient_dynamic_state_text(
            patient_agent=patient,
            doctor_agent=doctor,
        )
        return self._render_prompt_template(
            prompt_tpl,
            {
                "doctor": str(getattr(doctor, "name", "") or ""),
                "patient": str(getattr(patient, "name", "") or ""),
                "now": self._fmt_dt(self._now()),
                "patient_state_summary": str(patient_state_summary or ""),
                "known_agents": "\n".join(["- {}".format(name) for name in known_agents]),
                "tasks_json": task_json,
            },
        )

    def _call_environment_model_json(
        self,
        prompt_text: str,
        retry: int,
        doctor_agent: Any,
        env_policy: Dict[str, Any],
    ) -> Dict[str, Any]:
        route = str((env_policy or {}).get("route", "forced_llm") or "forced_llm").strip().lower()
        if route == "think_llm":
            return self._call_think_llm_json(
                prompt_text=prompt_text,
                retry=retry,
                doctor_agent=doctor_agent,
                caller="environment_task_outcome_think_llm",
            )
        runtime_cfg = self._resolve_environment_llm_runtime(env_policy)
        retry_count = max(1, int(retry or runtime_cfg.get("retry", 2) or 2))
        cache_key = "{}/{}/{}".format(
            runtime_cfg.get("provider", ""),
            runtime_cfg.get("model", ""),
            runtime_cfg.get("base_url", ""),
        )
        if self._environment_llm is None or self._environment_llm_key != cache_key:
            self._environment_llm = create_llm_model(runtime_cfg)
            self._environment_llm_key = cache_key
        payload = self._environment_llm.completion(
            prompt_text,
            retry=retry_count,
            callback=self._json_loads_loose,
            failsafe=None,
            caller="environment_task_outcome",
            temperature=float(runtime_cfg.get("temperature", 0.4) or 0.4),
        )
        if not isinstance(payload, dict):
            raise ConsultRecordError(reason="environment_model_json_parse_failed", retryable=False)
        return payload

    def _resolve_environment_llm_runtime(self, env_policy: Dict[str, Any]) -> Dict[str, Any]:
        route = str((env_policy or {}).get("route", "forced_llm") or "forced_llm").strip().lower()
        if route == "custom_llm":
            llm_cfg = copy.deepcopy((env_policy or {}).get("llm", {}) or {})
            if not isinstance(llm_cfg, dict) or not llm_cfg:
                raise ConsultRecordError(reason="environment_model_llm_missing", retryable=False)
            api_key_env = str(llm_cfg.get("api_key_env", "") or "").strip()
            if api_key_env and not str(llm_cfg.get("api_key", "") or "").strip():
                llm_cfg["api_key"] = str(os.getenv(api_key_env, "") or "")
            if not str(llm_cfg.get("api_key", "") or "").strip():
                llm_cfg["api_key"] = "EMPTY"
            llm_cfg["retry"] = self._safe_int(llm_cfg.get("retry", (env_policy or {}).get("retry", 2)), 2)
            return llm_cfg
        runtime_cfg = self.get_forced_llm_runtime_config()
        if not runtime_cfg:
            raise ConsultRecordError(reason="environment_model_forced_llm_unavailable", retryable=True)
        runtime_cfg = copy.deepcopy(runtime_cfg)
        runtime_cfg["retry"] = self._safe_int((env_policy or {}).get("retry", runtime_cfg.get("retry", 2)), runtime_cfg.get("retry", 2))
        return runtime_cfg

    def _normalize_environment_task_outcomes(
        self,
        payload: Dict[str, Any],
        tasks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        raw = payload if isinstance(payload, dict) else {}
        outcomes = raw.get("outcomes", raw.get("tasks", []))
        if not isinstance(outcomes, list):
            return []
        normalized = []
        max_tasks = len(tasks or [])
        for idx, item in enumerate(outcomes):
            if not isinstance(item, dict):
                continue
            raw_task_index = item.get("task_index", item.get("index", idx))
            try:
                task_index = int(raw_task_index)
            except Exception:
                task_index = idx
            if task_index < 0 or task_index >= max_tasks:
                continue
            event_memory = str(item.get("event", "") or "").strip()
            raw_agents = item.get("agents", item.get("social_agents", []))
            if isinstance(raw_agents, str):
                agents = [raw_agents]
            elif isinstance(raw_agents, (list, tuple, set)):
                agents = list(raw_agents)
            else:
                agents = []
            normalized_agents = []
            for raw_agent in agents:
                agent_name = str(raw_agent or "").strip()
                if agent_name and agent_name not in normalized_agents:
                    normalized_agents.append(agent_name)
            if not event_memory:
                continue
            normalized.append(
                {
                    "task_index": task_index,
                    "event": event_memory,
                    "agents": normalized_agents,
                }
            )
        return normalized

    def _build_environment_memory_payloads(
        self,
        outcomes: List[Dict[str, Any]],
        tasks: List[Dict[str, Any]],
        doctor: Any,
        patient: Any,
        agents_map: Dict[str, Any],
        task_batch_id: str,
        meeting_id: str,
        meeting_kind: str,
        env_policy: Dict[str, Any],
    ) -> tuple:
        patient_name = str(getattr(patient, "name", "") or "")
        doctor_name = str(getattr(doctor, "name", "") or "")
        known_agents = agents_map if isinstance(agents_map, dict) else {}
        max_social = int(env_policy.get("max_agents_per_task", 3) or 3)
        event_poignancy = int(env_policy.get("poignancy", 5) or 5)
        expire_days = int(env_policy.get("expire_days", 30) or 30)
        step_no = int(self.config.get("step", 0) or 0)
        payloads_by_agent: Dict[str, Dict[str, Any]] = {}
        summary_lines = []

        def _payload_for(agent_name: str) -> Dict[str, Any]:
            if agent_name not in payloads_by_agent:
                payloads_by_agent[agent_name] = {
                    "request_id": "envtask_{}_{}".format(task_batch_id, agent_name),
                    "source": "environment_task_outcome",
                    "target_agent": agent_name,
                    "options": {
                        "allow_partial_success": True,
                        "dedup_mode": "global",
                    },
                    "meta": {
                        "step": step_no,
                        "scene": "environment_task_result",
                        "meeting_id": str(meeting_id or ""),
                        "meeting_kind": str(meeting_kind or ""),
                        "task_batch_id": str(task_batch_id or ""),
                        "doctor": doctor_name,
                        "patient": patient_name,
                    },
                    "items": [],
                }
            return payloads_by_agent[agent_name]

        for outcome in outcomes:
            task_index = int(outcome.get("task_index", 0) or 0)
            task = tasks[task_index] if 0 <= task_index < len(tasks) else {}
            task_id = str(task.get("task_id", "task_{}".format(task_index + 1)) or "")
            describe = str(task.get("describe", "") or "")
            summary_lines.append(
                "任务{}({}): {} -> {}".format(
                    task_index + 1,
                    task_id,
                    describe,
                    str(outcome.get("event", "") or ""),
                )
            )
            event_memory = str(outcome.get("event", "") or "").strip()
            if event_memory:
                _payload_for(patient_name)["items"].append(
                    {
                        "item_id": "{}_event_{}_patient".format(task_id, task_index + 1),
                        "node_type": "event",
                        "subject": patient_name,
                        "predicate": "此时",
                        "object": "执行医生作业后形成结果",
                        "describe": event_memory,
                        "poignancy": event_poignancy,
                        "expire_days": expire_days,
                        "tags": ["environment_task_result", "doctor_order"],
                        "meta": {
                            "task_id": task_id,
                            "task_index": task_index,
                            "task_describe": describe,
                        },
                    }
                )
            social_count = 0
            for agent_name in outcome.get("agents", []) or []:
                if social_count >= max_social:
                    break
                agent_name = str(agent_name or "").strip()
                if not agent_name:
                    continue
                if agent_name == patient_name:
                    continue
                if agent_name not in known_agents:
                    self._log_highlight(
                        "ENV_TASK_SOCIAL_MEMORY_SKIP task_batch_id={} task_id={} agent={} reason=unknown_agent".format(
                            task_batch_id,
                            task_id,
                            agent_name,
                        )
                    )
                    continue
                _payload_for(agent_name)["items"].append(
                    {
                        "item_id": "{}_social_event_{}_{}".format(task_id, task_index + 1, social_count + 1),
                        "node_type": "event",
                        "subject": agent_name,
                        "predicate": "此时",
                        "object": "参与了患者医生作业相关互动",
                        "describe": event_memory,
                        "poignancy": event_poignancy,
                        "expire_days": expire_days,
                        "tags": ["environment_task_result", "doctor_order", "social_sync"],
                        "meta": {
                            "task_id": task_id,
                            "task_index": task_index,
                            "patient": patient_name,
                            "task_describe": describe,
                        },
                    }
                )
                social_count += 1

        payloads = [
            payload
            for payload in payloads_by_agent.values()
            if isinstance(payload.get("items"), list) and payload.get("items")
        ]
        return payloads, summary_lines

    def _count_injected_memory_results(self, result: Dict[str, Any]) -> int:
        if not isinstance(result, dict):
            return 0
        total = 0
        for item in result.get("results", []) or []:
            if isinstance(item, dict):
                total += int(item.get("injected", 0) or 0)
        return total

    def _append_environment_task_audit(
        self,
        task_batch_id: str,
        meeting_id: str,
        doctor: str,
        patient: str,
        tasks: List[Dict[str, Any]],
        outcomes: List[Dict[str, Any]],
        injection_result: Dict[str, Any],
    ) -> None:
        self._ensure_state_schema()
        task_state = self.state.setdefault("environment_task_state", {})
        if not isinstance(task_state, dict):
            task_state = {}
            self.state["environment_task_state"] = task_state
        audit = task_state.setdefault("audit", [])
        if not isinstance(audit, list):
            audit = []
            task_state["audit"] = audit
        audit.append(
            {
                "task_batch_id": str(task_batch_id or ""),
                "meeting_id": str(meeting_id or ""),
                "doctor": str(doctor or ""),
                "patient": str(patient or ""),
                "created_at": self._fmt_dt(self._now()),
                "tasks": copy.deepcopy(tasks),
                "outcomes": copy.deepcopy(outcomes),
                "injection_status": str((injection_result or {}).get("status", "") if isinstance(injection_result, dict) else ""),
                "injected": self._count_injected_memory_results(injection_result),
            }
        )
        if len(audit) > 500:
            del audit[:-500]

    def _handle_conditional_reflection_environment_task_result(
        self,
        patient: Any,
        doctor: Any,
        task_batch_id: str,
        meeting_id: str,
        meeting_kind: str,
        tasks: List[Dict[str, Any]],
        outcomes: List[Dict[str, Any]],
        outcome_summary: str,
    ) -> None:
        if patient is None or not hasattr(patient, "trigger_reflection"):
            return
        cfg = self._reflection_conditional_cfg()
        if not self._safe_bool(cfg.get("enabled", True), True):
            self._log_highlight(
                "REFLECTION_ENV_TASK_SKIP task_batch_id={} reason=conditional_disabled".format(task_batch_id)
            )
            return
        rules = cfg.get("rules", [])
        if not isinstance(rules, list):
            rules = []
        matched = 0
        for idx, rule in enumerate(rules):
            if not isinstance(rule, dict):
                continue
            if not self._safe_bool(rule.get("enabled", True), True):
                continue
            if str(rule.get("event", "") or "").strip() != "environment_task_result":
                continue
            target_role = str(rule.get("target_role", "patient") or "patient").strip()
            if target_role not in {"patient", "all"}:
                continue
            if not self._reflection_rule_matches_meeting_kind(rule, meeting_kind):
                continue
            matched += 1
            once_per_batch = self._safe_bool(rule.get("once_per_task_batch", True), True)
            if once_per_batch and self._has_reflected_for_environment_task_batch(task_batch_id):
                self._log_highlight(
                    "REFLECTION_ENV_TASK_SKIP task_batch_id={} rule_index={} reason=already_reflected".format(
                        task_batch_id,
                        idx,
                    )
                )
                continue
            context = {
                "event": "environment_task_result",
                "meeting_kind": str(meeting_kind or ""),
                "meeting_id": str(meeting_id or ""),
                "task_batch_id": str(task_batch_id or ""),
                "target_role": "patient",
                "doctor": str(getattr(doctor, "name", "") or ""),
                "patient": str(getattr(patient, "name", "") or ""),
                "task_count": len(tasks or []),
                "outcome_count": len(outcomes or []),
                "chat_summary": str(outcome_summary or ""),
                "behavior_events": [
                    str(item.get("event", "") or "").strip()
                    for item in outcomes or []
                    if isinstance(item, dict) and str(item.get("event", "") or "").strip()
                ][:5],
                "step": int(self.config.get("step", 0) or 0),
                "started_at": self._fmt_dt(self._now()),
            }
            try:
                result = patient.trigger_reflection(
                    trigger_source="environment_task_result",
                    trigger_context=context,
                )
                self._mark_environment_task_reflected(task_batch_id, result)
                self._log_highlight(
                    "REFLECTION_ENV_TASK_DONE task_batch_id={} target={} triggered={} reason={}".format(
                        task_batch_id,
                        str(getattr(patient, "name", "") or ""),
                        bool((result or {}).get("triggered", False)) if isinstance(result, dict) else bool(result),
                        str((result or {}).get("reason", "") if isinstance(result, dict) else ""),
                    )
                )
            except Exception as exc:
                self._log_highlight(
                    "REFLECTION_ENV_TASK_ERROR task_batch_id={} target={} detail={}".format(
                        task_batch_id,
                        str(getattr(patient, "name", "") or ""),
                        str(exc),
                    )
                )
        if matched <= 0:
            self._log_highlight(
                "REFLECTION_ENV_TASK_SKIP task_batch_id={} reason=no_rule_matched".format(task_batch_id)
            )

    def _has_reflected_for_environment_task_batch(self, task_batch_id: str) -> bool:
        task_id = str(task_batch_id or "").strip()
        if not task_id:
            return False
        self._ensure_state_schema()
        task_state = self.state.setdefault("environment_task_state", {})
        reflected = task_state.setdefault("reflected_batches", {})
        return isinstance(reflected, dict) and task_id in reflected

    def _mark_environment_task_reflected(self, task_batch_id: str, result: Any) -> None:
        task_id = str(task_batch_id or "").strip()
        if not task_id:
            return
        self._ensure_state_schema()
        task_state = self.state.setdefault("environment_task_state", {})
        reflected = task_state.setdefault("reflected_batches", {})
        if not isinstance(reflected, dict):
            reflected = {}
            task_state["reflected_batches"] = reflected
        reflected[task_id] = {
            "triggered_at": self._fmt_dt(self._now()),
            "triggered": bool((result or {}).get("triggered", False)) if isinstance(result, dict) else bool(result),
            "reason": str((result or {}).get("reason", "") if isinstance(result, dict) else ""),
        }

    def _apply_environment_task_outcomes(
        self,
        doctor: Any,
        patient: Any,
        tasks: List[Dict[str, Any]],
        meeting_id: str,
        meeting_kind: str,
        start_time: Any,
        env_policy: Dict[str, Any],
    ) -> None:
        if not tasks:
            return
        agents_map = dict(self._agents_ref) if isinstance(self._agents_ref, dict) else {}
        for agent in (doctor, patient):
            name = str(getattr(agent, "name", "") or "")
            if name and name not in agents_map:
                agents_map[name] = agent

        task_batch_id = self._build_environment_task_batch_id(patient, meeting_id)
        prompt_text = self._build_environment_task_prompt(
            doctor=doctor,
            patient=patient,
            tasks=tasks,
            task_batch_id=task_batch_id,
            agents_map=agents_map,
            env_policy=env_policy,
        )
        if not prompt_text:
            self._log_highlight(
                "ENV_TASK_OUTCOME_SKIP task_batch_id={} reason=prompt_empty".format(task_batch_id)
            )
            return
        try:
            payload = self._call_environment_model_json(
                prompt_text=prompt_text,
                retry=int(env_policy.get("retry", 2) or 2),
                doctor_agent=doctor,
                env_policy=env_policy,
            )
            self.append_forced_prompt_trace_record(
                speaker=doctor,
                other=patient,
                role="environment_model_llm",
                prompt_text=prompt_text,
                output=payload,
                turn_no=-1,
                meta={
                    "source": "environment_task_outcome",
                    "caller": "environment_task_outcome",
                    "route": str(env_policy.get("route", "forced_llm") or "forced_llm"),
                    "retry": int(env_policy.get("retry", 2) or 2),
                    "prompt_file": str(env_policy.get("prompt_file", "") or ""),
                    "task_count": len(tasks or []),
                },
                meeting_id=str(meeting_id or ""),
                pair_key=build_pair_key(
                    str(getattr(doctor, "name", "") or ""),
                    str(getattr(patient, "name", "") or ""),
                ),
            )
        except Exception as exc:
            self.append_forced_prompt_trace_record(
                speaker=doctor,
                other=patient,
                role="environment_model_llm",
                prompt_text=prompt_text,
                output={
                    "error": str(exc),
                    "status": "failed",
                },
                turn_no=-1,
                meta={
                    "source": "environment_task_outcome",
                    "caller": "environment_task_outcome",
                    "route": str(env_policy.get("route", "forced_llm") or "forced_llm"),
                    "retry": int(env_policy.get("retry", 2) or 2),
                    "prompt_file": str(env_policy.get("prompt_file", "") or ""),
                    "task_count": len(tasks or []),
                    "status": "failed",
                },
                meeting_id=str(meeting_id or ""),
                pair_key=build_pair_key(
                    str(getattr(doctor, "name", "") or ""),
                    str(getattr(patient, "name", "") or ""),
                ),
            )
            self._log_highlight(
                "ENV_TASK_OUTCOME_ERROR task_batch_id={} reason={}".format(
                    task_batch_id,
                    str(exc),
                )
            )
            return

        outcomes = self._normalize_environment_task_outcomes(payload, tasks)
        if not outcomes:
            self._log_highlight(
                "ENV_TASK_OUTCOME_EMPTY task_batch_id={} tasks={}".format(
                    task_batch_id,
                    len(tasks),
                )
            )
            return

        payloads, summary_lines = self._build_environment_memory_payloads(
            outcomes=outcomes,
            tasks=tasks,
            doctor=doctor,
            patient=patient,
            agents_map=agents_map,
            task_batch_id=task_batch_id,
            meeting_id=meeting_id,
            meeting_kind=meeting_kind,
            env_policy=env_policy,
        )
        if not payloads:
            self._log_highlight(
                "ENV_TASK_MEMORY_SKIP task_batch_id={} reason=payloads_empty".format(task_batch_id)
            )
            return

        result = self.memory_injection.inject_many(agents=agents_map, payloads=payloads, now=start_time)
        injected_count = self._count_injected_memory_results(result)
        self._append_environment_task_audit(
            task_batch_id=task_batch_id,
            meeting_id=meeting_id,
            doctor=str(getattr(doctor, "name", "") or ""),
            patient=str(getattr(patient, "name", "") or ""),
            tasks=tasks,
            outcomes=outcomes,
            injection_result=result,
        )
        self._log_highlight(
            "ENV_TASK_MEMORY_DONE task_batch_id={} payloads={} injected={} status={}".format(
                task_batch_id,
                len(payloads),
                injected_count,
                str(result.get("status", "") if isinstance(result, dict) else ""),
            )
        )
        if injected_count > 0 and self._safe_bool(env_policy.get("reflect_after_injection", True), True):
            self._handle_conditional_reflection_environment_task_result(
                patient=patient,
                doctor=doctor,
                task_batch_id=task_batch_id,
                meeting_id=meeting_id,
                meeting_kind=meeting_kind,
                tasks=tasks,
                outcomes=outcomes,
                outcome_summary="\n".join(summary_lines),
            )

    def _resolve_active_meeting(self, a: Any, b: Any) -> Optional[Dict[str, Any]]:
        self._ensure_state_schema()
        active = self.state.setdefault("active_meetings", {})
        meeting_ids = []
        for agent in (a, b):
            if not agent:
                continue
            self._ensure_agent_state(agent)
            lock = agent.status.get("intervention", {}).get("lock", {})
            if not isinstance(lock, dict):
                continue
            meeting_id = str(lock.get("meeting_id", "") or "").strip()
            if meeting_id and meeting_id not in meeting_ids:
                meeting_ids.append(meeting_id)
        for meeting_id in meeting_ids:
            meeting = active.get(meeting_id)
            if not isinstance(meeting, dict):
                continue
            doctor_name = str(meeting.get("doctor", "") or "").strip()
            patient_name = str(meeting.get("patient", "") or "").strip()
            pair = {doctor_name, patient_name}
            if pair and {str(getattr(a, "name", "") or ""), str(getattr(b, "name", "") or "")} == pair:
                return meeting
        return None

    def _resolve_global_doctor_patient_pair(self, a: Any, b: Any):
        doctor_name = self.doctor
        patient_candidates = set(self.patients)

        if doctor_name and a.name == doctor_name and (not patient_candidates or b.name in patient_candidates):
            return a, b
        if doctor_name and b.name == doctor_name and (not patient_candidates or a.name in patient_candidates):
            return b, a
        return None, None

    def resolve_meeting_context(self, speaker: Any, other: Any, forced: bool = False) -> Dict[str, Any]:
        meeting = self._resolve_active_meeting(speaker, other)
        if isinstance(meeting, dict):
            return meeting
        if bool(forced):
            return {}
        doctor, patient = self._resolve_global_doctor_patient_pair(speaker, other)
        if not doctor or not patient:
            return {}
        return {
            "meeting_id": "",
            "doctor": doctor.name,
            "patient": patient.name,
            "meeting_kind": "doctor_consult",
            "prompt_file": "",
            "meeting_source": "global_fallback",
        }

    def is_meeting_doctor(self, speaker: Any, other: Any, forced: bool = False) -> bool:
        meeting = self.resolve_meeting_context(speaker, other, forced=forced)
        if not isinstance(meeting, dict):
            return False
        return str(getattr(speaker, "name", "") or "") == str(meeting.get("doctor", "") or "")

    def get_meeting_prompt_file(self, speaker: Any, other: Any, forced: bool = False) -> str:
        meeting = self.resolve_meeting_context(speaker, other, forced=forced)
        if not isinstance(meeting, dict):
            return ""
        return str(meeting.get("prompt_file", "") or "").strip()

    def get_resident_chat_prompt_injection(self, speaker: Any, other: Any) -> str:
        """Return the configured resident-side prefix for any chat with the target patient."""
        intervention_cfg = self.config.get("intervention", {}) or {}
        injection_cfg = intervention_cfg.get("resident_chat_prompt_injection", {}) or {}
        if not isinstance(injection_cfg, dict) or not bool(injection_cfg.get("enabled", False)):
            return ""

        target_patient = str(injection_cfg.get("target_patient", "") or "").strip()
        speaker_name = str(getattr(speaker, "name", "") or "").strip()
        other_name = str(getattr(other, "name", "") or "").strip()
        if not target_patient or other_name != target_patient or speaker_name == target_patient:
            return ""

        prompt_file = str(injection_cfg.get("prompt_file", "") or "").strip()
        if not prompt_file:
            return ""
        try:
            prompt_tpl = self._load_prompt_txt_or_raise(prompt_file)
            prompt_text = self._render_prompt_template(
                prompt_tpl,
                {
                    "speaker": speaker_name,
                    "other": other_name,
                    "doctor": speaker_name,
                    "patient": target_patient,
                    "meeting_kind": "resident_chat",
                    "meeting_id": "",
                    "current_time": self._resolve_trace_step_time(),
                },
            )
        except Exception as exc:
            self._log_highlight(
                "RESIDENT_CHAT_PROMPT_LOAD_ERROR speaker={} other={} prompt_file={} detail={}".format(
                    speaker_name,
                    other_name,
                    prompt_file,
                    str(exc),
                )
            )
            return ""
        prompt_text = str(prompt_text or "").strip()
        if not prompt_text:
            return ""
        return "<RESIDENT_CHAT_PROMPT_INJECTION>\n{}\n</RESIDENT_CHAT_PROMPT_INJECTION>".format(
            prompt_text
        )

    def get_meeting_prompt_injection(self, speaker: Any, other: Any, forced: bool = False) -> str:
        prompt_blocks = [self.get_resident_chat_prompt_injection(speaker, other)]
        prompt_blocks = [block for block in prompt_blocks if block]
        if not self.enabled or (not bool(forced)):
            return "\n".join(prompt_blocks)
        prompt_file = self.get_meeting_prompt_file(speaker, other, forced=forced)
        if not prompt_file:
            return "\n".join(prompt_blocks)
        meeting = self.resolve_meeting_context(speaker, other, forced=forced)
        if not isinstance(meeting, dict):
            return "\n".join(prompt_blocks)
        prompt_target = str(meeting.get("prompt_target", "all") or "all").strip().lower()
        speaker_name = str(getattr(speaker, "name", "") or "")
        if prompt_target == "doctor" and speaker_name != str(meeting.get("doctor", "") or ""):
            return "\n".join(prompt_blocks)
        if prompt_target == "patient" and speaker_name != str(meeting.get("patient", "") or ""):
            return "\n".join(prompt_blocks)
        try:
            prompt_tpl = self._load_prompt_txt_or_raise(prompt_file)
            prompt_text = self._render_prompt_template(
                prompt_tpl,
                {
                    "speaker": str(getattr(speaker, "name", "") or ""),
                    "other": str(getattr(other, "name", "") or ""),
                    "doctor": str(meeting.get("doctor", "") or ""),
                    "patient": str(meeting.get("patient", "") or ""),
                    "meeting_kind": str(meeting.get("meeting_kind", "") or ""),
                    "meeting_id": str(meeting.get("meeting_id", "") or ""),
                    "current_time": self._resolve_trace_step_time(),
                },
            )
        except Exception as exc:
            self._log_highlight(
                "MEETING_PROMPT_LOAD_ERROR speaker={} other={} prompt_file={} detail={}".format(
                    getattr(speaker, "name", ""),
                    getattr(other, "name", ""),
                    prompt_file,
                    str(exc),
                )
            )
            return "\n".join(prompt_blocks)
        prompt_text = str(prompt_text or "").strip()
        if not prompt_text:
            return "\n".join(prompt_blocks)
        prompt_blocks.append(
            "<MEETING_PROMPT_INJECTION>\n{}\n</MEETING_PROMPT_INJECTION>".format(prompt_text)
        )
        return "\n".join(prompt_blocks)

    def _resolve_doctor_patient_pair(self, a: Any, b: Any):
        meeting = self._resolve_active_meeting(a, b)
        if isinstance(meeting, dict):
            doctor_name = str(meeting.get("doctor", "") or "")
            patient_name = str(meeting.get("patient", "") or "")
            if str(getattr(a, "name", "") or "") == doctor_name and str(getattr(b, "name", "") or "") == patient_name:
                return a, b
            if str(getattr(b, "name", "") or "") == doctor_name and str(getattr(a, "name", "") or "") == patient_name:
                return b, a
        return self._resolve_global_doctor_patient_pair(a, b)

    def _resolve_latest_chat_node_id(self, agent: Any) -> str:
        try:
            associate = getattr(agent, "associate", None)
            memory = getattr(associate, "memory", {}) if associate is not None else {}
            chats = memory.get("chat", []) if isinstance(memory, dict) else []
            if isinstance(chats, list) and chats:
                return str(chats[0] or "").strip()
        except Exception:
            return ""
        return ""

    def _is_post_treatment_followup_active_for_pair(self, doctor_name: str, patient_name: str) -> bool:
        manager = self.session_prompt_injection
        if not manager or (not bool(getattr(manager, "enabled", False))):
            return False
        if not hasattr(manager, "is_post_treatment_followup_active"):
            return False
        try:
            return bool(manager.is_post_treatment_followup_active(doctor_name, patient_name))
        except Exception:
            return False

    def _should_skip_session_eval_for_post_treatment_followup(self) -> bool:
        manager = self.session_prompt_injection
        if not manager or (not bool(getattr(manager, "enabled", False))):
            return False
        if not hasattr(manager, "should_skip_session_eval_for_post_treatment_followup"):
            return False
        try:
            return bool(manager.should_skip_session_eval_for_post_treatment_followup())
        except Exception:
            return False

    def _resolve_session_prompt_current_session(self, doctor_name: str, patient_name: str) -> str:
        manager = self.session_prompt_injection
        if not manager or (not bool(getattr(manager, "enabled", False))):
            return ""
        try:
            state = manager.resolve_current_session(doctor_name, patient_name)
        except Exception:
            return ""
        if not isinstance(state, dict):
            return ""
        return str(state.get("current_session", "") or "")

    def _is_session_completed_for_pair(self, doctor_name: str, patient_name: str) -> bool:
        if not self.stop_rule_scheduling_on_session_completed:
            return False
        doctor = str(doctor_name or "").strip()
        patient = str(patient_name or "").strip()
        if (not doctor) or (not patient):
            return False
        if self._is_post_treatment_followup_active_for_pair(doctor, patient):
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
            doctor = str(parts[0] or "").strip()
            patient = str(parts[1] or "").strip()
            if (not patient) or (patient in seen):
                continue
            if self._is_post_treatment_followup_active_for_pair(doctor, patient):
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
        self._ensure_step_flags_state_schema()
        resident_chat_state = self.state.get("resident_chat_state", {})
        if not isinstance(resident_chat_state, dict):
            resident_chat_state = {}
            self.state["resident_chat_state"] = resident_chat_state
        if not isinstance(resident_chat_state.get("rules"), dict):
            resident_chat_state["rules"] = {}
        if not isinstance(resident_chat_state.get("completed_counts_by_patient"), dict):
            resident_chat_state["completed_counts_by_patient"] = {}
        self._ensure_completed_meeting_state_schema()
        self._ensure_controller_processed_meeting_state_schema()
        environment_task_state = self.state.get("environment_task_state", {})
        if not isinstance(environment_task_state, dict):
            environment_task_state = {}
            self.state["environment_task_state"] = environment_task_state
        if not isinstance(environment_task_state.get("audit"), list):
            environment_task_state["audit"] = []
        if not isinstance(environment_task_state.get("reflected_batches"), dict):
            environment_task_state["reflected_batches"] = {}
        meeting_seq = self.state.get("meeting_seq", 0)
        try:
            meeting_seq = int(meeting_seq)
        except Exception:
            meeting_seq = 0
        if meeting_seq < 0:
            meeting_seq = 0
        self.state["meeting_seq"] = meeting_seq
        self._ensure_consult_record_state_schema()
        self._ensure_dialog_judge_trace_state_schema()
        self._ensure_session_eval_state_schema()
        self._ensure_subgoal_shadow_state_schema()
        self._ensure_subgoal_progress_state_schema()
        self._ensure_progressive_d_state_schema()
        self._ensure_forced_prompt_trace_state_schema()

    def _ensure_step_flags_state_schema(self) -> None:
        state = self.state.setdefault("step_flags", {})
        if not isinstance(state, dict):
            state = {}
            self.state["step_flags"] = state
        for key in (
            "forced_consult_happened",
            "treatment_completed_this_step",
            "staged_eval_triggered_this_step",
        ):
            state[key] = bool(state.get(key, False))

    def _reset_step_flags(self) -> None:
        self._ensure_state_schema()
        flags = self.state.setdefault("step_flags", {})
        flags["forced_consult_happened"] = False
        flags["treatment_completed_this_step"] = False
        flags["staged_eval_triggered_this_step"] = False

    def mark_step_flag(self, flag_name: str, value: bool = True) -> None:
        self._ensure_state_schema()
        flags = self.state.setdefault("step_flags", {})
        flags[str(flag_name or "").strip()] = bool(value)

    def get_step_flags(self) -> Dict[str, Any]:
        self._ensure_state_schema()
        flags = self.state.get("step_flags", {})
        if not isinstance(flags, dict):
            return {}
        return copy.deepcopy(flags)

    def _mark_resident_chat_completed(self, patient_name: str) -> None:
        patient = str(patient_name or "").strip()
        if not patient:
            return
        self._ensure_state_schema()
        resident_chat_state = self.state.setdefault("resident_chat_state", {})
        completed_counts = resident_chat_state.setdefault("completed_counts_by_patient", {})
        if not isinstance(completed_counts, dict):
            completed_counts = {}
            resident_chat_state["completed_counts_by_patient"] = completed_counts
        current = self._safe_int(completed_counts.get(patient, 0), 0)
        completed_counts[patient] = max(0, current) + 1
        self._log_highlight(
            "RESIDENT_CHAT_COMPLETED_COUNT patient={} count={}".format(
                patient,
                completed_counts[patient],
            )
        )

    def _ensure_completed_meeting_state_schema(self) -> None:
        state = self.state.setdefault("completed_meeting_state", {})
        if not isinstance(state, dict):
            state = {}
            self.state["completed_meeting_state"] = state
        if not isinstance(state.get("counts_by_pair"), dict):
            state["counts_by_pair"] = {}
        if not isinstance(state.get("meeting_ids_by_pair"), dict):
            state["meeting_ids_by_pair"] = {}
        if not isinstance(state.get("records_by_meeting_id"), dict):
            state["records_by_meeting_id"] = {}

    def _ensure_controller_processed_meeting_state_schema(self) -> None:
        state = self.state.setdefault(
            "controller_processed_meeting_state",
            {},
        )
        if not isinstance(state, dict):
            state = {}
            self.state["controller_processed_meeting_state"] = state
        meeting_ids_by_pair = state.get("meeting_ids_by_pair")
        if not isinstance(meeting_ids_by_pair, dict):
            meeting_ids_by_pair = {}
            state["meeting_ids_by_pair"] = meeting_ids_by_pair
        if bool(state.get("legacy_history_migrated", False)):
            return

        for source_name in ("session_eval_state", "subgoal_shadow_state"):
            source = self.state.get(source_name, {})
            history_by_pair = (
                source.get("history_by_pair", {})
                if isinstance(source, dict)
                else {}
            )
            if not isinstance(history_by_pair, dict):
                continue
            for pair_key, history in history_by_pair.items():
                if not isinstance(history, list):
                    continue
                ids = meeting_ids_by_pair.setdefault(str(pair_key or ""), [])
                if not isinstance(ids, list):
                    ids = []
                    meeting_ids_by_pair[str(pair_key or "")] = ids
                for item in history:
                    if not isinstance(item, dict):
                        continue
                    meeting_id = str(item.get("meeting_id", "") or "").strip()
                    if meeting_id and meeting_id not in ids:
                        ids.append(meeting_id)
        state["legacy_history_migrated"] = True

    def _is_controller_meeting_processed(
        self,
        pair_key: str,
        meeting_id: str,
    ) -> bool:
        target = str(meeting_id or "").strip()
        if not target:
            return False
        self._ensure_controller_processed_meeting_state_schema()
        state = self.state.setdefault(
            "controller_processed_meeting_state",
            {},
        )
        ids = state.setdefault("meeting_ids_by_pair", {}).get(
            str(pair_key or ""),
            [],
        )
        return isinstance(ids, list) and target in ids

    def _mark_controller_meeting_processed(
        self,
        pair_key: str,
        meeting_id: str,
    ) -> None:
        target = str(meeting_id or "").strip()
        pair = str(pair_key or "").strip()
        if not pair or not target:
            return
        self._ensure_controller_processed_meeting_state_schema()
        state = self.state.setdefault(
            "controller_processed_meeting_state",
            {},
        )
        ids = state.setdefault("meeting_ids_by_pair", {}).setdefault(
            pair,
            [],
        )
        if not isinstance(ids, list):
            ids = []
            state["meeting_ids_by_pair"][pair] = ids
        if target not in ids:
            ids.append(target)

    def _mark_completed_meeting(
        self,
        doctor_name: str,
        patient_name: str,
        meeting_id: str,
        meeting_kind: str,
        finished_at: Any,
    ) -> None:
        doctor = str(doctor_name or "").strip()
        patient = str(patient_name or "").strip()
        mid = str(meeting_id or "").strip()
        if (not doctor) or (not patient) or (not mid):
            return
        self._ensure_state_schema()
        completed_state = self.state.setdefault("completed_meeting_state", {})
        counts_by_pair = completed_state.setdefault("counts_by_pair", {})
        meeting_ids_by_pair = completed_state.setdefault("meeting_ids_by_pair", {})
        records_by_meeting_id = completed_state.setdefault("records_by_meeting_id", {})
        if not isinstance(counts_by_pair, dict):
            counts_by_pair = {}
            completed_state["counts_by_pair"] = counts_by_pair
        if not isinstance(meeting_ids_by_pair, dict):
            meeting_ids_by_pair = {}
            completed_state["meeting_ids_by_pair"] = meeting_ids_by_pair
        if not isinstance(records_by_meeting_id, dict):
            records_by_meeting_id = {}
            completed_state["records_by_meeting_id"] = records_by_meeting_id
        if mid in records_by_meeting_id:
            return

        pair_key = build_pair_key(doctor, patient)
        ids = meeting_ids_by_pair.setdefault(pair_key, [])
        if not isinstance(ids, list):
            ids = []
            meeting_ids_by_pair[pair_key] = ids
        if mid not in ids:
            ids.append(mid)
        counts_by_pair[pair_key] = len(ids)
        records_by_meeting_id[mid] = {
            "meeting_id": mid,
            "pair_key": pair_key,
            "doctor": doctor,
            "patient": patient,
            "meeting_kind": str(meeting_kind or "doctor_consult"),
            "finished_at": self._fmt_dt(finished_at),
        }
        self._log_highlight(
            "COMPLETED_MEETING_COUNT pair_key={} count={} meeting_id={}".format(
                pair_key,
                counts_by_pair[pair_key],
                mid,
            )
        )

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

    def _ensure_dialog_judge_trace_state_schema(self) -> None:
        state = self.state.setdefault("dialog_judge_trace_state", {})
        if not isinstance(state, dict):
            state = {}
            self.state["dialog_judge_trace_state"] = state
        if not isinstance(state.get("sessions"), list):
            state["sessions"] = []

    def _ensure_session_eval_state_schema(self) -> None:
        state = self.state.setdefault("session_eval_state", {})
        if not isinstance(state, dict):
            state = {}
            self.state["session_eval_state"] = state
        if not isinstance(state.get("latest_reason_by_pair"), dict):
            state["latest_reason_by_pair"] = {}
        if not isinstance(state.get("history_by_pair"), dict):
            state["history_by_pair"] = {}
        if (
            self.get_cbt_controller_mode() == "minimal"
            and not isinstance(state.get("minimal_by_pair"), dict)
        ):
            state["minimal_by_pair"] = {}
        self._drop_legacy_patient_evidence_fields(state)

    def _ensure_subgoal_shadow_state_schema(self) -> None:
        state = self.state.get("subgoal_shadow_state")
        if not isinstance(state, dict):
            return
        if not isinstance(state.get("latest_by_pair"), dict):
            state["latest_by_pair"] = {}
        if not isinstance(state.get("history_by_pair"), dict):
            state["history_by_pair"] = {}
        self._drop_legacy_patient_evidence_fields(state)

    def _ensure_subgoal_progress_state_schema(self) -> None:
        enabled = bool(self.get_progressive_d_runtime_policy().get("enabled", False))
        if not enabled:
            return
        if not isinstance(self.state.get("subgoal_progress_by_pair"), dict):
            self.state["subgoal_progress_by_pair"] = {}
        self._drop_legacy_patient_evidence_fields(
            self.state["subgoal_progress_by_pair"]
        )

    def _ensure_progressive_d_state_schema(self) -> None:
        if not self._is_progressive_d_v3_configured():
            return
        tracker = self.state.setdefault("progressive_d_tracker_state", {})
        if not isinstance(tracker, dict):
            tracker = {}
            self.state["progressive_d_tracker_state"] = tracker
        if not isinstance(tracker.get("by_pair"), dict):
            tracker["by_pair"] = {}
        risk = self.state.setdefault("risk_check_state_by_pair", {})
        if not isinstance(risk, dict):
            self.state["risk_check_state_by_pair"] = {}
        control = self.state.setdefault("progressive_d_control_state", {})
        if not isinstance(control, dict):
            control = {}
            self.state["progressive_d_control_state"] = control
        if not isinstance(control.get("latest_by_pair"), dict):
            control["latest_by_pair"] = {}
        if not isinstance(control.get("history_by_pair"), dict):
            control["history_by_pair"] = {}

    @classmethod
    def _drop_legacy_patient_evidence_fields(cls, value: Any) -> None:
        """Remove deprecated evaluator quotes before a resumed state is saved."""

        if isinstance(value, dict):
            value.pop("patient_evidence", None)
            for child in value.values():
                cls._drop_legacy_patient_evidence_fields(child)
        elif isinstance(value, list):
            for child in value:
                cls._drop_legacy_patient_evidence_fields(child)

    def _ensure_forced_prompt_trace_state_schema(self) -> None:
        state = self.state.setdefault("forced_prompt_trace_state", {})
        if not isinstance(state, dict):
            state = {}
            self.state["forced_prompt_trace_state"] = state
        if not isinstance(state.get("sessions"), list):
            state["sessions"] = []

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
                "meeting_kind": str(rule.get("meeting_kind", "doctor_consult") or "doctor_consult"),
                "prompt_file": str(rule.get("prompt_file", "") or ""),
                "prompt_target": str(rule.get("prompt_target", "all") or "all"),
                "meeting_source": str(rule.get("meeting_source", "meeting_rules") or "meeting_rules"),
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
            "meeting_kind": str(rule.get("meeting_kind", "doctor_consult") or "doctor_consult"),
            "prompt_file": str(rule.get("prompt_file", "") or ""),
            "prompt_target": str(rule.get("prompt_target", "all") or "all"),
            "meeting_source": str(rule.get("meeting_source", "meeting_rules") or "meeting_rules"),
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

    def _consult_history_enabled(self) -> bool:
        cfg = self.consult_history_cfg if isinstance(self.consult_history_cfg, dict) else {}
        return bool(self.enabled and self._safe_bool(cfg.get("enabled", False), False))

    def _pair_key_to_history_slug(self, pair_key: str) -> str:
        slug = str(pair_key or "").strip().replace("::", "__")
        for old, new in (("/", "_"), ("\\", "_"), (":", "_"), (" ", "_")):
            slug = slug.replace(old, new)
        return slug or "pair_unknown"

    def _get_consult_history_root(self, agent: Any) -> str:
        storage_root = str(getattr(agent, "storage_root", "") or "").strip()
        if not storage_root:
            return ""
        storage_dir = os.path.dirname(storage_root)
        checkpoint_root = os.path.dirname(storage_dir)
        if not checkpoint_root:
            return ""
        return os.path.join(checkpoint_root, "consult_history")

    def _get_consult_history_pair_dir(self, agent: Any, pair_key: str) -> str:
        root = self._get_consult_history_root(agent)
        if not root:
            return ""
        return os.path.join(root, self._pair_key_to_history_slug(pair_key))

    def _get_consult_history_manifest_path(self, agent: Any, pair_key: str) -> str:
        pair_dir = self._get_consult_history_pair_dir(agent, pair_key)
        if not pair_dir:
            return ""
        return os.path.join(pair_dir, "manifest.json")

    def _get_consult_history_index_dir(self, agent: Any, pair_key: str) -> str:
        pair_dir = self._get_consult_history_pair_dir(agent, pair_key)
        if not pair_dir:
            return ""
        return os.path.join(pair_dir, "index")

    def _default_consult_history_manifest(self) -> Dict[str, Any]:
        return {
            "records_by_id": {},
            "meeting_to_record": {},
            "dedup_keys": {},
            "write_audit": [],
            "record_seq": 0,
        }

    def _load_consult_history_manifest(self, agent: Any, pair_key: str) -> Dict[str, Any]:
        manifest = self._default_consult_history_manifest()
        path = self._get_consult_history_manifest_path(agent, pair_key)
        if not path or (not os.path.exists(path)):
            return manifest
        try:
            loaded = utils.load_dict(path)
        except Exception:
            loaded = {}
        if not isinstance(loaded, dict):
            loaded = {}
        for key, default in manifest.items():
            value = loaded.get(key, default)
            if isinstance(default, dict):
                manifest[key] = value if isinstance(value, dict) else copy.deepcopy(default)
            elif isinstance(default, list):
                manifest[key] = value if isinstance(value, list) else list(default)
            else:
                manifest[key] = value
        try:
            manifest["record_seq"] = max(0, int(manifest.get("record_seq", 0) or 0))
        except Exception:
            manifest["record_seq"] = 0
        return manifest

    def _save_consult_history_manifest(self, agent: Any, pair_key: str, manifest: Dict[str, Any]) -> str:
        path = self._get_consult_history_manifest_path(agent, pair_key)
        if not path:
            raise ConsultRecordError(reason="consult_history_manifest_path_missing", retryable=False)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        utils.save_dict(manifest if isinstance(manifest, dict) else self._default_consult_history_manifest(), path)
        return path

    def _consult_history_record_relpath(self, record_id: str) -> str:
        return os.path.join("records", "{}.json".format(str(record_id or "").strip()))

    def _consult_history_record_path(self, agent: Any, pair_key: str, record_id: str) -> str:
        pair_dir = self._get_consult_history_pair_dir(agent, pair_key)
        if not pair_dir:
            return ""
        return os.path.join(pair_dir, self._consult_history_record_relpath(record_id))

    def _next_consult_history_record_id(self, manifest: Dict[str, Any]) -> str:
        seq = 0
        try:
            seq = int((manifest or {}).get("record_seq", 0) or 0)
        except Exception:
            seq = 0
        seq = max(0, seq) + 1
        if isinstance(manifest, dict):
            manifest["record_seq"] = seq
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return "hist_{}_{}".format(ts, str(seq).zfill(3))

    def _get_or_init_consult_history_index(self, agent: Any, pair_key: str) -> Optional[LlamaIndex]:
        index_dir = self._get_consult_history_index_dir(agent, pair_key)
        if not index_dir:
            return None
        cache_key = os.path.normpath(index_dir)
        index_obj = self._consult_history_indexes.get(cache_key)
        if index_obj is not None:
            return index_obj
        embedding_cfg = copy.deepcopy(getattr(agent, "associate_embedding_config", {}) or {})
        if not isinstance(embedding_cfg, dict) or not embedding_cfg:
            raise ConsultRecordError(reason="consult_history_embedding_missing", retryable=False)
        index_obj = LlamaIndex(embedding_cfg, path=index_dir)
        self._consult_history_indexes[cache_key] = index_obj
        return index_obj

    def _load_consult_history_record_by_id(
        self,
        agent: Any,
        pair_key: str,
        record_id: str,
        manifest: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        current_manifest = manifest if isinstance(manifest, dict) else self._load_consult_history_manifest(agent, pair_key)
        records_by_id = current_manifest.get("records_by_id", {}) if isinstance(current_manifest, dict) else {}
        if not isinstance(records_by_id, dict):
            return None
        rel_path = str(records_by_id.get(str(record_id or "").strip(), "") or "").strip()
        if not rel_path:
            return None
        pair_dir = self._get_consult_history_pair_dir(agent, pair_key)
        if not pair_dir:
            return None
        abs_path = os.path.join(pair_dir, rel_path)
        if not os.path.exists(abs_path):
            return None
        try:
            loaded = utils.load_dict(abs_path)
        except Exception:
            return None
        return loaded if isinstance(loaded, dict) else None

    def _rebuild_consult_history_index_from_manifest(
        self,
        agent: Any,
        pair_key: str,
        manifest: Dict[str, Any],
    ) -> None:
        index_obj = self._get_or_init_consult_history_index(agent, pair_key)
        if index_obj is None:
            return
        records_by_id = manifest.get("records_by_id", {}) if isinstance(manifest, dict) else {}
        if not isinstance(records_by_id, dict):
            return
        rebuilt = 0
        for record_id in records_by_id.keys():
            record_id = str(record_id or "").strip()
            if not record_id:
                continue
            if index_obj.has_node(record_id):
                continue
            record = self._load_consult_history_record_by_id(agent, pair_key, record_id, manifest=manifest)
            if not isinstance(record, dict):
                continue
            chat_summary = str(record.get("chat_summary", "") or "").strip()
            if not chat_summary:
                continue
            participants = record.get("participants", {}) if isinstance(record.get("participants", {}), dict) else {}
            metadata = {
                "meeting_id": str(record.get("meeting_id", "") or "").strip(),
                "pair_key": str(record.get("pair_key", "") or "").strip(),
                "doctor": str(participants.get("doctor", "") or "").strip(),
                "patient": str(participants.get("patient", "") or "").strip(),
                "session_started_at": str(record.get("session_started_at", "") or "").strip(),
            }
            index_obj.add_node(chat_summary, metadata=metadata, id=record_id)
            rebuilt += 1
        if rebuilt > 0:
            index_obj.save()
            self._log_highlight(
                "CONSULT_HISTORY_INDEX_REBUILD pair_key={} rebuilt_nodes={}".format(
                    pair_key,
                    rebuilt,
                )
            )

    def _append_consult_history_audit(
        self,
        manifest: Dict[str, Any],
        status: str,
        reason: str,
        record_id: str = "",
        meeting_id: str = "",
        message: str = "",
    ) -> None:
        if not isinstance(manifest, dict):
            return
        audit = manifest.setdefault("write_audit", [])
        if not isinstance(audit, list):
            audit = []
            manifest["write_audit"] = audit
        audit.append(
            {
                "ts": self._fmt_dt(utils.get_timer().get_date()),
                "status": str(status or ""),
                "reason": str(reason or ""),
                "record_id": str(record_id or ""),
                "meeting_id": str(meeting_id or ""),
                "message": str(message or ""),
            }
        )
        if len(audit) > 2000:
            manifest["write_audit"] = audit[-2000:]

    def _write_consult_history_record(
        self,
        agent: Any,
        pair_key: str,
        record: Dict[str, Any],
        summary: str,
    ) -> Dict[str, Any]:
        manifest = self._load_consult_history_manifest(agent, pair_key)
        meeting_to_record = manifest.setdefault("meeting_to_record", {})
        if not isinstance(meeting_to_record, dict):
            meeting_to_record = {}
            manifest["meeting_to_record"] = meeting_to_record
        dedup_keys = manifest.setdefault("dedup_keys", {})
        if not isinstance(dedup_keys, dict):
            dedup_keys = {}
            manifest["dedup_keys"] = dedup_keys
        records_by_id = manifest.setdefault("records_by_id", {})
        if not isinstance(records_by_id, dict):
            records_by_id = {}
            manifest["records_by_id"] = records_by_id

        prepared = dict(record) if isinstance(record, dict) else {}
        record_id = str(prepared.get("record_id", "") or "").strip() or self._next_consult_history_record_id(manifest)
        prepared["record_id"] = record_id
        current = validate_consult_history_record(prepared)

        meeting_id = str(current.get("meeting_id", "") or "").strip()
        dedup_key = build_dedup_key(meeting_id, summary or current.get("chat_summary", ""))
        if meeting_id and meeting_id in meeting_to_record:
            record_id = str(meeting_to_record.get(meeting_id, "") or "")
            self._append_consult_history_audit(manifest, "skipped", "idempotent_hit_meeting", record_id=record_id, meeting_id=meeting_id)
            self._save_consult_history_manifest(agent, pair_key, manifest)
            return {"status": "skipped", "reason": "idempotent_hit_meeting", "record_id": record_id}
        if dedup_key in dedup_keys:
            record_id = str(dedup_keys.get(dedup_key, "") or "")
            self._append_consult_history_audit(manifest, "skipped", "dedup_key_hit", record_id=record_id, meeting_id=meeting_id)
            self._save_consult_history_manifest(agent, pair_key, manifest)
            return {"status": "skipped", "reason": "dedup_key_hit", "record_id": record_id}

        rel_path = self._consult_history_record_relpath(record_id)
        abs_path = self._consult_history_record_path(agent, pair_key, record_id)
        if not abs_path:
            raise ConsultRecordError(reason="consult_history_record_path_missing", retryable=False)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        utils.save_dict(current, abs_path)

        records_by_id[record_id] = rel_path
        if meeting_id:
            meeting_to_record[meeting_id] = record_id
        dedup_keys[dedup_key] = record_id

        write_reason = "write_success"
        try:
            index_obj = self._get_or_init_consult_history_index(agent, pair_key)
            if index_obj is None:
                raise ConsultRecordError(reason="consult_history_index_unavailable", retryable=False)
            if not index_obj.has_node(record_id):
                participants = current.get("participants", {}) if isinstance(current.get("participants", {}), dict) else {}
                index_obj.add_node(
                    str(current.get("chat_summary", "") or ""),
                    metadata={
                        "meeting_id": meeting_id,
                        "pair_key": str(current.get("pair_key", "") or "").strip(),
                        "doctor": str(participants.get("doctor", "") or "").strip(),
                        "patient": str(participants.get("patient", "") or "").strip(),
                        "session_started_at": str(current.get("session_started_at", "") or "").strip(),
                    },
                    id=record_id,
                )
                index_obj.save()
        except Exception as exc:
            write_reason = "write_success_index_pending"
            self._log_highlight(
                "CONSULT_HISTORY_INDEX_DEFER pair_key={} record_id={} detail={}".format(
                    pair_key,
                    record_id,
                    str(exc),
                )
            )

        self._append_consult_history_audit(manifest, "success", write_reason, record_id=record_id, meeting_id=meeting_id)
        self._save_consult_history_manifest(agent, pair_key, manifest)
        return {"status": "success", "reason": write_reason, "record_id": record_id}

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

    def _call_forced_llm_json(
        self,
        prompt_text: str,
        retry: int,
        caller: str = "consult_record_forced_llm",
    ) -> Dict[str, Any]:
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
            "LLM_CALL start provider={} model={} retry={} caller={}".format(
                runtime_cfg.get("provider", ""),
                runtime_cfg.get("model", ""),
                retry_count,
                str(caller or "consult_record_forced_llm"),
            )
        )
        payload = self._consult_record_llm.completion(
            prompt_text,
            retry=retry_count,
            callback=self._json_loads_loose,
            failsafe=None,
            caller=str(caller or "consult_record_forced_llm"),
            temperature=float(runtime_cfg.get("temperature", 0.5) or 0.5),
        )
        if not isinstance(payload, dict):
            self._log_consult("LLM_CALL end success=false reason=json_parse_failed")
            raise ConsultRecordError(reason="json_parse_failed", retryable=False)
        self._log_consult("LLM_CALL end success=true")
        return payload

    def _call_forced_llm_text(
        self,
        prompt_text: str,
        retry: int,
        caller: str = "consult_history_forced_llm",
    ) -> str:
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
        payload = self._consult_record_llm.completion(
            prompt_text,
            retry=retry_count,
            failsafe="",
            caller=str(caller or "consult_history_forced_llm"),
            temperature=float(runtime_cfg.get("temperature", 0.5) or 0.5),
        )
        text = str(payload or "").strip()
        if not text:
            raise ConsultRecordError(reason="forced_llm_empty_text", retryable=False)
        return text

    def _extract_latest_other_utterance(self, chats: Any, other_name: str) -> str:
        target = str(other_name or "").strip()
        if not target:
            return ""
        for item in reversed(list(chats or [])):
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            speaker = str(item[0] or "").strip()
            text = str(item[1] or "").strip()
            if speaker == target and text:
                return text
        return ""

    def _normalize_consult_history_gate_output(
        self,
        payload: Any,
        latest_utterance: str,
    ) -> Dict[str, Any]:
        raw = payload if isinstance(payload, dict) else {}
        need = raw.get("need_retrieval")
        if isinstance(need, bool):
            need_retrieval = need
        elif isinstance(need, str):
            lowered = need.strip().lower()
            if lowered == "true":
                need_retrieval = True
            elif lowered == "false":
                need_retrieval = False
            else:
                need_retrieval = False
        else:
            need_retrieval = False

        raw_query = raw.get("query", "")
        query = raw_query.strip() if isinstance(raw_query, str) else ""
        if need_retrieval and (not query):
            query = str(latest_utterance or "").strip()
        if not need_retrieval:
            query = ""

        raw_reason = raw.get("reason", "")
        reason = raw_reason.strip() if isinstance(raw_reason, str) else ""
        return {
            "need_retrieval": bool(need_retrieval),
            "query": query,
            "reason": reason,
        }

    def _consult_history_gate_with_trace(
        self,
        speaker: Any,
        other: Any,
        chats: Any,
        latest_utterance: str,
        policy: Dict[str, Any],
        pair_key: str,
    ) -> Dict[str, Any]:
        prompt_tpl = self._load_prompt_txt_or_raise(str(policy.get("gate_prompt_file", "") or ""))
        prompt_text = self._render_prompt_template(
            prompt_tpl,
            {
                "speaker": str(getattr(speaker, "name", "") or ""),
                "other": str(getattr(other, "name", "") or ""),
                "pair_key": str(pair_key or ""),
                "latest_utterance": str(latest_utterance or ""),
                "conversation": to_conversation_text(chats or []),
            },
        )
        try:
            raw = self._call_think_llm_json(
                prompt_text=prompt_text,
                retry=int(policy.get("gate_retry", 2) or 2),
                doctor_agent=speaker,
                caller="consult_history_gate",
            )
        except ConsultRecordError as exc:
            self._log_highlight(
                "CONSULT_HISTORY_GATE_FALLBACK speaker={} other={} reason={}".format(
                    getattr(speaker, "name", ""),
                    getattr(other, "name", ""),
                    str(exc.reason or "json_parse_failed"),
                )
            )
            raw = {"need_retrieval": False, "query": "", "reason": str(exc.reason or "json_parse_failed")}
        normalized = self._normalize_consult_history_gate_output(raw, latest_utterance)
        self._log_highlight(
            "CONSULT_HISTORY_GATE speaker={} other={} need_retrieval={} query_len={} reason={}".format(
                getattr(speaker, "name", ""),
                getattr(other, "name", ""),
                bool(normalized.get("need_retrieval", False)),
                len(str(normalized.get("query", "") or "")),
                normalized.get("reason", ""),
            )
        )
        return {
            "prompt_text": str(prompt_text or ""),
            "gate_output": copy.deepcopy(normalized if isinstance(normalized, dict) else {}),
        }

    def _retrieve_consult_history_candidates(
        self,
        agent: Any,
        pair_key: str,
        query: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        if not str(query or "").strip():
            return []
        manifest = self._load_consult_history_manifest(agent, pair_key)
        records_by_id = manifest.get("records_by_id", {}) if isinstance(manifest, dict) else {}
        if not isinstance(records_by_id, dict) or not records_by_id:
            return []
        index_obj = self._get_or_init_consult_history_index(agent, pair_key)
        if index_obj is None:
            return []
        self._rebuild_consult_history_index_from_manifest(agent, pair_key, manifest)
        node_ids = [str(rid or "").strip() for rid in records_by_id.keys() if str(rid or "").strip()]
        if not node_ids:
            return []
        similarity_top_k = max(1, min(int(top_k or 1), len(node_ids)))
        nodes = index_obj.retrieve(str(query or "").strip(), similarity_top_k=similarity_top_k, node_ids=node_ids)
        results: List[Dict[str, Any]] = []
        for node in nodes or []:
            record_id = str(getattr(node, "id_", "") or "").strip()
            if not record_id:
                continue
            record = self._load_consult_history_record_by_id(agent, pair_key, record_id, manifest=manifest)
            if not isinstance(record, dict):
                continue
            try:
                score = float(getattr(node, "score", 0.0) or 0.0)
            except Exception:
                score = 0.0
            results.append({
                "record_id": record_id,
                "score": score,
                "record": record,
            })
        self._log_highlight(
            "CONSULT_HISTORY_RETRIEVE pair_key={} query_len={} hits={} top_k={}".format(
                str(pair_key or ""),
                len(str(query or "")),
                len(results),
                similarity_top_k,
            )
        )
        return results

    def _render_consult_history_hits_for_prompt(self, hits: List[Dict[str, Any]]) -> str:
        blocks: List[str] = []
        for idx, item in enumerate(hits or [], start=1):
            record = item.get("record", {}) if isinstance(item, dict) else {}
            if not isinstance(record, dict):
                continue
            participants = record.get("participants", {}) if isinstance(record.get("participants", {}), dict) else {}
            blocks.append(
                "[命中记录 {}]\nrecord_id={}\nmeeting_id={}\npair_key={}\ndoctor={}\npatient={}\nsession_started_at={}\nscore={}\nchat_summary={}\ntranscript=\n{}".format(
                    idx,
                    str(record.get("record_id", "") or "").strip(),
                    str(record.get("meeting_id", "") or "").strip(),
                    str(record.get("pair_key", "") or "").strip(),
                    str(participants.get("doctor", "") or "").strip(),
                    str(participants.get("patient", "") or "").strip(),
                    str(record.get("session_started_at", "") or "").strip(),
                    str(item.get("score", 0.0)),
                    str(record.get("chat_summary", "") or "").strip(),
                    str(record.get("transcript", "") or "").strip(),
                )
            )
        return "\n\n".join(blocks).strip()

    def _build_consult_history_trace_hits(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        trace_hits: List[Dict[str, Any]] = []
        for idx, item in enumerate(hits or [], start=1):
            item_dict = item if isinstance(item, dict) else {}
            record = item_dict.get("record", {})
            if not isinstance(record, dict):
                continue
            score = item_dict.get("score", 0.0)
            try:
                score = float(score or 0.0)
            except Exception:
                score = 0.0
            trace_hits.append(
                {
                    "rank": idx,
                    "record_id": str(record.get("record_id", "") or item_dict.get("record_id", "") or "").strip(),
                    "session_started_at": str(record.get("session_started_at", "") or "").strip(),
                    "chat_summary": str(record.get("chat_summary", "") or "").strip(),
                    "score": score,
                }
            )
        return trace_hits

    def _summarize_consult_history_hits_with_trace(
        self,
        speaker: Any,
        other: Any,
        chats: Any,
        pair_key: str,
        query: str,
        hits: List[Dict[str, Any]],
        policy: Dict[str, Any],
    ) -> Dict[str, Any]:
        prompt_tpl = self._load_prompt_txt_or_raise(str(policy.get("summary_prompt_file", "") or ""))
        prompt_text = self._render_prompt_template(
            prompt_tpl,
            {
                "speaker": str(getattr(speaker, "name", "") or ""),
                "other": str(getattr(other, "name", "") or ""),
                "pair_key": str(pair_key or ""),
                "query": str(query or "").strip(),
                "conversation": to_conversation_text(chats or []),
                "consult_history_hits": self._render_consult_history_hits_for_prompt(hits),
            },
        )
        route = str(policy.get("summary_route", "forced_llm") or "forced_llm").strip().lower()
        retry = int(policy.get("summary_retry", 2) or 2)
        if route == "think_llm":
            text = self._call_think_llm_text(
                prompt_text=prompt_text,
                retry=retry,
                doctor_agent=speaker,
                caller="consult_history_summary",
            )
        else:
            text = self._call_forced_llm_text(
                prompt_text=prompt_text,
                retry=retry,
                caller="consult_history_summary_forced_llm",
            )
        text = str(text or "").strip()
        memory_block = ""
        if text:
            memory_block = "<consult_history_memory>\n{}\n</consult_history_memory>".format(text)
        return {
            "prompt_text": str(prompt_text or ""),
            "summary_output": text,
            "memory_block": memory_block,
        }

    def _build_consult_history_trace_context(
        self,
        speaker: Any,
        other: Any,
        chats: Any,
        forced: bool = False,
        turn_no: int = 1,
    ) -> Dict[str, Any]:
        trace_context: Dict[str, Any] = {
            "evaluated": False,
            "memory_block": "",
            "trace_prompt_text": "",
            "gate_prompt_text": "",
            "summary_prompt_text": "",
            "gate_output": {},
            "retrieval_hits": [],
            "summary_output": "",
        }
        if not bool(forced):
            return trace_context
        policy = self.get_consult_history_runtime_policy()
        if not bool(policy.get("enabled", False)):
            return trace_context
        doctor, patient = self._resolve_doctor_patient_pair(speaker, other)
        if not doctor or not patient:
            return trace_context
        if int(turn_no or 1) <= 1:
            return trace_context
        pair_key = build_pair_key(str(getattr(doctor, "name", "") or ""), str(getattr(patient, "name", "") or ""))
        latest_utterance = self._extract_latest_other_utterance(chats, getattr(other, "name", ""))
        if not latest_utterance:
            return trace_context
        gate_trace = self._consult_history_gate_with_trace(
            speaker=speaker,
            other=other,
            chats=chats,
            latest_utterance=latest_utterance,
            policy=policy,
            pair_key=pair_key,
        )
        gate_output = gate_trace.get("gate_output", {}) if isinstance(gate_trace, dict) else {}
        gate_prompt_text = str(gate_trace.get("prompt_text", "") or "") if isinstance(gate_trace, dict) else ""
        trace_context["evaluated"] = True
        trace_context["gate_output"] = copy.deepcopy(gate_output if isinstance(gate_output, dict) else {})
        trace_context["gate_prompt_text"] = gate_prompt_text
        trace_prompt_sections = []
        if gate_prompt_text:
            trace_prompt_sections.append("[consult_history_gate_prompt]\n{}".format(gate_prompt_text))
        if not bool((gate_output or {}).get("need_retrieval", False)):
            trace_context["trace_prompt_text"] = "\n\n".join(trace_prompt_sections).strip()
            return trace_context
        query = str((gate_output or {}).get("query", "") or "").strip()
        if not query:
            trace_context["trace_prompt_text"] = "\n\n".join(trace_prompt_sections).strip()
            return trace_context
        hits = self._retrieve_consult_history_candidates(
            agent=speaker,
            pair_key=pair_key,
            query=query,
            top_k=int(policy.get("retrieve_top_k", 3) or 3),
        )
        trace_context["retrieval_hits"] = self._build_consult_history_trace_hits(hits)
        if not hits:
            trace_context["trace_prompt_text"] = "\n\n".join(trace_prompt_sections).strip()
            return trace_context
        summary_trace = self._summarize_consult_history_hits_with_trace(
            speaker=speaker,
            other=other,
            chats=chats,
            pair_key=pair_key,
            query=query,
            hits=hits,
            policy=policy,
        )
        summary_prompt_text = str(summary_trace.get("prompt_text", "") or "") if isinstance(summary_trace, dict) else ""
        if summary_prompt_text:
            trace_prompt_sections.append("[consult_history_summary_prompt]\n{}".format(summary_prompt_text))
        trace_context["summary_prompt_text"] = summary_prompt_text
        trace_context["summary_output"] = str(summary_trace.get("summary_output", "") or "") if isinstance(summary_trace, dict) else ""
        trace_context["memory_block"] = str(summary_trace.get("memory_block", "") or "") if isinstance(summary_trace, dict) else ""
        trace_context["trace_prompt_text"] = "\n\n".join(trace_prompt_sections).strip()
        return trace_context

    def get_consult_history_memory_block(
        self,
        speaker: Any,
        other: Any,
        chats: Any,
        forced: bool = False,
        turn_no: int = 1,
        is_initiator: bool = False,
    ) -> str:
        del is_initiator
        trace_context = self._build_consult_history_trace_context(
            speaker=speaker,
            other=other,
            chats=chats,
            forced=forced,
            turn_no=turn_no,
        )
        return str(trace_context.get("memory_block", "") or "") if isinstance(trace_context, dict) else ""

    def _handle_consult_history_after_chat(
        self,
        speaker: Any,
        other: Any,
        chats: Any,
        summary: str,
        start_time: Any,
        meeting_id: str,
    ) -> None:
        if not self._consult_history_enabled():
            return
        if not str(meeting_id or "").strip():
            return
        doctor, patient = self._resolve_doctor_patient_pair(speaker, other)
        if not doctor or not patient:
            return
        pair_key = build_pair_key(str(getattr(doctor, "name", "") or ""), str(getattr(patient, "name", "") or ""))
        transcript = to_conversation_text(chats or [])
        chat_summary = str(summary or "").strip()
        if not transcript or not chat_summary:
            return
        try:
            record = build_consult_history_record(
                record_id="",
                meeting_id=str(meeting_id or "").strip(),
                session_started_at=self._fmt_iso8601_with_tz(start_time),
                doctor=str(getattr(doctor, "name", "") or ""),
                patient=str(getattr(patient, "name", "") or ""),
                pair_key=pair_key,
                chat_summary=chat_summary,
                transcript=transcript,
            )
            write_result = self._write_consult_history_record(
                agent=speaker,
                pair_key=pair_key,
                record=record,
                summary=chat_summary,
            )
            self._log_highlight(
                "CONSULT_HISTORY_WRITE pair_key={} status={} reason={} record_id={} meeting_id={}".format(
                    pair_key,
                    write_result.get("status", ""),
                    write_result.get("reason", ""),
                    write_result.get("record_id", ""),
                    str(meeting_id or ""),
                )
            )
        except Exception as exc:
            self._log_highlight(
                "CONSULT_HISTORY_WRITE_ERROR pair_key={} meeting_id={} detail={}".format(
                    pair_key,
                    str(meeting_id or ""),
                    str(exc),
                )
            )

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

    def _next_consult_record_id(self) -> str:
        self._ensure_consult_record_state_schema()
        state = self.state.setdefault("consult_record_state", {})
        records = state.setdefault("records_by_id", {})
        seq = len(records) + 1
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return "rec_{}_{}".format(ts, str(seq).zfill(3))

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

            soap_text_max_len = self._resolve_consult_soap_text_max_len()
            structure_retry = self._safe_int(consult_cfg.get("structure_retry", 1), 1)
            if structure_retry < 0:
                structure_retry = 0
            dropped: List[str] = []
            attempt = 0
            while True:
                attempt += 1
                llm_raw = self._call_forced_llm_json(
                    prompt_text,
                    retry=retry,
                    caller="consult_record_generate",
                )
                projected, dropped = project_whitelist(llm_raw)
                self._log_consult(
                    "PARSE project done attempt={} dropped_count={} dropped_paths={}".format(
                        attempt,
                        len(dropped),
                        "|".join(dropped) if dropped else "-",
                    )
                )
                try:
                    validated_soap = validate_soap_only(projected, soap_text_max_len=soap_text_max_len)
                    self._log_consult(
                        "VALIDATE soap success pre_max_len={} attempt={}".format(
                            soap_text_max_len,
                            attempt,
                        )
                    )
                    break
                except ConsultRecordValidationError as err:
                    reason = str(getattr(err, "reason", ""))
                    if reason != "soap_structure_invalid":
                        raise
                    if attempt > (structure_retry + 1):
                        self._append_consult_audit(
                            pair_key=pair_key,
                            meeting_id=str(meeting_id or ""),
                            record_id="",
                            status="failed",
                            reason="soap_structure_invalid",
                            message="validate_failed_after_retry attempts={}".format(attempt),
                            retryable=False,
                        )
                        self._log_consult(
                            "VALIDATE soap failed reason=soap_structure_invalid attempts={} max_retry={}".format(
                                attempt,
                                structure_retry,
                            )
                        )
                        return
                    self._log_consult(
                        "VALIDATE soap retry attempt={} reason=soap_structure_invalid".format(
                            attempt,
                        )
                    )
                    continue

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
