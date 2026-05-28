"""Prompt template helpers for the depression module."""

from __future__ import annotations

import os
import copy
import json
from string import Template
from typing import Any, Dict, Optional


class DepressionPromptTemplates:
    """Load depression prompt templates from data/prompts."""

    def __init__(self, prompt_root: Optional[str] = None):
        if prompt_root:
            self.prompt_root = os.path.abspath(prompt_root)
        else:
            self.prompt_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "data", "prompts")
            )

    def render(self, template: str, data: Optional[Dict[str, Any]] = None) -> str:
        content = self._load_text(template)
        if not content:
            return ""

        try:
            return Template(content).safe_substitute(data if isinstance(data, dict) else {})
        except Exception:
            return content

    def render_section(
        self,
        template: str,
        section: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> str:
        content = self._load_text(template)
        section_content = self._extract_section(content, section)
        if not section_content:
            return ""

        try:
            return Template(section_content).safe_substitute(data if isinstance(data, dict) else {})
        except Exception:
            return section_content

    def _load_text(self, template: str) -> str:
        path = self._data_path(template, ".txt")
        try:
            with open(path, "r", encoding="utf-8") as file:
                return file.read()
        except (FileNotFoundError, UnicodeDecodeError, OSError):
            return ""

    @staticmethod
    def _extract_section(content: str, section: str) -> str:
        target = str(section or "").strip()
        if not target:
            return ""

        current = ""
        lines = []
        for line in str(content or "").splitlines():
            stripped = line.strip()
            if stripped.startswith("@@ "):
                if current == target:
                    break
                current = stripped[3:].strip()
                continue
            if current == target:
                lines.append(line)
        return "\n".join(lines).strip("\n")

    def load_json(self, template: str, default: Optional[Any] = None) -> Any:
        path = self._data_path(template, ".json")
        try:
            with open(path, "r", encoding="utf-8") as file:
                data = json.load(file)
            return data
        except (FileNotFoundError, UnicodeDecodeError, OSError, json.JSONDecodeError):
            return copy.deepcopy(default)

    def _data_path(self, template: str, suffix: str) -> str:
        name = str(template or "").strip()
        if not name.endswith(suffix):
            name = f"{name}{suffix}"
        path = os.path.abspath(os.path.join(self.prompt_root, name))
        if not path.startswith(self.prompt_root + os.sep):
            return os.path.join(self.prompt_root, f"__invalid_template__{suffix}")
        return path


_DEFAULT_STORE = DepressionPromptTemplates()


def render_prompt(template: str, data: Optional[Dict[str, Any]] = None) -> str:
    return _DEFAULT_STORE.render(template, data)


def render_prompt_section(
    template: str,
    section: str,
    data: Optional[Dict[str, Any]] = None,
) -> str:
    return _DEFAULT_STORE.render_section(template, section, data)


def load_prompt_json(template: str, default: Optional[Any] = None) -> Any:
    return _DEFAULT_STORE.load_json(template, default)
