import os
os.environ["OPENAI_API_KEY"] = "sk-6cb4c63b197d463d94969ec253cec887"  ### DeepSeek API Key ###

class ExpertLLM:
    def __init__(self, api_key=None, model="deepseek-chat", base_url="https://api.deepseek.com", timeout=60):
        from openai import OpenAI

        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError("Missing OpenAI API key. Set OPENAI_API_KEY or pass api_key.")

        self._model = model
        self._client = OpenAI(api_key=resolved_key, base_url=base_url, timeout=timeout)

    def generate(self, user_prompt, system_prompt=None, temperature=0.2, **kwargs):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            **kwargs,
        )
        if response.choices:
            return response.choices[0].message.content
        return ""

    def chat(self, messages, system_prompt=None, temperature=0.2, **kwargs):
        if not isinstance(messages, list):
            raise TypeError("messages must be a list of role/content dicts.")

        merged = []
        if system_prompt:
            merged.append({"role": "system", "content": system_prompt})
        merged.extend(messages)

        response = self._client.chat.completions.create(
            model=self._model,
            messages=merged,
            temperature=temperature,
            **kwargs,
        )
        if response.choices:
            return response.choices[0].message.content
        return ""
