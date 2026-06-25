def build_chat_prompt(
    tokenizer,
    instruction: str,
    user_text: str,
    uses_chat_template: bool = True,
    system_role_supported: bool = True,
    reasoning_enabled: bool = False,
) -> str:
    if not uses_chat_template:
        return f"Instruction:\n{instruction}\n\nText:\n{user_text}"

    if system_role_supported:
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_text},
        ]
    else:
        messages = [
            {"role": "user", "content": f"Instruction:\n{instruction}\n\nText:\n{user_text}"},
        ]

    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if reasoning_enabled:
        kwargs["enable_thinking"] = True
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def build_injected_user_text(base_text: str, injection_text: str, separator: str = "\n\n") -> str:
    return f"{base_text}{separator}{injection_text}"
