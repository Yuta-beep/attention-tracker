def build_chat_prompt(tokenizer, instruction: str, user_text: str, system_role_supported: bool = True) -> str:
    if system_role_supported:
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_text},
        ]
    else:
        messages = [
            {"role": "user", "content": f"Instruction:\n{instruction}\n\nText:\n{user_text}"},
        ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_injected_user_text(base_text: str, injection_text: str, separator: str = "\n\n") -> str:
    return f"{base_text}{separator}{injection_text}"
