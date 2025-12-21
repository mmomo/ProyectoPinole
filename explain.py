from ollama import chat
from prompts import SYSTEM_PROMPTS, build_user_prompt

def explain_code(code: str, mode: str):
    response = chat(
        model="qwen2.5-coder:3b",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPTS[mode]
            },
            {
                "role": "user",
                "content": build_user_prompt(code)
            }
        ]
    )

    return response["message"]["content"]

if __name__ == "__main__":
    with open("DashboardView.swift") as f:
        code = f.read()

    for mode in ["junior", "senior", "review"]:
        print(f"\n===== {mode.upper()} ======\n")
        print(explain_code(code, mode))
        print("\n")