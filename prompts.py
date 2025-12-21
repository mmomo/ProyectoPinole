from rules import SWIFT_REVIEW_RULES

REVIEW_PROMPT = f"""
Analyze the following Swift code using the rules.

Rules:
{SWIFT_REVIEW_RULES}

Respond ONLY in this format:

ISSUES:
- <issue>
- <code fragment with that issue and where is located (file and line number)>

SUGGESTIONS:
- <suggestion>
- <example of how to fix it>

POSITIVE NOTES:
- <positive>

"""

SYSTEM_PROMPTS = {
    "junior": """
You are a patient senior iOS engineer.
Explain the code step by step.
Avoid jargon.
Explain what the code does and why.
""",

    "senior": """
You are an experienced iOS engineer.
Explain the intent of the code.
Focus on tradeoffs, design choices, and potential issues.
Assume the reader knows Swift.
""",

    "review": """
You are an iOS code reviewer.
Do not explain line by line.
Evaluate code quality, clarity, and risks.
Be concise and constructive.
"""
}

def build_user_prompt(code: str) -> str:
    return f"""
            Analyze the following Swift code:

            {code}
            """