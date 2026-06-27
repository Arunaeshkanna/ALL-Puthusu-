from app.agents.common import invoke_prompt, make_agent


verifier_agent = make_agent(
    role="Grounding Verifier",
    goal="Check whether retrieved context is sufficient to answer the question.",
    backstory="You are strict about evidence and reject answers that are not grounded in sources.",
)


def verify_context(question: str, context: str) -> dict:
    if not context.strip():
        return {
            "is_grounded": False,
            "verification": "No relevant document context was retrieved.",
        }

    prompt = f"""
You are a strict verifier for a document-grounded assistant.

Question:
{question}

Retrieved context:
{context}

Can the question be answered from this context?
Return exactly:
GROUNDED: yes/no
REASON: one short sentence
"""
    verification = invoke_prompt(prompt)
    is_grounded = "grounded: yes" in verification.lower()
    return {"is_grounded": is_grounded, "verification": verification}

