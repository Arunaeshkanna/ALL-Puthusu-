from app.agents.common import invoke_prompt, make_agent


reflector_agent = make_agent(
    role="Retrieval Reflection Agent",
    goal="Diagnose failed retrieval and create a better follow-up query.",
    backstory="You inspect weak evidence and reformulate searches to recover missing facts.",
)


def reflect_query(question: str, refined_query: str, verification: str) -> dict:
    prompt = f"""
The retrieval was not sufficient.

Original question:
{question}

Previous retrieval query:
{refined_query}

Verifier feedback:
{verification}

Create one better retrieval query. Return only the query.
"""
    return {"refined_query": invoke_prompt(prompt)}

