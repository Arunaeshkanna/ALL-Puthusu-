from app.agents.common import invoke_prompt, make_agent


query_rewriter_agent = make_agent(
    role="Query Optimization Expert",
    goal="Rewrite user questions into clear retrieval queries.",
    backstory="You improve search queries while preserving the user's original intent.",
)


def rewrite_query(question: str, plan: str) -> dict:
    prompt = f"""
Rewrite the question for semantic document retrieval.

Plan:
{plan}

Original question:
{question}

Rules:
- Preserve exact intent.
- Add key entities from the plan if useful.
- Return only the improved query.
"""
    return {"refined_query": invoke_prompt(prompt)}

