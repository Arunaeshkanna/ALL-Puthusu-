from app.agents.common import invoke_prompt, make_agent


report_writer_agent = make_agent(
    role="Document-Based Report Writer",
    goal="Write clear final answers using only retrieved source context.",
    backstory="You produce concise, well-structured, source-grounded answers.",
)


NOT_FOUND_MESSAGE = "The answer is not available in the uploaded documents."


def write_report(
    question: str,
    context: str,
    sources: list[str],
    verification: str,
    answer_style: str = "detailed",
) -> dict:
    if not context.strip():
        return {
            "answer": NOT_FOUND_MESSAGE,
            "confidence": 0.0,
        }

    detail_rules = """
- Write a detailed answer, not a one-line response.
- Start with a direct answer or short summary.
- Then explain the important points in separate paragraphs or bullet points.
- Include relevant details, definitions, steps, comparisons, examples, or conditions found in the context.
- If the context contains partial information, explain what is available and what is missing.
- Aim for 3-6 well-developed paragraphs or 8-14 useful bullet points when the context supports it.
"""
    concise_rules = """
- Keep the answer clear and precise.
- Use 5-10 lines when possible.
"""

    prompt = f"""
You are a STRICT document-based assistant.

Rules:
- Answer ONLY using the retrieved context.
- If the answer is not clearly present, say exactly:
  "{NOT_FOUND_MESSAGE}"
- Do not use external knowledge.
- Mention source names naturally only when useful.
{detail_rules if answer_style == "detailed" else concise_rules}

Question:
{question}

Verification:
{verification}

Retrieved context:
{context}

Sources:
{", ".join(sources) if sources else "None"}

Final answer:
"""
    answer = invoke_prompt(prompt)
    failed = NOT_FOUND_MESSAGE.lower() in answer.lower()
    return {
        "answer": answer,
        "confidence": 0.0 if failed else 0.85,
    }
