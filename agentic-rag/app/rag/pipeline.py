from app.graph.workflow import run_agentic_rag


def generate_strict_answer(query: str, selected_source: str | None = None):
    result = run_agentic_rag(
        query,
        force_single_pass=True,
        answer_style="detailed",
        selected_source=selected_source,
    )
    return result["answer"], result["sources"], result["confidence"]
