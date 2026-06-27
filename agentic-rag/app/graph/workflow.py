from typing import TypedDict

from langgraph.graph import END, StateGraph

from app.agents.planner import plan_query
from app.agents.query_rewriter import rewrite_query
from app.agents.reflector import reflect_query
from app.agents.report_writer import write_report
from app.agents.retriever import retrieve_context
from app.agents.verifier import verify_context
from app.config import MAX_REFLECTIONS
from app.memory.memory import memory


class AgenticRagState(TypedDict, total=False):
    question: str
    chat_history: str
    plan: str
    refined_query: str
    context: str
    sources: list[str]
    retrieved_count: int
    is_grounded: bool
    verification: str
    reflections: int
    reflection_attempts: list[str]
    answer: str
    confidence: float
    force_single_pass: bool
    answer_style: str
    selected_source: str | None
    final_reasoning_summary: str


def _plan_node(state: AgenticRagState) -> AgenticRagState:
    result = plan_query(state["question"], state.get("chat_history", ""))
    return {**state, **result}


def _rewrite_node(state: AgenticRagState) -> AgenticRagState:
    result = rewrite_query(state["question"], state["plan"])
    return {**state, **result}


def _retrieve_node(state: AgenticRagState) -> AgenticRagState:
    result = retrieve_context(
        state["refined_query"],
        source_filter=state.get("selected_source"),
    )
    return {**state, **result}


def _verify_node(state: AgenticRagState) -> AgenticRagState:
    result = verify_context(state["question"], state.get("context", ""))
    return {**state, **result}


def _reflect_node(state: AgenticRagState) -> AgenticRagState:
    result = reflect_query(
        state["question"],
        state["refined_query"],
        state.get("verification", ""),
    )
    return {
        **state,
        **result,
        "reflections": state.get("reflections", 0) + 1,
        "reflection_attempts": [
            *state.get("reflection_attempts", []),
            result.get("refined_query", ""),
        ],
    }


def _write_node(state: AgenticRagState) -> AgenticRagState:
    result = write_report(
        state["question"],
        state.get("context", ""),
        state.get("sources", []),
        state.get("verification", ""),
        state.get("answer_style", "detailed"),
    )
    final_summary = _build_final_reasoning_summary({**state, **result})
    return {**state, **result, "final_reasoning_summary": final_summary}


def _build_final_reasoning_summary(state: AgenticRagState) -> str:
    source_scope = state.get("selected_source") or "all indexed documents"
    source_text = ", ".join(state.get("sources", [])) or "no retrieved sources"
    grounded = "grounded" if state.get("is_grounded") else "not fully grounded"
    return (
        f"The final answer was generated from {source_scope}. "
        f"The retriever found {state.get('retrieved_count', 0)} relevant chunks "
        f"from {source_text}. The verifier marked the evidence as {grounded}. "
        f"Reflection attempts used: {state.get('reflections', 0)}."
    )


def _route_after_verify(state: AgenticRagState) -> str:
    if state.get("force_single_pass"):
        return "write"
    if state.get("is_grounded"):
        return "write"
    if state.get("reflections", 0) >= MAX_REFLECTIONS:
        return "write"
    return "reflect"


def build_workflow():
    graph = StateGraph(AgenticRagState)
    graph.add_node("planner_node", _plan_node)
    graph.add_node("rewrite_node", _rewrite_node)
    graph.add_node("retrieve_node", _retrieve_node)
    graph.add_node("verify_node", _verify_node)
    graph.add_node("reflect_node", _reflect_node)
    graph.add_node("write_node", _write_node)

    graph.set_entry_point("planner_node")
    graph.add_edge("planner_node", "rewrite_node")
    graph.add_edge("rewrite_node", "retrieve_node")
    graph.add_edge("retrieve_node", "verify_node")
    graph.add_conditional_edges(
        "verify_node",
        _route_after_verify,
        {"reflect": "reflect_node", "write": "write_node"},
    )
    graph.add_edge("reflect_node", "retrieve_node")
    graph.add_edge("write_node", END)
    return graph.compile()


workflow = build_workflow()


def run_agentic_rag(
    question: str,
    force_single_pass: bool = False,
    answer_style: str = "detailed",
    selected_source: str | None = None,
) -> dict:
    state = workflow.invoke(
        {
            "question": question,
            "chat_history": memory.as_text(),
            "reflections": 0,
            "force_single_pass": force_single_pass,
            "answer_style": answer_style,
            "selected_source": selected_source,
            "reflection_attempts": [],
        }
    )

    memory.add("user", question)
    memory.add("assistant", state.get("answer", ""))

    return {
        "answer": state.get("answer", ""),
        "sources": state.get("sources", []),
        "confidence": state.get("confidence", 0.0),
        "plan": state.get("plan", ""),
        "refined_query": state.get("refined_query", ""),
        "verification": state.get("verification", ""),
        "reflections": state.get("reflections", 0),
        "reflection_attempts": state.get("reflection_attempts", []),
        "retrieved_count": state.get("retrieved_count", 0),
        "selected_source": state.get("selected_source"),
        "final_reasoning_summary": state.get("final_reasoning_summary", ""),
    }
