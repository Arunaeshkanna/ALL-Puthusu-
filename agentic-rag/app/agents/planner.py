from app.agents.common import invoke_prompt, make_agent


planner_agent = make_agent(
    role="RAG Research Planner",
    goal="Decide the best retrieval and verification strategy for a user question.",
    backstory="You break document questions into precise search intentions before retrieval.",
)


def plan_query(question: str, chat_history: str = "") -> dict:
    prompt = f"""
You are a planning agent for a document-grounded RAG system.

Conversation history:
{chat_history or "No previous history."}

User question:
{question}

Return a compact plan with:
1. Search intent
2. Key entities
3. Whether web/youtube/url tools may help, only if the user explicitly provided such input
4. Answer format needed
"""
    return {"plan": invoke_prompt(prompt)}

