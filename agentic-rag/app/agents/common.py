from dataclasses import dataclass

from app.config import get_llm


@dataclass(frozen=True)
class RoleAgent:
    role: str
    goal: str
    backstory: str


def make_agent(role: str, goal: str, backstory: str) -> RoleAgent:
    return RoleAgent(role=role, goal=goal, backstory=backstory)


def invoke_prompt(prompt: str) -> str:
    return get_llm().invoke(prompt).content.strip()
