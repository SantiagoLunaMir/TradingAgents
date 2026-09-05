"""Research Manager: turns the bull/bear debate into a structured investment plan for the trader."""

from __future__ import annotations

from tradingagents.agents.schemas import ResearchPlan, render_research_plan
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)


def create_research_manager(llm):
    structured_llm = bind_structured(llm, ResearchPlan, "Research Manager")

    def research_manager_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)
        history = state["investment_debate_state"].get("history", "")

        investment_debate_state = state["investment_debate_state"]

        prompt = f"""As the Research Manager and debate facilitator, your role is to critically evaluate this round of debate and deliver a clear, actionable investment plan for the trader.

{instrument_context}

---

**Rating Scale** (use exactly one) — each tier maps to a concrete position outcome, not just a mood:
- **Buy**: Enter or add to the position at full size today. There is no later top-up — Buy and Overweight execute identically, so reserve Buy for when the bull case is strong enough to act on immediately.
- **Overweight**: Also enters or adds to the position at full size today (the same mechanical outcome as Buy) — choose this when full exposure is warranted but the case is constructive rather than urgent or emphatic.
- **Hold**: Maintain the current position; no trade is placed.
- **Underweight**: Trims the current position to half its size — pick this for a deteriorating-but-not-abandon view, not as a synonym for "somewhat cautious."
- **Sell**: Exits the position completely.

Commit to a clear stance whenever the debate's strongest arguments warrant one; reserve Hold for situations where the evidence on both sides is genuinely balanced. Because Buy and Overweight size identically, choose between them on how emphatic the case is, not on how much exposure you want — exposure is the same either way.

---

**Debate History:**
{history}""" + get_language_instruction()

        investment_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_research_plan,
            "Research Manager",
        )

        new_investment_debate_state = {
            "judge_decision": investment_plan,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": investment_plan,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": investment_plan,
        }

    return research_manager_node
