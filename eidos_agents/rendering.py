"""Human-readable companions for machine-readable task receipts."""

from __future__ import annotations

from .schemas import FinalDecision


def final_decision_markdown(decision: FinalDecision) -> str:
    def bullets(items: list[str]) -> str:
        return "\n".join(f"- {item}" for item in items) if items else "- None"

    verdict = decision.auditor_verdict or "not applicable"
    return f"""# Eidos Agent Lab Decision — {decision.task_id}

## TASK

`{decision.task_id}`

## DECISION

{decision.decision}

## WHAT CHANGED

{bullets(decision.what_changed)}

## WHAT WE LEARNED

{bullets(decision.what_we_learned)}

## EVIDENCE

{bullets(decision.evidence)}

## WHAT REMAINS UNCERTAIN

{bullets(decision.what_remains_uncertain)}

## AUDITOR VERDICT

{verdict}

## REGRESSIONS

{bullets(decision.regressions)}

## COST

See `{decision.cost_receipt_ref}`. Cost is estimated only when current reviewed pricing exists.

## RECOMMENDED NEXT ACTION

{decision.recommended_next_action}

## HUMAN ACTION REQUIRED

{bullets(decision.human_action_required)}
"""
