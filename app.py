"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py

Goal: wrap the LangGraph built in exercises 1–4 in a web UI that adapts to
the confidence bucket of each PR.

Routing thresholds (common/schemas.py):
    > 72%        auto_approve     UI shows a success card; reviewer does nothing
    58 – 72%     human_approval   UI shows Approve / Reject / Edit buttons
    <  58%       escalate         UI shows a question form for the reviewer,
                                  then the refined review for a final confirm
"""

from __future__ import annotations

import asyncio
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from common.db import db_conn, db_path
from common.schemas import ReviewState
from exercises.exercise_4_audit import (
    node_analyze,
    node_auto_approve,
    node_commit,
    node_escalate,
    node_fetch_pr,
    node_human_approval,
    node_route,
    node_synthesize,
)


load_dotenv()
st.set_page_config(page_title="HITL PR Review", layout="wide")

RISK_ICON = {"low": "🟢", "med": "🟡", "high": "🔴"}


# ─── Graph ─────────────────────────────────────────────────────────────────
async def node_commit_confirmed(state):
    """Like exercise 4's commit, but the escalation path also needs the
    reviewer's final confirm (human_choice == "approve") before posting."""
    if state.get("escalation_answers") and state.get("human_choice") != "approve":
        return await node_commit({**state, "escalation_answers": None})
    return await node_commit(state)


def build_graph(checkpointer):
    """Exercise 4 nodes, with escalate → synthesize → human_approval (final confirm) → commit."""
    g = StateGraph(ReviewState)
    for name, fn in [
        ("fetch_pr", node_fetch_pr), ("analyze", node_analyze), ("route", node_route),
        ("auto_approve", node_auto_approve), ("human_approval", node_human_approval),
        ("commit", node_commit_confirmed), ("escalate", node_escalate),
        ("synthesize", node_synthesize),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "fetch_pr")
    g.add_edge("fetch_pr", "analyze")
    g.add_edge("analyze", "route")
    g.add_conditional_edges(
        "route", lambda s: s["decision"],
        {"auto_approve": "auto_approve", "human_approval": "human_approval", "escalate": "escalate"},
    )
    g.add_edge("auto_approve", END)
    g.add_edge("escalate", "synthesize")
    g.add_edge("synthesize", "human_approval")
    g.add_edge("human_approval", "commit")
    g.add_edge("commit", END)
    return g.compile(checkpointer=checkpointer)


async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    """Invoke the graph once. Returns the final result or {'__interrupt__': ...}."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}
        if resume_value is None:
            return await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        return await app.ainvoke(Command(resume=resume_value), cfg)


async def load_thread(thread_id: str) -> tuple[dict, dict | None]:
    """Read a thread's checkpoint: (state values, pending interrupt payload or None)."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        snap = await build_graph(cp).aget_state({"configurable": {"thread_id": thread_id}})
    pending = [i.value for t in snap.tasks for i in t.interrupts]
    return snap.values, (pending[0] if pending else None)


async def recent_threads(limit: int = 15) -> list[dict]:
    """Same query as `audit.replay.list_threads`, returned as rows instead of printed."""
    async with db_conn() as conn:
        async with conn.execute(
            """
            SELECT thread_id,
                   pr_url,
                   MAX(timestamp) AS last_event,
                   CASE MAX(CASE risk_level WHEN 'high' THEN 3 WHEN 'med' THEN 2 ELSE 1 END)
                        WHEN 3 THEN 'high' WHEN 2 THEN 'med' ELSE 'low' END AS worst_risk,
                   COUNT(*)       AS events
              FROM audit_events
             GROUP BY thread_id, pr_url
             ORDER BY MAX(timestamp) DESC
             LIMIT ?
            """,
            (limit,),
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]


def apply_result(result: dict) -> None:
    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
        st.session_state.final = None
    else:
        st.session_state.interrupt_payload = None
        st.session_state.final = result


# ─── Session state ─────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "pr_url" not in st.session_state:
    st.session_state.pr_url = ""
if "interrupt_payload" not in st.session_state:
    st.session_state.interrupt_payload = None
if "final" not in st.session_state:
    st.session_state.final = None

# Restore the thread from the URL (?thread=...) so a page reload keeps the session.
if st.session_state.thread_id is None and st.query_params.get("thread"):
    tid = st.query_params["thread"]
    values, pending = asyncio.run(load_thread(tid))
    if values:
        st.session_state.thread_id = tid
        st.session_state.pr_url = values.get("pr_url", "")
        st.session_state.interrupt_payload = pending
        st.session_state.final = None if pending else values


st.title("HITL PR Review Agent")


# ─── Sidebar — recent sessions ─────────────────────────────────────────────
with st.sidebar:
    st.header("Recent sessions")
    try:
        threads = asyncio.run(recent_threads())
    except Exception as e:  # e.g. DB locked mid-write
        threads = []
        st.caption(f"Could not read audit_events: {e}")
    if not threads:
        st.caption("No sessions yet — run a review.")
    for t in threads:
        pr_short = t["pr_url"].split("github.com/")[-1]
        label = f"{RISK_ICON.get(t['worst_risk'], '⚪')} {pr_short}  ·  {t['thread_id'][:8]}"
        current = t["thread_id"] == st.session_state.thread_id
        if st.button(label, key=f"thread-{t['thread_id']}", use_container_width=True,
                     type="primary" if current else "secondary"):
            values, pending = asyncio.run(load_thread(t["thread_id"]))
            st.session_state.thread_id = t["thread_id"]
            st.session_state.pr_url = t["pr_url"]
            st.session_state.interrupt_payload = pending
            st.session_state.final = None if pending else values
            st.query_params["thread"] = t["thread_id"]
            st.rerun()
        st.caption(f"{t['events']} events · last {t['last_event'][:19].replace('T', ' ')}")


# ─── Top form — start a new review ─────────────────────────────────────────
with st.form("start"):
    pr_url = st.text_input(
        "PR URL", value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    submitted = st.form_submit_button("Run review")


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def render_approval_card(payload: dict) -> dict | None:
    """58–72% bucket (and the final confirm after escalation): LLM review + 3 buttons."""
    conf = payload["confidence"]
    if payload.get("refined"):
        st.subheader(f"Refined review — final confirm (confidence {conf:.0%})")
        st.caption("The agent re-analyzed the PR using your answers. Approve to post it.")
    else:
        st.subheader(f"Approval requested — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])

    for c in payload.get("comments", []):
        st.markdown(f"- **[{c['severity']}]** `{c['file']}:{c.get('line') or '?'}` — {c['body']}")

    with st.expander("Diff"):
        st.code(payload.get("diff_preview", ""), language="diff")

    feedback = st.text_input("Feedback (optional)", key="approval_feedback")
    col1, col2, col3 = st.columns(3)
    if col1.button("Approve", type="primary", use_container_width=True):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("Reject", use_container_width=True):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("Edit", use_container_width=True):
        if not feedback.strip():
            st.warning("Write what should change in the feedback box before choosing Edit.")
            return None
        return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    """< 58% bucket: show risk factors + question form. Return {question: answer} or None."""
    conf = payload["confidence"]
    st.subheader(f"Strong escalation — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    if payload.get("risk_factors"):
        st.error("**Risk factors**\n\n" + "\n".join(f"- {r}" for r in payload["risk_factors"]))
    st.markdown(payload["summary"])

    with st.form("escalation"):
        st.markdown("**The agent needs your input on these questions:**")
        answers: dict[str, str] = {}
        for i, q in enumerate(payload["questions"]):
            answers[q] = st.text_area(q, key=f"esc_q_{i}", height=80)
        if st.form_submit_button("Submit answers", type="primary"):
            if any(not a.strip() for a in answers.values()):
                st.warning("Please answer every question.")
                return None
            return {q: a.strip() for q, a in answers.items()}
    return None


def render_final(final: dict) -> None:
    action = final.get("final_action", "?")
    a = final.get("analysis")
    pr = st.session_state.pr_url
    if action.startswith("auto"):
        if action == "auto_committed":
            st.success(f"Auto-approved at {a.confidence:.0%} confidence. The agent posted the review "
                       "without human review.")
        else:
            st.error(f"Auto-approve branch ran but posting failed ({action}).")
    elif action == "committed":
        st.success("Review approved and posted to the PR.")
    elif action == "rejected":
        st.warning(f"Not posted (reviewer chose **{final.get('human_choice') or 'reject'}**).")
    else:
        st.error(f"final_action = {action}")

    if a is not None:
        st.markdown(f"**Confidence:** {a.confidence:.0%}  \n**Summary:** {a.summary}")
        for c in a.comments:
            st.markdown(f"- **[{c.severity}]** `{c.file}:{c.line or '?'}` — {c.body}")
    if final.get("escalation_answers"):
        with st.expander("Reviewer answers"):
            for q, ans in final["escalation_answers"].items():
                st.markdown(f"**{q}**  \n{ans}")
    if action in ("committed", "auto_committed"):
        st.link_button("View comment on GitHub", pr)


# ─── Main flow ─────────────────────────────────────────────────────────────
if submitted and pr_url:
    st.session_state.pr_url = pr_url
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None
    st.query_params["thread"] = st.session_state.thread_id

    try:
        with st.spinner("Fetching PR + asking the LLM..."):
            result = asyncio.run(run_graph(pr_url, st.session_state.thread_id))
        apply_result(result)
    except Exception as e:
        st.error(f"Review failed: {e}")

# Render the current interrupt card, if any
payload = st.session_state.interrupt_payload
if payload is not None:
    kind = payload["kind"]
    answer = render_approval_card(payload) if kind == "approval_request" else render_escalation_card(payload)
    if answer is not None:
        try:
            with st.spinner("Resuming..."):
                result = asyncio.run(run_graph(
                    st.session_state.pr_url, st.session_state.thread_id, resume_value=answer,
                ))
            apply_result(result)
            st.rerun()
        except Exception as e:
            st.error(f"Resume failed: {e}")

# Render final state, if reached
if st.session_state.final is not None:
    render_final(st.session_state.final)

if st.session_state.thread_id:
    st.caption(f"thread_id = {st.session_state.thread_id}  ·  replay: "
               f"`uv run python -m audit.replay --thread {st.session_state.thread_id}`")
