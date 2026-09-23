import os
import json
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.messages import HumanMessage, ToolMessage

from agent import get_agent_app


load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

agent_executor = get_agent_app()

fastapi_app = FastAPI()


def run_triage(thread_id: str, message: str):
    if not thread_id.strip():
        return "Please enter a Thread ID.", "ERROR"

    if not message.strip():
        return "Please enter an incident description.", "ERROR"

    config = {
        "configurable": {
            "thread_id": thread_id.strip()
        }
    }

    result = agent_executor.invoke(
        {
            "messages": [
                HumanMessage(content=message)
            ]
        },
        config=config,
    )

    snapshot = agent_executor.get_state(config)

    if snapshot.next and "sensitive_tools" in snapshot.next:
        pending_call = snapshot.values["messages"][-1].tool_calls[0]

        status_msg = (
            "**ACTION REQUIRED: AWAITING APPROVAL**\n\n"
            f"- **Action:** `{pending_call['name']}`\n\n"
            f"- **Parameters:**\n```json\n"
            f"{json.dumps(pending_call['args'], indent=2)}\n"
            f"```\n\n"
            "**Click Approve or Reject below.**"
        )

        return status_msg, "AWAITING_APPROVAL"

    return (
        snapshot.values["messages"][-1].content,
        "COMPLETED",
    )


def handle_decision(
    thread_id: str,
    decision: str,
    reason: str,
):
    if not thread_id.strip():
        return "Missing Thread ID.", "ERROR"

    config = {
        "configurable": {
            "thread_id": thread_id.strip()
        }
    }

    snapshot = agent_executor.get_state(config)

    if not snapshot.next or "sensitive_tools" not in snapshot.next:
        return (
            "No pending sensitive action found for this Thread ID.",
            "NO_PENDING_ACTION",
        )

    if decision == "Approve":
        result = agent_executor.invoke(
            None,
            config=config,
        )

        return (
            f"**Approved and executed:**\n\n"
            f"{result['messages'][-1].content}",
            "RESOLVED",
        )

    pending_call = snapshot.values["messages"][-1].tool_calls[0]

    rejection_msg = ToolMessage(
        tool_call_id=pending_call["id"],
        content=f"Rejected by engineer: {reason or 'Denied'}",
    )

    agent_executor.update_state(
        config,
        {
            "messages": [
                rejection_msg
            ]
        },
        as_node="sensitive_tools",
    )

    result = agent_executor.invoke(
        None,
        config=config,
    )

    return (
        f"**Action rejected:**\n\n"
        f"{result['messages'][-1].content}",
        "REJECTED",
    )


with gr.Blocks(
    title="Autonomous Incident Triage Agent"
) as demo:

    gr.Markdown(
        "# Autonomous Incident Triage Agent"
    )

    gr.Markdown(
        "Investigates incidents, searches internal runbooks, "
        "and pauses before critical operations."
    )

    with gr.Row():

        with gr.Column():

            thread_input = gr.Textbox(
                label="Thread ID",
                value="incident-001",
                placeholder="e.g. incident-001",
            )

            prompt_input = gr.Textbox(
                label="Incident Description",
                lines=5,
                placeholder=(
                    "Example: Auth service is timing out "
                    "and manual intervention is required."
                ),
            )

            triage_btn = gr.Button(
                "Trigger Triage",
                variant="primary",
            )

            gr.Markdown(
                "### Human-in-the-Loop Controls"
            )

            reason_input = gr.Textbox(
                label="Rejection Reason (Optional)",
                placeholder="Reason for rejecting the escalation",
            )

            with gr.Row():

                approve_btn = gr.Button(
                    "Approve Escalation",
                    variant="stop",
                )

                reject_btn = gr.Button(
                    "Reject Action",
                    variant="secondary",
                )

        with gr.Column():

            status_output = gr.Label(
                label="Workflow State",
                value="READY",
            )

            response_output = gr.Markdown(
                label="Agent Log / Output"
            )


    triage_btn.click(
        fn=run_triage,
        inputs=[
            thread_input,
            prompt_input,
        ],
        outputs=[
            response_output,
            status_output,
        ],
    )


    approve_btn.click(
        fn=lambda thread_id: handle_decision(
            thread_id,
            "Approve",
            "",
        ),
        inputs=[
            thread_input,
        ],
        outputs=[
            response_output,
            status_output,
        ],
    )


    reject_btn.click(
        fn=lambda thread_id, reason: handle_decision(
            thread_id,
            "Reject",
            reason,
        ),
        inputs=[
            thread_input,
            reason_input,
        ],
        outputs=[
            response_output,
            status_output,
        ],
    )


app = gr.mount_gradio_app(
    fastapi_app,
    demo,
    path="/",
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 7860)),
    )