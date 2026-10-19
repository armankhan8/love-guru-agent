import logging
import os
from pathlib import Path
from threading import Lock
from time import perf_counter
from uuid import uuid4

import gradio as gr
from dotenv import load_dotenv
from openai.types.responses import ResponseTextDeltaEvent
from agents import (
    Agent, Runner, SQLiteSession, ModelSettings, RunConfig,
    input_guardrail, GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
)

load_dotenv()
MODEL = "gpt-4.1-mini"
DB_PATH = Path(__file__).with_name("chat_history.db")
MAX_CHARS = 1500
MAX_MESSAGES = 10
MAX_DEMO_RUNS = 100
run_count = 0
quota_lock = Lock()
run_config = RunConfig(tracing_disabled=True)

def new_id():
    return str(uuid4())

@input_guardrail(run_in_parallel=False)
def check_length(ctx, agent, user_input):
    # Session history may make user_input a list.
    # Check only the latest message, supplied in local context.
    length = len(ctx.context["message"])
    return GuardrailFunctionOutput(
        output_info={"characters": length},
        tripwire_triggered=length > MAX_CHARS,
    )

reply_writer = Agent(
    name="Reply Writer",
    model=MODEL,
    model_settings=ModelSettings(max_tokens=160),
    instructions=(
        "Write one short, natural Hinglish reply using the supplied situation. "
        "Keep it respectful and low-pressure. Do not invent personal details. "
        "Respect rejection and boundaries; never encourage repeated unwanted contact. "
        "Return only the suggested draft. You do not send messages."
    ),
)

assistant = Agent(
    name="Love Guru",
    model=MODEL,
    model_settings=ModelSettings(max_tokens=250, parallel_tool_calls=False),
    instructions=(
        "Help with everyday dating situations in simple Hinglish. "
        "For ordinary advice, use one gentle joke and one practical suggestion. "
        "Do not insult people or pretend to know someone's feelings. "
        "Respect consent and boundaries. Do not encourage manipulation or pressure. "
        "For serious distress, skip jokes and respond supportively. "
        "Use conversation history to understand follow-up questions. "
        "When asked to write a reply, call write_reply once with all relevant "
        "context from the conversation. Do not emit a preamble before the tool. "
        "Then present the draft briefly. Never claim to have sent it."
    ),
    input_guardrails=[check_length],
    tools=[
        reply_writer.as_tool(
            tool_name="write_reply",
            tool_description=(
                "Draft a respectful Hinglish message. Include the situation, "
                "the message being replied to, and any requested tone."
            ),
            max_turns=2,
            run_config=run_config,
        ),
    ],
)

async def respond(message, history, session_id):
    global run_count
    message = message.strip()
    if not message:
        yield message, history, "Pehle apni situation likho."
        return
    if not os.getenv("OPENAI_API_KEY"):
        yield message, history, "App ki API key configure nahi hui."
        return
    if sum(item["role"] == "user" for item in history) >= MAX_MESSAGES:
        yield message, history, "Is chat ke 10 messages ho gaye. New Chat kholo."
        return

    # Global, process-local cap: New Chat does not reset it.
    with quota_lock:
        allowed = run_count < MAX_DEMO_RUNS
        if allowed:
            run_count += 1
    if not allowed:
        yield message, history, "Demo limit reached. Please try later."
        return

    session = SQLiteSession(session_id, str(DB_PATH))
    result = None
    started = perf_counter()
    base = history + [{"role": "user", "content": message}]
    text = ""
    status = "Love Guru soch raha hai..."
    yield "", base, status

    try:
        result = Runner.run_streamed(
            assistant,
            message,
            session=session,
            context={"message": message},
            max_turns=3,
            run_config=run_config,
        )
        async for event in result.stream_events():
            if (
                event.type == "raw_response_event"
                and isinstance(event.data, ResponseTextDeltaEvent)
            ):
                text += event.data.delta
                yield "", base + [
                    {"role": "assistant", "content": text}
                ], status
            elif event.type == "run_item_stream_event" and event.name == "tool_called":
                status = "Reply Writer draft bana raha hai..."
                yield "", base, status

        elapsed = perf_counter() - started
        usage = result.context_wrapper.usage
        answer = str(result.final_output or text)
        stats = (
            f"Time: {elapsed:.2f}s | Input: {usage.input_tokens} | "
            f"Output: {usage.output_tokens} | Total: {usage.total_tokens} tokens"
        )
        yield "", base + [
            {"role": "assistant", "content": answer}
        ], stats
    except InputGuardrailTripwireTriggered:
        yield message, history, f"Message {MAX_CHARS} characters ke andar rakho."
    except Exception as exc:
        logging.warning("Agent run failed: %s", type(exc).__name__)
        yield "", base + [
            {"role": "assistant", "content": "Reply complete nahi hua. New Chat se try karo."}
        ], "Run failed; complete usage unavailable."
    finally:
        if result is not None and not result.is_complete:
            result.cancel()
        session.close()

def reset_chat():
    return [], "", new_id(), "Nayi chat ready."

with gr.Blocks() as demo:
    gr.Markdown("# Love Guru 💬")
    gr.Markdown("Thoda humour, thodi sensible advice. Demo—personal details share mat karo.")
    session_id = gr.State(value=new_id)
    new_chat = gr.Button("＋ New Chat")
    chatbot = gr.Chatbot(height=400)
    message = gr.Textbox(
        label="Apni situation batao",
        placeholder="Usne bola abhi busy hoon...",
        lines=2,
    )
    send = gr.Button("Send", variant="primary")
    status = gr.Markdown("Ready.")
    gr.Examples(
        examples=[
            ["Usne bola abhi busy hoon. Ab kya karun?"],
            ["Ek short respectful reply likh do."],
        ],
        inputs=message,
    )

    for event in (send.click, message.submit):
        event(
            fn=respond,
            inputs=[message, chatbot, session_id],
            outputs=[message, chatbot, status],
            concurrency_id="demo_actions",
            concurrency_limit=1,
            api_name=False,
        )

    new_chat.click(
        fn=reset_chat,
        outputs=[chatbot, message, session_id, status],
        concurrency_id="demo_actions",
        concurrency_limit=1,
        api_name=False,
    )

if __name__ == "__main__":
    demo.queue(max_size=10).launch()
