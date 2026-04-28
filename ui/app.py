from __future__ import annotations

import logging
import time
from typing import Any

import gradio as gr
from ui.client import APIClient
from ui.components import debug_panel, trace_live
from ui.config import UISettings, get_ui_settings

logger = logging.getLogger(__name__)


_CUSTOM_CSS = """
/* Layout polish for the multi-agent research assistant */
.gradio-container {
    max-width: 1600px !important;
    margin: 0 auto;
}

/* Chat panel */
#chat-panel .message {
    font-size: 14px;
    line-height: 1.6;
}

/* Live trace */
#trace-live {
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 12px;
    background: rgba(0, 0, 0, 0.02);
    padding: 12px;
    border-radius: 6px;
    border-left: 3px solid #4a9eff;
    max-height: 320px;
    overflow-y: auto;
}

#trace-live p {
    margin: 4px 0;
}

/* Debug panel */
#debug-panel {
    background: rgba(0, 0, 0, 0.02);
    border-radius: 8px;
    padding: 8px;
}

#debug-panel .tabs {
    background: transparent;
}

/* Status badges */
.status-ok { color: #22c55e; }
.status-warn { color: #eab308; }
.status-error { color: #ef4444; }

/* Compact tables */
.gradio-container table {
    font-size: 12px;
}

/* Subtle borders */
.gr-block {
    border-radius: 6px;
}
"""


class UIHandlers:
    def __init__(self, settings: UISettings):
        self._settings = settings
        self._client = APIClient(
            api_url=settings.api_url,
            ws_url=settings.ws_url,
            api_key=settings.api_key,
            timeout=settings.request_timeout,
        )

    async def check_health(self) -> str:
        try:
            health = await self._client.health()
            ready = await self._client.readiness()

            status_emoji = "✓" if ready.get("status") == "ok" else "⚠"

            uptime_s = health.get("uptime_seconds", 0)
            uptime_str = f"{uptime_s:.0f}s" if uptime_s < 60 else f"{uptime_s / 60:.1f}m"

            qdrant = "✓" if ready.get("qdrant_reachable") else "✗"
            llm = "✓" if ready.get("llm_reachable") else "✗"

            return (
                f"**API Status:** {status_emoji} `{ready.get('status', '?')}`  "
                f"·  Qdrant {qdrant}  ·  LLM {llm}  ·  Uptime: {uptime_str}"
            )
        except Exception as e:
            return f"**API Status:** ✗ Cannot reach `{self._settings.api_url}` ({e})"

    async def fetch_stats(self) -> str:
        try:
            stats = await self._client.stats()
            return (
                f"**Collection:** `{stats.get('collection', '?')}`  "
                f"·  Chunks: **{stats.get('total_chunks', 0)}**  "
                f"·  Documents: **{stats.get('total_documents', 0)}**  "
                f"·  Embeddings: `{stats.get('embedding_model', '?')}`  "
                f"·  LLM: `{stats.get('llm_model', '?')}`"
            )
        except Exception as e:
            return f"_Could not fetch stats: {e}_"

    async def submit_query_streaming(
        self,
        query: str,
        mode: str,
        history: list,
    ):
        if not query.strip():
            yield history, "_Empty query_", {}, []
            return

        history = history + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": "_Processing..._"},
        ]

        trace_lines: list[str] = []
        final_response: dict[str, Any] = {}
        chunks_table: list[list[Any]] = []

        yield history, _format_trace(trace_lines), final_response, chunks_table

        resolved_mode = None if (not mode or mode == "auto") else mode

        try:
            async for event in self._client.stream_query(
                query=query,
                mode=resolved_mode,
                include_trace=True,
            ):
                event_type = event.get("type", "")

                line = trace_live.format_node_event(event)
                if line:
                    trace_lines.append(line)

                if event_type == "final":
                    response = event.get("response", {})
                    final_response = response
                    answer = response.get("answer", "")

                    if history and history[-1]["role"] == "assistant":
                        history[-1]["content"] = answer or "_(no answer)_"

                    chunks_table = debug_panel.render_chunks_table(response)

                elif event_type == "error":
                    error = event.get("error", "Unknown error")
                    if history and history[-1]["role"] == "assistant":
                        history[-1]["content"] = f"❌ **Error:** {error}"

                yield history, _format_trace(trace_lines), final_response, chunks_table

        except Exception as e:
            logger.exception("Streaming query failed")
            error_msg = f"❌ **Connection error:** {e}"
            if history and history[-1]["role"] == "assistant":
                history[-1]["content"] = error_msg
            trace_lines.append(error_msg)
            yield history, _format_trace(trace_lines), final_response, chunks_table

    def clear_chat(self) -> tuple[list, str, dict, list]:
        """Reset all UI state."""
        return [], trace_live.empty_trace(), {}, []


def _format_trace(lines: list[str]) -> str:
    if not lines:
        return trace_live.empty_trace()
    numbered = [f"`{i + 1}.` {line}" for i, line in enumerate(lines)]
    return "<br/><br/>".join(numbered)


def create_app(settings: UISettings | None = None) -> gr.Blocks:
    settings = settings or get_ui_settings()
    handlers = UIHandlers(settings)

    cognitive_modes = [
        "auto",
        "factual_qa",
        "summarization",
        "critical_review",
        "comparison",
        "methodological_audit",
        "idea_generation",
    ]

    with gr.Blocks(
        title="Multi-Agent Research Assistant",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate",
        ),
        css=_CUSTOM_CSS,
    ) as app:
        gr.Markdown(
            """
            # Multi-Agent Research Assistant
            _RAG-based scientific assistant with LangGraph multi-agent orchestration_
            """,
        )

        with gr.Row():
            api_status = gr.Markdown("**API Status:** _checking..._")
            corpus_stats = gr.Markdown("**Corpus:** _loading..._")

        with gr.Accordion("⚙ Settings", open=False):
            with gr.Row():
                with gr.Column(scale=2):
                    api_url_display = gr.Textbox(
                        label="API URL",
                        value=settings.api_url,
                        interactive=False,
                        scale=2,
                    )
                with gr.Column(scale=1):
                    refresh_btn = gr.Button("Refresh status", size="sm")

        with gr.Row(equal_height=False):
            with gr.Column(scale=6):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    elem_id="chat-panel",
                    height=480,
                    type="messages",
                    show_copy_button=True,
                    avatar_images=(None, None),
                    latex_delimiters=[
                        {
                            "left": "$$",
                            "right": "$$",
                            "display": True,
                        },
                        {"left": "$", "right": "$", "display": False},
                        {"left": "\\(", "right": "\\)", "display": False},
                        {"left": "\\[", "right": "\\]", "display": True},
                    ],
                )

                with gr.Row():
                    query_input = gr.Textbox(
                        label="Your query",
                        placeholder="e.g. How does LoRA reduce trainable parameters?",
                        lines=2,
                        max_lines=5,
                        scale=4,
                    )
                    mode_dropdown = gr.Dropdown(
                        label="Mode",
                        choices=cognitive_modes,
                        value="auto",
                        scale=1,
                    )

                with gr.Row():
                    submit_btn = gr.Button("Send", variant="primary", scale=2)
                    clear_btn = gr.Button("Clear", scale=1)

                gr.Markdown("### Live trace")
                trace_display = gr.Markdown(
                    trace_live.empty_trace(),
                    elem_id="trace-live",
                )

            with gr.Column(scale=4):
                gr.Markdown("### Debug Panel")

                response_state = gr.State({})

                with gr.Tabs(elem_id="debug-panel"):
                    with gr.Tab("Routing"):
                        routing_md = gr.Markdown("_No data yet_")

                    with gr.Tab("Chunks"):
                        chunks_df = gr.Dataframe(
                            headers=["#", "Chunk ID", "Section", "Document", "Score", "Preview"],
                            datatype=["number", "str", "str", "str", "number", "str"],
                            value=[],
                            interactive=False,
                            wrap=True,
                        )

                    with gr.Tab("Critic"):
                        critic_md = gr.Markdown("_No data yet_")

                    with gr.Tab("Latency"):
                        latency_md = gr.Markdown("_No data yet_")

                    with gr.Tab("Events"):
                        events_md = gr.Markdown("_No data yet_")

                    with gr.Tab("Raw JSON"):
                        raw_json = gr.JSON(value={})

                    with gr.Tab("Prompts"):
                        gr.Markdown("_System / user prompts used by the Generator (debug)_")

        def _update_debug_tabs(response: dict):
            return (
                debug_panel.render_routing(response),
                debug_panel.render_critic(response),
                debug_panel.render_latency(response),
                debug_panel.render_events_timeline(response),
                response,
            )

        response_state.change(
            fn=_update_debug_tabs,
            inputs=[response_state],
            outputs=[routing_md, critic_md, latency_md, events_md, raw_json],
        )

        submit_event = submit_btn.click(
            fn=handlers.submit_query_streaming,
            inputs=[query_input, mode_dropdown, chatbot],
            outputs=[chatbot, trace_display, response_state, chunks_df],
        )
        submit_event.then(lambda: "", outputs=[query_input])

        enter_event = query_input.submit(
            fn=handlers.submit_query_streaming,
            inputs=[query_input, mode_dropdown, chatbot],
            outputs=[chatbot, trace_display, response_state, chunks_df],
        )
        enter_event.then(lambda: "", outputs=[query_input])

        clear_btn.click(
            fn=handlers.clear_chat,
            outputs=[chatbot, trace_display, response_state, chunks_df],
        )

        refresh_btn.click(fn=handlers.check_health, outputs=[api_status])
        refresh_btn.click(fn=handlers.fetch_stats, outputs=[corpus_stats])

        app.load(fn=handlers.check_health, outputs=[api_status])
        app.load(fn=handlers.fetch_stats, outputs=[corpus_stats])

    return app
