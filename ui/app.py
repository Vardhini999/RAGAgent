"""
Gradio UI for the Agentic RAG pipeline.
Talks to the FastAPI backend via HTTP.
Deployable to Hugging Face Spaces.
"""
import gradio as gr
import httpx
import os

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
TIMEOUT = 120.0


def ingest_file(file):
    if file is None:
        return "No file provided."
    with open(file.name, "rb") as f:
        content = f.read()
    filename = os.path.basename(file.name)
    mime = "application/pdf" if filename.endswith(".pdf") else "text/plain"
    try:
        resp = httpx.post(
            f"{API_BASE}/ingest",
            files={"file": (filename, content, mime)},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return (
            f"Ingested **{data['filename']}**\n"
            f"- Chunks added: {data['chunks_added']}\n"
            f"- Total in store: {data['total_chunks_in_store']}"
        )
    except Exception as e:
        return f"Error: {e}"


def ask(query: str, history: list):
    if not query.strip():
        yield history, ""
        return
    try:
        resp = httpx.post(
            f"{API_BASE}/query",
            json={"query": query},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        mode_emoji = {"single": "⚡", "multi": "🔀", "cache": "⚡ cache"}.get(data["mode"], "")
        meta = (
            f"\n\n---\n"
            f"**Mode:** {data['mode']} {mode_emoji} | "
            f"**Confidence:** {data['confidence']:.0%} | "
            f"**Cache hit:** {'yes' if data['cache_hit'] else 'no'}"
        )
        if data.get("grader_score") is not None:
            meta += f" | **Grader:** {data['grader_score']:.2f}"
        if data["hitl_required"]:
            meta += " | ⚠️ Human review requested"
        if data["citations"]:
            meta += f"\n**Sources:** {', '.join(data['citations'])}"

        answer = data["answer"] + meta
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})
        yield history, ""
    except Exception as e:
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": f"Error: {e}"})
        yield history, ""


def cache_stats():
    try:
        resp = httpx.get(f"{API_BASE}/cache/stats", timeout=10)
        data = resp.json()
        return f"Hits: {data['hits']} | Misses: {data['misses']} | Hit rate: {data['hit_rate']:.1%}"
    except Exception as e:
        return f"Error: {e}"


with gr.Blocks(title="Agentic RAG") as demo:
    gr.Markdown("# Converse With Your Documents")
    gr.Markdown(
        "Upload a document and then ask your questions away. "
        "You can also see some cool information in addition to what you've asked for!"
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Upload Document")
            file_input = gr.File(label="PDF or TXT", file_types=[".pdf", ".txt"])
            ingest_btn = gr.Button("Ingest", variant="primary")
            ingest_status = gr.Markdown()
            ingest_btn.click(ingest_file, inputs=file_input, outputs=ingest_status)

            gr.Markdown("### Cache Stats")
            stats_btn = gr.Button("Refresh")
            stats_out = gr.Markdown()
            stats_btn.click(cache_stats, outputs=stats_out)

        with gr.Column(scale=2):
            gr.Markdown("### 2. Ask Questions")
            chatbot = gr.Chatbot(height=450, label="Conversation")
            query_box = gr.Textbox(
                placeholder="Ask anything about your document...",
                label="Query",
                lines=2,
            )
            with gr.Row():
                submit_btn = gr.Button("Ask", variant="primary")
                clear_btn = gr.Button("Clear")

            submit_btn.click(ask, inputs=[query_box, chatbot], outputs=[chatbot, query_box])
            query_box.submit(ask, inputs=[query_box, chatbot], outputs=[chatbot, query_box])
            clear_btn.click(lambda: ([], ""), outputs=[chatbot, query_box])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
