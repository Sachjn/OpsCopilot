import gradio as gr
import requests
import os
API = os.getenv('BACKEND_API','http://localhost:8000')

def ingest_text(title, text):
    try:
        r = requests.post(f"{API}/ingest/text", json={'title': title, 'text': text})
        return str(r.json())
    except Exception as e:
        return str(e)

def ask(q):
    try:
        r = requests.post(f"{API}/query", json={'query': q, 'top_k': 4})
        return r.json().get('answer', str(r.json()))
    except Exception as e:
        return str(e)

with gr.Blocks() as demo:
    gr.Markdown('# OpsCopilot — Gradio Space Demo')
    with gr.Tab('Ingest'):
        title = gr.Textbox(label='Title')
        text = gr.Textbox(label='Document Text', lines=10)
        ingest_btn = gr.Button('Ingest')
        out = gr.Textbox(label='Response')
        ingest_btn.click(fn=ingest_text, inputs=[title, text], outputs=out)
    with gr.Tab('Ask'):
        q = gr.Textbox(label='Question')
        ask_btn = gr.Button('Ask')
        ans = gr.Textbox(label='Answer', lines=6)
        ask_btn.click(fn=ask, inputs=q, outputs=ans)

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=7860, share=False)
