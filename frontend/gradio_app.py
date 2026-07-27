import gradio as gr
import requests

API = "http://localhost:8000/generate"

def chat(prompt):

    r = requests.post(API, params={"Text": prompt})

    return r.json()["response"]

demo = gr.Interface(
    fn=chat,
    inputs=gr.Textbox(lines=5),
    outputs="text",
    title="Local LLM Chat"
)

demo.launch()