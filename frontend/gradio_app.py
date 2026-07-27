import gradio as gr
import requests

API = "http://localhost:8000/generate"


def classify_news(text):

    response = requests.post(
        API,
        json={"text": text}
    )

    if response.status_code != 200:
        return f"Backend Error ({response.status_code}):\n{response.text}"

    return response.json()["response"]


demo = gr.Interface(
    fn=classify_news,
    inputs=gr.Textbox(
        lines=8,
        placeholder="Enter a financial news article or headline..."
    ),
    outputs=gr.Textbox(label="Predicted Sentiment"),
    title="Financial News Sentiment Classification",
    description="Enter a financial news headline or article. The fine-tuned LLM will classify it as Positive, Negative, or Neutral."
)

demo.launch()