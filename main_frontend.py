from frontend.gradio_app import demo
from configs.config import (
    FRONTEND_HOST,
    FRONTEND_PORT,
)

if __name__ == "__main__":

    demo.launch(
        server_name=FRONTEND_HOST,
        server_port=FRONTEND_PORT,
        share=False
    )