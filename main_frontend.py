from frontend.gradio_app import interface

if __name__ == "__main__":

    interface.launch(
        server_name="0.0.0.0",
        server_port=7860
    )