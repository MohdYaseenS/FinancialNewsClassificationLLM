def extract_pred_label(text):
    try:
        if "<|im_start|>assistant" in text:
            text = text.split("<|im_start|>assistant")[-1]

        if "<|im_end|>" in text:
            text = text.split("<|im_end|>")[0]

        if "<think>" in text:
            text = text.split("</think>")[-1]

        lines = [l.strip().lower() for l in text.split("\n") if l.strip()]

        for line in reversed(lines):
            if line in ["positive", "negative", "neutral"]:
                return line

    except Exception as e:
        print("Prediction extraction error:", e)

    return "unknown"