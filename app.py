import gradio as gr
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

# Model ID from Hugging Face Hub
MODEL_ID = "Pruthvi369i/llama_3.2_vision_MedVQA"

# Load model & processor
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Define inference function
def answer_medical_question(image, question):
    try:
        inputs = processor(images=image, text=question, return_tensors="pt").to(device)
        outputs = model.generate(**inputs)
        answer = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return answer
    except Exception as e:
        return f"Error processing request: {e}"

# Create Gradio UI
demo = gr.Interface(
    fn=answer_medical_question,
    inputs=[
        gr.Image(type="pil", label="Upload Radiograph"),
        gr.Textbox(label="Ask a medical question about the image"),
    ],
    outputs=gr.Textbox(label="Answer"),
    title="Medical Visual Question Answering",
    description="Upload a radiograph and ask a medical question. This AI will provide an answer based on the image.",
)

if __name__ == "__main__":
    demo.launch(share=True)
