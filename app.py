import gradio as gr
from transformers import pipeline

# Load fine-tuned model from Hugging Face Hub (without device argument)
model_id = "Pruthvi369i/llama_3.2_vision_MedVQA"
pipe = pipeline("visual-question-answering", model=model_id)

def answer_medical_question(image, question):
    """Processes the medical image and question to return an answer."""
    try:
        result = pipe(image=image, question=question)

        # Ensure result is a valid format
        if isinstance(result, list) and len(result) > 0:
            return result[0]["answer"]
        elif isinstance(result, dict) and "answer" in result:
            return result["answer"]
        else:
            return "⚠️ Error: No valid answer found."

    except Exception as e:
        return f"❌ An error occurred: {str(e)}"

# Create Gradio Interface
demo = gr.Interface(
    fn=answer_medical_question,
    inputs=[
        gr.Image(type="pil", label="📷 Upload Medical Image"),
        gr.Textbox(label="💡 Ask a medical question about the image")
    ],
    outputs=gr.Textbox(label="🧠 AI Answer"),
    title="🩺 Medical Visual Question Answering (MedVQA)",
    description="Upload a medical image and ask a question. This AI model (LLaMA 3.2 Vision) will provide a response based on its training on medical VQA datasets.",
    examples=[
        ["example_img.jpg", "What abnormality is visible in this image?"],
        ["example_img2.jpg", "Is this scan normal or abnormal?"]
    ],
    theme="default"
)

if __name__ == "__main__":
    demo.launch()
