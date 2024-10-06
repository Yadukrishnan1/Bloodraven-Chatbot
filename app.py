import gradio as gr

from src.model_loader import load_model
from src.retriever import setup_retriever
from src.llm_chain import setup_llm_chain
from src.utils import generate_response
from huggingface_hub import InferenceClient

"""
For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
"""
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")



def main():
    # Load necessary components
    retriever = setup_retriever()
    llm_chain = setup_llm_chain()
    # Set up retrieval-augmented generation chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | llm_chain)

    interface = gr.Interface
    (
        fn=lambda question: generate_response(question, rag_chain),
        inputs=gr.Textbox(placeholder="Ask your question...", label="Your Question", lines=2),
        outputs=gr.Textbox(label="Bloodraven's Response", lines=5, interactive=False),
        title="Bloodraven Chatbot",
        description="Ask any question and receive cryptic, prophetic answers from the Bloodraven from A Song of Ice and Fire.",
        css="""
            .gradio-container {
                font-family: 'Arial', sans-serif;  /* Simple, clear font */
                background-color: #f9f9f9;  /* Light gray background */
                color: #333333;  /* Dark gray text for contrast */
            }
            .input_textbox, .output_textbox {
                border-radius: 5px;  /* Rounded corners */
                border: 2px solid #007bff;  /* Bright blue border */
                background-color: #ffffff;  /* White background for input/output boxes */
                color: #333333;  /* Dark gray text for visibility */
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);  /* Subtle shadow */
            }
            .input_textbox:focus, .output_textbox:focus {
                border-color: #ff4500;  /* Bright orange-red border on focus */
            }
            .title {
                font-size: 36px;  /* Larger title */
                font-weight: bold;  /* Bold title */
                text-align: center;  /* Centered title */
                color: #007bff;  /* Bright blue for title */
            }
            .description {
                font-size: 20px;  /* Larger description font */
                text-align: center;  /* Centered description */
                margin-bottom: 20px;  /* Space below description */
                color: #555555;  /* Medium gray for description */
            }
            .footer {
                text-align: center;  /* Center footer text */
                margin-top: 20px;  /* Space above footer */
                color: #888888;  /* Light gray for footer text */
            }
        """,
        examples=[
            ["How are Daenerys and Jon Snow related?"],
            ["What is the fate of Bran Stark?"],
            ["Tell me about the Iron Throne."],
        ],
    )

    interface.launch()

if __name__ == "__main__":
    main()
