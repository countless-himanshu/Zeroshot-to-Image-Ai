import gradio as gr
from datetime import datetime

def generate_from_text(text, seed, num_steps, _=gr.Progress(track_tqdm=True)):
    result = ov_pipe(text, num_inference_steps=num_steps, seed=seed)
    return result["sample"][0]

def generate_from_image(img, text, seed, num_steps, strength, _=gr.Progress(track_tqdm=True)):
    result = ov_pipe(text, img, num_inference_steps=num_steps, seed=seed, strength=strength)
    return result["sample"][0]

# Create a custom header and footer for the web page
def custom_header():
    return f"""
    <div style='text-align: center; padding: 10px; background-color: #2a9d8f;'>
        <h1 style='color: white;'>ZeroShot Text-to-Image Generator</h1>
        <p style='color: #f4a261;'>Generate high-quality images from text or initial images with ease!</p>
    </div>
    """

def custom_footer():
    return f"""
    <div style='text-align: center; padding: 10px; background-color: #264653;'>
        <p style='color: white;'>© {datetime.now().year} ZeroShot Text-to-Image Generator by Nitin Mane and Team | Powered by OpenVINO™</p>
    </div>
    """

# Interactive demo for the Hugging Face hosting
with gr.Blocks() as demo:
    demo.title = "ZeroShot Text-to-Image Generator with OpenVINO™"
    demo.description = "This interactive demo allows you to generate images from text or modify existing images using our custom Stable Diffusion-based model."
    demo.header = custom_header()
    demo.footer = custom_footer()

    with gr.Tab("Text-to-Image Generation"):
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(lines=3, label="Enter your text prompt", placeholder="E.g. A fantasy landscape with castles and dragons")
                seed_input = gr.Slider(0, 10000000, value=42, label="Random Seed (for reproducibility)")
                steps_input = gr.Slider(1, 50, value=30, step=1, label="Number of Steps")
            with gr.Column():
                output_image = gr.Image(label="Generated Image", type="pil")
        generate_button = gr.Button("Generate Image")
        generate_button.click(fn=generate_from_text, inputs=[text_input, seed_input, steps_input], outputs=output_image)

    with gr.Tab("Image-to-Image Modification"):
        with gr.Row():
            with gr.Column():
                img_input = gr.Image(label="Upload an Image to Modify", type="pil")
                text_prompt = gr.Textbox(lines=3, label="Text Prompt to Guide Modification", placeholder="E.g. Add watercolor painting effect")
                seed_input = gr.Slider(0, 1024, value=123, label="Random Seed")
                num_steps_input = gr.Slider(1, 50, value=15, step=1, label="Number of Steps")
                strength_input = gr.Slider(0.0, 1.0, value=0.6, step=0.1, label="Transformation Strength")
            with gr.Column():
                modified_output = gr.Image(label="Modified Image")
        modify_button = gr.Button("Apply Modifications")
        modify_button.click(fn=generate_from_image, inputs=[img_input, text_prompt, seed_input, num_steps_input, strength_input], outputs=modified_output)

    # Add an extra panel for feature highlights
    with gr.Accordion("Features of ZeroShot Text-to-Image Generator", open=False):
        gr.Markdown("""
        - **Text-to-Image Generation**: Create beautiful images from any text prompt.
        - **Image-to-Image Modification**: Modify existing images using a text prompt to create artistic variations.
        - **Stable Diffusion Based**: Leveraging the latest advances in text-to-image technology with the Stable Diffusion model.
        - **OpenVINO™ Optimization**: Optimized for Intel hardware to provide fast and efficient image generation.
        - **Reproducibility**: Use random seeds to generate consistent results.
        """)

    gr.Markdown("<br><br>", visible=False)  # Spacer for aesthetics

# Launch the demo for hosting on Hugging Face Spaces
try:
    demo.queue().launch(debug=True, share=True)
except Exception:
    demo.queue().launch(debug=True, server_name="0.0.0.0", server_port=7860, share=True)
