import torch
from diffusers import StableDiffusionPipeline
from transformers import LoRAModel

def load_stable_diffusion_model():
    # Load Stable Diffusion 1.5
    model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-5")
    
    # Load LoRA for fine-tuning
    lora_model = LoRAModel.from_pretrained("lcm-lora-sdv1.5")
    
    # Integrate LoRA model with Stable Diffusion
    model.text_encoder = lora_model.merge(model.text_encoder)
    
    return model

if __name__ == "__main__":
    model = load_stable_diffusion_model()
    print("Stable Diffusion + LoRA model loaded successfully!")
