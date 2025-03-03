from model.stable_diffusion import load_stable_diffusion_model
from trainer import train_model

if __name__ == "__main__":
    # Load the model
    model = load_stable_diffusion_model()

    # Train the model
    train_model()
    
    print("Training complete and model is ready for inference!")
