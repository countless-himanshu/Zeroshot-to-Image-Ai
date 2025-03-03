from datasets import load_dataset
from transformers import CLIPTokenizer

def load_and_preprocess_data():
    # Load LAION-400M dataset
    dataset = load_dataset('laion/laion400m', split='train')
    
    # Initialize CLIP tokenizer for text preprocessing
    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')

    def preprocess_batch(batch):
        # Tokenize text
        tokenized_texts = tokenizer(batch['TEXT'], return_tensors='pt', padding=True, truncation=True)
        return {'text': tokenized_texts, 'image': batch['URL']}
    
    return dataset.map(preprocess_batch)

if __name__ == "__main__":
    dataset = load_and_preprocess_data()
    print("Data preprocessed successfully!")
