# ZeroShot Text-to-Image Generator Technical Report

## Table of Contents
- [Introduction](#introduction)
  - [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Data](#data)
  - [Dataset Details](#dataset-details)
- [Configuration](#configuration)
- [Feature Engineering](#feature-engineering)
- [Architecture Development](#architecture-development)
- [Model Metric Analysis](#model-metric-analysis)
- [Testing](#testing)
- [Deployment](#deployment)
  - [Prototype Testing](#prototype-testing)
- [Text-to-Image Generation with Stable Diffusion and OpenVINO™](#text-to-image-generation-with-stable-diffusion-and-openvino)
  - [Prerequisites](#prerequisites)
  - [Create PyTorch Models Pipeline](#create-pytorch-models-pipeline)
  - [Convert Models to OpenVINO IR Format](#convert-models-to-openvino-ir-format)
    - [Text Encoder](#text-encoder)
    - [U-net](#u-net)
    - [VAE](#vae)
  - [Prepare Inference Pipeline](#prepare-inference-pipeline)
  - [Configure Inference Pipeline](#configure-inference-pipeline)
    - [Text-to-Image Generation](#text-to-image-generation)
    - [Image-to-Image Generation](#image-to-image-generation)
  - [Interactive Demo](#interactive-demo)
- [Contributing](#contributing)
  - [Contributors](#contributors)
- [Versioning](#versioning)
- [License](#license)
- [Bugs/Issues](#bugs-issues)

&nbsp;


## Introduction

The ZeroShot Text-to-Image Generator is a deep learning project developed under the CS550 Machine Learning course at IIT Bhilai. The primary objective is to create a model capable of generating high-quality images from textual descriptions without the need for labeled datasets. This report provides insights into the architecture, training, and testing phases, as well as the motivation behind the development of this model.

### Model Architecture

The ZeroShot Text-to-Image Generator employs a hybrid model combining transformer-based text encoding with Generative Adversarial Networks (GANs) for image generation. The transformer model effectively captures the semantic relationships in text inputs, while the GAN synthesizes high-quality images from the encoded information. This approach enables the generation of realistic images even for prompts that the model has not been trained on.

The backbone of this project leverages the Stable Diffusion model, a text-to-image latent diffusion model developed by researchers from [CompVis](https://github.com/CompVis), [Stability AI](https://stability.ai/), and [LAION](https://laion.ai/). Stable Diffusion is trained on 512x512 images from a subset of the [LAION-5B](https://laion.ai/blog/laion-5b/) database, utilizing a frozen CLIP ViT-L/14 text encoder to condition the model on text prompts. It features an 860M parameter U-Net and a 123M parameter text encoder, making it relatively lightweight and able to run on many consumer GPUs. More information can be found in the [model card](https://huggingface.co/CompVis/stable-diffusion).

&nbsp;

## Installation

To install the ZeroShot Text-to-Image Generator, please refer to the [Installation Guide](notebook/Installation.md). The guide includes step-by-step instructions for setting up the required environment, installing dependencies, and configuring the model for usage.

&nbsp;

## Data

The ZeroShot Text-to-Image Generator relies on textual descriptions to generate images, rather than requiring traditional labeled datasets.

### Dataset Details

For training and evaluation, the project uses a subset of descriptive prompts derived from the LAION-5B dataset. This subset helps demonstrate the model's zero-shot capabilities by providing a diverse range of input prompts without explicitly labeled training examples.

&nbsp;

## Configuration

The configuration details for setting up the ZeroShot Text-to-Image Generator can be found in the [configuration/config.json](configuration/config.json) file. This file contains parameters for model settings, data paths, and system requirements.

&nbsp;

## Feature Engineering

Feature engineering for the ZeroShot Text-to-Image Generator involved extracting meaningful representations from the input text to condition the image generation process. The text prompts were tokenized using a tokenizer, followed by encoding with the CLIP ViT-L/14 text encoder. This allowed for a rich semantic representation of textual inputs that could be used to guide the image generation process.

Latent vectors representing the semantic content of the text prompts were used as conditioning inputs for the generation pipeline. Negative prompts, such as "blurry, low quality," were also provided to help the model avoid undesirable features and to further enhance the quality of generated images by eliminating unwanted noise.

In addition to text embeddings, random noise generation (seeded for reproducibility) was utilized to initiate the latent diffusion process, which provided the model with a stochastic starting point for image generation. The resulting latent vectors from this process were crucial in determining the ultimate quality of the generated images.

&nbsp;

## Architecture Development

The architecture development for the ZeroShot Text-to-Image Generator is based on the adaptation of the Stable Diffusion model, which utilizes a diffusion process in the latent space for generating high-quality images from text descriptions.

### Tokenization and Text Encoding
The architecture begins with tokenizing the input prompts, which consist of both positive and negative descriptions. The tokenizer breaks down the input text into tokens, which are subsequently passed to the CLIP ViT-L/14 text encoder. The encoder transforms the tokens into rich semantic embeddings that represent the intended content of the generated image. These text embeddings are then utilized as conditioning inputs for subsequent processing.

### Latent Noise Generation
A random noise generator is used to create initial latent noise. This noise serves as a starting point for the diffusion process. By applying a fixed random seed, the model ensures reproducibility of results. The latent noise, combined with the text embeddings, forms the basis for generating images that are consistent with the input prompts.

### Text-Conditioned U-Net
The U-Net architecture is the core of the generation pipeline. The latent noise and text embeddings are fed into the U-Net, which has been adapted to perform text-conditioned denoising. The U-Net consists of an encoder-decoder architecture, where the encoder progressively compresses the latent noise representation, and the decoder reconstructs the image by reversing the compression process. Cross-attention layers are added between the encoder and decoder to ensure that the generated images adhere closely to the input text prompts.

### Scheduler Algorithm
The scheduler algorithm orchestrates the latent diffusion process, guiding the U-Net through multiple denoising steps to gradually convert the initial noise into a high-quality image. The scheduler determines the step size and the direction of change at each iteration, ensuring that the latent representation evolves in a manner that aligns with the intended visual content. The denoising process is repeated for a fixed number of iterations to produce the conditioned latent representation.

### VAE Decoder
The VAE (Variational Autoencoder) decoder is the final component of the architecture. After the latent representation has been conditioned and refined by the U-Net, the VAE decoder is employed to transform the latent space representation into an actual image. This step involves decoding the compressed latent vectors into a grid of pixel values, resulting in the final visual output.

### Cross-Attention Mechanism
To maintain fidelity to the input prompts, the model incorporates a cross-attention mechanism that allows the U-Net to access the text embeddings at multiple stages of the denoising process. This ensures that both global and local features in the generated images are consistent with the semantics of the textual input. The cross-attention mechanism plays a crucial role in capturing details such as object shapes, textures, and relationships between visual elements.

### Negative Prompt Conditioning
Negative prompts, such as "blurry" or "low quality," are used to prevent certain characteristics from appearing in the generated image. The negative conditioning mechanism allows the model to understand undesirable features and avoid including them during the generation process. This feature enhances the quality and specificity of the generated images by explicitly removing unwanted attributes.

### Image Generation Output
The final image is produced by the VAE decoder and can be visualized or saved as needed. The output image is conditioned on both the positive prompts that describe the desired characteristics and the negative prompts that specify what to avoid. This dual conditioning ensures that the generated images align well with user expectations, resulting in high-quality and contextually accurate visuals.

### Reproducibility and Random Seeds
To ensure reproducibility, the architecture incorporates a random seed for noise generation. This allows users to generate the same output image consistently for a given input prompt, which is particularly useful for benchmarking and debugging purposes. The use of random seeds provides control over the stochastic nature of the diffusion process and helps maintain consistent results.

&nbsp;

## Model Metric Analysis

The evaluation of the ZeroShot Text-to-Image Generator was conducted using several key metrics to assess the quality of the generated images:
- **Frechet Inception Distance (FID)**: Used to measure the visual quality and similarity of generated images to real images.
- **Semantic Consistency**: Evaluates how accurately the generated images reflect the meaning of the input text prompts.
- **Inference Time**: Measures the time required to generate an image from a given text prompt, providing insights into model efficiency.

The results indicate that the model effectively captures the context from textual prompts and produces visually coherent images, suitable for a wide variety of applications.

&nbsp;

## Testing

Testing is conducted to evaluate the model's ability to generate visually accurate images from a wide range of textual descriptions. The model's outputs were validated based on their alignment with the provided prompts, ensuring semantic consistency and visual coherence. A detailed breakdown of testing metrics and results can be found in the [Testing Section](notebook/Testing.md).

&nbsp;

## Deployment

### Prototype Testing

The ZeroShot Text-to-Image Generator has been deployed as a real-time prototype, allowing users to interact with the model by providing their own prompts. Users can explore the prototype and assess the model's performance directly through our Hugging Face Space. Please visit [Hugging Face link](#) for more details.

&nbsp;

## Contributing

We welcome contributions to improve the ZeroShot Text-to-Image Generator. Please refer to our [Contributing Guide](CONTRIBUTING.md) for instructions on how to contribute to this project.

### Contributors

- **Nitin Mane** - Developer and lead for model training and architecture design.
- **Group Members**: Himanshu Rana, Deepak Kumar, Mahesh Kesgire, Mohd Mukheet.

&nbsp;



## Bugs/Issues

If you encounter any bugs or have feature requests, please submit an issue in the [GitHub Issues](https://github.com/Zeroshot-Dreamers/ZeroShot-Text-to-Image/issues) section. Refer to our [CONTRIBUTING.md](CONTRIBUTING.md) for more details on submitting issues.

&nbsp;
