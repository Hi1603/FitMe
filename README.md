# FitMe
An AI system using Stable Diffusion and LoRA to generate realistic fitness transformations.

📖 About The Project
This project is an AI-powered system designed to generate realistic body transformations for fitness visualization. It leverages a Stable Diffusion-based inpainting model to modify user-uploaded images, providing a motivational tool for personal health. The core innovation lies in the fine-tuning of Low-Rank Adaptation (LoRA) weights on a custom fitness dataset to significantly improve anatomical realism, achieving a 7.7% reduction in perceived visual distortion compared to the baseline SD v1.5 model.

🛠️ Built With
Core Frameworks: PyTorch

AI/ML Models: Stable Diffusion (Inpainting), LoRA (fine-tuned with Kohyaa Trainer), CLIPSeg

Frontend & Interface: Gradio

Key Libraries: Hugging Face PEFT, Pandas, NumPy

📈 Key Results & Outcomes
Improved Realism: Achieved a 7.7% reduction in perceived visual distortion (measured by BRISQUE score) compared to the baseline Stable Diffusion v1.5 model.

Quantitative Validation: Rigorously evaluated model outputs using CLIP, FID, BRISQUE, and NIQE metrics to track and validate improvements in image quality and consistency.

User-Friendly Interface: Developed a web-based UI with Gradio, allowing users to easily upload images and control the intensity of their desired fitness transformation.

📄 Project Documentation
You can view the full project report here:
