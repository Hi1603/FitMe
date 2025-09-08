import zipfile
import os
import numpy as np
import shutil
import torch
import clip
from PIL import Image
from torchvision import transforms
from cleanfid import fid
from skimage import io
from piq import DBCNN
from brisque import BRISQUE
from skimage.color import rgb2gray
from skimage.filters import gaussian

data_dir = "./data"
input_zip = "INPUTimg.zip"
output_zip = "OUTPUTimg.zip"
input_dir = os.path.join(data_dir, "input")
output_dir = os.path.join(data_dir, "output")

def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

extract_zip(input_zip, input_dir)
extract_zip(output_zip, output_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Function to compute CLIP score
def compute_clip_similarity(image1, image2):
    image1 = preprocess(Image.open(image1)).unsqueeze(0).to(device)
    image2 = preprocess(Image.open(image2)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features1 = clip_model.encode_image(image1)
        features2 = clip_model.encode_image(image2)
    
    return torch.cosine_similarity(features1, features2).item()

# Compute FID score
fid_score = fid.compute_fid(input_dir, output_dir)
dbcnn_model = DBCNN().to(device)
brisque_model = BRISQUE()

def evaluate_images():
    scores = []
    for filename in os.listdir(input_dir):
        input_img_path = os.path.join(input_dir, filename)
        output_img_path = os.path.join(output_dir, filename)
        
        if os.path.exists(output_img_path):
            # Load images
            img1 = io.imread(input_img_path)
            img2 = io.imread(output_img_path)
            
            # CLIP Similarity
            clip_score = compute_clip_similarity(input_img_path, output_img_path)
            
            # DBCNN
            img1_torch = transform(Image.open(input_img_path)).unsqueeze(0).to(device)
            img2_torch = transform(Image.open(output_img_path)).unsqueeze(0).to(device)
            dbcnn_score = torch.mean(dbcnn_model(img1_torch)).item()
            
            # BRISQUE
            brisque_score = brisque_model.score(img2)
            
            # NIQE
            niqe_score = gaussian(rgb2gray(img2), sigma=1.0).mean()
            
            scores.append((filename, clip_score, dbcnn_score, brisque_score, niqe_score))
    
    return scores

# Run evaluation
evaluation_results = evaluate_images()

# Print results
print(f"FID Score: {fid_score}")
print("Evaluation Results (per image):")
for res in evaluation_results:
    print(f"{res[0]} -> CLIP: {res[1]:.4f}, DBCNN: {res[2]:.4f}, BRISQUE: {res[3]:.4f}, NIQE: {res[4]:.4f}")
