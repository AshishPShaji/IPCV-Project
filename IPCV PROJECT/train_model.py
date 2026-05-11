import os
import cv2
import numpy as np
import pickle
from sklearn.mixture import GaussianMixture
from glob import glob

# Set absolute paths to dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "Face_Dataset", "Pratheepan_Dataset")
GROUND_TRUTH_DIR = os.path.join(BASE_DIR, "Face_Dataset", "Ground_Truth")

# Number of pixels to sample (to avoid memory crashing) - adjust if needed
MAX_PIXELS_PER_CLASS = 1_000_000

def get_image_paths():
    """Retrieve matched pairs of (original, ground_truth) images."""
    pairs = []
    
    # Check FacePhoto
    face_photos = glob(os.path.join(DATASET_DIR, "FacePhoto", "*.jpg")) + glob(os.path.join(DATASET_DIR, "FacePhoto", "*.png"))
    for img_path in face_photos:
        filename = os.path.basename(img_path).split('.')[0] + ".png" # GT files are usually png
        gt_path = os.path.join(GROUND_TRUTH_DIR, "GroundT_FacePhoto", filename)
        if os.path.exists(gt_path):
            pairs.append((img_path, gt_path))
            
    # Check FamilyPhoto
    family_photos = glob(os.path.join(DATASET_DIR, "FamilyPhoto", "*.jpg")) + glob(os.path.join(DATASET_DIR, "FamilyPhoto", "*.png"))
    for img_path in family_photos:
        filename = os.path.basename(img_path).split('.')[0] + ".png"
        gt_path = os.path.join(GROUND_TRUTH_DIR, "GroundT_FamilyPhoto", filename)
        if os.path.exists(gt_path):
            pairs.append((img_path, gt_path))
            
    return pairs

def extract_pixels(pairs):
    skin_pixels = []
    nonskin_pixels = []
    
    for img_path, gt_path in pairs:
        print(f"Processing {os.path.basename(img_path)}...")
        
        # Read image and convert to YCbCr
        img_bgr = cv2.imread(img_path)
        if img_bgr is None: continue
            
        img_ycbcr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        
        # Read ground truth (binary mask)
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask is None: continue
            
        # Ensure dimensions match (sometimes GT can be slightly off in Pratheepan dataset)
        if img_ycbcr.shape[:2] != gt_mask.shape[:2]:
            gt_mask = cv2.resize(gt_mask, (img_ycbcr.shape[1], img_ycbcr.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Assuming ground truth has white (255) for skin, black (0) for non-skin
        skin_mask = gt_mask > 128
        nonskin_mask = gt_mask <= 128
        
        # Extract pixels and reshape
        skin_val = img_ycbcr[skin_mask].reshape(-1, 3)
        nonskin_val = img_ycbcr[nonskin_mask].reshape(-1, 3)
        
        # Use random subset from each image so we don't run out of memory immediately
        if len(skin_val) > 0:
            np.random.shuffle(skin_val)
            skin_pixels.append(skin_val[:5000]) # Sample max 5000 per image
            
        if len(nonskin_val) > 0:
            np.random.shuffle(nonskin_val)
            nonskin_pixels.append(nonskin_val[:5000])
            
    # Concatenate all features
    skin_features = np.vstack(skin_pixels) if skin_pixels else np.empty((0,3))
    nonskin_features = np.vstack(nonskin_pixels) if nonskin_pixels else np.empty((0,3))
    
    # Global random sampling if still too large
    if len(skin_features) > MAX_PIXELS_PER_CLASS:
        np.random.shuffle(skin_features)
        skin_features = skin_features[:MAX_PIXELS_PER_CLASS]
        
    if len(nonskin_features) > MAX_PIXELS_PER_CLASS:
        np.random.shuffle(nonskin_features)
        nonskin_features = nonskin_features[:MAX_PIXELS_PER_CLASS]
        
    return skin_features, nonskin_features

def main():
    print("Finding dataset images...")
    pairs = get_image_paths()
    print(f"Found {len(pairs)} image pairs.")
    if not pairs:
        print("No valid image-ground truth pairs found. Exiting.")
        return
        
    print("Extracting YCbCr features...")
    skin_features, nonskin_features = extract_pixels(pairs)
    print(f"Skin pixel count: {len(skin_features)}")
    print(f"Non-skin pixel count: {len(nonskin_features)}")
    
    print("Training Skin GMM (n_components=2)...")
    # n_components=2 or 3 is usually good for a single class color distribution
    skin_gmm = GaussianMixture(n_components=3, covariance_type='full', max_iter=100)
    skin_gmm.fit(skin_features)
    
    print("Training Non-Skin GMM (n_components=2)...")
    nonskin_gmm = GaussianMixture(n_components=3, covariance_type='full', max_iter=100)
    nonskin_gmm.fit(nonskin_features)
    
    print("Saving models...")
    with open(os.path.join(BASE_DIR, 'skin_gmm.pkl'), 'wb') as f:
        pickle.dump(skin_gmm, f)
        
    with open(os.path.join(BASE_DIR, 'nonskin_gmm.pkl'), 'wb') as f:
        pickle.dump(nonskin_gmm, f)
        
    print("Training complete! Models saved as skin_gmm.pkl and nonskin_gmm.pkl")

if __name__ == '__main__':
    main()
