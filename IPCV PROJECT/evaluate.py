import os
import cv2
import numpy as np
import pickle
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, classification_report
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'Face_Dataset', 'Pratheepan_Dataset')
GT_DIR = os.path.join(BASE_DIR, 'Face_Dataset', 'Ground_Truth')
OUT_DIR = os.path.join(BASE_DIR, 'screenshots')
os.makedirs(OUT_DIR, exist_ok=True)

def load_models():
    with open(os.path.join(BASE_DIR, 'skin_gmm.pkl'), 'rb') as f:
        skin_gmm = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'nonskin_gmm.pkl'), 'rb') as f:
        nonskin_gmm = pickle.load(f)
    return skin_gmm, nonskin_gmm

def get_test_samples(max_images=10, pixels_per_img=1000):
    pairs = []
    for subset in ['FacePhoto', 'FamilyPhoto']:
        imgs = glob(os.path.join(DATASET_DIR, subset, '*.jpg')) + \
               glob(os.path.join(DATASET_DIR, subset, '*.png'))
        for img_path in imgs:
            fname = os.path.basename(img_path).split('.')[0] + '.png'
            gt = os.path.join(GT_DIR, f'GroundT_{subset}', fname)
            if os.path.exists(gt):
                pairs.append((img_path, gt))
    
    if not pairs:
        return None, None
        
    np.random.shuffle(pairs)
    pairs = pairs[:max_images]
    
    X_test = []
    y_test = []
    
    for img_path, gt_path in pairs:
        img = cv2.imread(img_path)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if img is None or gt is None: continue
        
        if img.shape[:2] != gt.shape[:2]:
            gt = cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            
        ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        
        ycbcr_flat = ycbcr.reshape(-1, 3)
        gt_flat = (gt > 128).astype(int).flatten()
        
        indices = np.random.choice(len(ycbcr_flat), min(len(ycbcr_flat), pixels_per_img), replace=False)
        X_test.append(ycbcr_flat[indices])
        y_test.append(gt_flat[indices])
        
    if not X_test:
        return None, None
        
    return np.vstack(X_test), np.concatenate(y_test)

def main():
    print("Loading models and testing on dataset samples...")
    try:
        skin_gmm, nonskin_gmm = load_models()
    except Exception as e:
        print(f"Error loading models: {e}. Ensure skin_gmm.pkl and nonskin_gmm.pkl exist.")
        return

    X, y_true = get_test_samples()
    if X is None:
        print("Error: Could not find any test images in Face_Dataset. Check paths.")
        return
    
    print(f"Testing on {len(X)} pixels from random dataset images...")
    
    # Predict
    skin_log_prob = skin_gmm.score_samples(X)
    nonskin_log_prob = nonskin_gmm.score_samples(X)
    y_pred = (skin_log_prob > nonskin_log_prob).astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Non-Skin', 'Skin']))
    
    # Plotting
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Skin', 'Skin'], 
                yticklabels=['Non-Skin', 'Skin'],
                annot_kws={"size": 16})
    plt.title('Skin Detection Confusion Matrix (Adaptive GMM-YCbCr)', fontsize=16)
    plt.ylabel('Actual Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    
    save_path = os.path.join(OUT_DIR, 'screenshot8_confusion_matrix.png')
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"\n[OK] Confusion matrix heatmap saved to: {save_path}")

if __name__ == '__main__':
    main()
