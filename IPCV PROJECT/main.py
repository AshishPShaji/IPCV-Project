import os
import cv2
import numpy as np
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_models():
    """Load the trained GMM models from disk."""
    skin_gmm_path = os.path.join(BASE_DIR, 'skin_gmm.pkl')
    nonskin_gmm_path = os.path.join(BASE_DIR, 'nonskin_gmm.pkl')
    
    if not os.path.exists(skin_gmm_path) or not os.path.exists(nonskin_gmm_path):
        print("Models not found! Please run 'python train_model.py' first.")
        return None, None
        
    with open(skin_gmm_path, 'rb') as f:
        skin_gmm = pickle.load(f)
    with open(nonskin_gmm_path, 'rb') as f:
        nonskin_gmm = pickle.load(f)
        
    return skin_gmm, nonskin_gmm

def get_skin_mask(frame_ycbcr, skin_gmm, nonskin_gmm):
    """
    Generate a binary skin mask using a two-stage approach:
      1. Fast YCbCr range pre-filter to skip obvious non-skin pixels.
      2. GMM log-likelihood ratio only on candidate pixels.
    Returns: (prefilter_mask, raw_gmm_mask) both at same resolution as input.
    """
    h, w, _ = frame_ycbcr.shape
    pixels = frame_ycbcr.reshape(-1, 3).astype(np.float32)

    Y  = pixels[:, 0]
    Cb = pixels[:, 1]
    Cr = pixels[:, 2]

    candidate_mask = (
        (Y  >= 80)  & (Y  <= 220) &
        (Cb >= 135) & (Cb <= 170) &
        (Cr >= 80)  & (Cr <= 120)
    )

    # Pre-filter mask (Stage 1 output)
    prefilter_flat = (candidate_mask.astype(np.uint8) * 255)
    prefilter_mask = prefilter_flat.reshape((h, w))

    mask_flat = np.zeros(len(pixels), dtype=np.uint8)

    n_candidates = candidate_mask.sum()
    if n_candidates > 0:
        candidates = pixels[candidate_mask]
        skin_log_prob    = skin_gmm.score_samples(candidates)
        nonskin_log_prob = nonskin_gmm.score_samples(candidates)
        skin_pixels = (skin_log_prob > nonskin_log_prob).astype(np.uint8) * 255
        mask_flat[candidate_mask] = skin_pixels

    raw_gmm_mask = mask_flat.reshape((h, w))
    return prefilter_mask, raw_gmm_mask

def save_screenshots(raw_frame, prefilter_mask, raw_gmm_mask, final_mask, annotated_frame):
    """Save all 8 report screenshots to a screenshots/ folder."""
    out_dir = os.path.join(BASE_DIR, 'screenshots')
    os.makedirs(out_dir, exist_ok=True)

    # Screenshot 1: Raw webcam frame
    raw_clean = raw_frame.copy()
    cv2.imwrite(os.path.join(out_dir, 'screenshot1_raw_frame.png'), raw_clean)

    # Screenshot 2: Pre-filter mask
    cv2.imwrite(os.path.join(out_dir, 'screenshot2_prefilter_mask.png'), prefilter_mask)

    # Screenshot 3: Raw GMM mask (before morphology)
    cv2.imwrite(os.path.join(out_dir, 'screenshot3_raw_gmm_mask.png'), raw_gmm_mask)

    # Screenshot 4: Final mask (after morphology)
    cv2.imwrite(os.path.join(out_dir, 'screenshot4_final_mask.png'), final_mask)

    # Screenshot 5: Annotated tracking frame
    cv2.imwrite(os.path.join(out_dir, 'screenshot5_tracking.png'), annotated_frame)

    # Screenshot 6: Side-by-side (annotated frame | final mask coloured)
    mask_bgr = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
    side_by_side = np.hstack([
        cv2.resize(annotated_frame, (640, 480)),
        cv2.resize(mask_bgr,        (640, 480))
    ])
    cv2.imwrite(os.path.join(out_dir, 'screenshot6_side_by_side.png'), side_by_side)

    # Screenshot 7: Simulated dim-light (darken the annotated frame)
    dim = cv2.convertScaleAbs(annotated_frame, alpha=0.4, beta=0)
    cv2.imwrite(os.path.join(out_dir, 'screenshot7_low_light_sim.png'), dim)

    # Screenshot 8: placeholder note (confusion matrix needs test-set eval)
    note = np.ones((200, 640, 3), dtype=np.uint8) * 255
    cv2.putText(note, 'Run evaluate.py to generate', (40, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,180), 2)
    cv2.putText(note, 'confusion matrix (screenshot 8)', (40, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,180), 2)
    cv2.imwrite(os.path.join(out_dir, 'screenshot8_confusion_matrix_placeholder.png'), note)

    print(f"\n[OK] All 8 screenshots saved to: {out_dir}\n")

def main():
    skin_gmm, nonskin_gmm = load_models()
    if skin_gmm is None or nonskin_gmm is None:
        return

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    DOWN_FACTOR = 4

    print("Starting webcam stream...")
    print("  Press 's' to save all 8 screenshots for the report.")
    print("  Press 'q' to quit.\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from camera.")
            break
            
        frame = cv2.flip(frame, 1)

        # Save raw frame BEFORE watermark for Screenshot 1
        raw_frame = frame.copy()

        # Student ID watermark
        cv2.putText(frame, 'StudentID: 2116230701041', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
        frame_ycbcr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        h_orig, w_orig = frame_ycbcr.shape[:2]

        small_ycbcr = cv2.resize(
            frame_ycbcr,
            (w_orig // DOWN_FACTOR, h_orig // DOWN_FACTOR),
            interpolation=cv2.INTER_AREA
        )
        
        # Get both pre-filter and raw GMM masks
        small_prefilter, small_raw_gmm = get_skin_mask(small_ycbcr, skin_gmm, nonskin_gmm)

        # Upscale both intermediate masks
        prefilter_mask = cv2.resize(small_prefilter, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        raw_gmm_mask   = cv2.resize(small_raw_gmm,   (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

        # Final mask after morphology
        final_mask = raw_gmm_mask.copy()
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        # Annotate tracking frame
        annotated = frame.copy()
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 1000:
                x, y, bw, bh = cv2.boundingRect(largest_contour)
                cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                cv2.putText(annotated, 'Skin Detected', (x, max(y - 10, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Skin Mask (Adaptive GMM-YCbCr)", final_mask)
        cv2.imshow("Real-Time Tracking - 2116230701041", annotated)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_screenshots(raw_frame, prefilter_mask, raw_gmm_mask, final_mask, annotated)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()