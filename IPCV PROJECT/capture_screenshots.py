"""
capture_screenshots.py
Run this script to auto-save all 8 report screenshots.
No keypresses needed — it counts down 3 seconds then captures.
"""
import os, cv2, numpy as np, pickle, time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, 'screenshots')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load models ──────────────────────────────────────────────────────
def load_models():
    with open(os.path.join(BASE_DIR, 'skin_gmm.pkl'),    'rb') as f: sg = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'nonskin_gmm.pkl'), 'rb') as f: ng = pickle.load(f)
    return sg, ng

# ── Skin mask (returns all stages) ───────────────────────────────────
def process(frame_ycbcr, sg, ng):
    h, w, _ = frame_ycbcr.shape
    px = frame_ycbcr.reshape(-1, 3).astype(np.float32)
    Y, Cb, Cr = px[:,0], px[:,1], px[:,2]

    cand = (Y>=80)&(Y<=220)&(Cb>=135)&(Cb<=170)&(Cr>=80)&(Cr<=120)
    prefilter = (cand.astype(np.uint8)*255).reshape(h, w)

    gmm_flat = np.zeros(len(px), dtype=np.uint8)
    if cand.sum() > 0:
        sl = sg.score_samples(px[cand])
        nl = ng.score_samples(px[cand])
        gmm_flat[cand] = (sl > nl).astype(np.uint8) * 255
    raw_gmm = gmm_flat.reshape(h, w)

    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    final    = cv2.morphologyEx(raw_gmm, cv2.MORPH_OPEN,  kernel, iterations=1)
    final    = cv2.morphologyEx(final,   cv2.MORPH_CLOSE, kernel, iterations=3)
    return prefilter, raw_gmm, final

def save_all(raw, annotated, prefilter, raw_gmm, final):
    cv2.imwrite(os.path.join(OUT_DIR, 'screenshot1_raw_frame.png'),      raw)
    cv2.imwrite(os.path.join(OUT_DIR, 'screenshot2_prefilter_mask.png'), prefilter)
    cv2.imwrite(os.path.join(OUT_DIR, 'screenshot3_raw_gmm_mask.png'),   raw_gmm)
    cv2.imwrite(os.path.join(OUT_DIR, 'screenshot4_final_mask.png'),     final)
    cv2.imwrite(os.path.join(OUT_DIR, 'screenshot5_tracking.png'),       annotated)

    mask_bgr = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)
    sbs = np.hstack([cv2.resize(annotated,(640,480)), cv2.resize(mask_bgr,(640,480))])
    cv2.imwrite(os.path.join(OUT_DIR, 'screenshot6_side_by_side.png'),   sbs)

    dim = cv2.convertScaleAbs(annotated, alpha=0.4, beta=0)
    cv2.imwrite(os.path.join(OUT_DIR, 'screenshot7_low_light_sim.png'),  dim)

    note = np.ones((200,640,3), dtype=np.uint8)*255
    cv2.putText(note,'Confusion matrix: run evaluate.py',(30,100),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,180),2)
    cv2.imwrite(os.path.join(OUT_DIR,'screenshot8_confusion_matrix_placeholder.png'), note)

    print(f"\n[OK] All 8 screenshots saved to:\n     {OUT_DIR}\n")

# ── Main ─────────────────────────────────────────────────────────────
def main():
    print("Loading models...")
    sg, ng = load_models()
    print("Opening webcam...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam."); return

    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    DOWN     = 4
    captured = False

    print("\nGet into position — capturing in 3 seconds...\n")
    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        raw   = frame.copy()
        cv2.putText(frame,'StudentID: 2116230701041',(10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)

        elapsed = time.time() - start
        countdown = max(0, 3 - int(elapsed))

        # Process
        ycbcr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        h, w  = ycbcr.shape[:2]
        small = cv2.resize(ycbcr,(w//DOWN,h//DOWN), interpolation=cv2.INTER_AREA)
        pf_s, gm_s = process(small, sg, ng)[:2]  # prefilter + raw_gmm from small
        _, _, final_s = process(small, sg, ng)

        pf    = cv2.resize(pf_s,    (w,h), interpolation=cv2.INTER_NEAREST)
        raw_g = cv2.resize(gm_s,    (w,h), interpolation=cv2.INTER_NEAREST)
        final = cv2.resize(final_s, (w,h), interpolation=cv2.INTER_NEAREST)

        annotated = frame.copy()
        cnts, _ = cv2.findContours(final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            lc = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(lc) > 1000:
                x,y,bw,bh = cv2.boundingRect(lc)
                cv2.rectangle(annotated,(x,y),(x+bw,y+bh),(0,255,0),2)
                cv2.putText(annotated,'Skin Detected',(x,max(y-10,10)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

        # Countdown overlay
        if countdown > 0:
            cv2.putText(annotated, f'Capturing in {countdown}s...',
                        (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,200,255), 3)
        elif not captured:
            save_all(raw, annotated, pf, raw_g, final)
            captured = True
            cv2.putText(annotated,'SAVED! Closing in 2s...',
                        (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

        cv2.imshow('Skin Mask', final)
        cv2.imshow('Capture Preview — closes automatically', annotated)

        cv2.waitKey(1)

        if captured and (time.time() - start) > 6:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
