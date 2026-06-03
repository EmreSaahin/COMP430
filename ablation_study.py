import os
import glob
import cv2
import numpy as np

def calculate_entropy(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

def apply_gamma(image, gamma=1.2):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_norm = l / 255.0
    l_gam = np.uint8(np.power(l_norm, gamma) * 255)
    return cv2.cvtColor(cv2.merge((l_gam, a, b)), cv2.COLOR_LAB2BGR)

def apply_clahe(image, clip_limit=2.5, tile_grid_size=(8,8)):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_cl, a, b)), cv2.COLOR_LAB2BGR)

def apply_sharpening(image):
    gaussian = cv2.GaussianBlur(image, (3, 3), 1.0)
    return cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)

def make_white_panel_with_text(img, label_text):
    h, w = img.shape[:2]
    top, bottom, left, right = 15, 45, 10, 10
    white_color = [255, 255, 255]
    bordered = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=white_color)
    text_y = bordered.shape[0] - 15
    cv2.putText(bordered, label_text, (12, text_y), cv2.FONT_HERSHEY_COMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
    return bordered

def run_filtered_ablation_study(input_dir, output_dir, min_size=64):
    os.makedirs(output_dir, exist_ok=True)
    image_paths = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    
    scores = {"C1_Gamma_Only": [], "C2_CLAHE_Only": [], "C3_Sharpening_Only": [], "C4_Gamma_CLAHE": [], "C5_Gamma_Sharpening": [], "C6_CLAHE_Sharpening": [], "C7_Full_Pipeline": []}
    processed_count = 0

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None: continue
        h, w = img.shape[:2]
        if w < min_size or h < min_size: continue
            
        processed_count += 1
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        
        img_large = cv2.resize(img, (w*3, h*3), interpolation=cv2.INTER_LANCZOS4)
        
        lab = cv2.cvtColor(img_large, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_med = cv2.medianBlur(l, 3)
        base_img = cv2.cvtColor(cv2.merge((l_med, a, b)), cv2.COLOR_LAB2BGR)

        c1 = apply_gamma(base_img)
        c2 = apply_clahe(base_img)
        c3 = apply_sharpening(base_img)
        c4 = apply_clahe(apply_gamma(base_img))
        c5 = apply_sharpening(apply_gamma(base_img))
        c6 = apply_sharpening(apply_clahe(base_img))
        
        c7_mid = apply_sharpening(apply_clahe(apply_gamma(base_img)))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        c7 = cv2.morphologyEx(c7_mid, cv2.MORPH_CLOSE, kernel)

        scores["C1_Gamma_Only"].append(calculate_entropy(c1))
        scores["C2_CLAHE_Only"].append(calculate_entropy(c2))
        scores["C3_Sharpening_Only"].append(calculate_entropy(c3))
        scores["C4_Gamma_CLAHE"].append(calculate_entropy(c4))
        scores["C5_Gamma_Sharpening"].append(calculate_entropy(c5))
        scores["C6_CLAHE_Sharpening"].append(calculate_entropy(c6))
        scores["C7_Full_Pipeline"].append(calculate_entropy(c7))

        p1 = make_white_panel_with_text(c1, "C1:Gamma")
        p2 = make_white_panel_with_text(c2, "C2:CLAHE")
        p3 = make_white_panel_with_text(c3, "C3:Sharp")
        p4 = make_white_panel_with_text(c4, "C4:G+C")
        p5 = make_white_panel_with_text(c5, "C5:G+S")
        p6 = make_white_panel_with_text(c6, "C6:C+S")
        p7 = make_white_panel_with_text(c7, "C7:Full")

        spacer = np.zeros((p1.shape[0], 8, 3), dtype=np.uint8) + 255
        combined_ablation = np.hstack((p1, spacer, p2, spacer, p3, spacer, p4, spacer, p5, spacer, p6, spacer, p7))
        
        cv2.imwrite(os.path.join(output_dir, f"ablation_panel_{img_name}.png"), combined_ablation, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    print(f"{processed_count} images were saved to '{output_dir}'.")

if __name__ == "__main__":
    run_filtered_ablation_study("sample_dataset", "ablation_images", min_size=64)