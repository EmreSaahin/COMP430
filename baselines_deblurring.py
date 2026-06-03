import os
import glob
import cv2
import numpy as np

def calculate_entropy(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

def make_white_panel_with_text(img, label_text):
    h, w = img.shape[:2]
    top, bottom, left, right = 20, 50, 15, 15
    white_color = [255, 255, 255]
    
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
    bordered = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=white_color)
    text_y = bordered.shape[0] - 15
    cv2.putText(bordered, label_text, (20, text_y), cv2.FONT_HERSHEY_COMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return bordered

def run_filtered_baselines(input_dir, output_dir, min_size=64):
    os.makedirs(output_dir, exist_ok=True)
    image_paths = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    
    baseline_scores = {"Original_Degraded": [], "Global_HE_Baseline": [], "Gaussian_Blur_Baseline": [], "Our_Proposed_Pipeline": []}
    processed_count = 0
    
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None: continue
        h, w = img.shape[:2]
        if w < min_size or h < min_size: continue
            
        processed_count += 1
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        
        img_large = cv2.resize(img, (w*4, h*4), interpolation=cv2.INTER_LANCZOS4)
        
        baseline_scores["Original_Degraded"].append(calculate_entropy(img))
        p1 = make_white_panel_with_text(img_large, "Original Input")
        
        gray = cv2.cvtColor(img_large, cv2.COLOR_BGR2GRAY)
        ghe = cv2.equalizeHist(gray)
        baseline_scores["Global_HE_Baseline"].append(calculate_entropy(ghe))
        p2 = make_white_panel_with_text(ghe, "Global HE")
        
        gauss = cv2.GaussianBlur(img_large, (5, 5), 0)
        baseline_scores["Gaussian_Blur_Baseline"].append(calculate_entropy(gauss))
        p3 = make_white_panel_with_text(gauss, "Gaussian Blur")
        
        lab = cv2.cvtColor(img_large, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        l_enh = clahe.apply(l)
        res_img = cv2.cvtColor(cv2.merge((l_enh, a, b)), cv2.COLOR_LAB2BGR)
        baseline_scores["Our_Proposed_Pipeline"].append(calculate_entropy(res_img))
        p4 = make_white_panel_with_text(res_img, "Proposed Method")
        
        spacer = np.zeros((p1.shape[0], 10, 3), dtype=np.uint8) + 255
        combined_row = np.hstack((p1, spacer, p2, spacer, p3, spacer, p4))
        
        cv2.imwrite(os.path.join(output_dir, f"baseline_panel_{img_name}.png"), combined_row, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    print(f"{processed_count} images were saved to '{output_dir}'.")

if __name__ == "__main__":
    run_filtered_baselines("sample_dataset", "baseline_images", min_size=64)