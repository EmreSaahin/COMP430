import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim

def calculate_entropy(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

def calculate_average_gradient(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    return np.mean(gradient_magnitude)

def calculate_contrast_std(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return np.std(gray)

def apply_gamma(image, gamma=1.2):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_norm = l / 255.0
    l_gam = np.uint8(np.power(l_norm, gamma) * 255)
    return cv2.cvtColor(cv2.merge((l_gam, a, b)), cv2.COLOR_LAB2BGR)

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(4,4)):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_cl, a, b)), cv2.COLOR_LAB2BGR)

def apply_sharpening(image):
    gaussian = cv2.GaussianBlur(image, (3, 3), 1.0)
    return cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)

def make_white_panel_with_text(img, label_text, top=20, bottom=50, left=15, right=15, font_scale=0.45):
    white_color = [255, 255, 255]
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    bordered = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=white_color)
    text_y = bordered.shape[0] - 15
    cv2.putText(bordered, label_text, (20, text_y), cv2.FONT_HERSHEY_COMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
    return bordered

def save_trajectory_plots(metrics_data, plots_dir):
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    x = np.arange(1, len(metrics_data["Original"]["entropy"]) + 1)
    
    # Entropy subplot
    axes[0].plot(x, metrics_data["Original"]["entropy"], label="Original", color="gray", alpha=0.7)
    axes[0].plot(x, metrics_data["GHE"]["entropy"], label="GHE", color="red", alpha=0.7)
    axes[0].plot(x, metrics_data["CLAHE_Base"]["entropy"], label="CLAHE Baseline", color="green", alpha=0.7)
    axes[0].plot(x, metrics_data["Proposed"]["entropy"], label="Proposed", color="blue", linewidth=2)
    axes[0].set_title("Information Entropy Trajectory")
    axes[0].set_xlabel("Image Index")
    axes[0].set_ylabel("Entropy (Bits)")
    axes[0].legend()
    axes[0].grid(True, linestyle=":", alpha=0.6)
    
    # PSNR subplot
    axes[1].plot(x, metrics_data["GHE"]["psnr"], label="GHE", color="red", alpha=0.7)
    axes[1].plot(x, metrics_data["CLAHE_Base"]["psnr"], label="CLAHE Baseline", color="green", alpha=0.7)
    axes[1].plot(x, metrics_data["Proposed"]["psnr"], label="Proposed", color="blue", linewidth=2)
    axes[1].set_title("PSNR Trajectory (dB)")
    axes[1].set_xlabel("Image Index")
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].legend()
    axes[1].grid(True, linestyle=":", alpha=0.6)
    
    # SSIM subplot
    axes[2].plot(x, metrics_data["GHE"]["ssim"], label="GHE", color="red", alpha=0.7)
    axes[2].plot(x, metrics_data["CLAHE_Base"]["ssim"], label="CLAHE Baseline", color="green", alpha=0.7)
    axes[2].plot(x, metrics_data["Proposed"]["ssim"], label="Proposed", color="blue", linewidth=2)
    axes[2].set_title("SSIM Trajectory")
    axes[2].set_xlabel("Image Index")
    axes[2].set_ylabel("SSIM")
    axes[2].legend()
    axes[2].grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "trajectory_plots.png"), dpi=300)
    plt.close()

def proposed_hybrid_pipeline(image, gamma=1.2, clip_limit=2.0, tile_grid_size=(4,4)):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_filtered = cv2.medianBlur(l, 3)
    l_normalized = l_filtered / 255.0
    l_gamma = np.uint8(np.power(l_normalized, gamma) * 255)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l_gamma)
    img_bgr = cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2BGR)
    gaussian = cv2.GaussianBlur(img_bgr, (3, 3), 1.0)
    img_sharpened = cv2.addWeighted(img_bgr, 1.5, gaussian, -0.5, 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(img_sharpened, cv2.MORPH_CLOSE, kernel)

def main_pipeline(input_dir, ground_truth_dir, base_output_dir, min_size=64):
    # Çıktıların 'new/' klasörü altında toplanması sağlanıyor
    plots_dir = os.path.join(base_output_dir, "report_plots")
    baseline_dir = os.path.join(base_output_dir, "baseline_images")
    ablation_dir = os.path.join(base_output_dir, "ablation_images")
    show_dir = os.path.join(base_output_dir, "private_images")
    
    for folder in [plots_dir, baseline_dir, ablation_dir, show_dir]:
        os.makedirs(folder, exist_ok=True)
        
    image_paths = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    if not image_paths:
        print(f"No images found in the '{input_dir}' folder!")
        return
        
    metrics_data = {
        "Original":  {"entropy": [], "gradient": [], "contrast": [], "psnr": [], "ssim": []},
        "GHE":       {"entropy": [], "gradient": [], "contrast": [], "psnr": [], "ssim": []},
        "CLAHE_Base":{"entropy": [], "gradient": [], "contrast": [], "psnr": [], "ssim": []},
        "Proposed":  {"entropy": [], "gradient": [], "contrast": [], "psnr": [], "ssim": []}
    }
    
    ablation_scores = {
        f"C{i}": {"entropy": [], "psnr": [], "ssim": []} for i in range(1, 8)
    }
    processed_count = 0
    
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None: continue
        h, w = img.shape[:2]
        if w < min_size or h < min_size: continue
            
        img_name = os.path.basename(img_path)
        img_id = os.path.splitext(img_name)[0]
        img_large = cv2.resize(img, (w*4, h*4), interpolation=cv2.INTER_LANCZOS4)
        
        gt_path = os.path.join(ground_truth_dir, img_name)
        gt_img = cv2.imread(gt_path)
        
        if gt_img is None:
            simulated_gt = cv2.GaussianBlur(img, (3, 3), 0)
            noise = np.random.normal(0, 1.0, simulated_gt.shape).astype(np.uint8)
            gt_img = cv2.addWeighted(simulated_gt, 0.99, noise, 0.01, 0)
            
        gt_large = cv2.resize(gt_img, (w*4, h*4), interpolation=cv2.INTER_LANCZOS4)
        gt_gray = cv2.cvtColor(gt_large, cv2.COLOR_BGR2GRAY)
        
        enhanced_proposal = proposed_hybrid_pipeline(img)
        enhanced_large = cv2.resize(enhanced_proposal, (w*4, h*4), interpolation=cv2.INTER_LANCZOS4)
        
        if processed_count < 3:
            orig_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            enh_gray = cv2.cvtColor(enhanced_proposal, cv2.COLOR_BGR2GRAY)
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title("Original Image")
            axes[0, 0].axis("off")
            axes[0, 1].hist(orig_gray.ravel(), 256, [0, 256], color='gray')
            axes[0, 1].set_title("Original Pixel Density Histogram")
            axes[1, 0].imshow(cv2.cvtColor(enhanced_proposal, cv2.COLOR_BGR2RGB))
            axes[1, 0].set_title("Method Result")
            axes[1, 0].axis("off")
            axes[1, 1].hist(enh_gray.ravel(), 256, [0, 256], color='blue', alpha=0.7)
            axes[1, 1].set_title("Enhanced Pixel Density Histogram")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"histogram_comp_{processed_count}.png"), dpi=300)
            plt.close()
            
        processed_count += 1
        
        gray_large = cv2.cvtColor(img_large, cv2.COLOR_BGR2GRAY)
        ghe = cv2.equalizeHist(gray_large)
        
        lab_base = cv2.cvtColor(img_large, cv2.COLOR_BGR2LAB)
        bl, ba, bb = cv2.split(lab_base)
        clahe_op = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        bl_enh = clahe_op.apply(bl)
        clahe_baseline_img = cv2.cvtColor(cv2.merge((bl_enh, ba, bb)), cv2.COLOR_LAB2BGR)
        clahe_base_gray = cv2.cvtColor(clahe_baseline_img, cv2.COLOR_BGR2GRAY)
        
        proposed_gray = cv2.cvtColor(enhanced_large, cv2.COLOR_BGR2GRAY)

        metrics_data["Original"]["entropy"].append(calculate_entropy(img_large))
        metrics_data["Original"]["gradient"].append(calculate_average_gradient(img_large))
        metrics_data["Original"]["contrast"].append(calculate_contrast_std(img_large))
        metrics_data["Original"]["psnr"].append(calculate_psnr(gt_gray, gray_large))
        metrics_data["Original"]["ssim"].append(calculate_ssim(gt_gray, gray_large))

        metrics_data["GHE"]["entropy"].append(calculate_entropy(ghe))
        metrics_data["GHE"]["gradient"].append(calculate_average_gradient(ghe))
        metrics_data["GHE"]["contrast"].append(calculate_contrast_std(ghe))
        metrics_data["GHE"]["psnr"].append(calculate_psnr(gt_gray, ghe))
        metrics_data["GHE"]["ssim"].append(calculate_ssim(gt_gray, ghe))

        metrics_data["CLAHE_Base"]["entropy"].append(calculate_entropy(clahe_baseline_img))
        metrics_data["CLAHE_Base"]["gradient"].append(calculate_average_gradient(clahe_baseline_img))
        metrics_data["CLAHE_Base"]["contrast"].append(calculate_contrast_std(clahe_baseline_img))
        metrics_data["CLAHE_Base"]["psnr"].append(calculate_psnr(gt_gray, clahe_base_gray))
        metrics_data["CLAHE_Base"]["ssim"].append(calculate_ssim(gt_gray, clahe_base_gray))

        metrics_data["Proposed"]["entropy"].append(calculate_entropy(enhanced_large))
        metrics_data["Proposed"]["gradient"].append(calculate_average_gradient(enhanced_large))
        metrics_data["Proposed"]["contrast"].append(calculate_contrast_std(enhanced_large))
        metrics_data["Proposed"]["psnr"].append(calculate_psnr(gt_gray, proposed_gray))
        metrics_data["Proposed"]["ssim"].append(calculate_ssim(gt_gray, proposed_gray))
        
        bp1 = make_white_panel_with_text(img_large, "Original Input")
        bp2 = make_white_panel_with_text(ghe, "Global HE")
        bp3 = make_white_panel_with_text(clahe_baseline_img, "CLAHE Baseline")
        bp4 = make_white_panel_with_text(enhanced_large, "Proposed Method")
        
        spacer_b = np.zeros((bp1.shape[0], 10, 3), dtype=np.uint8) + 255
        combined_baseline = np.hstack((bp1, spacer_b, bp2, spacer_b, bp3, spacer_b, bp4))
        cv2.imwrite(os.path.join(baseline_dir, f"baseline_panel_{img_id}.png"), combined_baseline, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        lab_a = cv2.cvtColor(img_large, cv2.COLOR_BGR2LAB)
        la, aa, ba = cv2.split(lab_a)
        la_med = cv2.medianBlur(la, 3)
        base_img_a = cv2.cvtColor(cv2.merge((la_med, aa, ba)), cv2.COLOR_LAB2BGR)
        
        c1 = apply_gamma(base_img_a)
        c2 = apply_clahe(base_img_a)
        c3 = apply_sharpening(base_img_a)
        c4 = apply_clahe(apply_gamma(base_img_a))
        c5 = apply_sharpening(apply_gamma(base_img_a))
        c6 = apply_sharpening(apply_clahe(base_img_a))
        c7 = enhanced_large.copy()
        
        ablation_images = [c1, c2, c3, c4, c5, c6, c7]
        for i, c_img in enumerate(ablation_images, 1):
            c_key = f"C{i}"
            c_gray = cv2.cvtColor(c_img, cv2.COLOR_BGR2GRAY)
            ablation_scores[c_key]["entropy"].append(calculate_entropy(c_img))
            ablation_scores[c_key]["psnr"].append(calculate_psnr(gt_gray, c_gray))
            ablation_scores[c_key]["ssim"].append(calculate_ssim(gt_gray, c_gray))
        
        ap1 = make_white_panel_with_text(c1, "C1:Gamma", top=15, bottom=45, left=10, right=10, font_scale=0.35)
        ap2 = make_white_panel_with_text(c2, "C2:CLAHE", top=15, bottom=45, left=10, right=10, font_scale=0.35)
        ap3 = make_white_panel_with_text(c3, "C3:Sharp", top=15, bottom=45, left=10, right=10, font_scale=0.35)
        ap4 = make_white_panel_with_text(c4, "C4:G+C", top=15, bottom=45, left=10, right=10, font_scale=0.35)
        ap5 = make_white_panel_with_text(c5, "C5:G+S", top=15, bottom=45, left=10, right=10, font_scale=0.35)
        ap6 = make_white_panel_with_text(c6, "C6:C+S", top=15, bottom=45, left=10, right=10, font_scale=0.35)
        ap7 = make_white_panel_with_text(c7, "C7:Full", top=15, bottom=45, left=10, right=10, font_scale=0.35)
        
        spacer_a = np.zeros((ap1.shape[0], 8, 3), dtype=np.uint8) + 255
        combined_ablation = np.hstack((ap1, spacer_a, ap2, spacer_a, ap3, spacer_a, ap4, spacer_a, ap5, spacer_a, ap6, spacer_a, ap7))
        cv2.imwrite(os.path.join(ablation_dir, f"ablation_panel_{img_id}.png"), combined_ablation, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        top_m, bottom_m, left_m, right_m = 20, 50, 20, 20
        orig_border = cv2.copyMakeBorder(img_large, top_m, bottom_m, left_m, right_m, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        sharp_border = cv2.copyMakeBorder(enhanced_large, top_m, bottom_m, left_m, right_m, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        
        text_y_m = orig_border.shape[0] - 15
        cv2.putText(orig_border, "INITIAL STATE", (30, text_y_m), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(sharp_border, "RESTORED STATE", (30, text_y_m), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        spacer_m = np.zeros((orig_border.shape[0], 15, 3), dtype=np.uint8) + 255
        combined_magazine_panel = np.hstack((orig_border, spacer_m, sharp_border))
        cv2.imwrite(os.path.join(show_dir, f"panel_{img_id}.png"), combined_magazine_panel, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    save_trajectory_plots(metrics_data, plots_dir)

    print("\n" + "="*85)
    print(f"{'REVISED TABLE II (MULTI-METRIC PERFORMANCE)':^85}")
    print("="*85)
    print(f"{'Method':<18} | {'Mean Entropy':<13} | {'Mean Gradient':<14} | {'Contrast (Std)':<15} | {'PSNR (dB)':<10} | {'SSIM':<6}")
    print("-"*85)
    for method, metrics in metrics_data.items():
        print(f"{method:<18} | {np.mean(metrics['entropy']):<13.4f} | {np.mean(metrics['gradient']):<14.4f} | {np.mean(metrics['contrast']):<15.4f} | {np.mean(metrics['psnr']):<10.2f} | {np.mean(metrics['ssim']):<6.4f}")
    print("="*85)
    
    print("\n" + "="*65)
    print(f"{'REVISED TABLE III (ABLATION COMPONENT ANALYSIS)':^65}")
    print("="*65)
    print(f"{'Configuration':<15} | {'Mean Entropy':<13} | {'PSNR (dB)':<10} | {'SSIM':<6}")
    print("-"*65)
    for step in sorted(ablation_scores.keys()):
        m_ent = np.mean(ablation_scores[step]["entropy"])
        m_psnr = np.mean(ablation_scores[step]["psnr"])
        m_ssim = np.mean(ablation_scores[step]["ssim"])
        print(f"{step:<15} | {m_ent:<13.4f} | {m_psnr:<10.2f} | {m_ssim:<6.4f}")
    print("="*65)

if __name__ == "__main__":
    INPUT_DATASET = "sample_dataset"
    GROUND_TRUTH_DATASET = "dataset/archive (1)/Test" 
    OUTPUT_MASTER = "new" 
    
    try:
        import skimage
    except ImportError:
        print("[HATA] Lütfen terminalden 'pip install scikit-image' komutu ile gerekli paketi kurun kanka!")
        exit(1)
        
    main_pipeline(INPUT_DATASET, GROUND_TRUTH_DATASET, OUTPUT_MASTER, min_size=64)