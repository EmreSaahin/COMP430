import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_entropy(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    return entropy

def proposed_hybrid_pipeline(image, gamma=1.2, clip_limit=2.5, tile_grid_size=(8,8)):

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    l_filtered = cv2.medianBlur(l, 3)
    
    l_normalized = l_filtered / 255.0
    l_gamma = np.uint8(np.power(l_normalized, gamma) * 255)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l_gamma)
    
    img_enhanced_lum = cv2.merge((l_clahe, a, b))
    img_bgr = cv2.cvtColor(img_enhanced_lum, cv2.COLOR_LAB2BGR)
    
    gaussian = cv2.GaussianBlur(img_bgr, (3, 3), 1.0)
    img_sharpened = cv2.addWeighted(img_bgr, 1.5, gaussian, -0.5, 0)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    img_final = cv2.morphologyEx(img_sharpened, cv2.MORPH_CLOSE, kernel)
    
    return img_final

def save_histogram_comparison(orig_img, enh_img, save_path):

    orig_gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
    enh_gray = cv2.cvtColor(enh_img, cv2.COLOR_BGR2GRAY)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    

    axes[0, 0].imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")
    
    axes[0, 1].hist(orig_gray.ravel(), 256, [0, 256], color='gray')
    axes[0, 1].set_title("Original Pixel Density Histogram")
    axes[0, 1].set_xlim([0, 256])
    
    # Geliştirilmiş Görsel ve Histogramı
    axes[1, 0].imshow(cv2.cvtColor(enh_img, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("Method Result")
    axes[1, 0].axis("off")
    
    axes[1, 1].hist(enh_gray.ravel(), 256, [0, 256], color='blue', alpha=0.7)
    axes[1, 1].set_title("Enhanced Pixel Density Histogram")
    axes[1, 1].set_xlim([0, 256])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def save_overall_entropy_plot(orig_entropies, enh_entropies, save_path):

    plt.figure(figsize=(12, 6))
    x = np.arange(1, len(orig_entropies) + 1)
    
    plt.plot(x, orig_entropies, marker='o', linestyle='--', color='red', label='Original Entropy', alpha=0.7)
    plt.plot(x, enh_entropies, marker='s', linestyle='-', color='green', label='Enhanced Entropy', alpha=0.8)
    
    plt.title("Information Entropy Change (Before vs After)", fontsize=14)
    plt.xlabel("Image Index", fontsize=12)
    plt.ylabel("Entropy Value (Bits)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def process_dataset(input_dir, output_dir):
    
    enhanced_dir = os.path.join(output_dir, "enhanced_images")
    plots_dir = os.path.join(output_dir, "report_plots")
    
    os.makedirs(enhanced_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    image_paths = sorted(glob.glob(os.path.join(input_dir, "*.png")))
    
    if not image_paths:
        print(f"No images were found in the '{input_dir}' folder!")
        return
        
    original_entropies = []
    enhanced_entropies = []
    
    for idx, img_path in enumerate(image_paths):
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        
        if img is None:
            continue
            
        enhanced_img = proposed_hybrid_pipeline(img)
        
        ent_orig = calculate_entropy(img)
        ent_enh = calculate_entropy(enhanced_img)
        
        original_entropies.append(ent_orig)
        enhanced_entropies.append(ent_enh)
        
        cv2.imwrite(os.path.join(enhanced_dir, img_name), enhanced_img)

        if idx < 3:
            hist_save_path = os.path.join(plots_dir, f"histogram{idx}.png")
            save_histogram_comparison(img, enhanced_img, hist_save_path)

    entropy_plot_path = os.path.join(plots_dir, "overall_entropy_chart.png")
    save_overall_entropy_plot(original_entropies, enhanced_entropies, entropy_plot_path)

    print("\n--- Result---")
    print(f"Average Initial Entropy: {np.mean(original_entropies):.4f}")
    print(f"Average Improved Entropy: {np.mean(enhanced_entropies):.4f}")

if __name__ == "__main__":
    INPUT_FOLDER = "sample_test_dataset"
    OUTPUT_FOLDER = "pipeline_results"
    
    process_dataset(INPUT_FOLDER, OUTPUT_FOLDER)