import os
import glob
import shutil
import random

def create_random_test_dataset(source_dir, target_dir, num_images=50):

    print("--------------------------------------------------")

    if os.path.exists(target_dir):
        print(f"The '{target_dir}' folder already exists.")
        shutil.rmtree(target_dir)
    
    os.makedirs(target_dir)
    print(f"The folder '{target_dir}' has been created.")

    all_images = glob.glob(os.path.join(source_dir, "*.png"))
    total_found = len(all_images)
    
    if total_found == 0:
        print(f"No .png image was found in '{source_dir}'!!!!")
        return

    if total_found < num_images:
        num_images = total_found

    selected_images = random.sample(all_images, num_images)
    print(f"A total of {num_images} images were selected.")

    copied_count = 0
    for img_path in selected_images:
        file_name = os.path.basename(img_path)
        destination_path = os.path.join(target_dir, file_name)
        
        shutil.copy(img_path, destination_path)
        copied_count += 1

    
    print(f"{copied_count} images were transferred to the '{target_dir}' folder.")
    print("--------------------------------------------------")

if __name__ == "__main__":
    SOURCE_FOLDER = "dataset/archive (1)/Test"
    TARGET_FOLDER = "sample_dataset"
    
    create_random_test_dataset(SOURCE_FOLDER, TARGET_FOLDER, num_images=50)