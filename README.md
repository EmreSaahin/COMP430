# COMP430
Project assignment prepared for the COMP430 course.

## Project Structure and Modules
The codebase is designed with a modular architecture for sustainability and testability:
- `main.py`: The main execution script that controls the entire enhancement pipeline.
- `enhancement_pipeline.py`: Contains the core logic for enhancement algorithms (CLAHE, Gamma correction, etc.).
- `ablation_study.py`: An analysis module that evaluates the impact of different parameters on enhancement quality.
- `baselines_deblurring.py`: Implements baseline deblurring methods used for comparative analysis.
- `create_sample_dataset.py`: A utility script used to prepare the sample dataset for processing.
- `sample_dataset/`: Contains the sample images used for testing and analysis.

## Sample Enhancement Result
An example output demonstrating the effectiveness of the enhancement pipeline (CLAHE and unsharp masking):

![Enhanced Image](sample_dataset/12059.png)

## Installation and Usage
After installing the required dependencies, you can execute the main script:

```bash
pip install opencv-python numpy
python main.py
