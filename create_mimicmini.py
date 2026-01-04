"""
Create a mini MIMIC dataset for testing
Copies a small subset of MIMIC data to test training pipeline
"""
import os
import shutil
import pandas as pd
from pathlib import Path

# Source and destination paths
SOURCE_BASE = r"D:\Data\MIMIC"
DEST_BASE = r"D:\Data\mimicmini"

# Number of samples to include
NUM_SAMPLES = 50  # About 10-15 patients

def create_mini_dataset():
    """Create mini MIMIC dataset"""

    print("=" * 80)
    print("Creating MIMIC Mini Dataset")
    print("=" * 80)

    # Create destination directory
    os.makedirs(DEST_BASE, exist_ok=True)

    # Step 1: Load metadata and split info, select balanced samples
    print(f"\n1. Loading metadata and selecting {NUM_SAMPLES} samples (balanced across splits)...")
    metadata_path = os.path.join(SOURCE_BASE, "mimic-cxr-2.0.0-metadata.csv")
    split_path = os.path.join(SOURCE_BASE, "mimic-cxr-2.0.0-split.csv")

    metadata = pd.read_csv(metadata_path)
    split_info = pd.read_csv(split_path)

    # Merge metadata with split info (only keep dicom_id and split from split_info)
    split_info_minimal = split_info[['dicom_id', 'split']]
    metadata_with_split = metadata.merge(split_info_minimal, on='dicom_id', how='left')

    # Select samples from each split proportionally
    train_samples = metadata_with_split[metadata_with_split['split'] == 'train'].head(int(NUM_SAMPLES * 0.7))
    val_samples = metadata_with_split[metadata_with_split['split'] == 'validate'].head(int(NUM_SAMPLES * 0.15))
    test_samples = metadata_with_split[metadata_with_split['split'] == 'test'].head(int(NUM_SAMPLES * 0.15))

    metadata_mini = pd.concat([train_samples, val_samples, test_samples]).drop(columns=['split'], errors='ignore').reset_index(drop=True)

    print(f"   Selected {len(metadata_mini)} samples")
    print(f"   Unique patients: {metadata_mini['subject_id'].nunique()}")
    print(f"   Unique studies: {metadata_mini['study_id'].nunique()}")

    # Step 2: Save mini metadata
    print("\n2. Saving mini metadata CSV...")
    metadata_mini.to_csv(os.path.join(DEST_BASE, "mimic-cxr-2.0.0-metadata.csv"), index=False)

    # Step 3: Process chexpert labels
    print("\n3. Processing CheXpert labels...")
    chexpert_path = os.path.join(SOURCE_BASE, "mimic-cxr-2.0.0-chexpert.csv")
    chexpert = pd.read_csv(chexpert_path)

    # Filter by subject_id and study_id
    selected_studies = set(zip(metadata_mini['subject_id'], metadata_mini['study_id']))
    chexpert_mini = chexpert[chexpert.apply(lambda row: (row['subject_id'], row['study_id']) in selected_studies, axis=1)]
    chexpert_mini.to_csv(os.path.join(DEST_BASE, "mimic-cxr-2.0.0-chexpert.csv"), index=False)
    print(f"   Saved {len(chexpert_mini)} label records")

    # Step 4: Process split
    print("\n4. Processing split CSV...")
    split = split_info.copy()

    # Filter by dicom_id
    selected_dicoms = set(metadata_mini['dicom_id'])
    split_mini = split[split['dicom_id'].isin(selected_dicoms)]
    split_mini.to_csv(os.path.join(DEST_BASE, "mimic-cxr-2.0.0-split.csv"), index=False)
    print(f"   Saved {len(split_mini)} split records")
    print(f"   Train: {(split_mini['split'] == 'train').sum()}")
    print(f"   Validate: {(split_mini['split'] == 'validate').sum()}")
    print(f"   Test: {(split_mini['split'] == 'test').sum()}")

    # Step 5: Copy images
    print("\n5. Copying images...")
    dest_images_base = os.path.join(DEST_BASE, "MIMIC-CXR-JPG", "files")
    os.makedirs(dest_images_base, exist_ok=True)

    copied_images = 0
    for idx, row in metadata_mini.iterrows():
        subject_id = row['subject_id']
        study_id = row['study_id']
        dicom_id = row['dicom_id']

        # Construct paths
        p_prefix = f"p{str(subject_id)[:2]}"
        p_folder = f"p{subject_id}"
        s_folder = f"s{study_id}"

        src_img = os.path.join(SOURCE_BASE, "MIMIC-CXR-JPG", "files", p_prefix, p_folder, s_folder, f"{dicom_id}.jpg")
        dest_img = os.path.join(dest_images_base, p_prefix, p_folder, s_folder, f"{dicom_id}.jpg")

        if os.path.exists(src_img):
            os.makedirs(os.path.dirname(dest_img), exist_ok=True)
            shutil.copy2(src_img, dest_img)
            copied_images += 1
        else:
            print(f"   WARNING: Image not found: {src_img}")

    print(f"   Copied {copied_images} images")

    # Step 6: Copy reports
    print("\n6. Copying reports...")
    dest_reports_base = os.path.join(DEST_BASE, "reports", "files")
    os.makedirs(dest_reports_base, exist_ok=True)

    # Get unique studies
    unique_studies = metadata_mini[['subject_id', 'study_id']].drop_duplicates()

    copied_reports = 0
    for idx, row in unique_studies.iterrows():
        subject_id = row['subject_id']
        study_id = row['study_id']

        # Construct paths
        p_prefix = f"p{str(subject_id)[:2]}"
        p_folder = f"p{subject_id}"

        src_report = os.path.join(SOURCE_BASE, "reports", "files", p_prefix, p_folder, f"s{study_id}.txt")
        dest_report = os.path.join(dest_reports_base, p_prefix, p_folder, f"s{study_id}.txt")

        if os.path.exists(src_report):
            os.makedirs(os.path.dirname(dest_report), exist_ok=True)
            shutil.copy2(src_report, dest_report)
            copied_reports += 1
        else:
            print(f"   WARNING: Report not found: {src_report}")

    print(f"   Copied {copied_reports} reports")

    # Summary
    print("\n" + "=" * 80)
    print("MIMIC Mini Dataset Created Successfully!")
    print("=" * 80)
    print(f"Location: {DEST_BASE}")
    print(f"Samples: {len(metadata_mini)}")
    print(f"Patients: {metadata_mini['subject_id'].nunique()}")
    print(f"Studies: {metadata_mini['study_id'].nunique()}")
    print(f"Images: {copied_images}")
    print(f"Reports: {copied_reports}")
    print("\nYou can now test with:")
    print("  python main.py --method finetune --data_path D:\\Data\\mimicmini --epochs 2")
    print("=" * 80)


if __name__ == "__main__":
    create_mini_dataset()
