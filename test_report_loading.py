"""
Test script for MIMIC-CXR report loading functionality

This script tests the new report loading feature (Findings + Impression).
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from datasets import MIMICCXRDataLoader


def test_report_extraction():
    """Test report extraction from a sample report"""
    print("=" * 80)
    print("Testing Report Extraction")
    print("=" * 80)

    config = Config()
    loader = MIMICCXRDataLoader(config)

    # Test with a known report
    sample_report = """
                                 FINAL REPORT
 EXAMINATION:  CHEST (PA AND LAT)

 INDICATION:  Test patient

 TECHNIQUE:  Chest PA and lateral

 COMPARISON:  None.

 FINDINGS:

 There is no focal consolidation, pleural effusion or pneumothorax.  Bilateral
 nodular opacities that most likely represent nipple shadows.

 IMPRESSION:

 No acute cardiopulmonary process.
    """

    result = loader._extract_findings_impression(sample_report)
    print("\nExtracted text:")
    print(result)
    print("\nExpected: Findings + Impression combined")
    print("-" * 80)


def test_data_loading_with_reports():
    """Test loading MIMIC data with reports"""
    print("\n" + "=" * 80)
    print("Testing MIMIC-CXR Data Loading with Reports")
    print("=" * 80)

    config = Config()
    loader = MIMICCXRDataLoader(config)

    # Load a small subset for testing
    print("\nLoading MIMIC-CXR data (this will take a while)...")
    image_paths, labels, reports, split_info = loader.load_data(
        use_provided_split=True,
        label_policy='ignore_uncertain'
    )

    print(f"\n[OK] Loaded {len(image_paths)} images")
    print(f"[OK] Loaded {len(reports)} reports")
    print(f"[OK] Labels shape: {labels.shape}")

    # Check report statistics
    non_empty_reports = sum(1 for r in reports if r and r != "")
    print(f"\n[INFO] Non-empty reports: {non_empty_reports}/{len(reports)} ({100*non_empty_reports/len(reports):.1f}%)")

    # Show sample reports
    print("\n" + "=" * 80)
    print("Sample Reports (first 3 with text):")
    print("=" * 80)

    count = 0
    for i, report in enumerate(reports):
        if report and report != "" and count < 3:
            print(f"\n[Sample {count + 1}] Image: {os.path.basename(image_paths[i])}")
            print(f"Report: {report[:200]}...")  # First 200 chars
            count += 1
            if count >= 3:
                break

    print("\n" + "=" * 80)
    print("[OK] Report loading test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    # Test 1: Extract findings and impression
    test_report_extraction()

    # Test 2: Load full dataset with reports
    test_data_loading_with_reports()

    print("\nAll tests completed!")
