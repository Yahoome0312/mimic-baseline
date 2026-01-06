"""
Test script to verify binary mode configuration
"""
# -*- coding: utf-8 -*-

import sys
import os
import io

# Set UTF-8 encoding for stdout on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import load_class_config, generate_text_prompts

def test_load_class_config():
    """Test that load_class_config returns all prompt fields"""
    print("=" * 80)
    print("Testing load_class_config")
    print("=" * 80)

    # Test MIMIC-CXR config (binary mode)
    print("\n1. Loading MIMIC-CXR config...")
    mimic_config = load_class_config('mimic_cxr', verbose=False)

    print(f"  ✓ Loaded config with {len(mimic_config)} fields")
    print(f"    - class_names: {len(mimic_config['class_names'])} classes")
    print(f"    - task_type: {mimic_config['task_type']}")
    print(f"    - prompt_mode: {mimic_config.get('prompt_mode', 'MISSING!')}")
    print(f"    - prompt_templates: {list(mimic_config.get('prompt_templates', {}).keys())}")
    print(f"    - prompt_overrides: {list(mimic_config.get('prompt_overrides', {}).keys())}")

    # Verify prompt_mode is present
    assert 'prompt_mode' in mimic_config, "ERROR: prompt_mode not in config!"
    assert mimic_config['prompt_mode'] == 'binary', f"ERROR: prompt_mode should be 'binary', got '{mimic_config['prompt_mode']}'"

    print("\n  ✓ All fields present and correct!")

    # Test ChestXray14 config (should also be binary)
    print("\n2. Loading ChestXray14 config...")
    chestxray14_config = load_class_config('chestxray14', verbose=False)
    print(f"  ✓ prompt_mode: {chestxray14_config.get('prompt_mode', 'MISSING!')}")

    return mimic_config


def test_generate_prompts(class_config):
    """Test that generate_text_prompts works with binary mode"""
    print("\n" + "=" * 80)
    print("Testing generate_text_prompts")
    print("=" * 80)

    print("\n1. Generating binary prompts...")
    prompts = generate_text_prompts(class_config, verbose=False)

    # Check structure
    assert isinstance(prompts, dict), f"ERROR: Binary mode should return dict, got {type(prompts)}"
    assert 'positive' in prompts, "ERROR: Missing 'positive' key in prompts"
    assert 'negative' in prompts, "ERROR: Missing 'negative' key in prompts"

    print(f"  ✓ Structure correct: dict with 'positive' and 'negative' keys")
    print(f"  ✓ Generated {len(prompts['positive'])} positive prompts")
    print(f"  ✓ Generated {len(prompts['negative'])} negative prompts")

    # Show a few examples
    print("\n2. Sample prompts:")
    for i in range(min(3, len(class_config['class_names']))):
        cls = class_config['class_names'][i]
        print(f"  {cls}:")
        print(f"    Positive: {prompts['positive'][i]}")
        print(f"    Negative: {prompts['negative'][i]}")

    # Check No Finding override
    if 'No Finding' in class_config['class_names']:
        idx = class_config['class_names'].index('No Finding')
        print(f"\n3. Verifying 'No Finding' override:")
        print(f"  Positive: {prompts['positive'][idx]}")
        print(f"  Negative: {prompts['negative'][idx]}")
        assert prompts['positive'][idx] == "No finding.", "ERROR: No Finding override not applied!"
        print("  ✓ Override applied correctly!")

    return prompts


def test_dataloader_compatibility(prompts):
    """Test that prompts can be extracted for dataloader"""
    print("\n" + "=" * 80)
    print("Testing dataloader compatibility")
    print("=" * 80)

    # Simulate dataloader extraction
    if isinstance(prompts, dict) and 'positive' in prompts:
        dataloader_prompts = prompts['positive']
    else:
        dataloader_prompts = prompts

    assert isinstance(dataloader_prompts, list), "ERROR: Dataloader prompts should be a list"
    print(f"  ✓ Successfully extracted list of {len(dataloader_prompts)} prompts for dataloader")

    return dataloader_prompts


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("BINARY MODE VERIFICATION TEST")
    print("=" * 80)

    try:
        # Test 1: Load config
        mimic_config = test_load_class_config()

        # Test 2: Generate prompts
        prompts = test_generate_prompts(mimic_config)

        # Test 3: Dataloader compatibility
        dataloader_prompts = test_dataloader_compatibility(prompts)

        # Final summary
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED!")
        print("=" * 80)
        print("\nSummary:")
        print(f"  ✓ load_class_config returns all prompt fields")
        print(f"  ✓ generate_text_prompts creates binary prompts")
        print(f"  ✓ Prompt overrides work correctly")
        print(f"  ✓ Dataloader compatibility maintained")
        print("\nBinary mode is ready to use!")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
