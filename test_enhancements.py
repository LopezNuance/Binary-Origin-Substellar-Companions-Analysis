#!/usr/bin/env python
"""
Test script to verify all enhancements are working correctly
"""
import subprocess
import os
import sys
from pathlib import Path


def test_enhancement(cmd, expected_files, description):
    """Run a test and check outputs"""
    print(f"\nTesting: {description}")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ Failed with error:\n{result.stderr}")
        return False

    missing = []
    for f in expected_files:
        if not os.path.exists(f):
            missing.append(f)

    if missing:
        print(f"❌ Missing expected files: {missing}")
        return False

    print("✅ Success!")
    return True


def main():
    """Run all enhancement tests"""
    print("=" * 60)
    print("VLMS Enhancement Test Suite")
    print("=" * 60)

    # First test individual modules
    print("\nTesting individual modules...")

    module_tests = [
        {
            'cmd': ['python', 'tests/test_disk_migration.py'],
            'files': [],  # No output files for unit tests
            'desc': 'Disk migration module'
        },
        {
            'cmd': ['python', 'tests/test_system_schema.py'],
            'files': [],
            'desc': 'System schema module'
        },
        {
            'cmd': ['python', 'tests/test_statistical_analysis_enhancement.py'],
            'files': [],
            'desc': 'Statistical analysis enhancements'
        }
    ]

    passed = 0
    total_tests = len(module_tests)

    for test in module_tests:
        if test_enhancement(test['cmd'], test['files'], test['desc']):
            passed += 1

    print("\n" + "=" * 60)
    print(f"Module Tests Results: {passed}/{total_tests} tests passed")

    if passed == total_tests:
        print("🎉 All module tests passed!")
        print("\nNote: Integration tests require data files and would need")
        print("to be run with the full pipeline using commands like:")
        print("python source/panoptic_vlms_project.py --fetch --build-systems --regimes --disk-panel")
        return 0
    else:
        print("⚠️ Some module tests failed. Check the output above.")
        return 1


def test_imports():
    """Test that all new modules can be imported"""
    print("\nTesting imports...")

    try:
        sys.path.insert(0, 'source')

        import disk_migration
        print("✅ disk_migration imported successfully")

        import system_schema
        print("✅ system_schema imported successfully")

        from analysis import regime_clustering
        print("✅ analysis.regime_clustering imported successfully")

        from analysis import segmented_trend
        print("✅ analysis.segmented_trend imported successfully")

        # Test that hdbscan and ruptures can be imported (optional dependencies)
        try:
            import hdbscan
            print("✅ hdbscan available")
        except ImportError:
            print("⚠️ hdbscan not available - install with: pip install hdbscan")

        try:
            import ruptures
            print("✅ ruptures available")
        except ImportError:
            print("⚠️ ruptures not available - install with: pip install ruptures")

        return True

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_command_line_args():
    """Test that new command line arguments are accepted"""
    print("\nTesting command line arguments...")

    cmd = [
        'python', 'source/panoptic_vlms_project.py', '--help'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("❌ Help command failed")
        return False

    help_text = result.stdout

    # Check for new arguments
    new_args = [
        '--disk-panel', '--build-systems', '--regimes',
        '--kl-a0', '--Sigma1AU', '--disk-lifetime-myr'
    ]

    missing_args = []
    for arg in new_args:
        if arg not in help_text:
            missing_args.append(arg)

    if missing_args:
        print(f"❌ Missing arguments in help: {missing_args}")
        return False

    print("✅ All new command line arguments are available")
    return True


if __name__ == "__main__":
    print("Running comprehensive enhancement tests...\n")

    # Test imports first
    if not test_imports():
        print("Import tests failed, exiting.")
        sys.exit(1)

    # Test command line arguments
    if not test_command_line_args():
        print("Command line argument tests failed, exiting.")
        sys.exit(1)

    # Run main tests
    sys.exit(main())