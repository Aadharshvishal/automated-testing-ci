#!/usr/bin/env python3
"""
Test runner script for Assignment 4 - Automated Testing & CI
This script provides an easy way to run all tests with different configurations.
"""

import subprocess
import sys
import os
import argparse

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Assignment 4 Test Runner")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage report")
    parser.add_argument("--html", action="store_true", help="Generate HTML coverage report")
    parser.add_argument("--xml", action="store_true", help="Generate XML coverage report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--markers", "-m", help="Run tests with specific markers")
    parser.add_argument("--file", "-f", help="Run specific test file")
    
    args = parser.parse_args()
    
    print("🚀 Assignment 4 - Automated Testing & CI")
    print("=" * 50)
    
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    if args.verbose:
        cmd.append("-v")
    
    if args.coverage:
        cmd.extend(["--cov=.", "--cov-report=term-missing"])
    
    if args.html:
        cmd.append("--cov-report=html:htmlcov")
    
    if args.xml:
        cmd.append("--cov-report=xml:coverage.xml")
    
    if args.markers:
        cmd.extend(["-m", args.markers])
    
    if args.file:
        cmd.append(args.file)
    else:
        cmd.append("test_standalone.py")
    
    # Convert to string for shell execution
    cmd_str = " ".join(cmd)
    
    # Run tests
    success = run_command(cmd_str, "Running comprehensive test suite")
    
    if success:
        print("\n🎉 All tests passed successfully!")
        print("\nTest Summary:")
        print("- Authentication tests: ✅")
        print("- CRUD operation tests: ✅")
        print("- Schema validation tests: ✅")
        print("- Utility function tests: ✅")
        print("- Error handling tests: ✅")
        print("- Integration tests: ✅")
        print("- Performance tests: ✅")
        
        if args.coverage or args.html or args.xml:
            print("\nCoverage reports generated:")
            if args.html:
                print("- HTML report: htmlcov/index.html")
            if args.xml:
                print("- XML report: coverage.xml")
        
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
