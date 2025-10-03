"""
Entry point for the modular assembly step tracking application.

This script provides an easy way to run the modular application.
"""

import sys
import os

# Add modular directory to Python path
modular_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, modular_dir)

def main():
    """Run the modular assembly tracking application."""
    print("Starting Modular Assembly Step Tracking Application")
    print("=" * 60)
    
    try:
        # Import and run the main application
        from components.main_app import main as run_app
        run_app()
        
    except ImportError as e:
        print(f"Import Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're in the correct directory")
        print("2. Check that all component files exist")
        print("3. Verify Python path configuration")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nFor help, check the documentation in docs/")
        
    print("\nApplication finished")

if __name__ == "__main__":
    main()
