import os
import subprocess
import sys

def setup_environment():
    """Set up the environment for the Advanced Rhyming System"""
    print("Setting up Advanced Rhyming System environment...")

    # Create data directory
    data_dir = "rhyme_data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created data directory: {data_dir}")

    # Install dependencies
    dependencies = [
        "nltk",
        "pronouncing",
        "metaphone",
        "python-datamuse",
        "flask",
        "openai",
        "jellyfish",
        "textblob"
    ]

    print("Installing dependencies...")
    for dep in dependencies:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"  Installed {dep}")
        except subprocess.CalledProcessError:
            print(f"  Error installing {dep}")

    # Download NLTK data
    print("Downloading NLTK data...")
    import nltk
    try:
        nltk.download('cmudict')
        print("  Downloaded CMU Pronouncing Dictionary")
    except:
        print("  Error downloading CMU Pronouncing Dictionary")

    try:
        nltk.download('wordnet')
        print("  Downloaded WordNet")
    except:
        print("  Error downloading WordNet")

    print("\nSetup complete! You can now use the Advanced Rhyming System.")
    print("\nTo test the system, run:")
    print("  python test_rhyme_system.py --word love")
    print("\nTo start the API server, run:")
    print("  python rhyme_api.py")

if __name__ == '__main__':
    setup_environment()