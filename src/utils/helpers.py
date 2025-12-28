import os

def get_project_root():
    """Returns the absolute path to the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def ensure_directory(path):
    """Ensures that a directory exists."""
    if not os.path.exists(path):
        os.makedirs(path)
