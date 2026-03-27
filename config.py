import os

# -------------------------------------------------------------------------
# Directory Paths
# -------------------------------------------------------------------------
HISTORY_DIR = "history"
CASES_DIR = "cases"


# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------
def ensure_directories():
    """Ensure necessary directories exist."""
    os.makedirs(HISTORY_DIR, exist_ok=True)
    os.makedirs(CASES_DIR, exist_ok=True)


def get_available_cases() -> list:
    """Returns sorted list of available case names from the cases directory."""
    if not os.path.exists(CASES_DIR):
        return []
    return sorted([
        d for d in os.listdir(CASES_DIR)
        if os.path.isdir(os.path.join(CASES_DIR, d))
    ])
