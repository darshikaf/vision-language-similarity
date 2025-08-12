from pathlib import Path


def get_version():
    """Get version from _version file or fallback to default"""
    version_file = Path(__file__).parent / "_version"
    if version_file.exists():
        return version_file.read_text().strip()
    return "0.1.0-dev"


__version__ = get_version()
