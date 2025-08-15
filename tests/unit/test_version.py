import pytest
import re

from service import __version__


def test_version_exists():
    """Test that version is defined"""
    assert __version__ is not None
    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_version_format():
    """Test that version follows semantic versioning pattern"""
    # Should match patterns like: 0.1.0, 0.1.0-dev, 1.2.3.dev123456
    pattern = r'^\d+\.\d+\.\d+.*'
    assert re.match(pattern, __version__), f"Version '{__version__}' doesn't match expected format"
