import sys
from pathlib import Path

service_dir = Path(__file__).parent.parent
sys.path.insert(0, str(service_dir))

pytest_plugins = [
    "tests.fixtures.evaluator_fixtures",
    "tests.fixtures.data_fixtures", 
    "tests.fixtures.api_fixtures",
]
