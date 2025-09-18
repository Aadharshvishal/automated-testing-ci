"""
Pytest configuration and shared fixtures for Assignment 4 tests.
"""

import pytest
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return {
        "id": 1,
        "name": "Test User",
        "email": "test@example.com"
    }

@pytest.fixture
def sample_list():
    """Sample list for testing."""
    return [1, 2, 3, 4, 5]

@pytest.fixture
def sample_dict():
    """Sample dictionary for testing."""
    return {"key1": "value1", "key2": "value2"}

@pytest.fixture
def sample_string():
    """Sample string for testing."""
    return "Hello, World!"

@pytest.fixture
def sample_numbers():
    """Sample numbers for testing."""
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return "There are 123 apples and 456 oranges"

@pytest.fixture
def sample_url():
    """Sample URL for testing."""
    return "https://example.com/path?query=1"

@pytest.fixture
def sample_path():
    """Sample path for testing."""
    return "/tmp/test.txt"

@pytest.fixture
def sample_tasks():
    """Sample tasks for testing."""
    return ["Task 1", "Task 2", "Task 3"]

@pytest.fixture
def sample_choices():
    """Sample choices for testing."""
    return [1, 2, 3, 4, 5]

@pytest.fixture
def sample_data_list():
    """Sample data list for testing."""
    return [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]

@pytest.fixture
def sample_numbers_for_math():
    """Sample numbers for math testing."""
    return [1, 2, 3, 4, 5]

@pytest.fixture
def sample_numbers_for_stats():
    """Sample numbers for statistics testing."""
    return [1, 2, 3, 4, 5]

@pytest.fixture
def sample_numbers_for_mode():
    """Sample numbers for mode testing."""
    return [1, 1, 2, 3]

@pytest.fixture
def sample_numbers_for_operations():
    """Sample numbers for operations testing."""
    return [1, 5, 3, 9, 2]

@pytest.fixture
def sample_numbers_for_arithmetic():
    """Sample numbers for arithmetic testing."""
    return [1, 2, 3, 4, 5]

@pytest.fixture
def sample_numbers_for_bitwise():
    """Sample numbers for bitwise testing."""
    return [5, 3]

@pytest.fixture
def sample_numbers_for_membership():
    """Sample numbers for membership testing."""
    return [1, 2, 3, 4, 5]

@pytest.fixture
def sample_string_for_membership():
    """Sample string for membership testing."""
    return "hello"

@pytest.fixture
def sample_objects_for_identity():
    """Sample objects for identity testing."""
    a = [1, 2, 3]
    b = [1, 2, 3]
    c = a
    return {"a": a, "b": b, "c": c}

@pytest.fixture
def sample_numbers_for_range():
    """Sample numbers for range testing."""
    return list(range(5))

@pytest.fixture
def sample_numbers_for_slice():
    """Sample numbers for slice testing."""
    return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

@pytest.fixture
def sample_numbers_for_comprehension():
    """Sample numbers for comprehension testing."""
    return [1, 2, 3, 4, 5]

@pytest.fixture
def sample_numbers_for_generator():
    """Sample numbers for generator testing."""
    return [1, 2, 3, 4, 5]

@pytest.fixture
def sample_numbers_for_lambda():
    """Sample numbers for lambda testing."""
    return [2, 3, 4, 5]

@pytest.fixture
def sample_numbers_for_closure():
    """Sample numbers for closure testing."""
    return [3, 10]

@pytest.fixture
def sample_numbers_for_decorator():
    """Sample numbers for decorator testing."""
    return [2, 3]

@pytest.fixture
def sample_numbers_for_exception():
    """Sample numbers for exception testing."""
    return [10, 2]

@pytest.fixture
def sample_numbers_for_assertion():
    """Sample numbers for assertion testing."""
    return [1, 2, 3]

@pytest.fixture
def sample_numbers_for_boolean():
    """Sample numbers for boolean testing."""
    return [True, False]

@pytest.fixture
def sample_numbers_for_comparison():
    """Sample numbers for comparison testing."""
    return [5, 3]

@pytest.fixture
def sample_numbers_for_arithmetic_operations():
    """Sample numbers for arithmetic operations testing."""
    return [2, 3, 4, 5, 10, 2, 3, 1, 2, 3, 8]

@pytest.fixture
def sample_numbers_for_bitwise_operations():
    """Sample numbers for bitwise operations testing."""
    return [5, 3, 1, 7, 6, -6, 10, 5]

@pytest.fixture
def sample_numbers_for_membership_operations():
    """Sample numbers for membership operations testing."""
    return [1, 2, 3, 4, 5, 6]

@pytest.fixture
def sample_string_for_membership_operations():
    """Sample string for membership operations testing."""
    return "hello"

@pytest.fixture
def sample_objects_for_identity_operations():
    """Sample objects for identity operations testing."""
    a = [1, 2, 3]
    b = [1, 2, 3]
    c = a
    return {"a": a, "b": b, "c": c}