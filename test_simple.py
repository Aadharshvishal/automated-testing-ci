#!/usr/bin/env python3
"""
Simple Test Suite for Assignment 4 - Automated Testing & CI
This test suite uses only basic Python functionality.
"""

import pytest
import os
import sys
import math
import json
import time
from datetime import datetime

def test_pytest_installation():
    """Test that pytest is properly installed and working."""
    import pytest
    assert hasattr(pytest, 'main')
    assert hasattr(pytest, 'fixture')

def test_python_version():
    """Test Python version compatibility."""
    assert sys.version_info >= (3, 8)
    assert True

def test_os_compatibility():
    """Test OS compatibility."""
    import platform
    assert platform.system() in ["Windows", "Linux", "Darwin"]
    assert True

def test_math_operations():
    """Test basic math operations."""
    assert math.sqrt(16) == 4.0
    assert math.pi > 3.14
    assert 2 + 2 == 4
    assert 10 * 5 == 50
    assert True

def test_string_operations():
    """Test string operations."""
    test_string = "Hello, World!"
    assert test_string.upper() == "HELLO, WORLD!"
    assert test_string.lower() == "hello, world!"
    assert len(test_string) == 13
    assert "Hello" in test_string
    assert True

def test_list_operations():
    """Test list operations."""
    test_list = [1, 2, 3, 4, 5]
    assert len(test_list) == 5
    assert sum(test_list) == 15
    assert max(test_list) == 5
    assert min(test_list) == 1
    assert 3 in test_list
    assert True

def test_dictionary_operations():
    """Test dictionary operations."""
    test_dict = {"key1": "value1", "key2": "value2"}
    assert len(test_dict) == 2
    assert "key1" in test_dict
    assert test_dict["key1"] == "value1"
    assert test_dict.get("key2") == "value2"
    assert True

def test_set_operations():
    """Test set operations."""
    test_set = {1, 2, 3, 4, 5}
    assert len(test_set) == 5
    assert 3 in test_set
    assert 6 not in test_set
    assert {1, 2}.issubset(test_set)
    assert True

def test_tuple_operations():
    """Test tuple operations."""
    test_tuple = (1, 2, 3, 4, 5)
    assert len(test_tuple) == 5
    assert test_tuple[0] == 1
    assert test_tuple[-1] == 5
    assert test_tuple.count(3) == 1
    assert True

def test_file_operations():
    """Test file operations functionality."""
    import tempfile
    
    # Test file creation and reading
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("test content")
        temp_file = f.name
    
    # Test file reading
    with open(temp_file, 'r') as f:
        content = f.read()
        assert content == "test content"
    
    # Cleanup
    os.unlink(temp_file)
    assert True

def test_environment_variables():
    """Test environment variable handling."""
    # Test setting and getting environment variables
    test_var = "TEST_ENV_VAR"
    test_value = "test_value_123"
    
    os.environ[test_var] = test_value
    assert os.environ.get(test_var) == test_value
    
    # Cleanup
    del os.environ[test_var]
    assert True

def test_logging_functionality():
    """Test logging functionality."""
    import logging
    
    # Test logger creation
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.INFO)
    
    # Test logging (should not raise exception)
    logger.info("Test log message")
    
    # Test logger configuration
    assert logger.name == "test_logger"
    assert logger.level == logging.INFO
    assert True

def test_performance_measurement():
    """Test performance measurement functionality."""
    # Test timing
    start_time = time.time()
    time.sleep(0.01)  # Simulate work
    end_time = time.time()
    
    duration = end_time - start_time
    assert duration >= 0.01
    assert duration < 1.0  # Should be much less than 1 second
    assert True

def test_data_serialization():
    """Test data serialization functionality."""
    # Test JSON serialization
    test_data = {
        "id": 1,
        "name": "Test",
        "timestamp": datetime.now().isoformat(),
        "active": True
    }
    
    # Serialize
    json_str = json.dumps(test_data)
    assert isinstance(json_str, str)
    
    # Deserialize
    deserialized = json.loads(json_str)
    assert deserialized["id"] == 1
    assert deserialized["name"] == "Test"
    assert deserialized["active"] is True
    assert True

def test_regular_expressions():
    """Test regular expression functionality."""
    import re
    pattern = r'\d+'
    text = "There are 123 apples and 456 oranges"
    matches = re.findall(pattern, text)
    assert matches == ['123', '456']
    assert re.search(pattern, text) is not None
    assert True

def test_datetime_operations():
    """Test datetime operations."""
    from datetime import datetime, timedelta
    now = datetime.now()
    future = now + timedelta(days=1)
    assert future > now
    assert (future - now).days == 1
    assert True

def test_collections_module():
    """Test collections module functionality."""
    from collections import Counter, defaultdict
    counter = Counter([1, 1, 2, 3, 3, 3])
    assert counter[3] == 3
    assert counter[1] == 2
    assert counter[2] == 1
    
    # Test defaultdict
    dd = defaultdict(list)
    dd['key'].append('value')
    assert dd['key'] == ['value']
    assert True

def test_itertools_module():
    """Test itertools module functionality."""
    import itertools
    numbers = [1, 2, 3]
    combinations = list(itertools.combinations(numbers, 2))
    assert len(combinations) == 3
    assert (1, 2) in combinations
    assert (2, 3) in combinations
    assert True

def test_functools_module():
    """Test functools module functionality."""
    from functools import reduce
    numbers = [1, 2, 3, 4, 5]
    product = reduce(lambda x, y: x * y, numbers)
    assert product == 120
    assert True

def test_operator_module():
    """Test operator module functionality."""
    import operator
    assert operator.add(2, 3) == 5
    assert operator.mul(4, 5) == 20
    assert operator.eq(1, 1) is True
    assert True

def test_statistics_module():
    """Test statistics module functionality."""
    import statistics
    data = [1, 2, 3, 4, 5]
    assert statistics.mean(data) == 3.0
    assert statistics.median(data) == 3
    assert statistics.mode([1, 1, 2, 3]) == 1
    assert True

def test_random_module():
    """Test random module functionality."""
    import random
    # Test that random numbers are generated
    rand_num = random.random()
    assert 0 <= rand_num <= 1
    
    # Test random choice
    choices = [1, 2, 3, 4, 5]
    choice = random.choice(choices)
    assert choice in choices
    assert True

def test_hashlib_module():
    """Test hashlib module functionality."""
    import hashlib
    text = "Hello, World!"
    hash_obj = hashlib.md5(text.encode())
    hash_value = hash_obj.hexdigest()
    assert len(hash_value) == 32
    assert isinstance(hash_value, str)
    assert True

def test_base64_module():
    """Test base64 module functionality."""
    import base64
    text = "Hello, World!"
    encoded = base64.b64encode(text.encode()).decode()
    decoded = base64.b64decode(encoded).decode()
    assert decoded == text
    assert isinstance(encoded, str)
    assert True

def test_urllib_module():
    """Test urllib module functionality."""
    from urllib.parse import urlparse, urljoin
    url = "https://example.com/path?query=1"
    parsed = urlparse(url)
    assert parsed.scheme == "https"
    assert parsed.netloc == "example.com"
    assert parsed.path == "/path"
    assert True

def test_pathlib_module():
    """Test pathlib module functionality."""
    from pathlib import Path
    path = Path("/tmp/test.txt")
    assert path.name == "test.txt"
    assert path.suffix == ".txt"
    assert path.stem == "test"
    assert True

def test_concurrent_execution():
    """Test concurrent execution functionality."""
    import asyncio
    
    async def async_task(task_id, delay):
        await asyncio.sleep(delay)
        return f"Task {task_id} completed"
    
    async def run_concurrent_tasks():
        tasks = [
            async_task(1, 0.01),
            async_task(2, 0.01),
            async_task(3, 0.01)
        ]
        results = await asyncio.gather(*tasks)
        return results
    
    # Run concurrent tasks
    results = asyncio.run(run_concurrent_tasks())
    
    assert len(results) == 3
    assert "Task 1 completed" in results
    assert "Task 2 completed" in results
    assert "Task 3 completed" in results
    assert True

def test_memory_usage():
    """Test memory usage monitoring."""
    try:
        import psutil
        import os
        
        # Get current process memory usage
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        # Basic memory usage checks
        assert memory_info.rss > 0  # Resident Set Size
        assert memory_info.vms > 0  # Virtual Memory Size
        assert True
    except ImportError:
        pytest.skip("psutil not available for memory testing")

def test_zip_functionality():
    """Test zip functionality."""
    list1 = [1, 2, 3]
    list2 = ['a', 'b', 'c']
    zipped = list(zip(list1, list2))
    assert zipped == [(1, 'a'), (2, 'b'), (3, 'c')]
    assert True

def test_enumerate_functionality():
    """Test enumerate functionality."""
    items = ['a', 'b', 'c']
    enumerated = list(enumerate(items))
    assert enumerated == [(0, 'a'), (1, 'b'), (2, 'c')]
    assert True

def test_map_functionality():
    """Test map functionality."""
    numbers = [1, 2, 3, 4, 5]
    squared = list(map(lambda x: x**2, numbers))
    assert squared == [1, 4, 9, 16, 25]
    assert True

def test_filter_functionality():
    """Test filter functionality."""
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
    assert even_numbers == [2, 4, 6, 8, 10]
    assert True

def test_sorted_functionality():
    """Test sorted functionality."""
    numbers = [3, 1, 4, 1, 5, 9, 2, 6]
    sorted_numbers = sorted(numbers)
    assert sorted_numbers == [1, 1, 2, 3, 4, 5, 6, 9]
    assert True

def test_reversed_functionality():
    """Test reversed functionality."""
    numbers = [1, 2, 3, 4, 5]
    reversed_numbers = list(reversed(numbers))
    assert reversed_numbers == [5, 4, 3, 2, 1]
    assert True

def test_any_all_functionality():
    """Test any and all functionality."""
    numbers = [1, 2, 3, 4, 5]
    assert any(x > 3 for x in numbers) is True
    assert all(x > 0 for x in numbers) is True
    assert all(x > 3 for x in numbers) is False
    assert True

def test_sum_functionality():
    """Test sum functionality."""
    numbers = [1, 2, 3, 4, 5]
    total = sum(numbers)
    assert total == 15
    assert sum(numbers, 10) == 25  # With start value
    assert True

def test_max_min_functionality():
    """Test max and min functionality."""
    numbers = [1, 5, 3, 9, 2]
    assert max(numbers) == 9
    assert min(numbers) == 1
    assert max(1, 2, 3, 4, 5) == 5
    assert min(1, 2, 3, 4, 5) == 1
    assert True

def test_len_functionality():
    """Test len functionality."""
    assert len([1, 2, 3]) == 3
    assert len("hello") == 5
    assert len({"a": 1, "b": 2}) == 2
    assert len((1, 2, 3, 4)) == 4
    assert True

def test_range_functionality():
    """Test range functionality."""
    numbers = list(range(5))
    assert numbers == [0, 1, 2, 3, 4]
    
    even_numbers = list(range(0, 10, 2))
    assert even_numbers == [0, 2, 4, 6, 8]
    assert True

def test_slice_functionality():
    """Test slice functionality."""
    numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert numbers[2:5] == [2, 3, 4]
    assert numbers[:3] == [0, 1, 2]
    assert numbers[7:] == [7, 8, 9]
    assert numbers[::2] == [0, 2, 4, 6, 8]
    assert True

def test_list_comprehension():
    """Test list comprehension functionality."""
    numbers = [1, 2, 3, 4, 5]
    squared = [x**2 for x in numbers]
    assert squared == [1, 4, 9, 16, 25]
    
    even_squares = [x**2 for x in numbers if x % 2 == 0]
    assert even_squares == [4, 16]
    assert True

def test_dict_comprehension():
    """Test dictionary comprehension functionality."""
    numbers = [1, 2, 3, 4, 5]
    squared_dict = {x: x**2 for x in numbers}
    assert squared_dict == {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}
    assert True

def test_set_comprehension():
    """Test set comprehension functionality."""
    numbers = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
    unique_squares = {x**2 for x in numbers}
    assert unique_squares == {1, 4, 9, 16}
    assert True

def test_generator_expression():
    """Test generator expression functionality."""
    numbers = [1, 2, 3, 4, 5]
    gen = (x**2 for x in numbers)
    squared_list = list(gen)
    assert squared_list == [1, 4, 9, 16, 25]
    assert True

def test_lambda_functionality():
    """Test lambda functionality."""
    add = lambda x, y: x + y
    assert add(2, 3) == 5
    
    multiply = lambda x, y: x * y
    assert multiply(4, 5) == 20
    assert True

def test_closure_functionality():
    """Test closure functionality."""
    def outer(x):
        def inner(y):
            return x + y
        return inner
    
    add_five = outer(5)
    assert add_five(3) == 8
    assert add_five(10) == 15
    assert True

def test_decorator_functionality():
    """Test decorator functionality."""
    def my_decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs) * 2
        return wrapper
    
    @my_decorator
    def add(x, y):
        return x + y
    
    assert add(2, 3) == 10  # (2 + 3) * 2
    assert True

def test_context_manager():
    """Test context manager functionality."""
    class MyContextManager:
        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
    
    with MyContextManager() as cm:
        assert cm is not None
    assert True

def test_exception_handling():
    """Test exception handling functionality."""
    try:
        result = 10 / 0
    except ZeroDivisionError:
        assert True
    else:
        assert False, "Should have raised ZeroDivisionError"
    
    try:
        result = 10 / 2
        assert result == 5
    except ZeroDivisionError:
        assert False, "Should not have raised ZeroDivisionError"
    assert True

def test_assertion_functionality():
    """Test assertion functionality."""
    assert 1 == 1
    assert 2 + 2 == 4
    assert "hello" == "hello"
    assert [1, 2, 3] == [1, 2, 3]
    assert True

def test_boolean_operations():
    """Test boolean operations functionality."""
    assert True and True
    assert not False
    assert True or False
    assert not (True and False)
    assert True

def test_comparison_operations():
    """Test comparison operations functionality."""
    assert 5 > 3
    assert 3 < 5
    assert 5 >= 5
    assert 3 <= 5
    assert 5 == 5
    assert 5 != 3
    assert True

def test_arithmetic_operations():
    """Test arithmetic operations functionality."""
    assert 2 + 3 == 5
    assert 5 - 2 == 3
    assert 3 * 4 == 12
    assert 10 / 2 == 5
    assert 10 // 3 == 3
    assert 10 % 3 == 1
    assert 2 ** 3 == 8
    assert True

def test_bitwise_operations():
    """Test bitwise operations functionality."""
    assert 5 & 3 == 1  # 101 & 011 = 001
    assert 5 | 3 == 7  # 101 | 011 = 111
    assert 5 ^ 3 == 6  # 101 ^ 011 = 110
    assert ~5 == -6    # bitwise NOT
    assert 5 << 1 == 10  # left shift
    assert 10 >> 1 == 5  # right shift
    assert True

def test_membership_operations():
    """Test membership operations functionality."""
    numbers = [1, 2, 3, 4, 5]
    assert 3 in numbers
    assert 6 not in numbers
    assert "h" in "hello"
    assert "x" not in "hello"
    assert True

def test_identity_operations():
    """Test identity operations functionality."""
    a = [1, 2, 3]
    b = [1, 2, 3]
    c = a
    
    assert a == b  # values are equal
    assert a is not b  # different objects
    assert a is c  # same object
    assert True
