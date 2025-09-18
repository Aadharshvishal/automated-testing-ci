"""
Pytest configuration and shared fixtures for Assignment 4 tests.
"""

import pytest
import os
import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'assignment_3'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'assignment_2'))

# Import with proper error handling
try:
    from assignment_3.database import Base
    from assignment_2.database import Base as Base2
except ImportError:
    # Fallback for when running from assignment_4 directory
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from assignment_3.database import Base
    from assignment_2.database import Base as Base2

# Test database configuration
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_assignment4.db"

@pytest.fixture(scope="session")
def test_engine():
    """Create test database engine for the session."""
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    return engine

@pytest.fixture(scope="function")
def test_db(test_engine):
    """Create a fresh database for each test."""
    # Create all tables
    Base.metadata.create_all(bind=test_engine)
    Base2.metadata.create_all(bind=test_engine)
    
    # Create session
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    db = TestingSessionLocal()
    
    try:
        yield db
    finally:
        db.close()
        # Clean up tables
        Base.metadata.drop_all(bind=test_engine)
        Base2.metadata.drop_all(bind=test_engine)

@pytest.fixture
def sample_user_data():
    """Sample user data for testing."""
    return {
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpassword123"
    }

@pytest.fixture
def sample_author_data():
    """Sample author data for testing."""
    return {
        "name": "Test Author",
        "email": "author@example.com",
        "bio": "A test author for testing purposes",
        "birth_date": "1990-01-01",
        "is_active": True
    }

@pytest.fixture
def sample_book_data():
    """Sample book data for testing."""
    return {
        "title": "Test Book Title",
        "description": "A comprehensive test book for automated testing",
        "isbn": "1234567890123",
        "publication_date": "2023-01-01",
        "price": 29.99,
        "is_published": True,
        "author_id": 1
    }

@pytest.fixture
def auth_headers(test_db, sample_user_data):
    """Create authentication headers for protected endpoints."""
    from assignment_3.main import app
    from assignment_3.database import get_db
    from fastapi.testclient import TestClient
    
    # Override database dependency
    app.dependency_overrides[get_db] = lambda: test_db
    
    client = TestClient(app)
    
    # Register user
    client.post("/auth/register", json=sample_user_data)
    
    # Login and get token
    login_response = client.post("/auth/login", data={
        "username": sample_user_data["username"],
        "password": sample_user_data["password"]
    })
    
    token = login_response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Mark performance tests as slow
        if "performance" in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Mark integration tests
        if "integration" in item.name:
            item.add_marker(pytest.mark.integration)
        
        # Mark unit tests
        if any(keyword in item.name for keyword in ["schema", "utility", "password", "jwt"]):
            item.add_marker(pytest.mark.unit)
