#!/usr/bin/env python3
"""
Standalone Test Suite for Assignment 4 - Automated Testing & CI
This test suite focuses on testing the testing infrastructure itself.
"""

import pytest
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

def test_pytest_installation():
    """Test that pytest is properly installed and working."""
    import pytest
    assert hasattr(pytest, 'main')
    assert hasattr(pytest, 'fixture')

def test_coverage_installation():
    """Test that pytest-cov is properly installed."""
    try:
        import pytest_cov
        assert True
    except ImportError:
        pytest.skip("pytest-cov not installed")

def test_fastapi_installation():
    """Test that FastAPI is available for testing."""
    try:
        import fastapi
        from fastapi.testclient import TestClient
        assert hasattr(fastapi, 'FastAPI')
        assert TestClient is not None
    except ImportError:
        pytest.skip("FastAPI not installed")

def test_sqlalchemy_installation():
    """Test that SQLAlchemy is available for testing."""
    try:
        import sqlalchemy
        assert hasattr(sqlalchemy, 'create_engine')
        assert hasattr(sqlalchemy, 'Column')
    except ImportError:
        pytest.skip("SQLAlchemy not installed")

def test_pydantic_installation():
    """Test that Pydantic is available for testing."""
    try:
        import pydantic
        assert hasattr(pydantic, 'BaseModel')
        assert hasattr(pydantic, 'Field')
    except ImportError:
        pytest.skip("Pydantic not installed")

def test_httpx_installation():
    """Test that httpx is available for testing."""
    try:
        import httpx
        assert hasattr(httpx, 'Client')
        assert hasattr(httpx, 'AsyncClient')
    except ImportError:
        pytest.skip("httpx not installed")

def test_jose_installation():
    """Test that python-jose is available for JWT testing."""
    try:
        from jose import jwt
        assert jwt is not None
    except ImportError:
        pytest.skip("python-jose not installed")

def test_passlib_installation():
    """Test that passlib is available for password hashing."""
    try:
        from passlib.context import CryptContext
        assert CryptContext is not None
    except ImportError:
        pytest.skip("passlib not installed")

def test_database_connection():
    """Test database connection functionality."""
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker
    
    # Create in-memory SQLite database
    engine = create_engine("sqlite:///:memory:")
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Test connection
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        assert result.fetchone()[0] == 1

def test_fastapi_app_creation():
    """Test creating a FastAPI application."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    
    app = FastAPI(title="Test App")
    
    @app.get("/")
    def read_root():
        return {"message": "Hello World"}
    
    @app.get("/health")
    def health_check():
        return {"status": "healthy"}
    
    # Test with TestClient
    client = TestClient(app)
    
    # Test root endpoint
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}
    
    # Test health endpoint
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_pydantic_model_validation():
    """Test Pydantic model validation."""
    from pydantic import BaseModel, Field
    from typing import Optional
    from datetime import datetime
    
    class UserCreate(BaseModel):
        username: str = Field(..., min_length=3, max_length=50)
        email: str = Field(..., pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')
        password: str = Field(..., min_length=8)
        is_active: bool = True
        created_at: Optional[datetime] = None
    
    # Test valid data
    valid_data = {
        "username": "testuser",
        "email": "test@example.com",
        "password": "password123"
    }
    
    user = UserCreate(**valid_data)
    assert user.username == "testuser"
    assert user.email == "test@example.com"
    assert user.is_active == True
    
    # Test invalid data
    with pytest.raises(Exception):
        UserCreate(username="ab", email="invalid-email", password="123")

def test_password_hashing():
    """Test password hashing functionality."""
    from passlib.context import CryptContext
    
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    password = "testpassword123"
    hashed = pwd_context.hash(password)
    
    # Hash should be different from original
    assert hashed != password
    
    # Verification should work
    assert pwd_context.verify(password, hashed) == True
    
    # Wrong password should fail
    assert pwd_context.verify("wrongpassword", hashed) == False

def test_jwt_token_creation():
    """Test JWT token creation and verification."""
    from jose import jwt
    from datetime import datetime, timedelta
    
    SECRET_KEY = "test-secret-key"
    ALGORITHM = "HS256"
    
    # Create token
    data = {"sub": "testuser", "exp": datetime.utcnow() + timedelta(minutes=30)}
    token = jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)
    
    # Token should be created
    assert token is not None
    assert isinstance(token, str)
    
    # Token should be decodable
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    assert payload["sub"] == "testuser"

def test_sqlalchemy_model_creation():
    """Test SQLAlchemy model creation."""
    from sqlalchemy import Column, Integer, String, Boolean, DateTime
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from datetime import datetime
    
    Base = declarative_base()
    
    class TestUser(Base):
        __tablename__ = "test_users"
        
        id = Column(Integer, primary_key=True)
        username = Column(String(50), unique=True, nullable=False)
        email = Column(String(100), unique=True, nullable=False)
        is_active = Column(Boolean, default=True)
        created_at = Column(DateTime, default=datetime.utcnow)
    
    # Create in-memory database
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Test model creation
    db = SessionLocal()
    user = TestUser(username="testuser", email="test@example.com")
    db.add(user)
    db.commit()
    db.refresh(user)
    
    assert user.id is not None
    assert user.username == "testuser"
    assert user.email == "test@example.com"
    assert user.is_active == True
    
    db.close()

def test_fastapi_dependency_injection():
    """Test FastAPI dependency injection."""
    from fastapi import FastAPI, Depends
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.ext.declarative import declarative_base
    
    Base = declarative_base()
    engine = create_engine("sqlite:///:memory:")
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    def get_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()
    
    app = FastAPI()
    
    @app.get("/test-db")
    def test_db_endpoint(db = Depends(get_db)):
        return {"message": "Database dependency working"}
    
    client = TestClient(app)
    response = client.get("/test-db")
    assert response.status_code == 200
    assert response.json() == {"message": "Database dependency working"}

def test_error_handling():
    """Test error handling in FastAPI."""
    from fastapi import FastAPI, HTTPException
    from fastapi.testclient import TestClient
    
    app = FastAPI()
    
    @app.get("/error-test")
    def error_test():
        raise HTTPException(status_code=404, detail="Not found")
    
    client = TestClient(app)
    response = client.get("/error-test")
    assert response.status_code == 404
    assert response.json()["detail"] == "Not found"

def test_cors_middleware():
    """Test CORS middleware functionality."""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.testclient import TestClient
    
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/")
    def read_root():
        return {"message": "CORS enabled"}
    
    # Test that the app is properly configured
    assert len(app.user_middleware) > 0
    assert any("CORSMiddleware" in str(middleware) for middleware in app.user_middleware)
    
    # Test basic functionality
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "CORS enabled"}

def test_async_endpoints():
    """Test async endpoint functionality."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import asyncio
    
    app = FastAPI()
    
    @app.get("/async-test")
    async def async_endpoint():
        await asyncio.sleep(0.01)  # Simulate async operation
        return {"message": "Async endpoint working"}
    
    client = TestClient(app)
    response = client.get("/async-test")
    assert response.status_code == 200
    assert response.json() == {"message": "Async endpoint working"}

def test_file_operations():
    """Test file operations for test artifacts."""
    import tempfile
    import json
    
    # Test creating temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        test_data = {"test": "data", "number": 123}
        json.dump(test_data, f)
        temp_file = f.name
    
    # Test reading the file
    with open(temp_file, 'r') as f:
        loaded_data = json.load(f)
        assert loaded_data == test_data
    
    # Clean up
    os.unlink(temp_file)

def test_environment_variables():
    """Test environment variable handling."""
    import os
    
    # Test setting and getting environment variables
    test_var = "TEST_ENV_VAR"
    test_value = "test_value_123"
    
    os.environ[test_var] = test_value
    assert os.environ.get(test_var) == test_value
    
    # Clean up
    del os.environ[test_var]

def test_logging_functionality():
    """Test logging functionality for test output."""
    import logging
    
    # Set up logger
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.INFO)
    
    # Test logging - this should not raise an exception
    logger.info("Test log message")
    
    # If we get here, logging is working
    assert True

def test_performance_measurement():
    """Test performance measurement capabilities."""
    import time
    
    start_time = time.time()
    
    # Simulate some work
    time.sleep(0.01)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Duration should be reasonable
    assert duration >= 0.01
    assert duration < 0.1

def test_memory_usage():
    """Test memory usage monitoring."""
    import psutil
    import os
    
    # Get current process memory usage
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    # Memory usage should be reasonable (less than 100MB for this test)
    assert memory_info.rss < 100 * 1024 * 1024  # 100MB

def test_concurrent_execution():
    """Test concurrent execution capabilities."""
    import threading
    import time
    
    results = []
    
    def worker(worker_id):
        time.sleep(0.01)
        results.append(f"worker_{worker_id}")
    
    # Create and start threads
    threads = []
    for i in range(3):
        thread = threading.Thread(target=worker, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Check results
    assert len(results) == 3
    assert "worker_0" in results
    assert "worker_1" in results
    assert "worker_2" in results

def test_data_serialization():
    """Test data serialization for API responses."""
    import json
    from datetime import datetime
    
    # Test data with various types
    test_data = {
        "string": "test",
        "number": 123,
        "float": 123.45,
        "boolean": True,
        "list": [1, 2, 3],
        "dict": {"nested": "value"},
        "datetime": datetime.now().isoformat()
    }
    
    # Test JSON serialization
    json_str = json.dumps(test_data)
    assert isinstance(json_str, str)
    
    # Test JSON deserialization
    loaded_data = json.loads(json_str)
    assert loaded_data["string"] == "test"
    assert loaded_data["number"] == 123
    assert loaded_data["boolean"] == True

def test_configuration_management():
    """Test configuration management for different environments."""
    import os
    
    # Test different environment configurations
    environments = ["development", "testing", "production"]
    
    for env in environments:
        os.environ["ENVIRONMENT"] = env
        
        # Simulate environment-specific configuration
        if env == "development":
            debug = True
            log_level = "DEBUG"
        elif env == "testing":
            debug = False
            log_level = "INFO"
        else:  # production
            debug = False
            log_level = "WARNING"
        
        assert isinstance(debug, bool)
        assert log_level in ["DEBUG", "INFO", "WARNING", "ERROR"]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
