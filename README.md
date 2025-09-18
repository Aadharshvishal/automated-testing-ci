# Assignment 4 - Automated Testing & CI

This assignment demonstrates comprehensive automated testing and continuous integration using pytest and GitHub Actions.

## Features

- **Comprehensive Test Suite**: 20+ pytest test cases covering:
  - API routes and endpoints
  - Pydantic schemas and validation
  - Authentication and authorization
  - Database operations
  - Error handling and edge cases
  - Utility functions
- **GitHub Actions CI/CD**: Automated testing on every push
- **PostgreSQL Integration**: Database testing with service containers
- **Test Coverage**: High coverage across all modules
- **Build Status**: Automated build status reporting

## Test Coverage

### API Routes Tests
- Authentication endpoints (login, register, token validation)
- Authors CRUD operations
- Books CRUD operations
- Votes system
- Error handling and edge cases

### Schema Tests
- Pydantic model validation
- Request/response schema validation
- Data type validation
- Required field validation

### Utility Tests
- Password hashing and verification
- JWT token creation and validation
- Database connection and operations
- Helper functions

## Running Tests Locally

### Prerequisites
- Python 3.8+
- PostgreSQL (for integration tests)
- pip

### Setup
```bash
cd assignment_4
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest test_comprehensive.py

# Run with verbose output
pytest -v
```

## GitHub Actions

The CI workflow (`.github/workflows/ci.yml`) automatically:
1. Checks out the code
2. Sets up Python environment
3. Installs dependencies
4. Spins up PostgreSQL service
5. Runs the complete test suite
6. Reports build status

## Build Status

[![CI](https://github.com/Aadharshvishal/automated-testing-ci/actions/workflows/ci.yml/badge.svg)](https://github.com/Aadharshvishal/automated-testing-ci/actions/workflows/ci.yml)

## Assignment Requirements Met

✅ **20+ pytest test cases** - Comprehensive test coverage  
✅ **Routes testing** - All API endpoints tested  
✅ **Schemas testing** - Pydantic models validated  
✅ **Utilities testing** - Helper functions tested  
✅ **GitHub Actions workflow** - Automated CI/CD pipeline  
✅ **PostgreSQL service** - Database integration testing  
✅ **Build failure on test failure** - Proper CI behavior  
✅ **Build status badge** - README integration  

## Test Structure

```
assignment_4/
├── test_standalone.py      # Main working test suite (26 tests)
├── test_comprehensive.py   # Comprehensive test suite
├── test_comprehensive_fixed.py # Fixed comprehensive test suite
├── conftest.py            # Pytest configuration and fixtures
├── run_tests.py           # Test runner script
├── setup.py               # Setup script
├── pytest.ini            # Pytest configuration
├── requirements.txt       # Test dependencies
├── .github/
│   └── workflows/
│       └── ci.yml        # GitHub Actions workflow
├── .gitignore            # Git ignore file
└── README.md             # This file
```
