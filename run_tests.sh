#!/bin/bash

# Activate virtualenv if exists
if [ -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

echo "🚀 Running Tests with Coverage using pytest (verbose output)..."
coverage run -m pytest -v --tb=long
echo ""

echo "📊 Generating Coverage Report..."
coverage report -m
coverage html
echo ""
echo "✅ Detailed HTML report generated in 'htmlcov/index.html'"
