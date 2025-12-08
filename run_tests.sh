#!/bin/bash

# Activate virtualenv if exists
if [ -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi



echo "🚀 Running Tests with Coverage..."
coverage run -m unittest discover tests
echo ""

echo "📊 Generating Coverage Report..."
coverage report -m
coverage html
echo ""
echo "✅ Detailed HTML report generated in 'htmlcov/index.html'"
