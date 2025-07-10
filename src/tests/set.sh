#!/bin/bash

# Call Transcript Analysis - Setup Script
# This script sets up the environment and provides easy commands to run the analysis

set -e  # Exit on any error

echo "🔧 Call Transcript Analysis - Setup & Run Script"
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Check Python version
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
        print_status "Python 3 found: $PYTHON_VERSION"
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
        if [[ $PYTHON_VERSION == 3.* ]]; then
            print_status "Python 3 found: $PYTHON_VERSION"
            PYTHON_CMD="python"
        else
            print_error "Python 3.8+ required, found: $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python not found. Please install Python 3.8+"
        exit 1
    fi
}

# Install dependencies
install_dependencies() {
    print_info "Installing dependencies..."
    
    if [ -f "requirements.txt" ]; then
        $PYTHON_CMD -m pip install -r requirements.txt
        print_status "Dependencies installed"
    else
        print_warning "requirements.txt not found, installing core dependencies..."
        $PYTHON_CMD -m pip install google-generativeai requests python-dotenv
        print_status "Core dependencies installed"
    fi
}

# Setup environment file
setup_env() {
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
            print_info "Created .env file from .env.example"
            print_warning "Please edit .env and add your API keys"
        else
            cat > .env << EOF
# Call Transcript Analysis - Environment Variables
# Add your API keys below

# Google Gemini API Key (recommended)
GEMINI_API_KEY=your_gemini_api_key_here

# Hugging Face API Token (alternative)
HF_TOKEN=your_huggingface_token_here
EOF
            print_info "Created .env file"
            print_warning "Please edit .env and add your API keys"
        fi
    else
        print_status "Environment file already exists"
    fi
}

# Create sample data
create_samples() {
    print_info "Creating sample data files..."
    
    if [ -f "main.py" ]; then
        $PYTHON_CMD -c "
from main import create_sample_files
create_sample_files()
print('Sample files created: sample_transcript.txt, sample_categories.json')
"
        print_status "Sample files created"
    else
        print_error "main.py not found. Please ensure all files are in the current directory."
        exit 1
    fi
}

# Run tests
run_tests() {
    print_info "Running tests..."
    
    if [ -f "test_workflow.py" ]; then
        $PYTHON_CMD test_workflow.py
        print_status "Tests completed"
    else
        print_warning "test_workflow.py not found, skipping tests"
    fi
}

# Check API keys
check_api_keys() {
    if [ -f ".env" ]; then
        source .env
        
        if [ -n "$GEMINI_API_KEY" ] && [ "$GEMINI_API_KEY" != "your_gemini_api_key_here" ]; then
            print_status "Gemini API key configured"
            API_AVAILABLE=true
        elif [ -n "$HF_TOKEN" ] && [ "$HF_TOKEN" != "your_huggingface_token_here" ]; then
            print_status "Hugging Face token configured"
            API_AVAILABLE=true
        else
            print_warning "No API keys configured. Please edit .env file."
            API_AVAILABLE=false
        fi
    else
        print_warning "No .env file found. API keys not configured."
        API_AVAILABLE=false
    fi
}

# Run analysis with sample data
run_sample_analysis() {
    if [ "$API_AVAILABLE" = true ]; then
        print_info "Running sample analysis..."
        
        if [ -f "sample_transcript.txt" ] && [ -f "sample_categories.json" ]; then
            $PYTHON_CMD main.py --transcript sample_transcript.txt --categories sample_categories.json --verbose
            print_status "Sample analysis completed"
        else
            print_error "Sample files not found. Run: $0 --create-samples"
        fi
    else
        print_warning "Cannot run analysis without API keys. Please configure .env file."
    fi
}

# Show usage
show_usage() {
    echo
    echo "Usage: $0 [OPTION]"
    echo
    echo "Options:"
    echo "  --setup           Full setup (install deps, create env, create samples)"
    echo "  --install         Install dependencies only"
    echo "  --create-env      Create .env file only"
    echo "  --create-samples  Create sample data files only"
    echo "  --test            Run tests only"
    echo "  --run-sample      Run analysis with sample data"
    echo "  --check           Check system requirements and configuration"
    echo "  --help            Show this help message"
    echo
    echo "Examples:"
    echo "  $0 --setup                    # Complete setup"
    echo "  $0 --run-sample               # Run with sample data"
    echo "  $0 --check                    # Check configuration"
    echo
    echo "Manual usage:"
    echo "  python main.py --transcript transcript.txt --categories categories.json"
    echo "  python main.py --transcript transcript.txt --categories categories.json --provider huggingface"
    echo
}

# Main execution
main() {
    case "$1" in
        --setup)
            check_python
            install_dependencies
            setup_env
            create_samples
            check_api_keys
            print_status "Setup completed!"
            echo
            print_info "Next steps:"
            echo "1. Edit .env file and add your API keys"
            echo "2. Run: $0 --run-sample"
            ;;
        --install)
            check_python
            install_dependencies
            ;;
        --create-env)
            setup_env
            ;;
        --create-samples)
            check_python
            create_samples
            ;;
        --test)
            check_python
            run_tests
            ;;
        --run-sample)
            check_python
            check_api_keys
            run_sample_analysis
            ;;
        --check)
            check_python
            check_api_keys
            if [ -f "main.py" ]; then
                print_status "main.py found"
            else
                print_error "main.py not found"
            fi
            if [ -f "requirements.txt" ]; then
                print_status "requirements.txt found"
            else
                print_warning "requirements.txt not found"
            fi
            ;;
        --help|"")
            show_usage
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"