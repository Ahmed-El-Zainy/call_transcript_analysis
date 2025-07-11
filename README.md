# Call Transcript Issue Classifier (AI-Driven)

An AI-powered tool that analyzes customer support call transcripts and automatically categorizes issues based on dynamic category structures. Uses advanced language models (Google Gemini or Hugging Face) to provide accurate, context-aware classification.

## ✨ Features

- **Dynamic Category Support**: Handles nested category structures of varying depths
- **Multiple AI Providers**: Supports both Google Gemini and Hugging Face models
- **Robust Error Handling**: Comprehensive validation and error recovery
- **Flexible Output**: JSON output with optional metadata
- **Production Ready**: Logging, configuration management, and proper error handling

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Google Gemini API Key OR Hugging Face API Token

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ahmed-El-Zainy/call_transcript_analysis.git
   cd call-transcript-analysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   # For Google Gemini (recommended)
   GEMINI_API_KEY=your_gemini_api_key_here
   
   # For Hugging Face
   HF_TOKEN=your_huggingface_token_here
   ```

   Or set environment variables directly:
   ```bash
   export GEMINI_API_KEY="your_gemini_api_key_here"
   export HF_TOKEN="your_huggingface_token_here"
   ```

### Basic Usage

```bash
# Using Google Gemini (default)
python main.py --transcript transcript.txt --categories categories.json

# Using Hugging Face
python main.py --transcript transcript.txt --categories categories.json --provider huggingface

# Save output to file
python main.py --transcript transcript.txt --categories categories.json --output results.json

# Enable verbose logging
python main.py --transcript transcript.txt --categories categories.json --verbose
```

## 📁 File Formats

### Input Files

#### 1. Transcript File (`transcript.txt`)
Plain text file containing the call transcript:

```
Agent: Hello, thank you for calling customer support. How can I help you today?

Customer: Hi, I'm having trouble with my internet connection. It's been really slow for the past few days.

Agent: I'm sorry to hear that. Let me check your account...
```

#### 2. Categories File (`categories.json`)
JSON file with hierarchical category structure:

```json
[
  {
    "name": "Network Issue",
    "subcategories": [
      {"name": "Slow Connection", "subcategories": []},
      {"name": "No Connection", "subcategories": []},
      {"name": "Intermittent Connection", "subcategories": []}
    ]
  },
  {
    "name": "Billing",
    "subcategories": [
      {
        "name": "Cost Inquiry",
        "subcategories": [
          {"name": "DSL", "subcategories": []},
          {"name": "Cable", "subcategories": []}
        ]
      },
      {"name": "Payment Issue", "subcategories": []}
    ]
  }
]
```

### Output Format

```json
{
  "analysis_results": [
    ["Network Issue", "Slow Connection"],
    ["Billing", "Cost Inquiry", "DSL"]
  ],
  "metadata": {
    "provider": "gemini",
    "transcript_length": 1234,
    "num_categories": 3,
    "num_issues_found": 2
  }
}
```

## 🔧 Advanced Usage

### Command Line Options

```bash
python main.py --help
```

Options:
- `--transcript`: Path to transcript file (required)
- `--categories`: Path to categories JSON file (required)
- `--provider`: AI provider (`gemini` or `huggingface`, default: `gemini`)
- `--output`: Output file path (default: stdout)
- `--verbose`: Enable verbose logging

### API Keys Setup

#### Google Gemini API
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Set the `GEMINI_API_KEY` environment variable

#### Hugging Face API
1. Visit [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Create a new token with "Read" permissions
3. Set the `HF_TOKEN` environment variable

## 🧪 Testing

### Create Sample Files

Run the script to generate sample files for testing:

```python
# In Python shell or script
from main import create_sample_files
create_sample_files()
```

This creates:
- `sample_transcript.txt`: Example call transcript
- `sample_categories.json`: Example category structure

### Run with Sample Data

```bash
python main.py --transcript sample_transcript.txt --categories sample_categories.json --verbose
```

## 🏗️ Architecture

### Workflow Overview

```
Input Files → Prompt Building → LLM API Call → Output Parsing → JSON Result
```

### Key Components

1. **TranscriptAnalyzer**: Main orchestrator class
2. **Input Loader**: Handles file reading and validation
3. **Prompt Builder**: Creates dynamic prompts from categories
4. **API Callers**: Interfaces with Gemini/Hugging Face
5. **Output Parser**: Validates and structures LLM responses

### Design Principles

- **Single API Call**: Efficient one-shot prompting approach
- **Dynamic Categories**: Handles varying category depths automatically
- **Error Resilience**: Comprehensive error handling and recovery
- **Provider Agnostic**: Easy switching between AI providers
- **Extensible**: Clean architecture for adding new providers

## 📊 Supported AI Models

### Google Gemini
- **Model**: `gemini-1.5-flash`
- **Strengths**: Excellent reasoning, fast response times
- **Best for**: Complex categorization tasks

### Hugging Face
- **Model**: `meta-llama/Meta-Llama-3-70B-Instruct`
- **Strengths**: Open source, customizable
- **Best for**: Privacy-sensitive applications

## 🔍 Troubleshooting

### Common Issues

1. **API Key Errors**
   ```
   ValueError: GEMINI_API_KEY environment variable not set
   ```
   **Solution**: Set the appropriate API key in your environment

2. **File Not Found**
   ```
   FileNotFoundError: [Errno 2] No such file or directory
   ```
   **Solution**: Check file paths and ensure files exist

3. **Invalid JSON in Categories**
   ```
   json.JSONDecodeError: Expecting value
   ```
   **Solution**: Validate your categories.json file format

4. **Empty Results**
   ```
   {"analysis_results": []}
   ```
   **Solution**: Check if issues in transcript match category structure

### Enable Debug Logging

```bash
python main.py --transcript transcript.txt --categories categories.json --verbose
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google AI for the Gemini API
- Hugging Face for the Transformers and Inference API
- Meta for the Llama models

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs with `--verbose` flag
3. Open an issue on GitHub with:
   - Error message
   - Input files (anonymized)
   - Command used
   - Environment details