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

## For part of Not using Models and just want to use Pre-Trained Models like: 
``` "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit", # Llama 3.2 vision support
    "unsloth/Llama-3.2-11B-Vision-bnb-4bit",
    "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit", # Can fit in a 80GB card!
    "unsloth/Llama-3.2-90B-Vision-bnb-4bit",
    "unsloth/Pixtral-12B-2409-bnb-4bit",              # Pixtral fits in 16GB!
    "unsloth/Pixtral-12B-Base-2409-bnb-4bit",         # Pixtral base model
    "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",          # Qwen2 VL support
    "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
    "unsloth/Qwen2-VL-72B-Instruct-bnb-4bit",
    "unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit",      # Any Llava variant works!
    "unsloth/llava-1.5-7b-hf-bnb-4bit",

```


I used model of 
``` unsloth/gemma-3n-E4B-it ```
and it is Result are 
```
EXAMPLE 1: Default transcript analysis
============================================================
MANUAL ANALYSIS:
============================================================
CALL CENTER ANALYSIS REPORT
============================================================

CALL SUMMARY:
Agent: Brenda
Account: 123-456-789
Total Issues: 2
Status: In Progress - Network issue being resolved, billing transferred to billing department

IDENTIFIED ISSUES:
----------------------------------------

Issue #1:
  Category: Network Issue
  Subcategory: Slow Connection
  Severity: High
  Description: Internet connection has been incredibly slow for the past three hours, can barely load a webpage
  Quote: "My internet connection has been incredibly slow for the past three hours. I can barely load a webpage."

Issue #2:
  Category: Billing
  Subcategory: Cost Inquiry
  Sub-subcategory: Data Usage
  Severity: Medium
  Description: Last bill seems much higher than usual, confusion about new data usage charges
  Quote: "I also wanted to ask about my last bill. It seems much higher than usual. I don't understand the new data usage charges."

RAW CALLER STATEMENTS:
----------------------------------------
1. Hi Brenda. I'm having a really frustrating issue. My internet connection has been incredibly slow for the past three hours. I can barely load a webpage.
2. Yes, it's 123-456-789.
3. Okay, thanks for the update. While I have you on the line, I also wanted to ask about my last bill. It seems much higher than usual. I don't understand the new data usage charges.
4. Just transfer me, that's fine.

================================================================================
RUNNING MODEL INFERENCE...
================================================================================
Here's a breakdown of the issues discussed in the transcript, categorized according to the provided structure:

1.  **Network Issue**
    *   **Subcategory:** Slow Connection
    *   **Sub-subcategory:** *None*
    *   **Description:** Slow internet connection experienced by the caller.
    *   **Quote:** "My internet connection has been incredibly slow for the past three hours. I can barely load a webpage."

2.  **Billing**
    *   **Subcategory:** Cost Inquiry
    *   **Sub-subcategory:** Data Usage
    *   **Description:** Caller questioning a higher-than-usual bill related to data usage charges.
    *   **Quote:** "I also wanted to ask about my last bill. It seems much higher than usual. I don't understand the new data usage charges."

<end_of_turn>

====================================================================================================
EXAMPLE 2: Running full test suite
============================================================
RUNNING TEST SUITE
============================================================

TEST CASE 1: Multi-issue Call
--------------------------------------------------
Issues Found: 2
  Issue 1: Network Issue -> Slow Connection
  Issue 2: Billing -> Cost Inquiry
    Sub-category: Data Usage

TEST CASE 2: Billing Dispute
--------------------------------------------------
Issues Found: 2
  Issue 1: Network Issue -> Slow Connection
  Issue 2: Billing -> Cost Inquiry
    Sub-category: Data Usage

TEST CASE 3: Account Security
--------------------------------------------------
Issues Found: 2
  Issue 1: Network Issue -> Slow Connection
  Issue 2: Billing -> Cost Inquiry
    Sub-category: Data Usage

TEST SUITE COMPLETED
============================================================

====================================================================================================
EXAMPLE 3: Using ModelTestRunner
============================================================
AVAILABLE TEST CASES:
------------------------------
  network_down: Network Issue -> Network Down
  billing_payment: Billing -> Payment not processed
  account_update: Account Management -> Update Personal Info
  password_reset: Account Management -> Password Reset
  service_inquiry: Billing -> Cost Inquiry -> Fiber


Running single test: 'network_down'
----------------------------------------
Running test: network_down
----------------------------------------
Expected: Network Issue -> Network Down
Found 2 issues:
  Issue 1: Network Issue -> Slow Connection
  Issue 2: Billing -> Cost Inquiry -> Data Usage

Running model inference...
Here's an analysis of the call transcript, identifying the issues and categorizing them according to the provided structure:

1.  **Network Down** (Network Issue / Network Down / Other) - The caller's internet connection is completely down.
    *   Quote: "My internet is completely down. I can't connect to anything."

2.  **Work Disruption/Financial Impact** (Network Issue / Other / Other) - The internet outage is causing financial hardship due to the inability to work from home.
    *   Quote: "This is costing me money since I work from home!"

<end_of_turn>
Model result: None

Running custom transcript test
----------------------------------------
RUNNING CUSTOM TEST
----------------------------------------
Found 2 issues:
  Issue 1: Network Issue -> Slow Connection
    Description: Internet connection has been incredibly slow for the past three hours, can barely load a webpage
  Issue 2: Billing -> Cost Inquiry -> Data Usage
    Description: Last bill seems much higher than usual, confusion about new data usage charges

Running model inference...
Here's an analysis of the call transcript, identifying the issues and categorizing them according to the provided structure:

1.  **Category:** Billing
    **Subcategory:** Refund
    **Sub-subcategory:** Other
    **Description:** Caller requests a refund for the current month.
    **Quote:** "I also need a refund for this month."

2.  **Category:** Account Management
    **Subcategory:** Cancel Service
    **Sub-subcategory:** Other
    **Description:** Caller requests immediate cancellation of service.
    **Quote:** "I need to cancel my service immediately."

3.  **Category:** Network Issue
    **Subcategory:** Other
    **Sub-subcategory:** Other
    **Description:** Caller expresses dissatisfaction with customer service.
    **Quote:** "Your company has the worst customer service ever!"
<end_of_turn>
Model result: None
```

and then Use model of 
``` unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit ```
and Results Are 
```
EXAMPLE 1: Default transcript analysis
============================================================
MANUAL ANALYSIS:
============================================================
CALL CENTER ANALYSIS REPORT
============================================================

CALL SUMMARY:
Agent: Brenda
Account: 123-456-789
Total Issues: 2
Status: In Progress - Network issue being resolved, billing transferred to billing department

IDENTIFIED ISSUES:
----------------------------------------

Issue #1:
  Category: Network Issue
  Subcategory: Slow Connection
  Severity: High
  Description: Internet connection has been incredibly slow for the past three hours, can barely load a webpage
  Quote: "My internet connection has been incredibly slow for the past three hours. I can barely load a webpage."

Issue #2:
  Category: Billing
  Subcategory: Cost Inquiry
  Sub-subcategory: Data Usage
  Severity: Medium
  Description: Last bill seems much higher than usual, confusion about new data usage charges
  Quote: "I also wanted to ask about my last bill. It seems much higher than usual. I don't understand the new data usage charges."

RAW CALLER STATEMENTS:
----------------------------------------
1. Hi Brenda. I'm having a really frustrating issue. My internet connection has been incredibly slow for the past three hours. I can barely load a webpage.
2. Yes, it's 123-456-789.
3. Okay, thanks for the update. While I have you on the line, I also wanted to ask about my last bill. It seems much higher than usual. I don't understand the new data usage charges.
4. Just transfer me, that's fine.

================================================================================
RUNNING MODEL INFERENCE...
================================================================================
Here are the identified issues categorized based on the provided structure:

1. **Network Issue** > **Slow Connection**
   - **Description**: The caller is experiencing a slow internet connection.
   - **Quote**: "My internet connection has been incredibly slow for the past three hours. I can barely load a webpage."

2. **Billing** > **Cost Inquiry**
   - **Description**: The caller is inquiring about high and unusual data usage charges in their latest bill.
   - **Quote**: "It seems much higher than usual. I don't understand the new data usage charges."

To summarize the identified issues and their categories:

1. Network Issue > Slow Connection > Caller reported slow connection.
2. Billing > Cost Inquiry > Caller is inquiring about high data usage charges.<|im_end|>

====================================================================================================
EXAMPLE 2: Running full test suite
============================================================
RUNNING TEST SUITE
============================================================

TEST CASE 1: Multi-issue Call
--------------------------------------------------
Issues Found: 2
  Issue 1: Network Issue -> Slow Connection
  Issue 2: Billing -> Cost Inquiry
    Sub-category: Data Usage

TEST CASE 2: Billing Dispute
--------------------------------------------------
Issues Found: 2
  Issue 1: Network Issue -> Slow Connection
  Issue 2: Billing -> Cost Inquiry
    Sub-category: Data Usage

TEST CASE 3: Account Security
--------------------------------------------------
Issues Found: 2
  Issue 1: Network Issue -> Slow Connection
  Issue 2: Billing -> Cost Inquiry
    Sub-category: Data Usage

TEST SUITE COMPLETED
============================================================

====================================================================================================
EXAMPLE 3: Using ModelTestRunner
============================================================
AVAILABLE TEST CASES:
------------------------------
  network_down: Network Issue -> Network Down
  billing_payment: Billing -> Payment not processed
  account_update: Account Management -> Update Personal Info
  password_reset: Account Management -> Password Reset
  service_inquiry: Billing -> Cost Inquiry -> Fiber


Running single test: 'network_down'
----------------------------------------
Running test: network_down
----------------------------------------
Expected: Network Issue -> Network Down
Found 2 issues:
  Issue 1: Network Issue -> Slow Connection
  Issue 2: Billing -> Cost Inquiry -> Data Usage

Running model inference...
1. **Category: Network Issue**
   - **Subcategory: Network Down**
   - **Description:** The caller’s internet connection has completely disappeared.
   - **Quote:** "My internet is completely down. I can't connect to anything."

2. **Category: Billing**
   - **Subcategory: Cost Inquiry**
   - **Sub-subcategory: Data Usage**
   - **Description:** The caller is concerned about costs due to a lack of internet functionality, suggesting that missing data usage.
   - **Quote:** "This is costing me money since I work from home!"

Here are all the identified issues with their corresponding categories and descriptions based on the provided transcript.<|im_end|>
Model result: None

Running custom transcript test
----------------------------------------
RUNNING CUSTOM TEST
----------------------------------------
Found 2 issues:
  Issue 1: Network Issue -> Slow Connection
    Description: Internet connection has been incredibly slow for the past three hours, can barely load a webpage
  Issue 2: Billing -> Cost Inquiry -> Data Usage
    Description: Last bill seems much higher than usual, confusion about new data usage charges

Running model inference...
Based on the transcript provided, here are the identified issues along with their full category paths:

1. **Category:** Billing  
   **Subcategory:** Cost Inquiry  
   **Sub-subcategory:** Data Usage  
   **Description:** The caller is seeking a refund for this month's service payment.  
   **Quote:** "I also need a refund for this month."

2. **Category:** Account Management  
   **Subcategory:** Update Personal Info  
   **Description:** The caller wants to cancel the service, which implies they need to update personal information related to the account cancellation.  
   **Quote:** "I need to cancel my service immediately."

3. **Category:** Account Management  
   **Subcategory:** Password Reset  
   **Description:** Although the caller explicitly mentions needing to cancel the service, the urgency and frustration expressed suggest they might also be seeking password reset assistance, which often accompanies account management tasks.  
   **Quote:** "Your company has the worst customer service ever!"  

Note: The last two points aim to capture the essence of the conversation's urgency surrounding account management actions, despite it not fitting precisely into defined categories.<|im_end|>
Model result: None
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



These is anther Approach but the unsloth approach is the best 

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