#!/usr/bin/env python3
"""
Call Transcript Analysis - AI-Powered Issue Classification
Main entry point for the application
"""

import json
import os
import argparse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
from pathlib import Path

# Third-party imports
try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    import requests
except ImportError:
    requests = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Data class for analysis results"""
    analysis_results: List[List[str]]
    metadata: Optional[Dict[str, Any]] = None


class TranscriptAnalyzer:
    """Main analyzer class that handles the complete workflow"""
    
    def __init__(self, provider: str = 'gemini'):
        self.provider = provider.lower()
        self._validate_provider()
        self._setup_api()
    
    def _validate_provider(self):
        """Validate the selected provider and check dependencies"""
        valid_providers = ['gemini', 'huggingface']
        if self.provider not in valid_providers:
            raise ValueError(f"Provider must be one of: {valid_providers}")
        
        if self.provider == 'gemini' and genai is None:
            raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
        
        if self.provider == 'huggingface' and requests is None:
            raise ImportError("requests not installed. Run: pip install requests")
    
    def _setup_api(self):
        """Setup API configurations"""
        if self.provider == 'gemini':
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel("gemini-1.5-flash")
        
        elif self.provider == 'huggingface':
            self.hf_token = os.getenv("HF_TOKEN")
            if not self.hf_token:
                raise ValueError("HF_TOKEN environment variable not set")
            self.hf_api_url = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-70B-Instruct"
    
    def load_inputs(self, transcript_path: str, categories_path: str) -> tuple[str, List[Dict]]:
        """Load transcript and categories from files"""
        try:
            # Load transcript
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript = f.read().strip()
            
            # Load categories
            with open(categories_path, 'r', encoding='utf-8') as f:
                categories = json.load(f)
            
            logger.info(f"Loaded transcript from {transcript_path}")
            logger.info(f"Loaded {len(categories)} categories from {categories_path}")
            
            return transcript, categories
            
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in categories file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading inputs: {e}")
            raise
    
    def build_prompt(self, transcript: str, categories: List[Dict]) -> str:
        """Build dynamic prompt for LLM analysis"""
        prompt = f"""You are an AI assistant that analyzes customer support call transcripts and categorizes issues based on a dynamic list of categories and subcategories.

Below is the transcript:
---
{transcript}
---

Here is the list of categories (in JSON format):
{json.dumps(categories, indent=2)}

Your task:
- Analyze the transcript carefully.
- Extract all mentioned issues or problems discussed in the call.
- For each issue, identify the most specific path (Category → Subcategory → Sub-subcategory if available).
- Return the result as a JSON list of arrays, where each array represents the path to a specific issue.

Examples of expected output format:
- [ ["Network Issue", "Slow Connection"] ]
- [ ["Billing", "Cost Inquiry", "DSL"], ["Technical Support", "Equipment", "Router"] ]

Important guidelines:
1. Only classify issues that are actually mentioned or discussed in the transcript
2. Use the most specific path available in the category structure
3. If multiple issues are discussed, include all of them
4. If no clear issue matches the categories, return an empty array: []
5. Return ONLY the JSON output, no additional text

Return only the JSON output."""
        
        return prompt
    
    def call_gemini(self, prompt: str) -> str:
        """Make API call to Google Gemini"""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=2048,
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            raise
    
    def call_huggingface(self, prompt: str) -> str:
        """Make API call to Hugging Face Inference API"""
        try:
            headers = {"Authorization": f"Bearer {self.hf_token}"}
            payload = {
                "inputs": prompt,
                "parameters": {
                    "temperature": 0.2,
                    "max_new_tokens": 2048,
                    "return_full_text": False
                }
            }
            
            response = requests.post(self.hf_api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "")
            else:
                raise ValueError("Unexpected response format from Hugging Face API")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Hugging Face API call failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Error processing Hugging Face response: {e}")
            raise
    
    def parse_output(self, raw_output: str) -> List[List[str]]:
        """Parse LLM output into structured format"""
        try:
            # Clean the output
            raw_output = raw_output.strip()
            
            # Find JSON array in the output
            start_idx = raw_output.find("[")
            end_idx = raw_output.rfind("]")
            
            if start_idx == -1 or end_idx == -1:
                logger.warning("No JSON array found in output")
                return []
            
            json_str = raw_output[start_idx:end_idx + 1]
            result = json.loads(json_str)
            
            # Validate the structure
            if not isinstance(result, list):
                logger.warning("Output is not a list")
                return []
            
            # Validate each item is a list of strings
            validated_result = []
            for item in result:
                if isinstance(item, list) and all(isinstance(x, str) for x in item):
                    validated_result.append(item)
                else:
                    logger.warning(f"Invalid item in results: {item}")
            
            return validated_result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON output: {e}")
            logger.debug(f"Raw output: {raw_output}")
            return []
        except Exception as e:
            logger.error(f"Error parsing output: {e}")
            return []
    
    def analyze(self, transcript_path: str, categories_path: str) -> AnalysisResult:
        """Main analysis method - orchestrates the entire workflow"""
        try:
            # Step 1: Load inputs
            transcript, categories = self.load_inputs(transcript_path, categories_path)
            
            # Step 2: Build prompt
            prompt = self.build_prompt(transcript, categories)
            
            # Step 3: Make API call
            logger.info(f"Making API call to {self.provider}...")
            if self.provider == 'gemini':
                raw_output = self.call_gemini(prompt)
            else:
                raw_output = self.call_huggingface(prompt)
            
            # Step 4: Parse output
            results = self.parse_output(raw_output)
            
            # Step 5: Create result object
            metadata = {
                "provider": self.provider,
                "transcript_length": len(transcript),
                "num_categories": len(categories),
                "num_issues_found": len(results)
            }
            
            logger.info(f"Analysis complete. Found {len(results)} issues.")
            return AnalysisResult(analysis_results=results, metadata=metadata)
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Call Transcript Analysis - AI-Powered Issue Classification")
    parser.add_argument("--transcript", required=True, help="Path to transcript.txt file")
    parser.add_argument("--categories", required=True, help="Path to categories.json file")
    parser.add_argument("--provider", choices=['gemini', 'huggingface'], default='gemini',
                       help="AI provider to use (default: gemini)")
    parser.add_argument("--output", help="Output file path (default: stdout)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize analyzer
        analyzer = TranscriptAnalyzer(provider=args.provider)
        
        # Run analysis
        result = analyzer.analyze(args.transcript, args.categories)
        
        # Prepare output
        output_data = {
            "analysis_results": result.analysis_results
        }
        
        if args.verbose and result.metadata:
            output_data["metadata"] = result.metadata
        
        # Output results
        output_json = json.dumps(output_data, indent=2)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output_json)
            logger.info(f"Results written to {args.output}")
        else:
            print(output_json)
    
    except Exception as e:
        logger.error(f"Application failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())


# Example usage and test functions
def create_sample_files():
    """Create sample files for testing"""
    
    # Sample transcript
    sample_transcript = """
    Agent: Hello, thank you for calling customer support. How can I help you today?
    
    Customer: Hi, I'm having trouble with my internet connection. It's been really slow for the past few days.
    
    Agent: I'm sorry to hear that. Let me check your account. Can you tell me your account number?
    
    Customer: Sure, it's 123456789. Also, I wanted to ask about my bill. I noticed an extra charge this month.
    
    Agent: I can see your account. For the slow connection, let me run a speed test on your line. 
    And regarding the billing question, I see there's a DSL upgrade fee that was added last month.
    
    Customer: I didn't request any upgrade. Can you remove that charge?
    
    Agent: Let me check with my supervisor about removing the DSL upgrade fee. 
    For the connection issue, I'm seeing some line interference. I'll schedule a technician visit.
    """
    
    # Sample categories
    sample_categories = [
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
        },
        {
            "name": "Technical Support",
            "subcategories": [
                {"name": "Equipment", "subcategories": [
                    {"name": "Router", "subcategories": []},
                    {"name": "Modem", "subcategories": []}
                ]},
                {"name": "Installation", "subcategories": []}
            ]
        }
    ]
    
    # Write sample files
    with open("sample_transcript.txt", "w") as f:
        f.write(sample_transcript)
    
    with open("sample_categories.json", "w") as f:
        json.dump(sample_categories, f, indent=2)
    
    print("Sample files created: sample_transcript.txt, sample_categories.json")


if __name__ == "__main__":
    # Uncomment the line below to create sample files
    # create_sample_files()
    exit(main())