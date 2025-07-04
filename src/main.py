import json
import os
import yaml
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import requests
from dataclasses import dataclass
import google.generativeai as genai
from transformers import pipeline
import openai


@dataclass
class AnalysisResult:
    """Data class for analysis results"""
    analysis_results: List[List[str]]


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass


class GeminiProvider(LLMProvider):
    """Gemini API provider"""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
    
    def generate_response(self, prompt: str) -> str:
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")


class DeepSeekProvider(LLMProvider):
    """DeepSeek API provider"""
    
    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
    
    def generate_response(self, prompt: str) -> str:
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1
            }
            
            response = requests.post(self.base_url, headers=headers, json=data)
            response.raise_for_status()
            
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            raise Exception(f"DeepSeek API error: {str(e)}")


class HuggingFaceProvider(LLMProvider):
    """Hugging Face transformers provider"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-large"):
        try:
            self.pipeline = pipeline("text-generation", model=model_name, device_map="auto")
        except Exception:
            # Fallback to a smaller model if the large one fails
            self.pipeline = pipeline("text-generation", model="gpt2", device_map="auto")
    
    def generate_response(self, prompt: str) -> str:
        try:
            # Truncate prompt if too long
            max_length = min(1000, len(prompt) + 500)
            response = self.pipeline(prompt, max_length=max_length, num_return_sequences=1, temperature=0.7)
            return response[0]["generated_text"][len(prompt):].strip()
        except Exception as e:
            raise Exception(f"Hugging Face error: {str(e)}")


class QwenProvider(LLMProvider):
    """Qwen API provider (using OpenAI-compatible API)"""
    
    def __init__(self, api_key: str, model: str = "qwen-turbo", base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
    
    def generate_response(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Qwen API error: {str(e)}")


class OpenAIProvider(LLMProvider):
    """OpenAI API provider"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def generate_response(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")


class TranscriptAnalyzer:
    """Main class for analyzing call transcripts"""
    
    def __init__(self, config_file: str = "config.yaml"):
        self.config = self.load_config(config_file)
        self.llm_provider = self.initialize_llm_provider()
        
    
    def load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Return default config if file not found
            return {
                "llm_provider": "gemini",
                "providers": {
                    "gemini": {
                        "api_key": "AIzaSyDxuh2N0WbNHlui66ZftNPBH9G3n1vqZ00",
                        "model": "gemini-1.5-flash"
                    },
                    "deepseek": {
                        "api_key": "",
                        "model": "deepseek-chat"
                    },
                    "huggingface": {
                        "model": "microsoft/DialoGPT-large"
                    },
                    "qwen": {
                        "api_key": "",
                        "model": "qwen-turbo",
                        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
                    },
                    "openai": {
                        "api_key": "",
                        "model": "gpt-3.5-turbo"
                    }
                }
            }
    
    def initialize_llm_provider(self) -> LLMProvider:
        """Initialize the LLM provider based on configuration"""
        provider_name = self.config.get("llm_provider", "gemini").lower()
        providers_config = self.config.get("providers", {})
        
        if provider_name == "gemini":
            config = providers_config.get("gemini", {})
            api_key = config.get("api_key") or os.getenv("GEMINI_API_KEY")
            model = config.get("model", "gemini-pro")
            if not api_key:
                raise ValueError("Gemini API key not found in config or environment")
            return GeminiProvider(api_key, model)
        
        elif provider_name == "grok":
            config = providers_config.get("gemini", {})
            api_key = config.get("api_key") or os.getenv("GROK_API_KEY")
            model = config.get("model", "grok-3")
            if not api_key:
                raise ValueError("Grok API key not found in config or environment")
            return GeminiProvider(api_key, model)
        
        elif provider_name == "deepseek":
            config = providers_config.get("deepseek", {})
            api_key = config.get("api_key") or os.getenv("DEEPSEEK_API_KEY")
            model = config.get("model", "deepseek-chat")
            if not api_key:
                raise ValueError("DeepSeek API key not found in config or environment")
            return DeepSeekProvider(api_key, model)
        
        elif provider_name == "huggingface":
            config = providers_config.get("huggingface", {})
            model = config.get("model", "microsoft/DialoGPT-large")
            return HuggingFaceProvider(model)
        
        elif provider_name == "qwen":
            config = providers_config.get("qwen", {})
            api_key = config.get("api_key") or os.getenv("QWEN_API_KEY")
            model = config.get("model", "qwen-turbo")
            base_url = config.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
            if not api_key:
                raise ValueError("Qwen API key not found in config or environment")
            return QwenProvider(api_key, model, base_url)
        
        elif provider_name == "openai":
            config = providers_config.get("openai", {})
            api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
            model = config.get("model", "gpt-3.5-turbo")
            if not api_key:
                raise ValueError("OpenAI API key not found in config or environment")
            return OpenAIProvider(api_key, model)
        
        else:
            raise ValueError(f"Unknown LLM provider: {provider_name}")
    
    def load_transcript(self, file_path: str) -> str:
        """Load transcript from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"Transcript file not found: {file_path}")
    
    def load_categories(self, file_path: str) -> List[Dict[str, Any]]:
        """Load categories from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Categories file not found: {file_path}")
    
    def build_category_structure(self, categories: List[Dict[str, Any]]) -> str:
        """Build a readable category structure for the prompt"""
        def format_category(cat: Dict[str, Any], level: int = 0) -> str:
            indent = "  " * level
            result = f"{indent}- {cat['value']}"
            
            if 'description' in cat:
                result += f": {cat['description']}"
            
            if 'subcategories' in cat:
                result += "\n"
                for subcat in cat['subcategories']:
                    result += format_category(subcat, level + 1) + "\n"
            
            return result.rstrip()
        
        structure = ""
        for category in categories:
            structure += format_category(category) + "\n"
        
        return structure.strip()
    
    def create_analysis_prompt(self, transcript: str, categories: List[Dict[str, Any]]) -> str:
        """Create the prompt for LLM analysis"""
        category_structure = self.build_category_structure(categories)
        
        prompt = f"""
You are an expert call transcript analyzer. Your task is to analyze a customer support call transcript and identify ALL issues discussed, categorizing them according to the provided category structure.

TRANSCRIPT:
{transcript}

CATEGORY STRUCTURE:
{category_structure}

INSTRUCTIONS:
1. Carefully read the entire transcript
2. Identify ALL issues/topics discussed by the caller
3. For each issue, find the MOST SPECIFIC category path that matches
4. If an issue doesn't fit any category, categorize it as "Other"
5. Return ONLY a valid JSON object with the specified format

REQUIRED OUTPUT FORMAT:
{{
  "analysis_results": [
    ["Category", "Subcategory"],
    ["Category", "Subcategory", "Sub-subcategory"]
  ]
}}

IMPORTANT RULES:
- Each result should be an array representing the path from main category to most specific subcategory
- Include ALL issues found in the transcript
- Use exact category names as provided in the structure
- If no issues are found, return: {{"analysis_results": []}}
- Return ONLY the JSON object, no additional text

Analyze the transcript now:
"""
        return prompt
    
    def parse_llm_response(self, response: str) -> AnalysisResult:
        """Parse LLM response and extract structured data"""
        try:
            # Try to find JSON in the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                # No JSON found, return empty result
                return AnalysisResult(analysis_results=[])
            
            json_str = response[start_idx:end_idx]
            data = json.loads(json_str)
            
            # Validate the structure
            if 'analysis_results' not in data:
                return AnalysisResult(analysis_results=[])
            
            return AnalysisResult(analysis_results=data['analysis_results'])
        
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not parse LLM response: {e}")
            return AnalysisResult(analysis_results=[])
    
    def analyze_transcript(self, transcript_file: str, categories_file: str) -> AnalysisResult:
        """Main method to analyze transcript"""
        try:
            # Load inputs
            transcript = self.load_transcript(transcript_file)
            categories = self.load_categories(categories_file)
            
            # Create prompt
            prompt = self.create_analysis_prompt(transcript, categories)
            
            # Get LLM response
            response = self.llm_provider.generate_response(prompt)
            
            # Parse response
            result = self.parse_llm_response(response)
            
            return result
        
        except Exception as e:
            print(f"Error during analysis: {e}")
            return AnalysisResult(analysis_results=[])
    
    def analyze_with_hardcoded_categories(self, transcript_file: str) -> AnalysisResult:
        """Part 1: Analyze with hardcoded categories for single issue"""
        hardcoded_categories = [
            {
                "value": "Network Issue",
                "subcategories": [
                    {"value": "Slow Connection"},
                    {"value": "Network Down"},
                    {"value": "Other"}
                ]
            },
            {
                "value": "Billing",
                "subcategories": [
                    {"value": "Cost Inquiry"},
                    {"value": "Payment Not Processed"},
                    {"value": "Other"}
                ]
            }
        ]
        
        try:
            transcript = self.load_transcript(transcript_file)
            prompt = self.create_analysis_prompt(transcript, hardcoded_categories)
            response = self.llm_provider.generate_response(prompt)
            result = self.parse_llm_response(response)
            
            # For Part 1, return only the first/primary issue
            if result.analysis_results:
                return AnalysisResult(analysis_results=[result.analysis_results[0]])
            else:
                return AnalysisResult(analysis_results=[])
                
        except Exception as e:
            print(f"Error during hardcoded analysis: {e}")
            return AnalysisResult(analysis_results=[])


def main():
    """Main function to run the analyzer"""
    # Initialize analyzer
    analyzer = TranscriptAnalyzer()
    
    # Part 1: Single category with hardcoded categories
    print("=== Part 1: Single Category Classification ===")
    result_part1 = analyzer.analyze_with_hardcoded_categories("src/tests/transcript.txt")
    print(json.dumps(result_part1.__dict__, indent=2))
    
    # Part 2 & 3: Multi-issue extraction with dynamic categories
    print("\n=== Part 2 & 3: Multi-Issue Extraction with Dynamic Categories ===")
    result_part23 = analyzer.analyze_transcript("src/tests/transcript.txt", "src/tests/categories.json")
    print(json.dumps(result_part23.__dict__, indent=2))


if __name__ == "__main__":
    main()