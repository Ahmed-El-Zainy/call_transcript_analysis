import json
import re
from typing import List, Dict, Any, Optional
from transformers import pipeline
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranscriptAnalyzer:
    def __init__(self, model_name: str = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"):
        """
        Initialize the transcript analyzer with the specified model.
        
        Args:
            model_name: The name of the model to use for text generation
        """
        try:
            self.pipe = pipeline("text-generation", model=model_name)
            logger.info(f"Model {model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def load_categories(self, categories_file: str) -> List[Dict[str, Any]]:
        """
        Load categories from JSON file.
        
        Args:
            categories_file: Path to the categories JSON file
            
        Returns:
            List of category dictionaries
        """
        try:
            with open(categories_file, 'r') as f:
                categories = json.load(f)
            logger.info(f"Loaded {len(categories)} categories from {categories_file}")
            return categories
        except Exception as e:
            logger.error(f"Failed to load categories: {e}")
            raise
    
    def load_transcript(self, transcript_file: str) -> str:
        """
        Load transcript from text file.
        
        Args:
            transcript_file: Path to the transcript text file
            
        Returns:
            Transcript content as string
        """
        try:
            with open(transcript_file, 'r') as f:
                transcript = f.read()
            logger.info(f"Loaded transcript from {transcript_file}")
            return transcript
        except Exception as e:
            logger.error(f"Failed to load transcript: {e}")
            raise
    
    def format_categories_for_prompt(self, categories: List[Dict[str, Any]]) -> str:
        """
        Format categories into a structured string for the LLM prompt.
        
        Args:
            categories: List of category dictionaries
            
        Returns:
            Formatted categories string
        """
        def format_category(cat: Dict[str, Any], level: int = 0) -> str:
            indent = "  " * level
            result = f"{indent}- {cat['value']}"
            if 'description' in cat:
                result += f": {cat['description']}"
            result += "\n"
            
            if 'subcategories' in cat:
                for subcat in cat['subcategories']:
                    result += format_category(subcat, level + 1)
            
            return result
        
        formatted = "Available Categories:\n"
        for category in categories:
            formatted += format_category(category)
        
        return formatted
    
    def create_analysis_prompt(self, transcript: str, categories: List[Dict[str, Any]]) -> str:
        """
        Create the analysis prompt for the LLM.
        
        Args:
            transcript: The call transcript
            categories: List of category dictionaries
            
        Returns:
            Formatted prompt string
        """
        categories_str = self.format_categories_for_prompt(categories)
        
        prompt = f"""You are an expert call center analyst. Your task is to analyze a call transcript and identify ALL issues discussed, categorizing them according to the provided category structure.

{categories_str}

TRANSCRIPT:
{transcript}

INSTRUCTIONS:
1. Carefully read the transcript and identify ALL issues or topics discussed
2. For each issue, find the MOST SPECIFIC category path that matches it
3. A category path is a sequence from main category down to the most specific subcategory
4. If an issue doesn't fit any specific subcategory, use "Other" if available
5. Return your analysis in the following JSON format:

{{
  "analysis_results": [
    ["Category1", "Subcategory1"],
    ["Category2", "Subcategory2", "Sub-subcategory"],
    ...
  ]
}}

EXAMPLE:
If someone mentions slow internet and billing questions about DSL costs, return:
{{
  "analysis_results": [
    ["Network Issue", "Slow Connection"],
    ["Billing", "Cost Inquiry", "DSL"]
  ]
}}

Now analyze the transcript and return ONLY the JSON response:"""
        
        return prompt
    
    def extract_json_from_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON from the LLM response.
        
        Args:
            response: Raw response from the LLM
            
        Returns:
            Parsed JSON dictionary or None if parsing fails
        """
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                logger.warning("No JSON found in response")
                return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            return None
    
    def validate_categories(self, results: List[List[str]], categories: List[Dict[str, Any]]) -> List[List[str]]:
        """
        Validate that the identified categories exist in the category structure.
        
        Args:
            results: List of category paths
            categories: Valid category structure
            
        Returns:
            Validated category paths
        """
        def find_category_path(path: List[str], cats: List[Dict[str, Any]]) -> bool:
            if not path:
                return True
            
            for cat in cats:
                if cat['value'] == path[0]:
                    if len(path) == 1:
                        return True
                    elif 'subcategories' in cat:
                        return find_category_path(path[1:], cat['subcategories'])
            return False
        
        validated_results = []
        for result in results:
            if find_category_path(result, categories):
                validated_results.append(result)
            else:
                logger.warning(f"Invalid category path: {result}")
        
        return validated_results
    
    def analyze_single_category(self, transcript: str) -> Dict[str, Any]:
        """
        Part 1: Analyze transcript for single primary category (static categories).
        
        Args:
            transcript: The call transcript
            
        Returns:
            Analysis result with single category
        """
        # Static categories for Part 1
        static_categories = [
            {
                "value": "Network Issue",
                "description": "The caller is facing network issues",
                "subcategories": [
                    {"value": "Slow Connection", "description": "The caller reported slow connection"},
                    {"value": "Network Down", "description": "The caller reported their network is down"},
                    {"value": "Other"}
                ]
            },
            {
                "value": "Billing",
                "description": "The caller made billing inquiries",
                "subcategories": [
                    {"value": "Cost Inquiry", "description": "The caller is inquiring about costs"},
                    {"value": "Payment Not Processed", "description": "Payment processing issues"},
                    {"value": "Other"}
                ]
            }
        ]
        
        prompt = self.create_analysis_prompt(transcript, static_categories)
        prompt += "\n\nFor Part 1, return only the SINGLE most prominent issue:"
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = self.pipe(messages, max_new_tokens=200, temperature=0.1)
            response_text = response[0]['generated_text'][-1]['content'] if isinstance(response[0]['generated_text'], list) else response[0]['generated_text']
            
            result = self.extract_json_from_response(response_text)
            if result and 'analysis_results' in result:
                # For Part 1, return only the first result
                if result['analysis_results']:
                    return {"analysis_results": [result['analysis_results'][0]]}
            
            return {"analysis_results": []}
            
        except Exception as e:
            logger.error(f"Error in single category analysis: {e}")
            return {"analysis_results": []}
    
    def analyze_multi_issues(self, transcript: str) -> Dict[str, Any]:
        """
        Part 2: Analyze transcript for multiple issues (static categories).
        
        Args:
            transcript: The call transcript
            
        Returns:
            Analysis result with multiple categories
        """
        # Static categories for Part 2
        static_categories = [
            {
                "value": "Network Issue",
                "description": "The caller is facing network issues",
                "subcategories": [
                    {"value": "Slow Connection", "description": "The caller reported slow connection"},
                    {"value": "Network Down", "description": "The caller reported their network is down"},
                    {"value": "Other"}
                ]
            },
            {
                "value": "Billing",
                "description": "The caller made billing inquiries",
                "subcategories": [
                    {"value": "Cost Inquiry", "description": "The caller is inquiring about costs"},
                    {"value": "Payment Not Processed", "description": "Payment processing issues"},
                    {"value": "Other"}
                ]
            }
        ]
        
        prompt = self.create_analysis_prompt(transcript, static_categories)
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = self.pipe(messages, max_new_tokens=300, temperature=0.1)
            response_text = response[0]['generated_text'][-1]['content'] if isinstance(response[0]['generated_text'], list) else response[0]['generated_text']
            
            result = self.extract_json_from_response(response_text)
            if result and 'analysis_results' in result:
                validated_results = self.validate_categories(result['analysis_results'], static_categories)
                return {"analysis_results": validated_results}
            
            return {"analysis_results": []}
            
        except Exception as e:
            logger.error(f"Error in multi-issue analysis: {e}")
            return {"analysis_results": []}
    
    def analyze_dynamic_categories(self, transcript: str, categories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Part 3: Analyze transcript with dynamic categories from JSON file.
        
        Args:
            transcript: The call transcript
            categories: Dynamic categories loaded from JSON
            
        Returns:
            Analysis result with dynamic categories
        """
        prompt = self.create_analysis_prompt(transcript, categories)
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = self.pipe(messages, max_new_tokens=400, temperature=0.1)
            response_text = response[0]['generated_text'][-1]['content'] if isinstance(response[0]['generated_text'], list) else response[0]['generated_text']
            
            result = self.extract_json_from_response(response_text)
            if result and 'analysis_results' in result:
                validated_results = self.validate_categories(result['analysis_results'], categories)
                return {"analysis_results": validated_results}
            
            return {"analysis_results": []}
            
        except Exception as e:
            logger.error(f"Error in dynamic category analysis: {e}")
            return {"analysis_results": []}
    
    def run_complete_analysis(self, transcript_file: str, categories_file: str) -> Dict[str, Any]:
        """
        Run the complete analysis pipeline (Part 3 - final solution).
        
        Args:
            transcript_file: Path to transcript file
            categories_file: Path to categories JSON file
            
        Returns:
            Final analysis result
        """
        transcript = self.load_transcript(transcript_file)
        categories = self.load_categories(categories_file)
        
        return self.analyze_dynamic_categories(transcript, categories)


def main():
    """
    Main function to run the transcript analyzer.
    """
    # Initialize the analyzer
    analyzer = TranscriptAnalyzer()
    
    # File paths (adjust as needed)
    transcript_file = "/content/transcript.txt"
    categories_file = "/content/categories.json"
    
    try:
        # Run Part 1: Single Category Analysis
        print("=== Part 1: Single Category Analysis ===")
        transcript = analyzer.load_transcript(transcript_file)
        result_part1 = analyzer.analyze_single_category(transcript)
        print(json.dumps(result_part1, indent=2))
        
        # Run Part 2: Multi-Issue Analysis
        print("\n=== Part 2: Multi-Issue Analysis ===")
        result_part2 = analyzer.analyze_multi_issues(transcript)
        print(json.dumps(result_part2, indent=2))
        
        # Run Part 3: Dynamic Categories Analysis
        print("\n=== Part 3: Dynamic Categories Analysis ===")
        result_part3 = analyzer.run_complete_analysis(transcript_file, categories_file)
        print(json.dumps(result_part3, indent=2))
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())