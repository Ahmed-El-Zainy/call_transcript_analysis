import json
import os
import yaml
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import requests
from dataclasses import dataclass
import gradio as gr
import tempfile

# Try to import optional dependencies
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from transformers import pipeline
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


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
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai package not installed")
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
        if not HUGGINGFACE_AVAILABLE:
            raise ImportError("transformers package not installed")
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
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed")
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
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed")
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


class GrokProvider(LLMProvider):
    """Grok (X.AI) API provider"""
    
    def __init__(self, api_key: str, model: str = "grok-beta"):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed for Grok")
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1"
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
            raise Exception(f"Grok API error: {str(e)}")


class TranscriptAnalyzer:
    """Main class for analyzing call transcripts"""
    
    def __init__(self):
        self.llm_provider = None
        
    def initialize_llm_provider(self, provider_name: str, api_key: str, model: str) -> LLMProvider:
        """Initialize the LLM provider based on selection"""
        provider_name = provider_name.lower()
        
        if provider_name == "gemini":
            if not api_key:
                raise ValueError("Gemini API key is required")
            return GeminiProvider(api_key, model)
        
        elif provider_name == "deepseek":
            if not api_key:
                raise ValueError("DeepSeek API key is required")
            return DeepSeekProvider(api_key, model)
        
        elif provider_name == "huggingface":
            return HuggingFaceProvider(model)
        
        elif provider_name == "qwen":
            if not api_key:
                raise ValueError("Qwen API key is required")
            return QwenProvider(api_key, model)
        
        elif provider_name == "openai":
            if not api_key:
                raise ValueError("OpenAI API key is required")
            return OpenAIProvider(api_key, model)
        
        else:
            raise ValueError(f"Unknown LLM provider: {provider_name}")
    
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
            return AnalysisResult(analysis_results=[])
    
    def analyze_transcript(self, transcript: str, categories: List[Dict[str, Any]], 
                          provider_name: str, api_key: str, model: str) -> tuple:
        """Main method to analyze transcript"""
        try:
            # Initialize provider
            self.llm_provider = self.initialize_llm_provider(provider_name, api_key, model)
            
            # Create prompt
            prompt = self.create_analysis_prompt(transcript, categories)
            
            # Get LLM response
            response = self.llm_provider.generate_response(prompt)
            
            # Parse response
            result = self.parse_llm_response(response)
            
            return result.__dict__, ""
        
        except Exception as e:
            return {"analysis_results": []}, f"Error during analysis: {str(e)}"


# Default categories for demo
DEFAULT_CATEGORIES = [
    {
        "value": "Network Issue",
        "subcategories": [
            {"value": "Slow Connection"},
            {"value": "Network Down"},
            {"value": "Intermittent Connection"},
            {"value": "Other"}
        ]
    },
    {
        "value": "Billing",
        "subcategories": [
            {"value": "Cost Inquiry"},
            {"value": "Payment Not Processed"},
            {"value": "Overcharge"},
            {"value": "Refund Request"},
            {"value": "Other"}
        ]
    },
    {
        "value": "Technical Support",
        "subcategories": [
            {"value": "Hardware Issue"},
            {"value": "Software Problem"},
            {"value": "Setup Help"},
            {"value": "Other"}
        ]
    },
    {
        "value": "Account Management",
        "subcategories": [
            {"value": "Password Reset"},
            {"value": "Account Locked"},
            {"value": "Profile Update"},
            {"value": "Other"}
        ]
    }
]

# Sample transcript
SAMPLE_TRANSCRIPT = """
Customer: Hi, I'm having trouble with my internet connection. It's been really slow for the past few days.
Agent: I'm sorry to hear about the slow internet. Let me help you with that. Can you tell me what speeds you're experiencing?
Customer: It's supposed to be 100 Mbps but I'm only getting around 10 Mbps. Also, I noticed my last bill was higher than usual.
Agent: I see two issues here. Let me check your connection first and then look into the billing concern.
Customer: Yes, I was charged an extra $20 this month and I don't know why.
Agent: I can see there was a service upgrade fee. Let me explain that and also run a speed test on your connection.
"""

def get_available_providers():
    """Get list of available providers based on installed packages"""
    providers = []
    
    if GEMINI_AVAILABLE:
        providers.append("Gemini")
    
    providers.append("DeepSeek")  # Only needs requests
    
    if HUGGINGFACE_AVAILABLE:
        providers.append("HuggingFace")
    
    if OPENAI_AVAILABLE:
        providers.extend(["OpenAI", "Grok", "Qwen"])
    
    return providers

def get_models_for_provider(provider):
    """Get available models for each provider"""
    models = {
        "Gemini": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"],
        "DeepSeek": ["deepseek-chat", "deepseek-coder"],
        "HuggingFace": ["microsoft/DialoGPT-large", "gpt2", "microsoft/DialoGPT-medium"],
        "OpenAI": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o"],
        "Grok": ["grok-3", "grok-1", "grok-2", "grok-3-mini"],
        "Qwen": ["qwen-turbo", "qwen-plus", "qwen-max"]
    }
    return models.get(provider, [])

def update_model_choices(provider):
    """Update model dropdown based on provider selection"""
    models = get_models_for_provider(provider)
    return gr.Dropdown(choices=models, value=models[0] if models else None)

def update_api_key_visibility(provider):
    """Show/hide API key input based on provider"""
    needs_api_key = provider in ["Gemini", "DeepSeek", "OpenAI", "Grok", "Qwen"]
    return gr.Textbox(
        label="API Key",
        placeholder="Enter your API key..." if needs_api_key else "No API key required",
        type="password" if needs_api_key else "text",
        visible=True,
        interactive=needs_api_key
    )

def analyze_transcript_gradio(transcript, categories_json, provider, api_key, model):
    """Gradio interface function for transcript analysis"""
    if not transcript.strip():
        return None, "Please enter a transcript to analyze."
    
    if not provider:
        return None, "Please select an LLM provider."
    
    if not model:
        return None, "Please select a model."
    
    try:
        # Parse categories
        if categories_json.strip():
            categories = json.loads(categories_json)
        else:
            categories = DEFAULT_CATEGORIES
        
        # Initialize analyzer
        analyzer = TranscriptAnalyzer()
        
        # Analyze transcript
        result, error = analyzer.analyze_transcript(transcript, categories, provider, api_key, model)
        
        if error:
            return None, error
        else:
            return result, "Analysis completed successfully!"
    
    except json.JSONDecodeError:
        return None, "Invalid JSON format in categories. Please check your JSON syntax."
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"

# Create Gradio interface
def create_gradio_app():
    """Create and return the Gradio interface"""
    
    with gr.Blocks(title="Transcript Analyzer", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎯 AI-Powered Call Transcript Analyzer")
        gr.Markdown("Analyze customer support call transcripts and automatically categorize issues using various LLM providers.")
        
        with gr.Tab("📊 Analyze Transcript"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 📝 Input")
                    transcript_input = gr.Textbox(
                        label="Call Transcript",
                        placeholder="Enter the call transcript here...",
                        lines=10,
                        value=SAMPLE_TRANSCRIPT
                    )
                    
                    with gr.Accordion("📋 Categories (JSON)", open=False):
                        categories_input = gr.Textbox(
                            label="Categories JSON (leave empty for default)",
                            placeholder="Enter categories in JSON format...",
                            lines=8,
                            value=""
                        )
                        
                        gr.Markdown("**Default categories will be used if left empty**")
                
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ Settings")
                    
                    provider_dropdown = gr.Dropdown(
                        label="LLM Provider",
                        choices=get_available_providers(),
                        value=get_available_providers()[0] if get_available_providers() else None
                    )
                    
                    model_dropdown = gr.Dropdown(
                        label="Model",
                        choices=[],
                        value=None
                    )
                    
                    api_key_input = gr.Textbox(
                        label="API Key",
                        placeholder="Enter your API key...",
                        type="password",
                        visible=True
                    )
                    
                    analyze_btn = gr.Button("🔍 Analyze Transcript", variant="primary")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📈 Results")
                    result_output = gr.JSON(label="Analysis Results", show_label=True)
                    error_output = gr.Textbox(label="Status/Errors", lines=3, show_label=True)
        
        with gr.Tab("📚 Help & Examples"):
            gr.Markdown("""
            ## How to Use
            
            1. **Choose your LLM Provider**: Select from available providers (Gemini, DeepSeek, OpenAI, etc.)
            2. **Select Model**: Choose the specific model you want to use
            3. **Enter API Key**: Provide your API key for the selected provider (if required)
            4. **Input Transcript**: Paste your call transcript in the text area
            5. **Categories (Optional)**: Define custom categories in JSON format, or use defaults
            6. **Analyze**: Click the analyze button to get results
            
            ## Sample Categories JSON Format
            ```json
            [
                {
                    "value": "Technical Issue",
                    "subcategories": [
                        {"value": "Hardware Problem"},
                        {"value": "Software Bug"},
                        {"value": "Other"}
                    ]
                },
                {
                    "value": "Billing",
                    "subcategories": [
                        {"value": "Payment Issue"},
                        {"value": "Overcharge"},
                        {"value": "Other"}
                    ]
                }
            ]
            ```
            
            ## Available Providers
            - **Gemini**: Google's AI model (requires API key)
            - **DeepSeek**: DeepSeek's AI model (requires API key)
            - **OpenAI**: GPT models (requires API key)
            - **Grok**: X.AI's Grok model (requires API key)
            - **Qwen**: Alibaba's AI model (requires API key)
            - **HuggingFace**: Open-source models (no API key required)
            
            ## Getting API Keys
            - **Gemini**: Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
            - **DeepSeek**: Visit [DeepSeek Platform](https://platform.deepseek.com/)
            - **OpenAI**: Visit [OpenAI Platform](https://platform.openai.com/api-keys)
            - **Grok**: Visit [X.AI Console](https://console.x.ai/) - Get your API key from X.AI
            - **Qwen**: Visit [Dashscope](https://dashscope.aliyuncs.com/)
            """)
        
        # Event handlers
        provider_dropdown.change(
            update_model_choices,
            inputs=[provider_dropdown],
            outputs=[model_dropdown]
        )
        
        provider_dropdown.change(
            update_api_key_visibility,
            inputs=[provider_dropdown],
            outputs=[api_key_input]
        )
        
        analyze_btn.click(
            analyze_transcript_gradio,
            inputs=[transcript_input, categories_input, provider_dropdown, api_key_input, model_dropdown],
            outputs=[result_output, error_output]
        )
        
        # Initialize model dropdown on load
        demo.load(
            update_model_choices,
            inputs=[provider_dropdown],
            outputs=[model_dropdown]
        )
    
    return demo

# Launch the app
if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )