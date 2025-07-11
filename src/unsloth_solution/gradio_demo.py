import gradio as gr
import json
import re
from typing import List, Dict, Any

def do_gemma_3n_inference(messages):
    """
    Placeholder for your actual Gemma 3n inference function
    Replace this with your actual implementation
    """
    # For demo purposes, return a mock response
    # In production, this would call your actual model
    return {
        "model_response": "Analysis completed using Gemma 3n model",
        "confidence": 0.95,
        "processing_time": "1.2s"
    }

class CallAnalyzer:
    def __init__(self):
        self.category_structure = [
            {
                "value": "Network Issue",
                "description": "The caller is facing network issues",
                "subcategories": [
                    {
                        "value": "Slow Connection",
                        "description": "The caller reported slow connection"
                    },
                    {
                        "value": "Network Down",
                        "description": "The caller reported their network is down"
                    },
                    {
                        "value": "Other"
                    }
                ]
            },
            {
                "value": "Billing",
                "description": "The caller made billing inquiries",
                "subcategories": [
                    {
                        "value": "Payment not processed",
                        "description": "The caller reported an error while paying for a service"
                    },
                    {
                        "value": "Cost Inquiry",
                        "description": "The caller is inquiring about costs for a particular service",
                        "subcategories": [
                            {
                                "value": "DSL"
                            },
                            {
                                "value": "5G"
                            },
                            {
                                "value": "Fiber"
                            },
                            {
                                "value": "Data Usage"
                            }
                        ]
                    },
                    {
                        "value": "Other"
                    }
                ]
            },
            {
                "value": "Account Management",
                "subcategories": [
                    {
                        "value": "Update Personal Info"
                    },
                    {
                        "value": "Password Reset"
                    },
                    {
                        "value": "Service Cancellation"
                    },
                    {
                        "value": "Other"
                    }
                ]
            },
            {
                "value": "Technical Support",
                "subcategories": [
                    {
                        "value": "Device Configuration"
                    },
                    {
                        "value": "Software Issues"
                    },
                    {
                        "value": "Hardware Problems"
                    },
                    {
                        "value": "Other"
                    }
                ]
            }
        ]
        
        self.system_prompt = """You are an expert call center analyst. Your task is to analyze call transcripts and identify ALL issues discussed, categorizing them according to the provided category structure.

Instructions:
1. Read through the entire transcript carefully
2. Identify every distinct issue or concern raised by the caller
3. Match each issue to the most appropriate category and subcategory
4. Be specific about the type of issue where the structure allows
5. If an issue doesn't fit perfectly, choose the closest match
6. List all identified issues with their full category path

Format your response as a structured list of issues with their categories."""

    def extract_caller_issues(self, transcript: str) -> List[str]:
        """Extract issues mentioned by the caller from the transcript"""
        issues = []
        lines = transcript.strip().split('\n')
        
        for line in lines:
            if line.strip().startswith('Caller:'):
                content = line.replace('Caller:', '').strip()
                if content:  # Only add non-empty content
                    issues.append(content)
        
        return issues

    def extract_agent_info(self, transcript: str) -> Dict[str, str]:
        """Extract agent information from the transcript"""
        agent_info = {"name": "Unknown", "department": "Unknown"}
        
        lines = transcript.strip().split('\n')
        for line in lines:
            if line.strip().startswith('Agent:'):
                content = line.replace('Agent:', '').strip()
                # Try to extract agent name
                if "my name is" in content.lower():
                    name_match = re.search(r'my name is (\w+)', content.lower())
                    if name_match:
                        agent_info["name"] = name_match.group(1).title()
                
                # Try to extract department
                if any(dept in content.lower() for dept in ['support', 'billing', 'technical', 'security']):
                    for dept in ['Tech Support', 'Billing', 'Technical Support', 'Security']:
                        if dept.lower() in content.lower():
                            agent_info["department"] = dept
                            break
                break
        
        return agent_info

    def extract_account_info(self, transcript: str) -> Dict[str, str]:
        """Extract account information from the transcript"""
        account_info = {}
        
        # Look for account numbers
        account_match = re.search(r'(\d{3}-\d{3}-\d{3}|\d{9,12})', transcript)
        if account_match:
            account_info["account_number"] = account_match.group(1)
        
        return account_info

    def categorize_issues(self, transcript: str) -> List[Dict[str, Any]]:
        """Automatically categorize issues based on transcript content"""
        issues = []
        caller_statements = self.extract_caller_issues(transcript)
        
        # Keywords for different categories
        network_keywords = ['internet', 'connection', 'slow', 'down', 'network', 'wifi', 'broadband']
        billing_keywords = ['bill', 'charge', 'payment', 'cost', 'price', 'refund', 'invoice']
        account_keywords = ['account', 'password', 'login', 'address', 'phone', 'email', 'cancel', 'update']
        technical_keywords = ['device', 'router', 'modem', 'software', 'hardware', 'configure', 'setup']
        
        for statement in caller_statements:
            statement_lower = statement.lower()
            
            # Check for network issues
            if any(keyword in statement_lower for keyword in network_keywords):
                issue = {
                    "category": "Network Issue",
                    "subcategory": "Slow Connection" if "slow" in statement_lower else "Network Down" if "down" in statement_lower else "Other",
                    "description": self.generate_issue_description(statement),
                    "severity": self.assess_severity(statement),
                    "quoted_text": statement
                }
                issues.append(issue)
            
            # Check for billing issues
            elif any(keyword in statement_lower for keyword in billing_keywords):
                subcategory = "Payment not processed" if "payment" in statement_lower else "Cost Inquiry"
                issue = {
                    "category": "Billing",
                    "subcategory": subcategory,
                    "description": self.generate_issue_description(statement),
                    "severity": self.assess_severity(statement),
                    "quoted_text": statement
                }
                
                # Add sub-subcategory for cost inquiry
                if subcategory == "Cost Inquiry":
                    if "data" in statement_lower:
                        issue["sub_subcategory"] = "Data Usage"
                    elif "fiber" in statement_lower:
                        issue["sub_subcategory"] = "Fiber"
                    elif "5g" in statement_lower:
                        issue["sub_subcategory"] = "5G"
                    elif "dsl" in statement_lower:
                        issue["sub_subcategory"] = "DSL"
                
                issues.append(issue)
            
            # Check for account management issues
            elif any(keyword in statement_lower for keyword in account_keywords):
                subcategory = "Password Reset" if "password" in statement_lower else \
                             "Update Personal Info" if any(word in statement_lower for word in ["address", "phone", "email", "update"]) else \
                             "Service Cancellation" if "cancel" in statement_lower else "Other"
                
                issue = {
                    "category": "Account Management",
                    "subcategory": subcategory,
                    "description": self.generate_issue_description(statement),
                    "severity": self.assess_severity(statement),
                    "quoted_text": statement
                }
                issues.append(issue)
            
            # Check for technical support issues
            elif any(keyword in statement_lower for keyword in technical_keywords):
                subcategory = "Device Configuration" if any(word in statement_lower for word in ["router", "modem", "configure", "setup"]) else \
                             "Software Issues" if "software" in statement_lower else \
                             "Hardware Problems" if "hardware" in statement_lower else "Other"
                
                issue = {
                    "category": "Technical Support",
                    "subcategory": subcategory,
                    "description": self.generate_issue_description(statement),
                    "severity": self.assess_severity(statement),
                    "quoted_text": statement
                }
                issues.append(issue)
        
        return issues

    def generate_issue_description(self, statement: str) -> str:
        """Generate a concise description from the caller statement"""
        # Remove common filler words and create a clean description
        cleaned = re.sub(r'\b(um|uh|like|you know|I mean)\b', '', statement, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Truncate if too long
        if len(cleaned) > 100:
            cleaned = cleaned[:97] + "..."
        
        return cleaned

    def assess_severity(self, statement: str) -> str:
        """Assess the severity of an issue based on keywords"""
        statement_lower = statement.lower()
        
        high_severity_keywords = ['completely', 'totally', 'can\'t', 'won\'t', 'broken', 'frustrated', 'angry', 'urgent']
        medium_severity_keywords = ['slow', 'issue', 'problem', 'trouble', 'difficult']
        
        if any(keyword in statement_lower for keyword in high_severity_keywords):
            return "High"
        elif any(keyword in statement_lower for keyword in medium_severity_keywords):
            return "Medium"
        else:
            return "Low"

    def analyze_transcript(self, transcript: str) -> Dict[str, Any]:
        """Analyze the transcript and return structured JSON output"""
        if not transcript or not transcript.strip():
            return {
                "error": "No transcript provided",
                "call_summary": {},
                "identified_issues": [],
                "caller_statements": [],
                "model_inference": {}
            }
        
        # Extract information
        caller_issues = self.extract_caller_issues(transcript)
        agent_info = self.extract_agent_info(transcript)
        account_info = self.extract_account_info(transcript)
        categorized_issues = self.categorize_issues(transcript)
        
        # Generate call summary
        call_summary = {
            "total_issues": len(categorized_issues),
            "agent_name": agent_info["name"],
            "agent_department": agent_info["department"],
            "caller_statements_count": len(caller_issues),
            "resolution_status": "Analysis Complete"
        }
        
        # Add account info if available
        if account_info:
            call_summary.update(account_info)
        
        # Run model inference
        model_result = self.run_model_inference(transcript)
        
        return {
            "call_summary": call_summary,
            "identified_issues": categorized_issues,
            "caller_statements": caller_issues,
            "model_inference": model_result,
            "category_structure": self.category_structure,
            "analysis_metadata": {
                "transcript_length": len(transcript),
                "processing_timestamp": "2024-01-01T00:00:00Z",  # In production, use actual timestamp
                "analyzer_version": "1.0.0"
            }
        }

    def run_model_inference(self, transcript: str):
        """Run the Gemma 3n model inference with the transcript"""
        prompt = f"""{self.system_prompt}

Transcript:
{transcript}

Category Structure:
{json.dumps(self.category_structure, indent=2)}

Analyze the transcript and identify all issues discussed."""
        
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }]
        
        return do_gemma_3n_inference(messages)

# Initialize the analyzer
analyzer = CallAnalyzer()

def analyze_call_transcript(transcript: str) -> str:
    """Main function for Gradio interface"""
    try:
        result = analyzer.analyze_transcript(transcript)
        return json.dumps(result, indent=2, ensure_ascii=False)
    except Exception as e:
        error_result = {
            "error": str(e),
            "call_summary": {},
            "identified_issues": [],
            "caller_statements": [],
            "model_inference": {}
        }
        return json.dumps(error_result, indent=2, ensure_ascii=False)

# Sample transcripts for testing
sample_transcripts = {
    "Network Issue": """Agent: Thank you for calling Tech Support, my name is Sarah. How can I help you today?
Caller: Hi Sarah. I'm having a really frustrating issue. My internet connection has been incredibly slow for the past three hours. I can barely load a webpage.
Agent: I'm sorry to hear that. I can definitely look into your network status. Can you please provide your account number?
Caller: Yes, it's 123-456-789.
Agent: Thank you. I'm pulling up your account now. I can see some network congestion in your area.""",
    
    "Billing Issue": """Agent: Billing department, this is Mike. How can I assist you?
Caller: Hi Mike, I'm calling about my last bill. It seems much higher than usual and I don't understand the new data usage charges.
Agent: I can help you with that billing inquiry. Let me pull up your account details.
Caller: I've been a customer for 5 years and this is the first time I've seen charges like this.
Agent: I understand your concern. Let me review your usage patterns.""",
    
    "Account Management": """Agent: Account services, this is Lisa speaking.
Caller: Hi Lisa, I need to update my personal information. I moved last month and need to change my address.
Agent: I can help you update your account details. I'll also need to verify your identity first.
Caller: Of course. I also forgot my password and can't log into my online account.
Agent: No problem, I can help with both the address update and password reset.""",
    
    "Multiple Issues": """Agent: Customer service, this is John. How may I help you?
Caller: Hi John, I have several issues. First, my internet has been down since yesterday morning.
Agent: I'm sorry to hear that. Let me check your connection status.
Caller: Also, I tried to make a payment online but it keeps getting rejected.
Agent: I can help with both the connection issue and the payment problem.
Caller: And I need to cancel my premium service package.
Agent: I'll be happy to assist with all three issues."""
}

# Create Gradio interface
def create_gradio_interface():
    with gr.Blocks(title="Call Center Analysis Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 📞 Call Center Transcript Analysis")
        gr.Markdown("Upload or paste a call center transcript to automatically categorize issues and generate structured JSON output.")
        
        with gr.Row():
            with gr.Column(scale=1):
                transcript_input = gr.Textbox(
                    label="Call Transcript",
                    placeholder="Paste your call transcript here...",
                    lines=15,
                    max_lines=20
                )
                
                with gr.Row():
                    analyze_btn = gr.Button("🔍 Analyze Transcript", variant="primary")
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                
                gr.Markdown("### 📋 Sample Transcripts")
                sample_dropdown = gr.Dropdown(
                    choices=list(sample_transcripts.keys()),
                    label="Load Sample Transcript",
                    value=None
                )
                load_sample_btn = gr.Button("📥 Load Sample", variant="secondary")
            
            with gr.Column(scale=1):
                output_json = gr.JSON(
                    label="Analysis Results (JSON)",
                    show_label=True
                )
                
                gr.Markdown("### 📊 Analysis Features")
                gr.Markdown("""
                - **Issue Detection**: Automatically identifies caller issues
                - **Categorization**: Classifies issues into predefined categories
                - **Severity Assessment**: Evaluates issue severity (High/Medium/Low)
                - **Agent Information**: Extracts agent name and department
                - **Account Details**: Identifies account numbers and customer info
                - **Model Integration**: Includes Gemma 3n model inference results
                """)
        
        # Event handlers
        analyze_btn.click(
            fn=analyze_call_transcript,
            inputs=[transcript_input],
            outputs=[output_json]
        )
        
        clear_btn.click(
            fn=lambda: ("", None),
            inputs=[],
            outputs=[transcript_input, output_json]
        )
        
        def load_sample_transcript(sample_name):
            if sample_name and sample_name in sample_transcripts:
                return sample_transcripts[sample_name]
            return ""
        
        load_sample_btn.click(
            fn=load_sample_transcript,
            inputs=[sample_dropdown],
            outputs=[transcript_input]
        )
        
        # Auto-analyze when sample is loaded
        sample_dropdown.change(
            fn=lambda x: sample_transcripts.get(x, ""),
            inputs=[sample_dropdown],
            outputs=[transcript_input]
        )
    
    return demo

# Launch the demo
if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )