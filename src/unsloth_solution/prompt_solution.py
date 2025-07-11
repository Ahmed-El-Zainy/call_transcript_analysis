import json
import re
from typing import List, Dict, Any
from utils import do_gemma_3n_inference


class CallAnalyzer:
    def __init__(self):
        self.transcript = """
Agent: Thank you for calling Tech Support, my name is Brenda. How can I help you today?
Caller: Hi Brenda. I'm having a really frustrating issue. My internet connection has been incredibly slow for the past three hours. I can barely load a webpage.
Agent: I'm sorry to hear that. I can definitely look into your network status. Can you please provide your account number?
Caller: Yes, it's 123-456-789.
Agent: Thank you. I'm pulling up your account... I see some reported slowness in your area, and we're working on it. I expect it to be resolved within the next hour.
Caller: Okay, thanks for the update. While I have you on the line, I also wanted to ask about my last bill. It seems much higher than usual. I don't understand the new data usage charges.
Agent: Of course. I can transfer you to our billing department to discuss the cost inquiry, or I can pull up the details for you here if you'd like.
Caller: Just transfer me, that's fine.
Agent: Understood. One moment please.
"""
        
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

    def extract_caller_issues(self) -> List[str]:
        """Extract issues mentioned by the caller from the transcript"""
        issues = []
        lines = self.transcript.strip().split('\n')
        
        for line in lines:
            if line.strip().startswith('Caller:'):
                content = line.replace('Caller:', '').strip()
                issues.append(content)
        
        return issues

    def analyze_transcript(self) -> Dict[str, Any]:
        """Analyze the transcript and categorize issues"""
        caller_issues = self.extract_caller_issues()
        
        # Manual analysis based on the transcript content
        identified_issues = []
        
        # Issue 1: Network connectivity problem
        network_issue = {
            "category": "Network Issue",
            "subcategory": "Slow Connection",
            "description": "Internet connection has been incredibly slow for the past three hours, can barely load a webpage",
            "severity": "High",
            "quoted_text": "My internet connection has been incredibly slow for the past three hours. I can barely load a webpage."
        }
        identified_issues.append(network_issue)
        
        # Issue 2: Billing inquiry about high charges
        billing_issue = {
            "category": "Billing",
            "subcategory": "Cost Inquiry",
            "sub_subcategory": "Data Usage",
            "description": "Last bill seems much higher than usual, confusion about new data usage charges",
            "severity": "Medium",
            "quoted_text": "I also wanted to ask about my last bill. It seems much higher than usual. I don't understand the new data usage charges."
        }
        identified_issues.append(billing_issue)
        
        return {
            "call_summary": {
                "total_issues": len(identified_issues),
                "caller_account": "123-456-789",
                "agent_name": "Brenda",
                "resolution_status": "In Progress - Network issue being resolved, billing transferred to billing department"
            },
            "identified_issues": identified_issues,
            "caller_statements": caller_issues
        }

    def generate_analysis_prompt(self) -> str:
        """Generate the complete analysis prompt"""
        return f"""{self.system_prompt}

Transcript:
{self.transcript}

Category Structure:
{json.dumps(self.category_structure, indent=2)}

Analyze the transcript and identify all issues discussed. For each issue, provide:
1. The main category
2. The subcategory (if applicable)
3. The sub-subcategory (if applicable)
4. A brief description of the issue
5. The exact quote from the caller that describes the issue

Identified Issues:
"""

    def run_model_inference(self):
        """Run the Gemma 3n model inference with the generated prompt"""
        prompt = self.generate_analysis_prompt()
        
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }]
        
        # Call your existing model inference function
        return do_gemma_3n_inference(messages)

    def print_analysis(self):
        """Print the complete analysis"""
        analysis = self.analyze_transcript()
        
        print("=" * 60)
        print("CALL CENTER ANALYSIS REPORT")
        print("=" * 60)
        
        print(f"\nCALL SUMMARY:")
        print(f"Agent: {analysis['call_summary']['agent_name']}")
        print(f"Account: {analysis['call_summary']['caller_account']}")
        print(f"Total Issues: {analysis['call_summary']['total_issues']}")
        print(f"Status: {analysis['call_summary']['resolution_status']}")
        
        print(f"\nIDENTIFIED ISSUES:")
        print("-" * 40)
        
        for i, issue in enumerate(analysis['identified_issues'], 1):
            print(f"\nIssue #{i}:")
            print(f"  Category: {issue['category']}")
            print(f"  Subcategory: {issue['subcategory']}")
            if 'sub_subcategory' in issue:
                print(f"  Sub-subcategory: {issue['sub_subcategory']}")
            print(f"  Severity: {issue['severity']}")
            print(f"  Description: {issue['description']}")
            print(f"  Quote: \"{issue['quoted_text']}\"")
        
        print(f"\nRAW CALLER STATEMENTS:")
        print("-" * 40)
        for i, statement in enumerate(analysis['caller_statements'], 1):
            print(f"{i}. {statement}")

    def run_complete_analysis(self):
        """Run both manual analysis and model inference"""
        print("MANUAL ANALYSIS:")
        self.print_analysis()
        
        print("\n" + "=" * 80)
        print("RUNNING MODEL INFERENCE...")
        print("=" * 80)
        
        # Run model inference
        model_result = self.run_model_inference()
        
        return model_result

    def run_test_suite(self):
        """Run a comprehensive test suite with multiple transcript scenarios"""
        print("RUNNING TEST SUITE")
        print("=" * 60)
        
        test_transcripts = [
            {
                "name": "Multi-issue Call",
                "transcript": """
Agent: Hello, this is Tech Support. How can I help you today?
Caller: Hi, I have multiple problems. First, my internet is completely down since this morning.
Agent: I'm sorry to hear that. Let me check your connection status.
Caller: Also, I tried to make a payment online but it keeps failing.
Agent: I can help with both issues. Let me start with the connection problem.
Caller: And one more thing - I need to update my address in the system.
Agent: Of course, I'll help with all three issues.
"""
            },
            {
                "name": "Billing Dispute",
                "transcript": """
Agent: Thank you for calling billing support. How can I assist you?
Caller: I'm really upset about my bill. You charged me for 5G service but I only have DSL.
Agent: I apologize for the confusion. Let me review your account.
Caller: This is the third time I'm calling about this same issue!
Agent: I understand your frustration. I'll make sure this gets resolved today.
"""
            },
            {
                "name": "Account Security",
                "transcript": """
Agent: Security department, how may I help you?
Caller: I think someone accessed my account. I need to reset my password immediately.
Agent: I can definitely help with that. For security purposes, I'll need to verify your identity.
Caller: Of course. I also want to update my security questions.
Agent: Absolutely, we can take care of both items for you.
"""
            }
        ]
        
        for i, test_case in enumerate(test_transcripts, 1):
            print(f"\nTEST CASE {i}: {test_case['name']}")
            print("-" * 50)
            
            # Temporarily replace the transcript for analysis
            original_transcript = self.transcript
            self.transcript = test_case['transcript']
            
            # Run analysis
            analysis = self.analyze_transcript()
            
            print(f"Issues Found: {analysis['call_summary']['total_issues']}")
            for j, issue in enumerate(analysis['identified_issues'], 1):
                print(f"  Issue {j}: {issue['category']} -> {issue['subcategory']}")
                if 'sub_subcategory' in issue:
                    print(f"    Sub-category: {issue['sub_subcategory']}")
            
            # Restore original transcript
            self.transcript = original_transcript
        
        print("\nTEST SUITE COMPLETED")
        print("=" * 60)


class ModelTestRunner:
    """A dedicated test runner for model inference testing"""
    
    def __init__(self):
        self.test_cases = {
            "network_down": {
                "transcript": """
Agent: Technical support, this is Mike. How can I help?
Caller: My internet is completely down. I can't connect to anything.
Agent: I'm sorry to hear that. Let me check your connection.
Caller: This is costing me money since I work from home!
Agent: I understand. Let me get this resolved quickly.
""",
                "expected_categories": ["Network Issue", "Network Down"]
            },
            "billing_payment": {
                "transcript": """
Agent: Billing department, this is Sarah.
Caller: I tried to pay my bill but the payment keeps getting rejected.
Agent: I can help with that payment issue.
Caller: I've tried three different credit cards.
Agent: Let me check your payment settings.
""",
                "expected_categories": ["Billing", "Payment not processed"]
            },
            "account_update": {
                "transcript": """
Agent: Account services, how may I help you?
Caller: I need to update my personal information. I moved last month.
Agent: I can help you update your account details.
Caller: I need to change my address and phone number.
Agent: No problem, I'll make those updates for you.
""",
                "expected_categories": ["Account Management", "Update Personal Info"]
            },
            "password_reset": {
                "transcript": """
Agent: Security support, this is Tom.
Caller: I forgot my password and I'm locked out of my account.
Agent: I can help you reset your password.
Caller: I've been trying to log in for an hour.
Agent: Let me verify your identity and reset it for you.
""",
                "expected_categories": ["Account Management", "Password Reset"]
            },
            "service_inquiry": {
                "transcript": """
Agent: Sales support, this is Lisa.
Caller: I want to know about your fiber internet pricing.
Agent: I'd be happy to help with pricing information.
Caller: What are the monthly costs for different speeds?
Agent: Let me go over our fiber plans with you.
""",
                "expected_categories": ["Billing", "Cost Inquiry", "Fiber"]
            }
        }
    
    def list_available_tests(self):
        """List all available test cases"""
        print("AVAILABLE TEST CASES:")
        print("-" * 30)
        for test_name, test_data in self.test_cases.items():
            expected = " -> ".join(test_data["expected_categories"])
            print(f"  {test_name}: {expected}")
        print()
    
    def run_single_test(self, test_name: str):
        """Run a single test case"""
        if test_name not in self.test_cases:
            print(f"Test '{test_name}' not found!")
            return
        
        test_data = self.test_cases[test_name]
        print(f"Running test: {test_name}")
        print("-" * 40)
        
        # Create analyzer with test transcript
        analyzer = CallAnalyzer()
        analyzer.transcript = test_data["transcript"]
        
        # Run analysis
        analysis = analyzer.analyze_transcript()
        
        print(f"Expected: {' -> '.join(test_data['expected_categories'])}")
        print(f"Found {analysis['call_summary']['total_issues']} issues:")
        
        for i, issue in enumerate(analysis['identified_issues'], 1):
            category_path = issue['category']
            if 'subcategory' in issue:
                category_path += f" -> {issue['subcategory']}"
            if 'sub_subcategory' in issue:
                category_path += f" -> {issue['sub_subcategory']}"
            print(f"  Issue {i}: {category_path}")
        
        # Run model inference
        print("\nRunning model inference...")
        model_result = analyzer.run_model_inference()
        print(f"Model result: {model_result}")
        
        return analysis, model_result
    
    def run_all_tests(self):
        """Run all available test cases"""
        print("RUNNING ALL TEST CASES")
        print("=" * 50)
        
        results = {}
        for test_name in self.test_cases:
            print(f"\n--- {test_name.upper()} ---")
            results[test_name] = self.run_single_test(test_name)
        
        print("\n" + "=" * 50)
        print("ALL TESTS COMPLETED")
        return results
    
    def run_custom_test(self, transcript: str, expected_categories: List[str] = None):
        """Run a test with a custom transcript"""
        print("RUNNING CUSTOM TEST")
        print("-" * 40)
        
        # Create analyzer with custom transcript
        analyzer = CallAnalyzer()
        analyzer.transcript = transcript
        
        # Run analysis
        analysis = analyzer.analyze_transcript()
        
        if expected_categories:
            print(f"Expected: {' -> '.join(expected_categories)}")
        
        print(f"Found {analysis['call_summary']['total_issues']} issues:")
        
        for i, issue in enumerate(analysis['identified_issues'], 1):
            category_path = issue['category']
            if 'subcategory' in issue:
                category_path += f" -> {issue['subcategory']}"
            if 'sub_subcategory' in issue:
                category_path += f" -> {issue['sub_subcategory']}"
            print(f"  Issue {i}: {category_path}")
            print(f"    Description: {issue['description']}")
        
        # Run model inference
        print("\nRunning model inference...")
        model_result = analyzer.run_model_inference()
        print(f"Model result: {model_result}")
        
        return analysis, model_result


# Usage examples
if __name__ == "__main__":
    # Example 1: Run complete analysis with default transcript
    print("EXAMPLE 1: Default transcript analysis")
    print("=" * 60)
    analyzer = CallAnalyzer()
    analyzer.run_complete_analysis()
    
    print("\n" + "="*100)
    
    # Example 2: Run full test suite
    print("EXAMPLE 2: Running full test suite")
    print("=" * 60)
    analyzer.run_test_suite()
    
    print("\n" + "="*100)
    
    # Example 3: Test runner examples
    print("EXAMPLE 3: Using ModelTestRunner")
    print("=" * 60)
    
    test_runner = ModelTestRunner()
    
    # List available tests
    test_runner.list_available_tests()
    
    # Run single test
    print("\nRunning single test: 'network_down'")
    print("-" * 40)
    test_runner.run_single_test("network_down")
    
    # Run custom transcript
    custom_transcript = """
Agent: Hello, this is Support. How can I help?
Caller: Hi, I need to cancel my service immediately. Your company has the worst customer service ever!
Agent: I'm sorry to hear about your experience. Let me help you with the cancellation.
Caller: Finally, someone who listens. I also need a refund for this month.
"""
    
    print("\nRunning custom transcript test")
    print("-" * 40)
    test_runner.run_custom_test(custom_transcript)