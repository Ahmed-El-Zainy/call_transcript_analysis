#!/usr/bin/env python3
"""
Test Script for Call Transcript Analysis Workflow
Tests the complete workflow with sample data and various scenarios
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from main_test import TranscriptAnalyzer, create_sample_files, AnalysisResult


class TestTranscriptAnalyzer(unittest.TestCase):
    """Test cases for the TranscriptAnalyzer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_transcript = """
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
        
        self.sample_categories = [
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
        
        # Create temporary files for testing
        self.temp_dir = tempfile.mkdtemp()
        self.transcript_path = os.path.join(self.temp_dir, "test_transcript.txt")
        self.categories_path = os.path.join(self.temp_dir, "test_categories.json")
        
        with open(self.transcript_path, 'w') as f:
            f.write(self.sample_transcript)
        
        with open(self.categories_path, 'w') as f:
            json.dump(self.sample_categories, f)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_inputs(self):
        """Test loading transcript and categories files"""
        # Mock API setup to avoid requiring real API keys
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            with patch('main.genai.configure'):
                with patch('main.genai.GenerativeModel'):
                    analyzer = TranscriptAnalyzer(provider='gemini')
                    
                    transcript, categories = analyzer.load_inputs(
                        self.transcript_path, 
                        self.categories_path
                    )
                    
                    self.assertEqual(transcript.strip(), self.sample_transcript.strip())
                    self.assertEqual(categories, self.sample_categories)
    
    def test_build_prompt(self):
        """Test prompt building functionality"""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            with patch('main.genai.configure'):
                with patch('main.genai.GenerativeModel'):
                    analyzer = TranscriptAnalyzer(provider='gemini')
                    
                    prompt = analyzer.build_prompt(self.sample_transcript, self.sample_categories)
                    
                    # Check that prompt contains key components
                    self.assertIn("transcript", prompt.lower())
                    self.assertIn("categories", prompt.lower())
                    self.assertIn("json", prompt.lower())
                    self.assertIn(self.sample_transcript.strip(), prompt)
                    self.assertIn("Network Issue", prompt)
                    self.assertIn("Billing", prompt)
    
    def test_parse_output_valid_json(self):
        """Test parsing valid JSON output"""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            with patch('main.genai.configure'):
                with patch('main.genai.GenerativeModel'):
                    analyzer = TranscriptAnalyzer(provider='gemini')
                    
                    # Test valid JSON output
                    valid_output = '''
                    Here is the analysis result:
                    [
                        ["Network Issue", "Slow Connection"],
                        ["Billing", "Cost Inquiry", "DSL"]
                    ]
                    '''
                    
                    result = analyzer.parse_output(valid_output)
                    expected = [
                        ["Network Issue", "Slow Connection"],
                        ["Billing", "Cost Inquiry", "DSL"]
                    ]
                    
                    self.assertEqual(result, expected)
    
    def test_parse_output_invalid_json(self):
        """Test parsing invalid JSON output"""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            with patch('main.genai.configure'):
                with patch('main.genai.GenerativeModel'):
                    analyzer = TranscriptAnalyzer(provider='gemini')
                    
                    # Test invalid JSON output
                    invalid_output = "This is not valid JSON"
                    result = analyzer.parse_output(invalid_output)
                    
                    self.assertEqual(result, [])
    
    def test_parse_output_empty_result(self):
        """Test parsing empty result"""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            with patch('main.genai.configure'):
                with patch('main.genai.GenerativeModel'):
                    analyzer = TranscriptAnalyzer(provider='gemini')
                    
                    # Test empty result
                    empty_output = "No issues found: []"
                    result = analyzer.parse_output(empty_output)
                    
                    self.assertEqual(result, [])
    
    @patch('main.genai.GenerativeModel')
    @patch('main.genai.configure')
    def test_gemini_integration(self, mock_configure, mock_model_class):
        """Test Gemini API integration"""
        # Mock the API response
        mock_response = MagicMock()
        mock_response.text = '[["Network Issue", "Slow Connection"], ["Billing", "Cost Inquiry", "DSL"]]'
        
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            analyzer = TranscriptAnalyzer(provider='gemini')
            result = analyzer.analyze(self.transcript_path, self.categories_path)
            
            # Verify the result
            self.assertIsInstance(result, AnalysisResult)
            self.assertEqual(len(result.analysis_results), 2)
            self.assertEqual(result.analysis_results[0], ["Network Issue", "Slow Connection"])
            self.assertEqual(result.analysis_results[1], ["Billing", "Cost Inquiry", "DSL"])
    
    @patch('main.requests.post')
    def test_huggingface_integration(self, mock_post):
        """Test Hugging Face API integration"""
        # Mock the API response
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"generated_text": '[["Network Issue", "Slow Connection"], ["Billing", "Cost Inquiry", "DSL"]]'}
        ]
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        with patch.dict(os.environ, {'HF_TOKEN': 'test_token'}):
            analyzer = TranscriptAnalyzer(provider='huggingface')
            result = analyzer.analyze(self.transcript_path, self.categories_path)
            
            # Verify the result
            self.assertIsInstance(result, AnalysisResult)
            self.assertEqual(len(result.analysis_results), 2)
            self.assertEqual(result.analysis_results[0], ["Network Issue", "Slow Connection"])
            self.assertEqual(result.analysis_results[1], ["Billing", "Cost Inquiry", "DSL"])
    
    def test_invalid_provider(self):
        """Test invalid provider handling"""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            with self.assertRaises(ValueError):
                TranscriptAnalyzer(provider='invalid_provider')
    
    def test_missing_api_key(self):
        """Test missing API key handling"""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                TranscriptAnalyzer(provider='gemini')
    
    def test_file_not_found(self):
        """Test file not found error handling"""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            with patch('main.genai.configure'):
                with patch('main.genai.GenerativeModel'):
                    analyzer = TranscriptAnalyzer(provider='gemini')
                    
                    with self.assertRaises(FileNotFoundError):
                        analyzer.load_inputs("nonexistent.txt", "nonexistent.json")


class TestWorkflowIntegration(unittest.TestCase):
    """Integration tests for the complete workflow"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create more comprehensive test data
        self.complex_transcript = """
        Agent: Good morning, TechSupport here. How can I help you?
        
        Customer: Hi, I have multiple issues. First, my internet keeps cutting out every few hours. 
        Second, I got charged twice for my monthly service. And third, my router seems to be overheating.
        
        Agent: I'm sorry to hear about these issues. Let me address each one:
        
        For the internet connectivity, are you on cable or DSL?
        
        Customer: I'm on cable internet, 100 Mbps plan.
        
        Agent: I can see intermittent connection drops in your area. There's a known issue with the cable infrastructure.
        I'll schedule a technician to check your line and replace any faulty equipment.
        
        For the billing, I can see you were charged twice in error. I'll process a refund immediately.
        
        Regarding the router overheating, that's a safety concern. I'll have the technician bring a replacement router.
        
        Customer: Great, when can someone come out?
        
        Agent: I have availability tomorrow between 2-4 PM. Does that work?
        
        Customer: Perfect, thank you!
        """
        
        self.complex_categories = [
            {
                "name": "Connectivity Issues",
                "subcategories": [
                    {
                        "name": "Intermittent Connection",
                        "subcategories": [
                            {"name": "Cable", "subcategories": []},
                            {"name": "DSL", "subcategories": []},
                            {"name": "Fiber", "subcategories": []}
                        ]
                    },
                    {
                        "name": "Complete Outage",
                        "subcategories": [
                            {"name": "Planned Maintenance", "subcategories": []},
                            {"name": "Infrastructure Issue", "subcategories": []}
                        ]
                    }
                ]
            },
            {
                "name": "Billing Issues",
                "subcategories": [
                    {
                        "name": "Billing Errors",
                        "subcategories": [
                            {"name": "Double Charge", "subcategories": []},
                            {"name": "Wrong Amount", "subcategories": []}
                        ]
                    },
                    {
                        "name": "Refund Requests",
                        "subcategories": [
                            {"name": "Service Credit", "subcategories": []},
                            {"name": "Overcharge Refund", "subcategories": []}
                        ]
                    }
                ]
            },
            {
                "name": "Equipment Problems",
                "subcategories": [
                    {
                        "name": "Hardware Issues",
                        "subcategories": [
                            {"name": "Router Problems", "subcategories": []},
                            {"name": "Modem Issues", "subcategories": []},
                            {"name": "Overheating", "subcategories": []}
                        ]
                    },
                    {
                        "name": "Equipment Replacement",
                        "subcategories": [
                            {"name": "Defective Unit", "subcategories": []},
                            {"name": "Upgrade", "subcategories": []}
                        ]
                    }
                ]
            }
        ]
        
        # Write test files
        self.transcript_path = os.path.join(self.temp_dir, "complex_transcript.txt")
        self.categories_path = os.path.join(self.temp_dir, "complex_categories.json")
        
        with open(self.transcript_path, 'w') as f:
            f.write(self.complex_transcript)
        
        with open(self.categories_path, 'w') as f:
            json.dump(self.complex_categories, f)
    
    def tearDown(self):
        """Clean up integration test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('main.genai.GenerativeModel')
    @patch('main.genai.configure')
    def test_complex_workflow(self, mock_configure, mock_model_class):
        """Test workflow with complex, multi-issue transcript"""
        # Mock comprehensive API response
        mock_response = MagicMock()
        mock_response.text = '''
        Based on the transcript analysis, here are the identified issues:
        [
            ["Connectivity Issues", "Intermittent Connection", "Cable"],
            ["Billing Issues", "Billing Errors", "Double Charge"],
            ["Billing Issues", "Refund Requests", "Overcharge Refund"],
            ["Equipment Problems", "Hardware Issues", "Overheating"],
            ["Equipment Problems", "Equipment Replacement", "Defective Unit"]
        ]
        '''
        
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            analyzer = TranscriptAnalyzer(provider='gemini')
            result = analyzer.analyze(self.transcript_path, self.categories_path)
            
            # Verify multiple issues were identified
            self.assertIsInstance(result, AnalysisResult)
            self.assertGreaterEqual(len(result.analysis_results), 3)
            
            # Check metadata
            self.assertIsNotNone(result.metadata)
            self.assertEqual(result.metadata['provider'], 'gemini')
            self.assertGreater(result.metadata['transcript_length'], 0)
            self.assertGreater(result.metadata['num_categories'], 0)


def run_manual_tests():
    """Run manual tests that require actual API keys"""
    print("Running manual tests...")
    print("Note: These tests require actual API keys set in environment variables")
    
    # Test with sample data
    print("\n1. Testing sample data creation...")
    create_sample_files()
    print("✓ Sample files created")
    
    # Test with Gemini (if API key available)
    gemini_key = os.getenv('GEMINI_API_KEY')
    if gemini_key:
        print("\n2. Testing Gemini integration...")
        try:
            analyzer = TranscriptAnalyzer(provider='gemini')
            result = analyzer.analyze('sample_transcript.txt', 'sample_categories.json')
            print(f"✓ Gemini test passed. Found {len(result.analysis_results)} issues")
            print(f"Results: {result.analysis_results}")
        except Exception as e:
            print(f"✗ Gemini test failed: {e}")
    else:
        print("\n2. Skipping Gemini test (no API key)")
    
    # Test with Hugging Face (if token available)
    hf_token = os.getenv('HF_TOKEN')
    if hf_token:
        print("\n3. Testing Hugging Face integration...")
        try:
            analyzer = TranscriptAnalyzer(provider='huggingface')
            result = analyzer.analyze('sample_transcript.txt', 'sample_categories.json')
            print(f"✓ Hugging Face test passed. Found {len(result.analysis_results)} issues")
            print(f"Results: {result.analysis_results}")
        except Exception as e:
            print(f"✗ Hugging Face test failed: {e}")
    else:
        print("\n3. Skipping Hugging Face test (no token)")
    
    # Clean up
    try:
        os.remove('sample_transcript.txt')
        os.remove('sample_categories.json')
        print("\n✓ Cleanup completed")
    except:
        pass


def run_performance_tests():
    """Run performance tests with various data sizes"""
    print("\nRunning performance tests...")
    
    # Test with different transcript lengths
    transcript_sizes = [
        ("Small", "Agent: Hello. Customer: My internet is slow. Agent: I'll help you."),
        ("Medium", "Agent: Hello. Customer: My internet is slow. " * 50),
        ("Large", "Agent: Hello. Customer: My internet is slow. " * 200)
    ]
    
    simple_categories = [
        {"name": "Network", "subcategories": [
            {"name": "Slow", "subcategories": []}
        ]}
    ]
    
    for size_name, transcript in transcript_sizes:
        print(f"\nTesting {size_name} transcript ({len(transcript)} chars)...")
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(transcript)
            transcript_path = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(simple_categories, f)
            categories_path = f.name
        
        try:
            with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
                with patch('main.genai.configure'):
                    with patch('main.genai.GenerativeModel') as mock_model_class:
                        mock_response = MagicMock()
                        mock_response.text = '[["Network", "Slow"]]'
                        mock_model = MagicMock()
                        mock_model.generate_content.return_value = mock_response
                        mock_model_class.return_value = mock_model
                        
                        import time
                        start_time = time.time()
                        
                        analyzer = TranscriptAnalyzer(provider='gemini')
                        result = analyzer.analyze(transcript_path, categories_path)
                        
                        end_time = time.time()
                        print(f"✓ {size_name} test completed in {end_time - start_time:.2f} seconds")
                        print(f"  Found {len(result.analysis_results)} issues")
        
        except Exception as e:
            print(f"✗ {size_name} test failed: {e}")
        
        finally:
            # Clean up
            os.unlink(transcript_path)
            os.unlink(categories_path)


if __name__ == "__main__":
    print("Call Transcript Analysis - Test Suite")
    print("=" * 50)
    
    # Run unit tests
    print("\n🧪 Running Unit Tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run manual tests (require API keys)
    print("\n🔧 Running Manual Tests...")
    run_manual_tests()
    
    # Run performance tests
    print("\n⚡ Running Performance Tests...")
    run_performance_tests()
    
    print("\n✅ All tests completed!")
    print("\nTo run with real API keys:")
    print("1. Set GEMINI_API_KEY or HF_TOKEN environment variables")
    print("2. Run: python test_workflow.py")
    print("3. Or run individual components:")
    print("   - python -m unittest test_workflow.TestTranscriptAnalyzer")
    print("   - python -m unittest test_workflow.TestWorkflowIntegration")