import requests
import json
import sseclient
import unittest

class TestGenerateEndpoints(unittest.TestCase):
    def setUp(self):
        self.base_url = "http://localhost:5000"  # Adjust if your server runs on a different port
        self.test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a simple hello world program in Python."}
        ]
        self.models = [
            # "gpt-4o-2024-11-20",
            # "claude-3-7-sonnet-20250219",
            # "deepseek-chat",
            # "deepseek-reasoner",
            # "o1-2024-12-17",
            "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
        ]

    def test_generate_endpoint(self):
        """Test the non-streaming generate endpoint"""
        for model in self.models:
            with self.subTest(model=model):
                try:
                    print(f"\nTesting model: {model}")  # Debug log
                    response = requests.post(
                        f"{self.base_url}/generate",
                        json={
                            "messages": self.test_messages,
                            "model": model
                        }
                    )
                    
                    print(f"Response status code: {response.status_code}")  # Debug log
                    print(f"Response content: {response.content}")  # Debug log
                    
                    self.assertEqual(response.status_code, 200, 
                        f"Expected status code 200, but got {response.status_code}\nResponse content: {response.content}")
                    
                    data = response.json()
                    print(f"Parsed JSON data: {data}")  # Debug log
                    
                    self.assertIn("message", data, 
                        f"Response missing 'message' key. Response data: {data}")
                    self.assertIsInstance(data["message"], str, 
                        f"Expected message to be string, got {type(data['message'])}")
                    self.assertGreater(len(data["message"]), 0, 
                        "Response message is empty")
                except Exception as e:
                    print(f"Exception occurred while testing {model}: {str(e)}")
                    raise

    def test_invalid_model(self):
        """Test response with an invalid model name"""
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "messages": self.test_messages,
                "model": "invalid-model-name"
            }
        )
        
        self.assertEqual(response.status_code, 200, 
            f"Expected status code 200, but got {response.status_code}")
        data = response.json()
        self.assertIn("message", data, 
            f"Response missing 'message' key. Response data: {data}")
        self.assertEqual(data["message"], "Model not supported.", 
            f"Expected 'Model not supported.' message, got '{data['message']}'")

    def test_missing_parameters(self):
        """Test response when required parameters are missing"""
        # Test missing messages
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "model": "gpt-3.5-turbo"
            }
        )
        self.assertEqual(response.status_code, 400,
            f"Expected status code 400 for missing messages, got {response.status_code}")

        # Test missing model
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "messages": self.test_messages
            }
        )
        self.assertEqual(response.status_code, 400,
            f"Expected status code 400 for missing model, got {response.status_code}")

if __name__ == "__main__":
    unittest.main()