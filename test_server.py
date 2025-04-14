import requests
import json
import sseclient
import unittest
import time

class TestGenerateEndpoints(unittest.TestCase):
    def setUp(self):
        self.base_url = "http://localhost:5000"  # Adjust if your server runs on a different port
        coding_problem = """In a mystic dungeon, n magicians are standing in a line. Each magician has an attribute that gives you energy. Some magicians can give you negative energy, which means taking energy from you. You have been cursed in such a way that after absorbing energy from magician i, you will be instantly transported to magician (i + k). This process will be repeated until you reach the magician where (i + k) does not exist. In other words, you will choose a starting point and then teleport with k jumps until you reach the end of the magicians' sequence, absorbing all the energy during the journey. You are given an array energy and an integer k. Return the maximum possible energy you can gain.

Example 1:

Input: energy = [5,2,-10,-5,1], k = 3 Output: 3 Explanation: We can gain a total energy of 3 by starting from magician 1 absorbing 2 + 1 = 3.

Example 2:

Input: energy = [-2,-3,-1], k = 2 Output: -1 Explanation: We can gain a total energy of -1 by starting from magician 2.

Constraints:

1 <= energy.length <= 10^5 -1000 <= energy[i] <= 1000 1 <= k <= energy.length - 1"""
        # Define standard test messages with system roles
        self.test_messages = [
            {"role": "system", "content": "You are a helpful assistant. Your job is to work on a problem with the user. Respond only the user's queries, you don't need to solve the problem immediately. Only think for 500 words at most."},
            {"role": "system", "content": f"Here's a coding problem that you are working on: {coding_problem}"},
            {"role": "user", "content": "Hey what do you think about this problem?"},
        ]
        
        # Define Claude-compatible messages (without system roles)
        self.claude_messages = [
            {"role": "user", "content": "You are a helpful assistant. Your job is to work on a problem with the user. Respond only the user's queries, you don't need to solve the problem immediately. Only think for 500 words at most.\n\nHere's a coding problem that you are working on: " + coding_problem},
            {"role": "assistant", "content": "I understand. I'll help you work through this coding problem without solving it immediately and keep my explanations concise."},
            {"role": "user", "content": "Hey what do you think about this problem?"},
        ]
        
        self.models = [
            # "gpt-4o-2024-11-20",
            # "claude-3-7-sonnet-20250219",
            # "deepseek-chat",
            # "deepseek-reasoner",
            # "o1-2024-12-17",
            # "gemini-2.5-pro-preview-03-25",
            # "deepseek-ai/DeepSeek-R1",
            "gpt-4.5-preview-2025-02-27",
            # "deepseek-ai/DeepSeek-V3",
            # "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
        ]
        
        # Define which models need special message formatting
        self.claude_model_prefixes = ["claude", "anthropic"]

    def get_messages_for_model(self, model):
        """Return the appropriate messages format based on the model"""
        # Check if the model is a Claude model
        if any(prefix in model.lower() for prefix in self.claude_model_prefixes):
            return self.claude_messages
        return self.test_messages

    def test_generate_endpoint(self):
        """Test the non-streaming generate endpoint"""
        for model in self.models:
            with self.subTest(model=model):
                try:
                    print(f"\nTesting model: {model}")  # Debug log
                    
                    # Use appropriate message format for the model
                    messages = self.get_messages_for_model(model)
                    
                    response = requests.post(
                        f"{self.base_url}/generate",
                        json={
                            "messages": messages,
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
                "messages": self.test_messages,  # Can use standard format for invalid model
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

    def test_generate_streaming_endpoint(self):
        """Test the streaming generate endpoint"""
        for model in self.models:
            with self.subTest(model=model):
                try:
                    print(f"\nTesting streaming model: {model}")  # Debug log
                    
                    # Use appropriate message format for the model
                    messages = self.get_messages_for_model(model)
                    
                    # Make request with stream=True to handle SSE
                    response = requests.post(
                        f"{self.base_url}/generate-streaming",
                        json={
                            "messages": messages,
                            "model": model
                        },
                        stream=True
                    )
                    
                    print(f"Response status code: {response.status_code}")  # Debug log
                    
                    self.assertEqual(response.status_code, 200, 
                        f"Expected status code 200, but got {response.status_code}")
                    
                    # Parse SSE response
                    client = sseclient.SSEClient(response)
                    message_parts = []
                    
                    # Collect first few chunks (limiting to avoid long tests)
                    chunk_count = 0
                    max_chunks = 5
                    
                    # Add a timeout for the streaming test
                    timeout = 10  # seconds
                    start_time = time.time()
                    
                    for event in client.events():
                        message_parts.append(event.data)
                        chunk_count += 1
                        if chunk_count >= max_chunks:
                            break
                        # Add a timeout check
                        if time.time() - start_time > timeout:
                            print(f"Timeout after {timeout} seconds waiting for chunks")
                            break
                    
                    # Check that we got some content
                    self.assertGreater(len(message_parts), 0, 
                        "No message chunks received from streaming endpoint")
                    
                    # Print the collected message parts for debugging
                    print(f"Collected {len(message_parts)} message chunks")
                    print(f"First chunk: {message_parts[0][:100]}...")
                    
                except Exception as e:
                    print(f"Exception occurred while testing streaming {model}: {str(e)}")
                    raise

    def test_invalid_model_streaming(self):
        """Test streaming response with an invalid model name"""
        response = requests.post(
            f"{self.base_url}/generate-streaming",
            json={
                "messages": self.test_messages,  # Can use standard format for invalid model
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

    def test_missing_parameters_streaming(self):
        """Test streaming response when required parameters are missing"""
        # Test missing messages
        response = requests.post(
            f"{self.base_url}/generate-streaming",
            json={
                "model": "gpt-3.5-turbo"
            }
        )
        self.assertEqual(response.status_code, 400,
            f"Expected status code 400 for missing messages, got {response.status_code}")

        # Test missing model
        response = requests.post(
            f"{self.base_url}/generate-streaming",
            json={
                "messages": self.test_messages
            }
        )
        self.assertEqual(response.status_code, 400,
            f"Expected status code 400 for missing model, got {response.status_code}")

if __name__ == "__main__":
    unittest.main()