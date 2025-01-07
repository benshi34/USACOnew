import subprocess
from flask import jsonify

def run_code(code, input, expected_output):
    try:
        result = subprocess.run(
            ["python3", "-c", code],  # Execute code as a string
            input=input,
            text=True,
            capture_output=True,
            timeout=10  # Prevent infinite loops
        )
    except subprocess.TimeoutExpired:
        return "Code Execution Timed Out"
    except Exception as e:
        return str(e)

    parsed_output = result.stdout.strip('\n')
    return {'output': parsed_output, 'passed': parsed_output == expected_output}

if __name__ == '__main__':
    code = '''class Solution:
    def fib(self, n: int) -> int:
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b

solution = Solution()
i = input()
print(solution.fib(int(i)))'''
    print(run_code(code, '3', '2'))