from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify
from flask_cors import CORS
# from USACOBench.evaluation.judges.usaco_judge import USACOJudge
from USACOBench.evaluation.judges.leetcode_judge import LeetCodeJudge
from utils import load_json
from USACOBench.evaluation import ResultType

app = Flask(__name__)
CORS(app)

# problem_dict = load_json('data/datasets/usaco_subset307_dict')

@app.route('/set-leetcode-auth', methods=['POST'])
def set_leetcode_auth():
    data = request.json
    leetcode_cookie = data.get('leetcode_cookie')
    leetcode_csrf_token = data.get('leetcode_csrf_token')

    if not leetcode_cookie or not leetcode_csrf_token:
        return jsonify({'error': 'Missing cookie or CSRF token'}), 400

    print(f"Received LeetCode cookie: {leetcode_cookie}")
    print(f"Received LeetCode CSRF token: {leetcode_csrf_token}")

    # In a real application, you might want to store these in a secure way
    # For example, in an encrypted database or a secure key-value store
    return jsonify({'message': 'LeetCode authentication info received successfully'}), 200


@app.route('/execute', methods=['POST'])
def execute_code():
    data = request.json
    code = data.get('model_output')
    problem_id = data.get('problem_id')
    platform = data.get('platform')
    
    # Clean out the delimiters:
    print('Executing!')
    
    try:
        # Execute the code and capture the output
        code = code.strip().replace('```Python', "").replace('```', "").replace('```python', "")
        assert code[0] != '`' and code[-1] != '`'
        if platform == 'usaco':
            # assert problem_id in problem_dict
            return jsonify({'output': 'USACO not supported at the moment.', 'passed': False})

        # if platform == 'usaco':
        #     judge = USACOJudge()
        #     result = judge._judge(problem_dict[problem_id], code, save_solution=False, mode='eval_all')
        elif platform == 'leetcode':
            judge = LeetCodeJudge()
            print(code)
            result = judge._leetcode_judge(code, problem_id)
        
        if result['result_type'] == ResultType.UNKNOWN:
            output = "Error occurred during evaluation."
            passed = False
        else:
            if platform == 'usaco':
                passed = result['num_passed'] == result['num_tests']
                output = f"Evaluation Results: \n Passed: {passed} \n"
                test_output_str = ""
                for result in result['result_list']:
                    test_output_str += result['status'] + '\n'
                output += test_output_str
            elif platform == 'leetcode':
                output = f"Evaluation Results: \n"
                output += result['status']
                passed = True if result['result_type'] == ResultType.ACCEPTED else False
    except Exception as e:
        output = 'Probably Parsing Error: Try Submitting Solution Again. \n' + str(e)
        passed = False

    return jsonify({'output': output, 'passed': passed})

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'message': 'Server is running!'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)