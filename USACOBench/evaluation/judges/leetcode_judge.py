from pathlib import Path
from typing import List, Dict, Any, Union
from .leetcode_utils import parse_leetcode_result_type
from USACOBench.evaluation.result_type import ResultType
from .judge import Judge
from .leetcode_utils import ProgrammingLanguage, LeetCodeEnv, LeetCodeSubmission

Problem = Dict[Any, Any]
Solution = Dict[str, Union[str, None]]
SolutionSet = List[Solution]
Result = Dict[str, str]
ResultSet = List[Result]
# construct judge sandbox directories if they don't exist
Path('judge_sandbox/solutions/leetcode').mkdir(parents=True, exist_ok=True)
class LeetCodeJudge(Judge):
    def __init__(self,
                 solutions_path: str = None,
                 sleep_length: float = 6):
        '''
        solutions_path: filepath for auxiliary solution files to be created
            and stored (cf-tool reads from file)
        sleep_length: amount to sleep after each submission to avoid
            rate limit (if None, random amount between 0-1 seconds)
        '''
        super().__init__(solutions_path, sleep_length)
    
    # TODO support non-py solutions (need to update submission language)
    def _judge(self,
               problem: Problem,
               problem_id: str,
               solution_code: str,
               language: str = 'Python3') -> Result:
        assert language == 'Python3' or language == "Python"
        if solution_code is None:
            return {
                'result_type': ResultType.UNKNOWN,
                'status': 'No submission',
                'judge_output': 'No submission'
            }
        return self._leetcode_judge(solution_code, problem_id)
    
    def _leetcode_judge(self, solution_code: str, question_slug: str) -> Result:
        '''
        Judges a Python3 Leetcode solution; returns result as a
            dictionary with status and judge_output strings
        solution_code: code to submit as a string
        question_slug: name of the question to judge
        '''
        lang = ProgrammingLanguage.PYTHON3 if '->' in solution_code else ProgrammingLanguage.PYTHON
        sub = LeetCodeSubmission(code=solution_code, lang=lang, question_slug=question_slug)
        env = LeetCodeEnv()
        status, reward, done, submission_result = env.step(sub)
        result_type, status = parse_leetcode_result_type((status, reward, done, submission_result))
        return {
            'result_type': result_type,
            'status': status,
            'judge_output': submission_result
        }