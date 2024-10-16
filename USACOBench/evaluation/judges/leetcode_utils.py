from USACOBench.evaluation.result_type import ResultType
from enum import Enum
from pydantic import BaseModel
from typing import Optional
import ast
import dotenv
import gym
import leetcode
import leetcode.auth
import os
import time
from datetime import datetime

dotenv.load_dotenv()

class ProgrammingLanguage(Enum):
    """
    Enum for valid LeetCodeSubmission programming languages
    """
    PYTHON = "python"
    PYTHON3 = "python3"

class LeetCodeSubmission(BaseModel):
    """
    Model for a Leetcode Code Submission
    """
    code: str
    lang: ProgrammingLanguage
    question_id: str
    question_slug: str
    question_id: Optional[str] = None
    timeout: int = 5

class Stats:
    def __init__(self):
        self.correct = 0
        self.timeout = 0
        self.syntax_error = 0
        self.runtime_error = 0
        self.test_case_fail = 0
        self.unknown = 0
        self.contextFound = 0
        self.similarQuestionFound = 0
        self.topicQuestionFound = 0

class LeetCodeEnv(gym.Env):
    """
    Gym environment for LeetCode submissions
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, cooldown=0):
        super(LeetCodeEnv, self).__init__()
        self.__configure_leetcode()
        self.reward = False
        self.last_run = None
        self.cooldown = cooldown  # To avoid rate limit

    def __configure_leetcode(self):
        configuration = leetcode.Configuration()
        # From Dev Tools/Application/Cookies/LEETCODE_SESSION
        leetcode_session = os.environ["LEETCODE_SESSION"]
        csrf_token = os.environ['CSRF_TOKEN']
        configuration.api_key["x-csrftoken"] = csrf_token
        configuration.api_key["csrftoken"] = csrf_token
        configuration.api_key["LEETCODE_SESSION"] = leetcode_session
        configuration.api_key["Referer"] = "https://leetcode.com"
        configuration.debug = False
        self.api_instance = leetcode.DefaultApi(leetcode.ApiClient(configuration))

    def step(self, action: LeetCodeSubmission):
        """
        Sends a submission to LeetCode and returns the result
        Args:
            action (LeetCodeSubmission): LeetCodeSubmission object
        Returns:
            status (str): 'Accepted' | 'Runtime Error'| 'Wrong Answer' | 'Submission Timed-Out' | 'Unknown'
            reward (bool): True if status is 'Accepted', False otherwise
            done (bool): True if status is 'Accepted', False otherwise
            submission_result (dict): LeetCode API response
        """
        submission_result = self.__send_submission(action)
        reward, status = self.__calculate_reward(submission_result)
        self.reward = reward
        done = self.is_done()
        return status, reward, done, submission_result
    
    def reset(self):
        self.reward = False

    def __send_submission(self, sub: LeetCodeSubmission):
        self.__wait_for_cooldown()
        if sub.question_id is None:
            sub.question_id = id_from_slug(sub.question_slug, self.api_instance)
        submission = leetcode.Submission(
            judge_type="large",
            typed_code=sub.code,
            question_id=sub.question_id,
            test_mode=False,
            lang=sub.lang.value,
        )
        submission_id = self.api_instance.problems_problem_submit_post(
            problem=sub.question_slug, body=submission
        )
        time.sleep(sub.timeout)
        submission_result = self.api_instance.submissions_detail_id_check_get(
            id=submission_id.submission_id
        )
        return submission_result
    
    def __calculate_reward(self, submission_result):
        if submission_result == {"state": "STARTED"}:
            status_msg = "Submission Timed-Out"
        elif (
            "status" in submission_result.keys()
            and submission_result["status"] == "PENDING"
        ):
            status_msg = "Submission Timed-Out"
        elif "status_msg" in submission_result.keys():
            status_msg = submission_result[
                "status_msg"
            ]  # 'Accepted' | 'Runtime Error'| 'Wrong Answer'
        else:
            status_msg = "Unknown"
        return status_msg == "Accepted", status_msg
    def __wait_for_cooldown(self):
        if self.last_run == None:
            self.last_run = datetime.now()
        else:
            while (datetime.now() - self.last_run).total_seconds() < self.cooldown:
                time.sleep(0.1)
            self.last_run = datetime.now()
    def is_done(self):
        return self.reward

def id_from_slug(slug: str, api_instance) -> str:
    """
    Retrieves the id of the question with the given slug
    """
    graphql_request = leetcode.GraphqlQuery(
      query="""
                  query getQuestionDetail($titleSlug: String!) {
                    question(titleSlug: $titleSlug) {
                      questionId
                    }
                  }
              """,
              variables={"titleSlug": slug},
              operation_name="getQuestionDetail",
      )
    response = ast.literal_eval(str(api_instance.graphql_post(body=graphql_request)))
    frontend_id = response['data']['question']['question_id']
    return frontend_id

# Need to fix to include memory error
def parse_leetcode_result_type(result) -> ResultType:
    if result:
        status, reward, _, submission_result = result
        print(status)
        print(reward)
        print(submission_result)
        if reward: 
            return ResultType.ACCEPTED, "Your code passed all test cases."
        elif 'timed' in status.lower():
            return ResultType.TIME_LIMIT_EXCEEDED, "Your code timed out. This means that either it was not efficient enough to pass the stress tests, or there is some coding error that caused the test cases to not run within the time limit."
        elif 'runtime' in submission_result['status_msg'].lower():
            return ResultType.RUNTIME_ERROR, submission_result['runtime_error']
        elif 'syntax' in submission_result['status_msg'].lower():
            return ResultType.COMPILATION_ERROR, ""
        elif 'wrong' in submission_result['status_msg'].lower():
            if submission_result['last_testcase'] and submission_result['code_output'] and submission_result['expected_output']:
                status_msg = f"""Your code did not pass all the test cases. Here was the test case it failed on: 
                {submission_result['last_testcase']} 
                The expected output was {submission_result['expected_output']}, but your code's output was {submission_result['code_output']}."""
            else:
                status_msg = """No further information available, but the code did not pass all the test cases."""
            return ResultType.WRONG_ANSWER, status_msg
        else:
            return ResultType.UNKNOWN, "The cause of error is unknown: a likely cause lies within parsing errors."
    else: 
        return ResultType.UNKNOWN, "The cause of error is unknown: a likely cause lies within parsing errors."

def extract_code(text_response):
    try:
        temp = text_response
        text_response = text_response.split('```python3')
        if len(text_response) == 1:
            text_response = text_response[0].split('```python')
        if len(text_response) == 1:
            text_response = text_response[0].split('```Python3')
        if len(text_response) == 1:
            text_response = text_response[0].split('```Python')
        if len(text_response) == 1:
            text_response = text_response[0].split('```')
            text_response = text_response[1:]
        else:
            text_response = text_response[1].split('```')
        return text_response[0]
    except Exception as error:
        print("FAILED EXTRACTING CODE")
        print(error)
        print("RESPONSE:")
        print(temp)
        return None