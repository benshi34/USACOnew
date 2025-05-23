from enum import Enum 

class RetrievalType(int, Enum):
    EPISODIC = 1
    SEMANTIC = 2
    EPISODIC_SEMANTIC = 3
    OPTIMAL = 4

def retrieval_prompt_fn(query, retrieval_type=RetrievalType.EPISODIC):
    if retrieval_type == RetrievalType.EPISODIC:
        descriptor = "multiple somewhat similar problems, as well as the solution to those similar problems"
    elif retrieval_type == RetrievalType.SEMANTIC:
        descriptor = "a textbook chapter relevant to the given problem"
    elif retrieval_type == RetrievalType.EPISODIC_SEMANTIC:
        descriptor = "multiple somewhat similar problems and solutions, as well as a textbook chapter relevant to the given problem"
    elif retrieval_type == RetrievalType.OPTIMAL:
        descriptor = "hints from a guide on how to solve the problem"
    else:
        raise Exception("Retrieval type not supported.")
    return f"""Please reply with a Python 3 solution to the below problem. Make sure to wrap your code in '```python' and '```' Markdown
delimiters, and include exactly one block of code with the entire solution
(in the final code step). You will also be given retrieved text, which represents {descriptor}. Feel free to use the given information to aid your problem solving process.
Reason through the problem and:
1. Restate the problem in plain English
2. Conceptualize a solution first in plain English
3. Output the final Python solution with your solution steps in comments.
No outside libraries are allowed.

[BEGIN RETRIEVED TEXT]
{query['retrieval_text']}
[END RETRIEVED TEXT]

Now it's your turn. Here is the problem you are to solve:
[BEGIN PROBLEM]
{query['problem_description']}
[END PROBLEM]"""

def reflexion_prompt_fn(query, retrieval=False):
    retrieval_text = ""
    if retrieval:
        retrieval_text = "You were also given a couple of similar problems to the problem above along with their solutions to aid you in solving the problem at hand. Here are the similar problems you were given:\n" + query['retrieval_text'] 
    
    return f"""You were previously solving a coding problems. Here is the problem that you were solving:
    [BEGIN PROBLEM]
    {query['problem_description']}
    [END PROBLEM]
    {retrieval_text}
    And here are all your past attempts, as well how your code fared on the unit tests for the problem:
    {query['reflection_buffer']}
    If the previous attempt is already correct and passed all the tests, then just return the previously solution. But if it doesn't pass all the tests, think carefully about where you went wrong in your latest solution, first outputting why you think you went wrong. Then, given your insights, try to fix the solution, outputting a block of correct python3 code to be executed and evaluated again. Make sure to wrap your code in '```python' and '```' Markdown delimiters.
    """

def solve_prompt_fn(query):
    return f"""Please reply with a Python 3 solution to the below problem. Make sure to wrap your code in '```python' and '```' Markdown
delimiters, and include exactly one block of code with the entire solution
(in the final code step). Do not use any outside libraries, as we will not be able to run your solution. Also: do not wrap your code in a main function as that causes issues.
Reason through the problem and:
1. Restate the problem in plain English
2. Conceptualize a solution first in plain English
3. Output the final Python solution with your solution steps in comments.

[BEGIN PROBLEM]
{query['problem_description'] }
[END PROBLEM]
"""

def solve_detailed_prompt_fn(query):
    return f"""Please reply with a Python 3 solution to the below problem. First, conceptualize a solution, detailing the steps and your thought process to approaching the problem. Structure your output like a textbook solution to a problem. Your solution will be used to educate others, so make sure to be sufficiently detailed. 
    Afterwards, generate code that will solve the problem. Do not generate pseudocode. Make sure to wrap your code in '```python' and '```' Markdown delimiters.
No outside libraries are allowed.

[BEGIN PROBLEM]
{query['problem_description'] }
[END PROBLEM]
"""