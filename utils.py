from typing import List, Dict, Any, Tuple, Union, Callable
import pickle
import json
from datetime import datetime
from typing import Dict, List
from USACOBench.evaluation.metrics import pass_at_k
from USACOBench.prompts import solve_prompt_fn, retrieval_prompt_fn, reflexion_prompt_fn, RetrievalType
from rank_bm25 import BM25Okapi
from evaluate import evaluate_model
from functools import partial
import random
from collections import defaultdict
import tiktoken
import os
from USACOBench.retrievers.retrievers import Retriever

os.environ['TIKTOKEN_CACHE_DIR'] = '/home/qbshi/cache'

Problem = Dict[Any, Any]
Solution = Dict[str, Union[str, None]]
SolutionSet = List[Solution]
SolutionDict = Dict[str, SolutionSet]
Result = Dict[str, str]
ResultSet = List[Result]
ResultDict = Dict[str, ResultSet]
Query = Dict[str, str]

# INCLUSIVE
CP_ID_CUTOFFS = {'2012': 139, '2013': 287, '2014': 438, '2015': 554, '2016': 650, '2017': 746, '2018': 842, '2019': 950, '2020': 1046, '2021': 1142, '2022': 1238, '2023': 1334, '2024': 1430}
MIN_YEAR = 2012
MAX_YEAR = 2024

# NOTE: these are technically not thread-safe, although they are up to
# microsecond approximation due to timestamps

def combine_list_dicts(solution_dict1, solution_dict2):
    res = dict()
    for problem_id in solution_dict1:
        res[problem_id] = solution_dict1[problem_id] + solution_dict2[problem_id]
    return res

def get_difficulty_dist(rs: ResultSet, problem_dict):
    '''
    Takes lists of result sets, returns the difficulty distribution as a tuple.
    '''
    bronze, silver, gold, platinum = 0, 0, 0, 0
    for result_set in rs:
        if is_correct_set(result_set):
            result = result_set[0]
            difficulty = problem_dict[result['problem_id']]['problem_level'].lower()
            if difficulty == 'bronze':
                bronze += 1
            elif difficulty == 'silver':
                silver += 1
            elif difficulty == 'gold':
                gold += 1
            elif difficulty == 'platinum':
                platinum += 1
            else:
                print(f"DIFFICULTY NOT FOUND ON PROBLEM {result['problem_id']}")
    
    return (bronze, silver, gold, platinum)

def get_date_dist(rs: ResultSet, problem_dict, difficulties: set = {'bronze', 'silver', 'gold', 'platinum'}):
    '''
    Takes lists of result sets, returns distribution of dates as tuples of (season, number correct).
    Buckets by season/school year of USACO competitions. 
    Optional difficulty argument to only display problem of a specific set of difficulties.
    '''
    distribution = defaultdict(int)
    for result_set in rs:
        curr_problem = problem_dict[result_set[0]['problem_id']]
        if curr_problem['problem_level'] not in difficulties:
            continue
        if is_correct_set(result_set):
            release_year = get_release_year(curr_problem)
            distribution[release_year] += 1
    return distribution

def get_overlap(rs1: ResultSet, rs2: ResultSet): 
    '''
    Takes two lists of result sets, return a list of problems that are solved within both result sets
    '''
    solved1 = set(get_solved(rs1))
    solved2 = set(get_solved(rs2))
    intersection = solved1.intersection(solved2)
    return list(intersection)

def get_solved(rs):
    solved = []
    for result_set in rs:
        if is_correct_set(result_set):
            solved.append(result_set[0]['problem_id'])
    return solved

def is_correct_set(r: ResultSet):
    for result in r:
        if is_correct(result):
            return True
    return False

def is_correct(r: Result):
    if 'num_passed' in r and r['num_passed'] == r['num_tests']:
        return True
    return False

def get_release_year(problem: Problem):
    cp_id = problem['cp_id']
    for year in range(MIN_YEAR, MAX_YEAR+1): 
        if int(cp_id) <= CP_ID_CUTOFFS[str(year)]:
            return year
    return MAX_YEAR

def count_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def save_json(obj, path, timestamp=True, verbose=True):
    if timestamp:
        timestamp_str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S_%f")
        fname = '{}_{}.json'.format(path, timestamp_str)
    else:
        fname = '{}.json'.format(path)
    print('Saved json at {}'.format(fname))
    with open(fname, 'w') as f:
        json.dump(obj, f)
    return fname

def load_json(fname, verbose=True):
    with open(fname + '.json', 'r') as f:
        return json.load(f)

def save_pickle(obj, path, timestamp=True, verbose=True):
    if timestamp:
        timestamp_str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S_%f")
        fname = '{}_{}.pickle'.format(path, timestamp_str)
    else:
        fname = '{}.pickle'.format(path)
    print('Saved pickle at {}'.format(fname))
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(fname, verbose=True):
    with open(fname + '.pickle', 'rb') as f:
        return pickle.load(f)

def get_rdict(rs, ss):
    '''
    Takes (index-matched) lists of result sets and solution sets and returns
        the relevant result dictionary, i.e. dictionary from problem_id to result set
        Assumes at least one attempt for each problem.
    '''
    rdict = {}
    for r, s in zip(rs, ss):
        rdict[s[0]['problem_id']] = r
    return rdict

def filter(result_sets, difficulties=['bronze'], problem_dict=None):
    output = []
    for result_set in result_sets:
        trues = False
        for difficulty in difficulties:
            trues = trues or problem_dict[result_set[0]['problem_id']]['problem_level'] == difficulty
        if trues:
            output.append(result_set)
    return output

def search(text, problem_descriptions, problem_ids):
    for i, problem in enumerate(problem_descriptions):
        if text == problem:
            return problem_ids[i]
    return None

def combine(result_sets1, result_sets2):
    combined = []
    for result_set in result_sets1:
        temp = result_set + find(result_set[0]['problem_id'], result_sets2)
        combined.append(temp)
    return combined

def find(problem_id, result_sets): 
    for result_set in result_sets:
        if result_set[0]['problem_id'] == problem_id:
            return result_set
    return None

def calculate_percentage_identical(queries, target_queries):
    target_queries_dict = {}
    score = 0
    for query in target_queries:
        target_queries_dict[query['problem_id']] = query
    for query in queries:
        if query['similar_problem_id'] == target_queries_dict[query['problem_id']]['similar_problem_id']:
            score += 1
    return score / len(queries) * 100

def filter_queries(in_queries, queries):
    in_set = set([query['problem_id'] for query in in_queries])
    result = []
    for query in queries:
        if query['problem_id'] in in_set:
            result.append(query)
    return result

def parse_execution_output(execution_output):
    result_list = execution_output['result_list']
    if not result_list:
        result_list_str = execution_output['status']
    else:
        result_list_str = '\n'.join([result['status'] for result in result_list])
    num_tests = execution_output['num_tests']
    num_passed = execution_output['num_passed']
    return f'Testing Output: \n {result_list_str} \n Number of tests: {num_tests} \n Number passed: {num_passed}'

def get_difficulty_performances(full_results, problem_dict, k=1):
    bronze = filter(full_results, ['bronze'], problem_dict=problem_dict)
    silver = filter(full_results, ['silver'], problem_dict=problem_dict)
    gold = filter(full_results, ['gold'], problem_dict=problem_dict)
    platinum = filter(full_results, ['platinum'], problem_dict=problem_dict)
    return pass_at_k(bronze, k=k)[0], pass_at_k(silver, k=k)[0], pass_at_k(gold, k=k)[0], pass_at_k(platinum, k=k)[0]


####################################################################################################
# Query Generation Functions
####################################################################################################

# General Retrieval Function
def generate_retrieval_queries(p: int, problem_dict: Dict[str, Problem], query_texts: List[tuple[str, str]], retriever: Retriever, loocv=True):
    # query_texts = (problem_id, text)
    fin_queries = []
    for query_text in query_texts:
        df_result = retriever.retrieve(query_text, num_docs=p)
        similar_problem_texts = list(df_result['docs'])
        similar_problem_ids = list(df_result['problem_id'])
        similar_problem_text = ""
        words = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh"]
        for i, text in enumerate(similar_problem_texts):
            similar_problem_text += f"\n\n {words[i]} problem and solution \n\n" + text 
        
        fin_queries.append({'problem_id': query_text[0], 'retrieval_text': '[BEGIN SIMILAR PROBLEMS]\n' + similar_problem_text + '\n[END SIMILAR PROBLEMS]\n', 'retrieval_problem_ids': similar_problem_ids, 'problem_description': problem_dict[query_text[0]]['description']})
    return fin_queries

# p is the number of problems retrieved
def generate_episodic_retrieval_queries(p, problem_dict, solution_sets, use_text=True):
    corpus = [problem_dict[problem_id]['description'] + '\nSolution: \n' + problem_dict[problem_id]['solution_english'] + '\nSolution Code: \n' + problem_dict[problem_id]['solution_python3'] for problem_id in problem_dict.keys()]
    solutions = solution_sets
    query_texts = []
    problem_ids = []
    for solution in solutions:
        solution_text = solution[0]['solution']
        problem_id = solution[0]['problem_id']
        if problem_id in problem_dict.keys():
            if use_text:
                query_texts.append(problem_dict[problem_id]['description'] + '\n' + solution_text)
            else:
                query_texts.append(problem_dict[problem_id]['description'] + '\n' + solution_text)
            problem_ids.append(problem_id)

    sim_prob_queries_new_code = []
    for i, problem_id in enumerate(problem_ids):
        curr_description = problem_dict[problem_id]['description'] + '\nSolution: \n' + problem_dict[problem_id]['solution_english'] + '\nSolution Code: \n' + problem_dict[problem_id]['solution_python3']
        duplicated = corpus[:]
        duplicated.remove(curr_description)
        tokenized_corpus = [doc.split(' ') for doc in duplicated]
        bm25 = BM25Okapi(tokenized_corpus)

        curr_query = query_texts[i]
        tokenized_query = curr_query.split(" ")
        similar_problem_texts = bm25.get_top_n(tokenized_query, duplicated, n=p)
        similar_problem_text = ""
        words = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh"]
        similar_problem_ids = []
        for i, text in enumerate(similar_problem_texts):
            similar_problem_text += f"\n\n {words[i]} problem and solution \n\n" + text 
            similar_problem_ids.append(search(text, corpus, list(problem_dict.keys())))

        for val in similar_problem_ids:
            if val == None:
                print(similar_problem_texts)
                assert 1 == 2
        sim_prob_queries_new_code.append({'problem_id': problem_id, 'retrieval_text': '[BEGIN SIMILAR PROBLEMS]\n' + similar_problem_text + '\n[END SIMILAR PROBLEMS]\n', 'retrieval_problem_ids': similar_problem_ids, 'problem_description': problem_dict[problem_id]['description']})
    save_json(sim_prob_queries_new_code, f'queries_firstsolve_{p}problem_episodic')
    return sim_prob_queries_new_code

def generate_random_retrieval_queries(p, problem_dict):
    problem_ids = list(problem_dict.keys())
    corpus = [problem_dict[problem_id]['description'] + '\nSolution: \n' + problem_dict[problem_id]['solution_english'] + '\nSolution Code: \n' + problem_dict[problem_id]['solution_python3'] for problem_id in problem_ids]
    sim_prob_queries_new_code = []
    for i, problem_id in enumerate(list(problem_dict.keys())):
        curr_description = problem_dict[problem_id]['description'] + '\nSolution: \n' + problem_dict[problem_id]['solution_english'] + '\nSolution Code: \n' + problem_dict[problem_id]['solution_python3']
        duplicated = corpus[:]
        duplicated.remove(curr_description)

        r_indices = random.sample(range(len(duplicated)), p)
        similar_problem_texts = [duplicated[indx] for indx in r_indices]
        similar_problem_text = ""
        words = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh"]
        similar_problem_ids = []
        for i, text in enumerate(similar_problem_texts):
            similar_problem_text += f"\n\n {words[i]} problem and solution \n\n" + text 
            similar_problem_ids.append(search(text, corpus, problem_ids))

        for val in similar_problem_ids:
            if val == None:
                print(similar_problem_texts)
                assert 1 == 2
        sim_prob_queries_new_code.append({'problem_id': problem_id, 'retrieval_text': '[BEGIN SIMILAR PROBLEMS]\n' + similar_problem_text + '\n[END SIMILAR PROBLEMS]\n', 'retrieval_problem_ids': similar_problem_ids, 'problem_description': problem_dict[problem_id]['description']})
    save_json(sim_prob_queries_new_code, f'queries_random_{p}problem_episodic')
    return sim_prob_queries_new_code

def generate_constant_retrieval_queries(p, problem_dict):
    problem_ids = list(problem_dict.keys())
    corpus = [problem_dict[problem_id]['description'] + '\nSolution: \n' + problem_dict[problem_id]['solution_english'] + '\nSolution Code: \n' + problem_dict[problem_id]['solution_python3'] for problem_id in problem_ids]
    sim_prob_queries_new_code = []
    r_indices = random.sample(range(len(corpus)), p)
    similar_problem_texts = [corpus[indx] for indx in r_indices]

    for i, problem_id in enumerate(list(problem_dict.keys())):       
        similar_problem_text = ""
        words = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh"]
        similar_problem_ids = []
        for i, text in enumerate(similar_problem_texts):
            similar_problem_text += f"\n\n {words[i]} problem and solution \n\n" + text 
            similar_problem_ids.append(search(text, corpus, problem_ids))

        for val in similar_problem_ids:
            if val == None:
                print(similar_problem_texts)
                assert 1 == 2
        sim_prob_queries_new_code.append({'problem_id': problem_id, 'retrieval_text': '[BEGIN SIMILAR PROBLEMS]\n' + similar_problem_text + '\n[END SIMILAR PROBLEMS]\n', 'retrieval_problem_ids': similar_problem_ids, 'problem_description': problem_dict[problem_id]['description']})
    save_json(sim_prob_queries_new_code, f'queries_constant_{p}problem_episodic')
    return sim_prob_queries_new_code

def generate_semantic_retrieval_queries(problem_dict, solution_sets, model_name):
    # Fetching the corpus to pass in: The corpus being textbook chapters
    textbook = load_json('data/corpuses/cpbook_v2')
    textbook_corpus = [article['full_article'] for article in textbook]
    tokenized_corpus = [chapter.split(' ') for chapter in textbook_corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    query_texts = dict()

    solutions = solution_sets
    query_texts = []
    problem_ids = []
    for solution in solutions:
        solution_text = solution[0]['solution']
        problem_id = solution[0]['problem_id']
        if problem_id in problem_dict.keys():
            query_texts.append(problem_dict[problem_id]['description'] + '\n' + solution_text)
            problem_ids.append(problem_id)

    textbook_queries_new_code = []
    for i, problem_id in enumerate(problem_ids):
        curr_query = query_texts[i]
        tokenized_query = curr_query.split(" ")
        if '3.5' in model_name:
            textbook_text = bm25.get_top_n(tokenized_query, textbook_corpus, n=1)[0][:7000]
        else:
            textbook_text = bm25.get_top_n(tokenized_query, textbook_corpus, n=1)[0]
        textbook_queries_new_code.append({'problem_id': problem_id, 'retrieval_text': '[BEGIN TEXTBOOK EXCERPT]\n' + textbook_text  + '\n[END TEXTBOOK EXCERPT]\n', 'problem_description': problem_dict[problem_id]['description']})
    save_json(textbook_queries_new_code, 'queries_firstsolve_semantic')
    return textbook_queries_new_code

def generate_episodic_semantic_retrieval_queries(num_problems_fetched, problem_dict, solution_sets, model_name):
    textbook_queries = generate_semantic_retrieval_queries(problem_dict, solution_sets, model_name)
    episodic_queries = generate_episodic_retrieval_queries(num_problems_fetched, problem_dict, solution_sets)
    textbook_queries_dict = dict()
    episodic_queries_dict = dict()

    for query in textbook_queries:
        textbook_queries_dict[query['problem_id']] = query
    for query in episodic_queries:
        episodic_queries_dict[query['problem_id']] = query
    
    final_queries = []
    for query in textbook_queries:
        retrieval_text_textbook = query['retrieval_text']
        episodic_query = episodic_queries_dict[query['problem_id']]
        retrieval_text_episodic = episodic_query['retrieval_text'] 
        retrieval_text = retrieval_text_episodic + '\n\n' + retrieval_text_textbook 
        final_queries.append({'retrieval_problem_ids': episodic_query['retrieval_problem_ids'], 'retrieval_text': retrieval_text, 'problem_id': query['problem_id'], 'problem_description': query['problem_description']})

    save_json(final_queries, 'queries_firstsolve_episodic_semantic')
    return final_queries

def generate_reflexion_queries(rdict, sdict, problem_dict, model_name, iteration, prev_queries_dict=None, retrieval=False):
    reflection_queries_dict = dict()

    for problem_id in sdict.keys():
        if problem_id in problem_dict.keys():
            for solution in sdict[problem_id][:1]:
                prev_buffer = ''
                if prev_queries_dict:
                    prev_buffer = prev_queries_dict[problem_id]['reflection_buffer']
                current_response = solution['solution']
                current_execution_output = rdict[solution['problem_id']][0]['result_list']
                num_samples = problem_dict[problem_id]['description'].count("SAMPLE INPUT")
                execution_output = ""
                if current_execution_output:
                    current_execution_output = current_execution_output[:num_samples]
                    for i, result in enumerate(current_execution_output):
                        execution_output += f"Test Case {i}\n" + result['status'] + "\n"
                else:
                    execution_output = "No submission, formatting error during judging."

                if retrieval:
                    retrieval_text = prev_queries_dict[problem_id]['retrieval_text']
                    retrieval_problem_ids = prev_queries_dict[problem_id]['retrieval_problem_ids']
                    reflection_queries_dict[problem_id] = {'problem_id': problem_id, 'reflection_buffer': prev_buffer + f'\n Reflection Response Number {iteration+1}: \n' + current_response + f'\n Reflection Response Execution Output Number {iteration+1}:\n' + execution_output, 'retrieval_text': retrieval_text, 'retrieval_problem_ids': retrieval_problem_ids, 'problem_description': problem_dict[problem_id]['description']}
                else:
                    reflection_queries_dict[problem_id] = {'problem_id': problem_id, 'reflection_buffer': prev_buffer + f'\n Reflection Response Number {iteration+1} \n' + current_response + f'\n Reflection Response Execution Output Number {iteration+1}:\n' + execution_output, 'problem_description': problem_dict[problem_id]['description']}
    if retrieval:
        name = f'queries_dict_{model_name}_retrieval_reflexion'
    else:
        name = f'queries_dict_{model_name}_reflexion'
    save_json(reflection_queries_dict, name, timestamp=False)
    return reflection_queries_dict

def calculate_final_rs(reflexions, problem_dict):
    rs = []
    for problem_id in reflexions[0].keys():
        num_samples = problem_dict[problem_id]['description'].count('SAMPLE INPUT')
        for i, reflexion_result in enumerate(reflexions):
            if reflexion_result[problem_id][0]['result_list']:
                sample_testing_results = reflexion_result[problem_id][0]['result_list'][:num_samples]
            elif i == len(reflexions)-1:
                rs.append(reflexion_result[problem_id])
                break
            else:
                continue
            passed = True
            for result in sample_testing_results:
                if result['result_type'] != 1:
                    passed = False
                    break
            if passed or i == len(reflexions)-1:
                rs.append(reflexion_result[problem_id])
                break
    return rs


####################################################################################################
# Run Functions
####################################################################################################


def run_solve(model_fn, model_name, problem_dict, attempts, return_queries=False):
    queries = []
    for problem_id in problem_dict.keys():
        queries.append({'problem_id': problem_id, 'problem_description': problem_dict[problem_id]['description']})

    rdict, sdict, rs, ss = evaluate_model(model_fn, solve_prompt_fn, queries=queries, verbose=True, attempts=attempts, problem_ids=list(problem_dict.keys()))
    model_name_save = model_name.replace('/', '-')
    save_json([rdict, sdict, rs, ss], f'results/results_{model_name_save}_solve_{attempts}attempts')
    return (rdict, sdict, rs, ss) if not return_queries else (rdict, sdict, rs, ss, queries)

def run_retrieval(model_fn, model_name, problem_dict, attempts, solution_sets, num_retrieved, retrieval_type, return_queries=False, use_text=True):
    if retrieval_type == RetrievalType.EPISODIC:
        queries = generate_episodic_retrieval_queries(num_retrieved, problem_dict, solution_sets, use_text=use_text)
    elif retrieval_type == RetrievalType.SEMANTIC:
        queries = generate_semantic_retrieval_queries(problem_dict, solution_sets, model_name)
    else:
        queries = generate_episodic_semantic_retrieval_queries(num_retrieved, problem_dict, solution_sets, model_name)

    r_prompt_fn = partial(retrieval_prompt_fn, retrieval_type=retrieval_type)
    rdict, sdict, rs, ss = evaluate_model(model_fn, r_prompt_fn, queries=queries, verbose=True, attempts=attempts, problem_ids=list(problem_dict.keys()))
    model_name_save = model_name.replace('/', '-')
    save_json([rdict, sdict, rs, ss], f'results/results_{model_name_save}_episodic_retrieval_{attempts}attempts')

    return (rdict, sdict, rs, ss) if not return_queries else (rdict, sdict, rs, ss, queries)

def run_reflexion(model_fn, model_name, problem_dict, attempts, prev_result_dict, prev_solution_dict, prev_queries_dict, iteration, return_queries=True, retrieval=False):
    new_reflexion_queries_dict = generate_reflexion_queries(prev_result_dict, prev_solution_dict, problem_dict, model_name, iteration, prev_queries_dict=prev_queries_dict, retrieval=retrieval)
    rdict, sdict, rs, ss = evaluate_model(model_fn, reflexion_prompt_fn, queries=list(new_reflexion_queries_dict.values()), verbose=True, attempts=attempts)
    model_name_save = model_name.replace('/', '-')
    save_json([rdict, sdict, rs, ss], f'results_{model_name_save}_reflexion_{str(iteration)}iteration')

    return (rdict, sdict, rs, ss) if not return_queries else (rdict, sdict, rs, ss, new_reflexion_queries_dict)