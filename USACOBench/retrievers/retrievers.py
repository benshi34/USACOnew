from abc import ABC, abstractmethod
import time
import random
import numpy as np
from tqdm import tqdm
from typing import Dict, List
from scipy.spatial import distance
import pandas as pd

from rank_bm25 import BM25Okapi
from openai import OpenAI

client = OpenAI()

from .utils import AggregationType

class Retriever(ABC):
    def __init__(self, docs):
        self.docs = docs
        
    @abstractmethod
    def retrieve(self, query, num_docs=10):
        pass

class OpenAIRetriever(Retriever):
    def __init__(self, docs: List[str], problem_ids: List[str], model_name='text-embedding-3-small'):
        # Problem ids matches to docs in index order
        self.docs = docs
        self.model_name = model_name
        
        self.df = pd.DataFrame()
        self.df['problem_id'] = problem_ids
        self.df['docs'] = docs
        self.df['embedding'] = [self.get_embedding(text) for text in docs]

    def get_embedding(self, text):
        text = text.replace("\n", " ")
        return client.embeddings.create(input = [text], model=self.model_name).data[0].embedding

    # query = (problem_id, text)
    def retrieve(self, query: tuple[str, str], num_docs=1, loocv=True):
        problem_id, query_text = query
        query_embedding = self.get_embedding(query_text)
        curr_df = self.df.copy()
        if loocv:
            curr_df = curr_df[~curr_df['problem_id'].isin([problem_id])]
        curr_df['similarities'] = curr_df.embedding.apply(lambda x: distance.cosine(query_embedding, x))
        res = curr_df.sort_values('similarities', ascending=False).head(num_docs)
        return res

class BM25Retriever(Retriever):
    def __init__(self, docs: List[str], problem_ids: List[str]):
        self.docs = docs
        self.df = pd.DataFrame()
        self.df['problem_id'] = problem_ids
        self.df['docs'] = docs
    
    # Query = (problem_id, text)
    def retrieve(self, query: tuple[str, str], num_docs=1, loocv=True):
        problem_id, query_text = query
        curr_df = self.df.copy()
        if loocv:
            curr_df = curr_df[~curr_df['problem_id'].isin([problem_id])]
        
        tokenized_document = [doc.split(' ') for doc in curr_df['docs']]
        bm25 = BM25Okapi(tokenized_document)
        
        tokenized_query = query_text.split(' ')
        similar_problem_texts = bm25.get_top_n(tokenized_query, list(curr_df['docs']), n=num_docs)
        # only the documents that have the texts desired
        curr_df = curr_df[curr_df['docs'].isin(similar_problem_texts)]
        return curr_df