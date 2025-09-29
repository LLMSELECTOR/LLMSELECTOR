import random 
import pandas as pd
import requests
from datasets import load_dataset

class DataLoader_MedCal(object):
    def __init__(self,random_state=2024,
                 num_query = 100000, # number of questions
                 cate='test_rules',
                 version='v2.0',
                ):
        self.random_state = random_state
        self.local_random = random.Random()  # Create a new random generator instance
        self.local_random.seed(self.random_state)  # Seed the local generator
        self.dataset = load_dataset("PTPReasoning/MedCalc-Bench-v1.0") # Fails with NonMatchingChecksumError
        self.problems = [entry for entry in self.dataset[cate]]
        self.num_query = min(num_query,len(self.problems))
    
    def get_query_list(self,category = 'dev'):
        queryset = [self._convert(idx) for idx in range(self.num_query)]
        return queryset

    def _convert(self,index):
        a = self.problems[index]
        query = f"{a['input']}"
        answer = a['Ground Truth Answer']
        return query, answer, 'NA', a['Row Number']
        
    def get_query_df(self,):
        q_list = self.get_query_list()
        df = pd.DataFrame(q_list, columns=['query', 'true_answer', 'image', 'ID'])
        return df