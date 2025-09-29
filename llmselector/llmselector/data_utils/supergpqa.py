import random 
import pandas as pd
import requests
from datasets import load_dataset

class DataLoader_SuperGPQA(object):
    def __init__(self,random_state=2024,
                 num_query = 1000, # number of questions
                 cate='physics',
                ):
        self.random_state = random_state
        self.local_random = random.Random()  # Create a new random generator instance
        self.local_random.seed(self.random_state)  # Seed the local generator
        dataset = load_dataset("m-a-p/SuperGPQA")['train']
        # Filter the dataset for the specified category
        self.problems = dataset.filter(lambda x: x['discipline'] == cate)
        self.num_query = min(num_query,len(self.problems))
    
    def get_query_list(self,category = 'dev'):
        queryset = [self._convert(idx) for idx in range(self.num_query)]
        return queryset

    def _convert(self,index):
        a = self.problems[index]

        # MMLU-Pro has 10 choices (A-J) and correct answer index
        question = a['question']
        choices = a['options']
        correct_answer_idx = a['answer_letter']
        #print(f"idx is::: {correct_answer_idx}")
        # Construct question string with 10 options
        max_i = len(choices)
        question_str = f"{question}\nHere are the options:" \
                       + "\n".join([f"({chr(65+i)}) {choices[i]}" for i in range(max_i)])
        correct_answer = correct_answer_idx  # Convert index to letter
        return question_str, correct_answer, 'NA', a['uuid'] 
        
    def get_query_df(self,):
        q_list = self.get_query_list()
        df = pd.DataFrame(q_list, columns=['query', 'true_answer', 'image', 'ID'])
        return df