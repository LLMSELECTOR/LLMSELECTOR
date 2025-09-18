import random 
import pandas as pd
from datasets import load_dataset
import hashlib

w_max = 3000  # your desired max width
h_max = 5000  # your desired max height

def is_image_small_enough(row):
    
    #w, h = row['query']['query_images'][0]['raw'].size
    try:
        w, h = row['query']['query_images'][0]['raw'].size
        #print(f"w and h {w}, {h}")
        return w < w_max and h < h_max
    except Exception:
        return False  # skip rows that don't have the expected structure


class DataLoader_Ocrbenchv2(object):
    def __init__(self,random_state=2024,
                 num_query = 100000, # number of questions
                category = []):
        self.random_state = random_state
        self.local_random = random.Random()  # Create a new random generator instance
        self.local_random.seed(self.random_state)  # Seed the local generator
        if(type(category)==str):
            category=[category]
        self.category = category
        self.num_query = num_query

    def get_query_list(self,category = 'execution'):
        dataset = load_dataset("lmms-lab/OCRBench-v2")
        types = dataset['test']['type']  # This is a list
        if(len(self.category)>0):
            matching_indices = [i for i, t in enumerate(types) if t in self.category]
            # Select only those rows
            subset = dataset['test'].select(matching_indices)
        else:
            print('use full benchmark')
            subset = dataset['test']
        problems = subset
        self.problems = problems
        #print(type(problems))
        #problems = problems.filter(is_image_small_enough)
        k = min(self.num_query, len(subset))
        #problems = problems.shuffle(seed=self.random_state).select(range(k))
        problems = problems.select(range(k))
        queryset = [self._convert(a,idx) for idx,a in enumerate(problems)]
        return queryset
        
    def _convert(self,a, index):
#        instruction = 'Answer the question concisely.\n'
        instruction = 'Answer the following question. Give your final answer x at the end as "final answer: x". \n'
        query = f"{a['question']}"    
        answer = a['answers']
        img = a['image']
        img_hash = self.get_image_hash(a['image'])
        return {"text":instruction+query,"query_images":[{"raw":img,"hash":img_hash}]}, answer, "NA", str(a['id'])

    def get_query_df(self,):
        q_list = self.get_query_list()
        df = pd.DataFrame(q_list, columns=['query', 'true_answer', 'image', 'ID'])
        self.df = df
        df = df[df.apply(is_image_small_enough,axis=1)]
        #df = df[~df["ID"].between(5800, 5999)] # remove these that cause filter errors
        df = df[~df["ID"].astype(int).between(5800, 5999)]
        df = df[~df["ID"].astype(int).between(4100, 4499)]
        df = df[~df["ID"].astype(int).between(7000, 7199)]

        return df
    
    def get_image_hash(self, img):
        #print(f"img mode is {img.mode}")
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_bytes = img.tobytes()
        img_hash = hashlib.sha256(img_bytes).hexdigest()
        return img_hash  # Returns a 32-bit int