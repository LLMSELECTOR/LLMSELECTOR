import random, hashlib, re
import pandas as pd
from datasets import load_dataset, concatenate_datasets


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


class DataLoader_MathVista(object):
    def __init__(self,random_state=2024,
                 num_query = 100000, # number of questions
                category = ['testmini'],
                ):
        self.random_state = random_state
        self.local_random = random.Random()  # Create a new random generator instance
        self.local_random.seed(self.random_state)  # Seed the local generator
        if(type(category)==str):
            category=[category]
        self.category = category
        self.num_query = num_query

    def get_query_list(self,category = 'execution'):
        problems = load_dataset("AI4Math/MathVista")[self.category[0]]
        #print(f"the number of queries {len(problems)}")
        self.problems = problems
        k = min(self.num_query, len(problems))
        #problems = problems.shuffle(seed=self.random_state).select(range(k))
        problems = problems.select(range(k))
        queryset = [self._convert(a,idx) for idx,a in enumerate(problems)]
        return queryset
        
    def _convert(self,a, index):
        answer_mapper = ['A','B','C','D','E','F','G','H','I','J','K']
#        instruction = 'Answer the question concisely.\n'
        instruction = 'Answer the following question. Give your final answer x at the end as "final answer: x". \n'
        query = f"{a['query']}"   
        if(a['choices']==None):
            answer = [a['answer']]
        else:
            #print(f"[choice is]:: {a['choices']}")
            #print(f"answer is {a['answer']}")
            for idx, item in enumerate(a['choices']):
                if a['answer'] == item:
                    answer = answer_mapper[idx]
        img_list = self.get_all_images(a)
        img_hash_list = [ self.get_image_hash(img) for img in img_list]
        q_images = [{"raw":self.get_image_RGB(img_list[i]),"hash":img_hash_list[i]} for i in range(len(img_list))]
        return {"text":instruction+query,"query_images":q_images}, answer, "NA", str(a['pid'])

    def get_all_images(self,item):
        return [item['decoded_image']]
    
    def get_query_df(self,):
        q_list = self.get_query_list()
        df = pd.DataFrame(q_list, columns=['query', 'true_answer', 'image', 'ID'])
        self.df = df
        df = df[df.apply(is_image_small_enough,axis=1)]
        return df
    
    def get_image_hash(self, img):
        #print(f"img mode is {img.mode}")
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_bytes = img.tobytes()
        img_hash = hashlib.sha256(img_bytes).hexdigest()
        return img_hash  # Returns a 32-bit int

    def get_image_RGB(self,img):
        if img.mode != "RGB":
            img = img.convert("RGB")        
        return img
    
def extract_image_tags(text):
    """
    Extracts 'imagei' from all occurrences of '<image i>' in the given text.

    Args:
        text (str): The input string.

    Returns:
        List[str]: A list of 'imagei' strings (without angle brackets or space).
    """
    matches = re.findall(r'<image (\d+)>', text)
    return [f'image_{i}' for i in matches]
