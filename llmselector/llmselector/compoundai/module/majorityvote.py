from .module import Module, DEFAULT_MODEL
from ..llm import Get_Generate
from ..compoundai import CompoundAI
import re
from collections import Counter
from .directgen import Generator, resize_to_max_pixels
import copy
DESCRIPTION_DEBATE = '''This compound AI system uses 6 modules to solve a question. The module 0, module 1, and module 2 generate initial answers respectively. Next, module 3, module 4, and module 5 debates with each other to update their answer based on the answer from the first three modules. Finally, a majority vote is taken over the updated answers to generate the ultimate answer to the original question.
'''

GEN_PROMPT = '''{query}'''

JUDGER_PROMPT=''' Please act as an impartial judge and evaluate the quality of the responses provided by AI assistants to the user question displayed below. You should choose the assistant that
follows the user’s instructions and answers the user’s question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity,
and level of detail of their responses. Begin your evaluation by comparing the provided responses and provide a short explanation. Avoid any position biases and ensure that the
order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of
the assistants. Be as objective as possible. 

Format: After providing your explanation, output your final verdict by strictly following this format. "[final selection]: i" if candidate answer i is the best. For example, if candidate answer 2 is the best, output "[final selection]: 2".

[User query]: {query}'''
class Merge(Module):
    def __init__(self,
                 regex = r'\([a-h][1-8]\)',
                 remove_left=1,
                 remove_right=1,
                 ):
        self.regex = regex
        self.remove_left= remove_left
        self.remove_right = remove_right
        pass
        
    def get_response(self,*args):
        ans_list = [extract_ans(arg,self.regex,remove_left=self.remove_left,remove_right=self.remove_right) for arg in args]
        counter = Counter(ans_list)
        # Find the string with the highest count
        majority, _ = counter.most_common(1)[0]
        #print(f"merge inputs are {args}")
        #print(f"extracted ans list is {ans_list}")
        for i in range(len(ans_list)):
            if(majority==ans_list[i]):
                return args[i]
        print(f"did not find the original one, return majority {majority}")
        return majority

def extract_ans(text,regex = r'\([a-h][1-8]\)',remove_left=1,remove_right=1):
    # Use regex to find all matches of the pattern
    matches = re.findall(regex, text,re.IGNORECASE)
    # Return the last match if available, otherwise an empty tuple
    if not matches:
        return '()'
    len1 = len(matches[-1])
    #print(f"match is {matches[-1]}")
    return f'{matches[-1][remove_left:len1-remove_right]}'

class MajorityVoteof5(CompoundAI):
    def __init__(self,
                 description = DESCRIPTION_DEBATE,
                 gen_prompt = GEN_PROMPT,
                 merge_regex= r'\A(.*)\Z',
                ):
        super().__init__(description=description)
        self.gen_prompt = gen_prompt
        self.merge_regex = merge_regex
        self.create_pipeline(pipeline= self._get_pipeline())
        pass
        
    def _get_pipeline(self):
        pipeline = [["query",0],
                           [Generator(add_space=0,gen_prompt=self.gen_prompt),[0]],
                           [Generator(add_space=1,gen_prompt=self.gen_prompt),[0]],
                           [Generator(add_space=2,gen_prompt=self.gen_prompt),[0]],
                           [Generator(add_space=3,gen_prompt=self.gen_prompt),[0]],
                           [Generator(add_space=4,gen_prompt=self.gen_prompt),[0]],

                           [Merge(regex=self.merge_regex,remove_left=0,remove_right=0),[1,2,3,4,5]],
                           ]
        return pipeline

class MajorityVoteofN(CompoundAI):
    def __init__(self,
                 description = DESCRIPTION_DEBATE,
                 gen_prompt = GEN_PROMPT,
                 merge_regex= r'\A(.*)\Z',
                 N=5,
                ):
        super().__init__(description=description)
        self.gen_prompt = gen_prompt
        self.N=N
        self.merge_regex = merge_regex
        self.create_pipeline(pipeline= self._get_pipeline())        
        pass
        
    def _get_pipeline(self):
        pipeline = [["query",0]]
        for i in range(self.N):
            pipeline.append([Generator(add_space=i,gen_prompt=self.gen_prompt),[0]])               
        pipeline.append( [Merge(regex=self.merge_regex,remove_left=0,remove_right=0),list(range(1,self.N+1))] ) 
                           
        return pipeline

class BestofN(CompoundAI):
    def __init__(self,
                 description = DESCRIPTION_DEBATE,
                 gen_prompt = GEN_PROMPT,
                 judger_prompt  = JUDGER_PROMPT,
                 merge_regex= r'\A(.*)\Z',
                 N=5,
                 debug=False,
                ):
        super().__init__(description=description)
        self.gen_prompt = gen_prompt
        self.judger_prompt = judger_prompt
        self.N=N
        self.debug=debug
        self.merge_regex = merge_regex
        self.create_pipeline(pipeline= self._get_pipeline())        
        pass
        
    def _get_pipeline(self):
        pipeline = [["query",0]]
        for i in range(self.N):
            pipeline.append([Generator(add_space=i,gen_prompt=self.gen_prompt),[0]])               
        pipeline.append( [LLMJudger(add_space=0,gen_prompt=self.judger_prompt,debug=self.debug),list(range(0,self.N+1))] ) 
                           
        return pipeline
    

class LLMJudger(Module):
    def __init__(self,
                add_space=0,
                gen_prompt = '''{query}''',
                max_tokens = 1000,
                model = DEFAULT_MODEL,
                debug=False,
                ):
        self.add_space = add_space
        self.gen_prompt = gen_prompt
        self.debug = debug
        super().__init__(max_tokens=max_tokens,model=model)
        pass

    def format_text(self,text="",*args):
        newtext = self.gen_prompt.format(query=text)
        newtext = " "*self.add_space + newtext
        for idx in range(len(args)):
            a1 = args[idx]
            newtext += f"[candidate answer {idx}]: {a1}\n"
        if(self.debug):
            print(f"[the input args is]::: {args}\n\n [judgement query is]::: {newtext}\n\n")
        if(self.add_space>10):
            print(f"add_space is: {self.add_space}")
            print(f"query is: {text}")
            print(f"newtext is: {newtext}")
        return newtext
    
    def get_response(self,query, *args):
        #time1 = time.time()
        query = copy.copy(query)  # Make a deep copy to avoid mutating original
        #print(f'copy time is {time.time()-time1}')
        if(type(query)==str):
            query = self.format_text(query)
        else:
            query['text'] = self.format_text(query['text'],*args)
            query['query_images'] = [resize_to_max_pixels(img) for img in query['query_images']]
        #time1 = time.time()
        self.query = query['text']
        r1 = Get_Generate(query,self.model,
                            max_tokens=self.max_tokens,
                           )
        final_answer = self.judge2answer(r1,*args)
        #print(f'gen time is {time.time()-time1}')
        return final_answer
    
    def judge2answer(self,r1,*args):
        match = re.search(r"\[final selection\]:\s*(\d+)", r1, re.IGNORECASE)
        if match:
            result = int(match.group(1))
        else:
            #print(f"cannot find the selection in judgement: {r1}")
            result = 0
        if(self.debug==True):
            print(f"[judgement is]::: {r1}\n\n [the index is]: {result} with {type(result)}")
        try:
            return args[int(result)]
        except:
            print(f"arg extract fails. Use the first")
            return args[0]
            print(f"prompt is f{self.query}")
            print(f"r1 is :::{r1}")
            print(f"args is::: {args}")
            print(f"result is ::: {result}")
            return args[int(result)]

        #print(result)  # Output: i
        return
    