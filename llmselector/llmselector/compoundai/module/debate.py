from .module import Module, DEFAULT_MODEL
from ..llm import Get_Generate
from ..compoundai import CompoundAI
import re
from collections import Counter
DESCRIPTION_DEBATE = '''This compound AI system uses 6 modules to solve a question. The module 0, module 1, and module 2 generate initial answers respectively. Next, module 3, module 4, and module 5 debates with each other to update their answer based on the answer from the first three modules. Finally, a majority vote is taken over the updated answers to generate the ultimate answer to the original question.
'''

PROMPT_TEMPLATE_DEBATE='''[User Question]:{query}
[Your Response]: {response}
[Instruction]: These are the solutions to the question from other agents: {other_responses}

Using the reasoning from other agents as additional advice, can you give an updated answer? Examine your solution and that of the other agents. Then state your final answer concisely at the end in the form (X), for example (hello world).
'''
PROMPT_TEMPLATE_INITDEBATE='''Please answer the following question.
[Question]: {query} 
[Instruction]: Format your final answer concisely as (X), e.g., (hello world).
'''

# Classes for debate
class Debator(Module):
    def __init__(self,
                 prompt_template_debate=PROMPT_TEMPLATE_DEBATE,
                 model = DEFAULT_MODEL,
                 max_tokens = 1000,
                 add_space=0,
                ):
        super().__init__(max_tokens=max_tokens,model=model)
        self.prompt_template_debate = prompt_template_debate
        self.add_space=add_space
        
    def get_response(self,query, response, *args):
        other_responses = ''
        for i in range(len(args)):
            arg = args[i]
            other_responses += f'[agent {i} response]: {arg} \n'
            
        return Get_Generate(self.prompt_template_debate.format(
            query=query,
            response=response,
            other_responses=other_responses),
                            self.model,
                           max_tokens=self.max_tokens,
                           ) 

class InitDebator(Module):
    def __init__(self,
                 prompt_template_initdebate = PROMPT_TEMPLATE_INITDEBATE,
                 model = DEFAULT_MODEL,
                 max_tokens = 1000,
                 add_space=0,

                ):
        super().__init__(max_tokens=max_tokens,model=model)
        self.prompt_template_initdebate = prompt_template_initdebate
        self.add_space=add_space

    def get_response(self,query,):
        return Get_Generate(self.prompt_template_initdebate.format(query=query)+' '*self.add_space,
                            self.model,
                           max_tokens=self.max_tokens,
                           )


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
        return majority

def extract_ans(text,regex = r'\([a-h][1-8]\)',remove_left=1,remove_right=1):
    # Use regex to find all matches of the pattern
    matches = re.findall(regex, text)
    # Return the last match if available, otherwise an empty tuple
    if not matches:
        return '()'
    len1 = len(matches[-1])
    return f'{matches[-1][remove_left:len1-remove_right]}'
    
    return f'{matches[-1][remove_left:remove_right]}' if matches else '()'

class MultiAgentDebate(CompoundAI):
    def __init__(self,
                 description=DESCRIPTION_DEBATE,
                ):
        super().__init__(description=description)
        self.create_pipeline(pipeline= self._get_pipeline())
        pass
        
    def _get_pipeline(self):
        pipeline = [["query",0],
                           [InitDebator(add_space=0,prompt_template_initdebate=PROMPT_TEMPLATE_INITDEBATE),[0]],
                           [InitDebator(add_space=1,prompt_template_initdebate=PROMPT_TEMPLATE_INITDEBATE),[0]],
                           [InitDebator(add_space=2,prompt_template_initdebate=PROMPT_TEMPLATE_INITDEBATE),[0]],
                           [Debator(prompt_template_debate=PROMPT_TEMPLATE_DEBATE),[0,1,2,3]],
                           [Debator(prompt_template_debate=PROMPT_TEMPLATE_DEBATE),[0,2,1,3]],
                           [Debator(prompt_template_debate=PROMPT_TEMPLATE_DEBATE),[0,3,1,2]],
                           [Merge(regex=r'\(.*?\)'),[4,5,6]],
                           ]
        return pipeline