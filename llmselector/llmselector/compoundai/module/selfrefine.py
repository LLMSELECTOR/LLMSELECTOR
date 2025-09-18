from .module import Module, DEFAULT_MODEL
from ..llm import Get_Generate
from ..compoundai import CompoundAI
import re
from collections import Counter

DESCRIPTION_SELFREFINE_OLD = '''This compound AI system uses 3 modules to solve a question. The module 0 generates an initial answer, module 1 gives some feedback to this initial answer. Finally, module 2 uses the initial answer and feedback to generate an updated answer.
'''
DESCRIPTION_SELFREFINE = '''This compound AI system uses 3 modules to solve a question. The module 0 generates an initial answer, module 1 gives some feedback to this initial answer. Finally, module 2 uses the initial answer and feedback to generate an updated answer.

Diagnosis Hint: In a self-refine system, errors often propogate from the beginning to the end.
Thus, if multiple modules may be responsible for the errors, then the early modules must also be responsible.
For example, if the final answer is wrong, and the module 0 makes a mistake, then even if module 1 and module 2 may be wrong as well, you should think that module 0 is wrong, since it is the root.

Also, note that the desired answer only shows the concept list. Therefore, do not worry about the format.
'''


PROMPT_TEMPLATE_FEEDBACK = "Below is a question and an initial answer. Is the predicted code output correct? If not, explain which reasoning steps in the initial answer leads to the mistakes. \nOriginal question:{task}\nModel answer:{answer}\nFeedback:"
PROMPT_TEMPLATE_REFINE =  "The below is a question, an initial answer, and some feedback. Generate a new step-by-step answer based on the feedback. Make sure that you fix all mistakes identified by the feedback.\nOriginal question:{task}\nInitial answer:{answer}\nFeedback:{feedback}\nNew answer:"

class Refiner(Module):
    def __init__(self,
                prompt_template_refine= PROMPT_TEMPLATE_REFINE,
                ):
        self.prompt_template_refine = prompt_template_refine
        self.model = DEFAULT_MODEL
        return
        
    def get_response(self,task, answer, feedback):
        refine_prompt = self.prompt_template_refine.format(
            task=task,feedback=feedback,answer=answer)
        return Get_Generate(refine_prompt,self.model) 
    
class Critic(Module):
    def __init__(self,
                prompt_template_feedback= PROMPT_TEMPLATE_FEEDBACK,
                ):
        self.prompt_template_feedback = prompt_template_feedback
        self.model = DEFAULT_MODEL
        return
    def get_response(self, task, answer):
        feedback_prompt = self.prompt_template_feedback.format(
            task=task,answer=answer)
        return Get_Generate(feedback_prompt,self.model) 

class Generator(Module):
    def get_response(self,query):
        return Get_Generate(query,self.model,
                            max_tokens=self.max_tokens,
                           )

class SelfRefine(CompoundAI):
    def __init__(self,
                 description=DESCRIPTION_SELFREFINE,
                 prompt_template_feedback=PROMPT_TEMPLATE_FEEDBACK,
                 prompt_template_refine=PROMPT_TEMPLATE_REFINE,
                ):
        super().__init__(description=description)
        self.prompt_template_feedback = prompt_template_feedback
        self.prompt_template_refine = prompt_template_refine
        self.create_pipeline(pipeline= self._get_pipeline())
        pass
        
    def _get_pipeline(self):
        pipeline = [["query",0],
                           [Generator(),[0]],
                           [Critic(prompt_template_feedback=self.prompt_template_feedback),[0,1]],
                           [Refiner(prompt_template_refine=self.prompt_template_refine), [0,1,2]],
                           ]
        return pipeline
    
class SelfRefineMultiRound(CompoundAI):
    def __init__(self,
                 description=DESCRIPTION_SELFREFINE,
                 prompt_template_feedback=PROMPT_TEMPLATE_FEEDBACK,
                 prompt_template_refine=PROMPT_TEMPLATE_REFINE,
                 round=1,
                ):
        super().__init__(description=description)
        self.prompt_template_feedback = prompt_template_feedback
        self.prompt_template_refine = prompt_template_refine
        self.round = round
        self.create_pipeline(pipeline= self._get_pipeline())
        pass
        
    def _get_pipeline(self):
        pipeline = [["query",0],
                    [Generator(),[0]],
                    ]
        for i in range(self.round):
            c1 = [Critic(prompt_template_feedback=self.prompt_template_feedback),[0,2*i+1]]           
            r1 = [Refiner(prompt_template_refine=self.prompt_template_refine), [0,2*i+1,2*i+2]]
            pipeline.append(c1)
            pipeline.append(r1)
        return pipeline
