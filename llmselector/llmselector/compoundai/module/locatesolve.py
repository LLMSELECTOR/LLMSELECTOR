from .module import Module, DEFAULT_MODEL
from ..llm import Get_Generate
from ..compoundai import CompoundAI
import re
from collections import Counter


DESCRIPTION_LOCATESOLVE = '''This compound AI system uses 2 modules to solve a LookupQA question. A LookupQA question contains a table consisting of "ID" and "Task" columns. The goal is to solve the task with a particular "ID". Module 0 extracts the task with a particular ID, and module 1 solves the task and returns the final answer. Keep in mind that the tasks shown in the table is typically different from each other.
'''

DESCRIPTION_LOCATESOLVE_OLD = '''This compound AI system uses 2 modules to solve a LookupQA question. A LookupQA question contains a table consisting of "ID" and "Task" columns. The goal is to solve the task with a particular "ID". Module 0 extracts the task with a particular ID, and module 1 solves the task and returns the final answer.

Diagnosis Hint: Here are some examples.

[Example 1]
[Module 0's output]: "What is 11+(10.9>10.11)?"
[Module 1's output]: "final answer: 11"
[The compound system"s output]: 11
[The desired correct answer]: 12
[Your task]: Determine whether MODULE 0 is responsible for the discrepancy.
[Your judgment]: the system answer is close (just by 1) to the desired answer. Thus the extracted task is likely correct. Hence, module 0 is likely not responsible.
[final answer: No]

[Example 2]
[Module 0's output]: "What is 11+(10.9>10.11)?"
[Module 1's output]: "final answer: 11"
[The compound system"s output]: 11
[The desired correct answer]: 12
[Your task]: Determine whether MODULE 1 is responsible for the discrepancy.
[Your judgment]: the system answer is close (just by 1) to the desired answer. Thus the extracted task is likely correct. Hence, module 1 is likely responsible.
[final answer: Yes]

[Example 3]
[Module 0's output]: "What is 89+(10.9>10.11)?"
[Module 1's output]: "final answer: 90"
[The compound system"s output]: 90
[The desired correct answer]: 34
[Your task]: Determine whether MODULE 0 is responsible for the discrepancy.
[Your judgment]: the system answer is far away (by 56) from the desired answer. Thus the extracted task is likely incorrect. Hence, module 0 is likely responsible.
[final answer: Yes]

[Example 4]
[Module 0's output]: "What is 89+(10.9>10.11)?"
[Module 1's output]: "final answer: 90"
[The compound system"s output]: 90
[The desired correct answer]: 34
[Your task]: Determine whether MODULE 1 is responsible for the discrepancy.
[Your judgment]: the system answer is far away from (by 56) the desired answer. Thus the extracted task is likely incorrect. Hence, module 1 is likely not responsible.
[final answer: No]


[Example 5]
[Module 0's output]: "What is 34+(10.9>10.11)?"
[Module 1's output]: "final answer: 34"
[The compound system"s output]: 34
[The desired correct answer]: 35
[Your task]: Determine whether MODULE 0 is responsible for the discrepancy.
[Your judgment]: the system answer is close (just by 1) to the desired answer. Thus the extracted task is likely correct. Hence, module 0 is likely not responsible.
[final answer: No]

[Example 6]
[Module 0's output]: "What is 34+(10.9>10.11)?"
[Module 1's output]: "final answer: 34"
[The compound system"s output]: 34
[The desired correct answer]: 35
[Your task]: Determine whether MODULE 1 is responsible for the discrepancy.
[Your judgment]: the system answer is close (just by 1) to the desired answer. Thus the extracted task is likely correct. Hence, module 1 is likely responsible.
[final answer: Yes]

End of Hint.
'''

PROMPT_TEMPLATE_EXTRACTTASK= '''Extract the task corresponding to the ID asked in the following question. Do not answer the original question directly or generate any other texts.
----------
Original question: {query}
----------
Your response:
'''
PROMPT_TEMPLATE_CONVERTTABLE = '''Below is a question that involves a table where the content of each column is shown separately. Rewrite the question so that the table is shown in a row-wise format. Make sure to align contents carefully. Do not change any other things in the question. Do not generate any other texts.
----------
{query}
----------
Your response:
'''
PROMPT_TEMPLATE_SOLVESUBTASK='''
{query} Format your answer x as "final answer: x".
'''

# Classes for Table task
class ConvertTable(Module):
    def __init__(self,
                 prompt_template_convertable=PROMPT_TEMPLATE_CONVERTTABLE,
                 max_tokens = 1000,
                ):
        super().__init__(max_tokens=max_tokens)
        self.prompt_template_convertable = prompt_template_convertable
    def get_response(self,query):
        return Get_Generate(self.prompt_template_convertable.format(query=query),self.model,
                           max_tokens=self.max_tokens,
                           
                           )    


class ExtractTask(Module):
    def __init__(self,
                 prompt_template_extracttask=PROMPT_TEMPLATE_EXTRACTTASK,
                                 max_tokens = 1000,

                ):
        super().__init__(max_tokens=max_tokens)
        self.prompt_template_extracttask = prompt_template_extracttask
    def get_response(self,query):
        return Get_Generate(self.prompt_template_extracttask.format(query=query),self.model,
                                                      max_tokens=self.max_tokens,

                           )  
        
class CompoundExtractTask(Module):
    def __init__(self,
                 prompt_template_extracttask=PROMPT_TEMPLATE_EXTRACTTASK,
                 prompt_template_convertable=PROMPT_TEMPLATE_CONVERTTABLE,
                                 max_tokens_extracktask = 1000,
                                 max_tokens_convertable = 1000,


                ):
        super().__init__(max_tokens=max_tokens_convertable)
        self.MyConvertTable = ConvertTable(
            prompt_template_convertable=prompt_template_convertable,
            max_tokens=max_tokens_convertable,
                                        )
        self.MyExtractTask = ExtractTask(prompt_template_extracttask=prompt_template_extracttask,
                                        max_tokens=max_tokens_extracktask)
    def get_response(self,query):
        self.MyConvertTable.setmodel(self.model)
        self.MyExtractTask.setmodel(self.model)
        q1 = self.MyConvertTable.get_response(query)
        q2 = self.MyExtractTask.get_response(q1)
        return q2  

class SolveSubTask(Module):
    def __init__(self,
                 prompt_template_solvesubtask=PROMPT_TEMPLATE_SOLVESUBTASK,
                                 max_tokens = 1000,

                ):
        super().__init__(max_tokens=max_tokens)
        self.prompt_template_solvesubtask = prompt_template_solvesubtask
    def get_response(self,query):
        return Get_Generate(self.prompt_template_solvesubtask.format(query=query),self.model,
                                                      max_tokens=self.max_tokens,

                           )

class LocateSolve(CompoundAI):
    def __init__(self,
                 description=DESCRIPTION_LOCATESOLVE,
                 prompt_template_extracttask=PROMPT_TEMPLATE_EXTRACTTASK,
                 prompt_template_convertable=PROMPT_TEMPLATE_CONVERTTABLE,
                prompt_template_solvesubtask=PROMPT_TEMPLATE_SOLVESUBTASK,
                ):
        super().__init__(description=description)
        self.prompt_template_extracttask = prompt_template_extracttask
        self.prompt_template_convertable = prompt_template_convertable
        self.prompt_template_solvesubtask = prompt_template_solvesubtask 
        self.create_pipeline(pipeline= self._get_pipeline())
        pass
        
    def _get_pipeline(self):
        pipeline = [["query",0],
               [CompoundExtractTask(max_tokens_convertable=4000,prompt_template_convertable=self.prompt_template_convertable),[0]],
               [SolveSubTask(prompt_template_solvesubtask=self.prompt_template_solvesubtask), [1]],
               ]
        return pipeline