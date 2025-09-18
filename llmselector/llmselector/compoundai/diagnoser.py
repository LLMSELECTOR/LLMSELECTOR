from .llm import Get_Generate
import re


DIAGNOSE_TEMPLATE = '''You are an error diagnosis expert for compound AI systems. 
Below is the description of a compound AI system, a query, the generations from each module of the compound AI system, the final output, and the desired answer. 

{data}

Does module {i} cause the mismatch between the desired answer and the true answer? Remember that the desired answer is absolutely correct. You should first analyze carefully why the final answer could be wrong, just by looking at the question and the true answer. Next, based on your analysis, figure out which module might be wrong. Do not attempt to directly solve any subtasks, in order to avoid your own biases.

You must analyze the system step by step first and then give your final answer.

If module {i} is indeed the issue, format your final decision as ‘[final answer: Yes]’. Otherwise, format it as ’[final answer: No]’. If the final output matches the desired answer, generate '[correct answer]'.
'''

DIAGNOSE_TEMPLATE = '''
You are an expert in troubleshooting multi-module AI systems.

Below you will find:
{data}

Your task
Determine whether MODULE {i} is responsible for the discrepancy between the system’s final output and the desired (ground-truth) answer.

Guidelines

Take the desired answer as absolutely correct.
Follow the judgment approach provided in the diagnosis examples.
Do not attempt to solve any subtasks yourself.
Reason step by step first, and then generate the final answer by the following format.

Decision format
• If MODULE {i} is the likely source of the error, output exactly:
[final answer: Yes]

• If the error is more plausibly due to another module, output exactly:
[final answer: No]

• If the final output already matches the desired answer, output exactly:
[correct answer]

'''


def extract_final_answer(text):
    # Remove special characters and spaces
    cleaned_text = re.sub(r'[^\w]', '', text)  # Retain only alphanumeric characters
    # Convert to lowercase for case-insensitive matching
    cleaned_text = cleaned_text.lower()
    
    # Match the required phrases
    if "finalansweryes" in cleaned_text:
        return 1
    elif "finalanswerno" in cleaned_text:
        return 0
    elif "correctanswer" in cleaned_text:
        return 0
    else:
        return 0.5
        
class Diagnoser(object):
    def __init__(self,
#                 diagnose_model = 'claude-3-5-sonnet-20240620',
                 diagnose_model = 'gpt-4o-2024-05-13',

                ):
        self.diagnoise_model = diagnose_model
        self.prompt = "None"
        return
    def get_prompt(self):
        return self.prompt
        
    def diagnose(self,
                 compoundaisystem:dict,
                 query:str,
                 answer:str,
                 true_answer:str,
                 module_id:int = 0,
                 diagnose_template:str = DIAGNOSE_TEMPLATE,
                 temperature=0,
                 show_prompt=False,
                ):
        '''
        determine if the mistake is due to the module with module_id or not.
        return 1 if yes, and 0 otherwise.
        compoundaisystem must contain two keys, description and module
        description is a text describing the system
        module contains the output from each module. 
        '''
        #print(f"** generate data")
        data = self._generate_data(
                       compoundaisystem=compoundaisystem,
                       query=query,
                       answer=answer,
                       true_answer=true_answer,
        )
        #print(f"** generate prompt")
        prompt = diagnose_template.format(data=data,i=module_id)
        self.prompt=prompt
        #prompt += f'[Your analysis]:'
        #print(f"********* show_prompt is {show_prompt}")
        if(show_prompt):
            #print(f"query is {query}")
            #print(f"answer is {answer}")
            print(f'***prompt*** {prompt}')
        #return 0
        analysis = Get_Generate(prompt,self.diagnoise_model,temperature=temperature)
        #print(f"analysis is {analysis}")
#        error = re.search(r'error:\s*(0|1)', analysis, re.IGNORECASE)
        '''
        error = re.search(r'error:\s*(0|1)', re.sub(r'[\*\[\]]', '', analysis), re.IGNORECASE)

        if error:
            error = int(error.group(1))
        else:
            error = 0.5
        '''
        error = extract_final_answer(analysis)
        return error, analysis

    def _generate_data(self,
                       compoundaisystem:dict,
                       query:str,
                       answer:str,
                       true_answer:str,
                      ):
        des = compoundaisystem['description']
        prompt = f'[Compound AI system]: "{des}"\n[question]: {query}\n'
        for i in range(len(compoundaisystem['module'])):
            output = compoundaisystem['module'][i]
            prompt += f'[Module {i}\'s output]: "{output}"\n'
        prompt += f'[The compound system"s output]: {answer} \n'    
        prompt += f'[The desired correct answer]: {true_answer} \n'
        prompt += f'[Your judgment]:'
        return prompt