import pandas as pd
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from .llm import Get_Generate
import numpy as np

def compute_score(compoundaisystems, data_df, metric,true_answer_name='true_answer',save_answer_prefix="answer",answer_convert=None):
    # Define the function for one row
    def process_row(row):
        return ai_system.generate(row['query'], metric, row['true_answer'])


    # Apply with parallel processing and progress bar
    def apply_parallel_with_progress(df, func, n_jobs=-1):
        # Initialize tqdm progress bar
        results = []
        with tqdm(total=len(df)) as pbar:
            # Using Parallel to process and update the progress bar in each iteration
            for result in Parallel(n_jobs=n_jobs)(
                    delayed(func)(row) for _, row in df.iterrows()):
                results.append(result)
                pbar.update(1)  # Update progress bar for each result
        return results


    # Apply the function with multi-threading and a progress bar
    def apply_multithreaded_with_progress(df, func, num_threads=8):
        results = []
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit all tasks to the executor
            futures = [executor.submit(func, row) for _, row in df.iterrows()]
            
            # Use tqdm to show progress as futures complete
            for future in tqdm(as_completed(futures), total=len(futures)):
                results.append(future.result())
        
        return results


    # Apply the function with multi-threading, a progress bar, and order preservation
    def apply_multithreaded_with_progress_ordered(df, func, num_threads=20):
        results = [None] * len(df)  # Initialize a list to hold results in the correct order
        # Reset the index of the DataFrame to ensure sequential indexing
        df_reset = df.reset_index(drop=True)
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit all tasks to the executor with indices to track order
            futures = {executor.submit(func, row): i for i, row in df_reset.iterrows()}
            
            # Use tqdm to show progress as futures complete
            for future in tqdm(as_completed(futures), total=len(futures)):
                idx = futures[future]  # Retrieve the original index
                results[idx] = future.result()  # Place result in the correct position
    
        return results

    # Initialize a list to store the results
    results = []

    # Iterate over each AI system in the dictionary
    for name, ai_system in compoundaisystems.items():
        # Generate answers using the AI system
#        data_df['answer'] = data_df['query'].apply(ai_system.generate)
        tqdm.pandas()
#        data_df['answer'] = data_df.progress_apply(
#            lambda row: ai_system.generate(row['query'], metric, row['true_answer']), axis=1)

        # Apply the function in parallel with a progress bar
        data_df[f'answer_{name}'] = apply_multithreaded_with_progress_ordered(data_df, process_row, num_threads=40)
        data_df[f'{save_answer_prefix}_{name}'] = apply_multithreaded_with_progress_ordered(data_df, process_row, num_threads=40)
        if(answer_convert):
            data_df[f'{save_answer_prefix}_{name}'] = data_df[f'{save_answer_prefix}_{name}'].apply(answer_convert)

        # Use Parallel and delayed to apply the function to each row
        # Apply the function in parallel with a progress bar
        #data_df['answer'] = apply_parallel_with_progress(data_df, process_row, n_jobs=8)

        #data_df['answer'] = Parallel(n_jobs=10)(delayed(process_row)(row) for _, row in tqdm(data_df.iterrows()))

        '''
        data_df['answer'] = data_df.swifter.apply(
    lambda row: ai_system.generate(row['query'], metric, row['true_answer']), axis=1)
        '''

        # Calculate scores using the specified metric
        #print(data_df)
        if(true_answer_name not in data_df): # It is not there, try prefix
            answer_name = f'{true_answer_name}_{name}'
        else:
            answer_name = true_answer_name
        data_df[f'score_{name}'] = data_df.progress_apply(lambda row: metric.get_score(row[f'answer_{name}'], row[answer_name]), axis=1)
        
        # Compute the mean score for the current AI system
        mean_score = data_df[f'score_{name}'].mean()
        
        # Append the result as a tuple (name, mean_score)
        results.append((name, mean_score))
    
    # Create a DataFrame from the results
    score_df = pd.DataFrame(results, columns=['Name', 'Mean_Score'])
    
    return score_df


def compute_tag(compoundaisystems, data_df, metric, tag='iter'):
    # Define the function for one row
    def process_row(row):
        result = ai_system.generate(row['query'], metric, row['true_answer'])
        #print(f"result is {result[tag]}")
        return result[tag]
    

    # Apply with parallel processing and progress bar
    def apply_parallel_with_progress(df, func, n_jobs=-1):
        # Initialize tqdm progress bar
        results = []
        with tqdm(total=len(df)) as pbar:
            # Using Parallel to process and update the progress bar in each iteration
            for result in Parallel(n_jobs=n_jobs)(
                    delayed(func)(row) for _, row in df.iterrows()):
                results.append(result)
                pbar.update(1)  # Update progress bar for each result
        return results


    # Apply the function with multi-threading and a progress bar
    def apply_multithreaded_with_progress(df, func, num_threads=8):
        results = []
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit all tasks to the executor
            futures = [executor.submit(func, row) for _, row in df.iterrows()]
            
            # Use tqdm to show progress as futures complete
            for future in tqdm(as_completed(futures), total=len(futures)):
                results.append(future.result())
        
        return results


    # Apply the function with multi-threading, a progress bar, and order preservation
    def apply_multithreaded_with_progress_ordered(df, func, num_threads=8):
        results = [None] * len(df)  # Initialize a list to hold results in the correct order
        # Reset the index of the DataFrame to ensure sequential indexing
        df_reset = df.reset_index(drop=True)
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit all tasks to the executor with indices to track order
            futures = {executor.submit(func, row): i for i, row in df_reset.iterrows()}
            
            # Use tqdm to show progress as futures complete
            for future in tqdm(as_completed(futures), total=len(futures)):
                idx = futures[future]  # Retrieve the original index
                results[idx] = future.result()  # Place result in the correct position
    
        return results

    # Initialize a list to store the results
    results = []

    # Iterate over each AI system in the dictionary
    for name, ai_system in compoundaisystems.items():
        # Generate answers using the AI system
#        data_df['answer'] = data_df['query'].apply(ai_system.generate)
        tqdm.pandas()
#        data_df['answer'] = data_df.progress_apply(
#            lambda row: ai_system.generate(row['query'], metric, row['true_answer']), axis=1)

        # Apply the function in parallel with a progress bar
        data_df[f'{tag}_{name}'] = apply_multithreaded_with_progress_ordered(data_df, process_row, num_threads=40)

        # Use Parallel and delayed to apply the function to each row
        # Apply the function in parallel with a progress bar
        #data_df['answer'] = apply_parallel_with_progress(data_df, process_row, n_jobs=8)

        #data_df['answer'] = Parallel(n_jobs=10)(delayed(process_row)(row) for _, row in tqdm(data_df.iterrows()))

        '''
        data_df['answer'] = data_df.swifter.apply(
    lambda row: ai_system.generate(row['query'], metric, row['true_answer']), axis=1)
        '''

        # Calculate scores using the specified metric
        #print(data_df)
        #data_df[f'{tag}_{name}'] = data_df.progress_apply(lambda row: metric.get_score(row['answer'], row['true_answer']), axis=1)
        
    return data_df
    


def strip_latex(response: str) -> str:
  if response.startswith("$") and response.endswith("$"):
    response = response[1:-1]
  if "boxed{" in response and response.endswith("}"):
    response = response[0:-1].split("boxed{")[1]
  if "text{" in response and response.endswith("}"):
    response = response[0:-1].split("text{")[1]
  if "texttt{" in response and response.endswith("}"):
    response = response[0:-1].split("texttt{")[1]
  return response


def extract_answer(sample: str) -> str:
  """Extracts the final answer from the sample."""
  answer_prefixes = [
      "The answer is:",
      "The final answer is ",
      "The final answer is: ",
      "The answer is "
  ]
  answer = sample.lower()
  for answer_prefix in answer_prefixes:
    answer_prefix = answer_prefix.lower()
    if answer_prefix in answer:
      answer = answer.split(answer_prefix)[-1].strip()
  if answer.endswith("."):
    answer = answer[:-1]
  return strip_latex(answer)

def fuzzy_match(prediction: str, reference: str) -> bool:
  """Fuzzy match function for BigBench Extra Hard."""
  if prediction == reference:
    return True

  # (a) vs a
  if len(prediction) == 3 and prediction[0] == "(" and prediction[-1] == ")":
    return prediction[1] == reference
  if len(reference) == 3 and reference[0] == "(" and reference[-1] == ")":
    return reference[1] == prediction

  # Numbers
  try:
    if float(prediction) == float(reference):
      return True
  except ValueError:
    pass

  # quote issues
  if prediction.replace("'", "") == reference.replace("'", ""):
    return True

  # Bracket issues
  if f"[{reference}]" == prediction or f"[{prediction}]" == reference:
    return True

  # Question mark issues
  if prediction.endswith("?") and prediction[:-1] == reference:
    return True

  return False


def preprocess_sample(sample: str) -> str:
  prediction = extract_answer(sample.strip()).lower()
  prediction = prediction.replace(", ", ",").replace("**", "")
  prediction = prediction.split("\n")[0]
  prediction = prediction[0:-1] if prediction.endswith(".") else prediction
  return prediction


def preprocess_reference(reference: str) -> str:
  reference = reference.strip().lower()
  reference = reference.replace(", ", ",")
  return reference


def evaluate_correctness(sample: str, reference: str) -> bool:
  prediction = preprocess_sample(sample)
  reference = preprocess_reference(reference)
  return fuzzy_match(prediction, reference)


class Metric(object):
    def __init__(self,name='mc'):
        self.name = name
    
    def get_score(self,answer, true_answer):
        scorer = metric_mapper[self.name]
        return scorer(answer, true_answer)

    def get_name(self,):
        return self.name
        
    

def get_score_concept_binary(answer,concepts):
    contain = [concept in answer for concept in concepts]
    return int(sum(contain)/len(contain))

def get_score_MC(answer,true_answer):
    return f"the answer is ({true_answer.lower()})" in answer.lower()

def get_score_em(answer,true_answer):
    return f"theansweris{remove_special_characters(true_answer.lower())}" in remove_special_characters(answer.lower())

def get_score_numeric(answer,true_answer):
    eps = 1e-5
    ans = extract_final_numeric_answer(answer)
    #print(type(ans),ans)
    #print(type(true_answer),true_answer)
    return (np.abs(ans-true_answer)<eps)

def get_score_matchone(answer,true_answer):
    return answer in true_answer

def get_score_extract_matchone(answer,true_answer):
    match = re.search(r"final answer:\s*(.+)", answer,re.IGNORECASE)
    if match:
        x = match.group(1)
        return x in true_answer
    else:
        #print(f"cannot find final answer in {answer}")
        return False
#    return answer in true_answer


import re

def get_score_em_direct(answer, true_answer):        
    return 1 if remove_special_characters(
        answer.lower()
                                         ) == remove_special_characters(true_answer.lower()) else 0

def get_score_answer_contain(answer, true_answer):        
    for a in true_answer:
         if (str(a).lower() in answer):
           return 1
    return 0 

def get_score_em_llm(answer, true_answer, llm_model='gpt-4o-2024-05-13'):
    prompt = f'[answer]: [{answer}]\n[True answer]:[{true_answer}]. Generate "correct" if they are semantically equivalent, and "wrong" otherwise.'
    score = Get_Generate(prompt, model_gen=llm_model)
    print(answer, true_answer, score)
    if("correct" in score):
        return 1
    elif("wrong" in score):
        return 0
    else:
        return 0
    
def remove_special_characters(s):
    # Keep only letters, digits, and the specified special characters
    return re.sub(r'[^A-Za-z0-9+\-*/]', '', s)

def get_score_concept(answer,concepts):
    #contain = [concept in answer for concept in concepts]
    contain = []
    for concept in concepts:
        contain.append(concept in answer)
        if(concept not in answer):
            print(concept)
    return sum(contain)/len(contain)

def get_score_count(answer, true_count):
    return count_words(answer)==true_count

def count_words(paragraph):
    # Split the paragraph into words
    words = paragraph.split()
    # Return the number of words
    return len(words)

def extract_final_numeric_answer(text):
    """
    Extracts a numerical value from the text if it contains the phrase 'final answer: x'.
    
    Args:
        text (str): The input text to search for the numerical value.
    
    Returns:
        float or None: The numerical value if found, otherwise None.
    """
    match = re.search(r'final answer:\s*([-+]?\d*\.?\d+)', text.lower(), re.IGNORECASE)
    if match:
        numeric_value = match.group(1)
        return float(numeric_value)  # Ensure it's converted to float
    return 0


def get_score_fuzzymatch(answer,true_answer):
    return evaluate_correctness(answer,true_answer)


metric_mapper = {
    "mc":get_score_MC,
    "concept":get_score_concept_binary,
    "em":get_score_em,
    "count":get_score_count,
    "em_direct":get_score_em_direct,
    'numeric_match':get_score_numeric,
    'match_one':get_score_matchone,
    'extract_match_one':get_score_extract_matchone,
    "em_LLM":get_score_em_llm,
    "em_direct_contain":get_score_answer_contain,
    "fuzzy_match":get_score_fuzzymatch,

}
