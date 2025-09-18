from itertools import product
from tqdm import tqdm
from .metric import get_score_count
import random, re, os, concurrent.futures
from sqlitedict import SqliteDict
# Make tqdm work with pandas apply
tqdm.pandas()
import pandas as pd
from .diagnoser import Diagnoser
from .engine_module import BacktrackerFull, AllocatorFixChain, CriticNaive

from collections import Counter
from functools import partial

from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor
import copy
import time

# core.py
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.hasHandlers():  # Prevent duplicate handlers when re-importing
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def mode_of_lists(list_of_lists):
    # Convert each inner list to a tuple so they can be used as keys in Counter
    tupled_lists = [tuple(lst) for lst in list_of_lists]
    
    # Use Counter to count occurrences of each tuple (representing a list)
    counter = Counter(tupled_lists)
    
    # Find the most common tuple (mode)
    mode_tuple, count = counter.most_common(1)[0]
    
    # Convert the tuple back to a list (or just return the tuple if needed)
    return list(mode_tuple)


class Optimizer(object):
    def __init__(self, 
                 model_list = ['gpt-4o-2024-05-13','claude-3-5-sonnet-20240620'],
                 max_budget=10000000,
                 parallel_eval=True,
                 max_worker=10,
                 backtracker = BacktrackerFull,
                 allocator = AllocatorFixChain,
                 critic = CriticNaive,
                 verbose = False,
                ):
        self.models = model_list
        self.max_budget=max_budget
        self.parallel_eval = parallel_eval
        self.max_worker=max_worker
        self.backtracker = backtracker
        self.allocator = allocator
        self.critic = critic
        self.verbose = verbose
        pass
        
    def optimize(self,
                 training_df,
                 metric,
                 compoundaisystem,
                ):
        pass
        
    def eval(self,
             data_df,
             AIAgent,
             M,
            ):
        # Function to apply the AIAgent.generate method
        def generate_answer(query):
            return AIAgent.generate(query)
        
        # Create a thread pool and process the queries in parallel
        def apply_parallel_with_progress(df, column, func, max_workers=20):
            queries = df[column].tolist()
            answers = []
        
            # Use ThreadPoolExecutor for multithreading
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit tasks and wrap in tqdm for a progress bar
                futures = list(tqdm(executor.map(func, queries), total=len(queries), desc="Processing"))
                answers.extend(futures)
            return answers
        time1 = time.time()
           
        if((not(self.parallel_eval)) or len(data_df)==1):
            data_df['answer'] = data_df['query'].apply(AIAgent.generate)
        else:
            # Apply the function in parallel
            #print(f"parallel eval with {self.max_worker} workers")
            data_df['answer'] = apply_parallel_with_progress(data_df, 'query', generate_answer,self.max_worker)
        #print(f"time to get answer {time.time()-time1}")

        
        time1 = time.time()
        data_df['score'] = data_df.apply(lambda row: M.get_score(row['answer'], row['true_answer']), axis=1)
        #print(f"time to get score {time.time()-time1}")
        return data_df['score'].mean()

    def set_budget(self,
                   max_budget=1000,
                  ):
        self.max_budget = max_budget
        return max_budget

    def set_models(self,
             combo,
             compoundaisystem,
             ):
        Myallow = self.allocator()
        Myallow.setup(combo)
        compoundaisystem.setup_ABC(allocator = Myallow, backtracker = self.backtracker(), critic = self.critic())
        return compoundaisystem

class OptimizerFullSearch(Optimizer):
    def optimize(self,
                 training_df,
                 metric,
                 compoundaisystem,
                ):
        def compute_score(combo):
            self.set_models(combo,compoundaisystem)
            score = self.eval(training_df,compoundaisystem,metric)
            return score
        # compute the score for all combinations
        T = len(compoundaisystem.get_pipeline())-1 # number of components
        all_permutations = list(product(self.models, repeat=T))
        random.shuffle(all_permutations)
        all_permutations = all_permutations[0:self.max_budget]
        # scores = [(combo, compute_score(combo)) for combo in tqdm(all_permutations)]       
        scores = []
        pbar = tqdm(all_permutations, desc="Scoring")

        for combo in pbar:
            score = compute_score(combo)
            scores.append((combo, score))
            pbar.set_postfix({'combo':combo,'score': score})

        # get the maximum score and the config
        max_combo, max_score = max(scores, key=lambda x: x[1])
        # set up the model
        print(f"[The finally learned model allocation is] :::{max_combo}")
        compoundaisystem = self.set_models(max_combo,compoundaisystem)
        return compoundaisystem

class OptimizerManual(Optimizer):        
    def optimize(self,
                 training_df,
                 metric,
                 compoundaisystem,
                 model_best=['gpt-4o'],
                ):
        self.set_models(model_best,compoundaisystem)
        return


    
class OptimizerLLMDiagnoser(Optimizer):
    def __init__(self, 
                 model_list = [
              'gpt-4o-2024-05-13','gpt-4-turbo-2024-04-09','gpt-4o-mini-2024-07-18',
              'claude-3-5-sonnet-20240620','claude-3-haiku-20240307',
              'gemini-1.5-pro','gemini-1.5-flash',
              'meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo','meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo','Qwen/Qwen2.5-72B-Instruct-Turbo',
              ],
                 max_budget=1000000,
                 parallel_eval=True,
                 max_worker=10,
                 backtracker = BacktrackerFull,
                 allocator = AllocatorFixChain,
                 critic = CriticNaive,
                 verbose = False,
                 
                 judge_model = 'claude-3-5-sonnet-20240620',
                 diag_model = 'gemini-1.5-pro',
                 seed = 0,
                 beta = 1,
                 alpha = 0, # no diagnoser
                 get_answer=0,
                 max_workers=10,
                ):
        super().__init__(
                model_list = model_list,
                 max_budget=max_budget,
                 parallel_eval=parallel_eval,
                 max_worker=max_worker,
                 backtracker = backtracker,
                 allocator = allocator,
                 critic = critic,
                 verbose = verbose)
        self.judge_model = judge_model
        self.diag = Diagnoser(diag_model)
        random.seed(seed)
        self.beta = beta
        self.alpha = alpha
        self.get_answer = get_answer
        self.max_workers = max_workers
        self.score_history = []
        pass
    
    def reset_score_history(self,):
        self.score_history = []
        return 
    def get_score_history(self,):
        return self.score_history
    
    def allocation2key(self,allocation):
        key = " ".join(allocation)
        return key

    def optimize(self,
                 training_df,
                 metric,
                 compoundaisystem,
                 show_progress=False,
                 get_answer=0,
                 max_iter=1000,
                 init_mi=[],
                ):
        def check_last_elements_same(L, T):
            logger.debug(f"L is {L}m and T is {T}")
            if T > len(L):  # If T is larger than the list, return False or handle appropriately
                return False
            last_elements = L[-T:]  # Slice to get the last T elements
            last_elements_as_tuples = [tuple(lst) for lst in last_elements]
            return len(set(last_elements_as_tuples)) == 1  # Check if all elements are the same using a set

        def aggregateallocation(allocations, allocation_list):
            mode = find_mode(allocations.to_list())
            #print(f"the most common allocation is {mode}") if self.verbose==True else None
            allocation_list.append(mode)
            return mode, allocation_list
     
        def find_mode(data):
            # Initialize a Counter to track the frequency of all possible values
            frequency = Counter()
        
            # Iterate over each sublist in the data
            for item in data:
                # Iterate over each possible value in the sublist
                for possible_value in item:
                    # Convert the value to a tuple to ensure hashability
                    frequency[tuple(possible_value)] += 1
        
            # Find the most common value (the mode)
            mode, count = frequency.most_common(1)[0]  # Gets the most common element and its count
            print(frequency) if self.verbose==True else None
            return list(mode)  # Convert the mode back into a list (if required)
        
        # initialization
        L = len(compoundaisystem.get_pipeline())-1
        c = 0
        M = len(self.models)-1
        B = self.max_budget
        iter = 0
        delta = 0
        allocator = list(tuple(random.choices(self.models, k=L)))
        if(len(init_mi)==L):
            allocator = init_mi
        allocator_list = [allocator]
        while(c<=B-M and delta == 0):
            print(allocator_list) if self.verbose else None
            # choose a module to optimize
            module_idx = self.update_module(iter=iter,L=L)
            # optimize the allocation w.r.t. the chosen module for each data point
            allocations = self.update_model(training_df, metric, compoundaisystem, allocator = allocator, module_idx = module_idx)
            # aggregate to one allocation
            allocator, allocator_list = aggregateallocation(allocations,allocator_list)
            print(f"----allocation list---- {allocator_list}") if self.verbose else None
            # check for stopping criteria
            c += M
            delta = check_last_elements_same(allocator_list,T=L+1)
            logger.debug(f"check last element is same: {delta}")
            iter += 1
        self.set_models(allocator_list[-1],compoundaisystem)
        logger.critical(f"allocation trace is: {allocator_list} with valid steps {len(allocator_list)-L-1}")
        print("final allocation list:",allocator_list) if self.verbose else None
        return {"allocation_trace":allocator_list,"steps":len(allocator_list)-L-1}

    def update_module(self, 
                   iter = 0,
                   module_idx = 0,
                   L = 3,
                  ):
        return (iter) % L
        
    def update_model(self,training_df, metric, compoundaisystem, allocator, module_idx = 0):
        def process_row(row):
            # compute the score for each possible model allocated to module_idx
            allocator_list = [allocator[:module_idx] + [model] + allocator[module_idx + 1:] for model in self.models]
            row_df = pd.DataFrame([row])
            scores = [(allocated_model,self.compute_score_onerow( row_df, metric,  compoundaisystem, allocated_model,module_idx)) for allocated_model in allocator_list]
            #print(scores) if self.verbose else None
            # Find the maximum score
            max_score = max(scores, key=lambda x: x[1])[1]
            # Filter all items with the maximum score
            max_combos = [combo for combo, score in scores if score == max_score]
            #print("all scoure max:",max_combos)
            # take the maximum
            max_combo, max_score = max(scores, key=lambda x: x[1])
            return max_combos

        def update_allocator(df, allocation):
            # Check if the 'allocator' column exists
            if 'allocator' not in df.columns:
                # Initialize with the allocation list
                df = df.copy()
                df.loc[:, 'allocator'] = [allocation] * len(df)

            else:
                # Update the 'allocator' column for each row
                def update_row(row):
                    if isinstance(row['allocator'], list) and len(row['allocator']) == 1:
                        # Use the first value to replace it
                        return [row['allocator'][0]]
                    else:
                        # Use allocation to replace it
                        return allocation
                
                df['allocator'] = df.apply(update_row, axis=1)
            
            return df
    
        def compute_score(combo):
            self.set_models(combo,compoundaisystem)
            score = self.eval(training_df,compoundaisystem,metric)
            return score



        training_df = update_allocator(training_df, allocator)

        '''
        # TODO: multi-thread requires a refactor
        # Currently it raises conflicts (due to handling compound ai systems' allocations)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks for parallel execution to process each row
            futures = {idx: executor.submit(process_row, row) for idx, (_, row) in enumerate(training_df.iterrows())}
    
            # Collect results as they are completed
            results = [None] * len(training_df)  # Placeholder for results
            for future in as_completed(futures.values()):
                # Identify the completed future's index
                idx = next(idx for idx, fut in futures.items() if fut == future)
                results[idx] = future.result()
    
        # Step 3: Add the results to the DataFrame
        training_df['allocator'] = results
        '''
        allocator_list = [allocator[:module_idx] + [model] + allocator[module_idx + 1:] for model in self.models]
        logger.debug(f"pre-compute the score... with allocations: {allocator_list}")
        scores = [(combo, compute_score(combo)) for combo in tqdm(allocator_list)]


        training_df['allocator'] = training_df.progress_apply(process_row, axis=1)
        return training_df['allocator']

    def compute_score_onerow(self,
                             training_df, 
                             metric, 
                             compoundaisystem,
                             allocated_model,
                             module_idx,
                            ):
        self.set_models(allocated_model,compoundaisystem)
        score = self.eval(training_df, compoundaisystem, metric)  # Pass the row to eval, not the whole df
        if(self.alpha==0):
            return score*self.beta
        score_diagnoser = self.diagnose( training_df, metric, compoundaisystem, allocated_model,module_idx)
        return score*self.beta + self.alpha*score_diagnoser
        
    def diagnose(self, training_df, metric, compoundaisystem,  allocated_model,module_idx):
        error, analysis = self.get_score_LLM_onequery(training_df.iloc[0],compoundaisystem,module_idx=module_idx)
        #print(f"allocated model:: {allocated_model} and diag index {module_idx}") if self.verbose else None
        #print(analysis) if self.verbose else None
        return 1-error

    def get_score_LLM_onequery(self,
                    query_full,
                    compoundaisystem,
                    module_idx=0,
                     ):
        ans = compoundaisystem.generate(query_full['query'])
        history = compoundaisystem.load_history()['trace']
        Info1 = {'description':compoundaisystem.get_description()}   
        Info1['module']= [t[1] for t in history] 
        #logger.debug(f"start diagnose {type(self.diag)}")
        show_prompt = logger.isEnabledFor(logging.DEBUG)
        logger.debug(f"show_prompt is {show_prompt}")

        error, analysis = self.diag.diagnose(
                compoundaisystem=Info1,
                 query=query_full['query'],
                 answer=ans,
                 true_answer=query_full['true_answer'],
                 module_id = module_idx,
            temperature=0,
            show_prompt=show_prompt,
                    )
        logger.debug(f"prompt***: {self.diag.get_prompt()}")
        self.score_history.append(
            {"query_full":query_full,
             "error":error,
             "analysis":analysis,
             "module_idx":module_idx,
             "Info1":Info1,
             }
                                  )
        Info1
        return error, analysis
 

class OptimizerLLMDiagnoser2Stage(OptimizerLLMDiagnoser):
    def __init__(self, 
                    model_list = [
                'gpt-4o-2024-05-13','gpt-4-turbo-2024-04-09','gpt-4o-mini-2024-07-18',
                'claude-3-5-sonnet-20240620','claude-3-haiku-20240307',
                'gemini-1.5-pro','gemini-1.5-flash',
                'meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo','meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo','Qwen/Qwen2.5-72B-Instruct-Turbo',
                ],
                    max_budget=1000000,
                    parallel_eval=True,
                    max_worker=10,
                    backtracker = BacktrackerFull,
                    allocator = AllocatorFixChain,
                    critic = CriticNaive,
                    verbose = False,
                    
                    judge_model = 'claude-3-5-sonnet-20240620',
                    diag_model = 'gemini-1.5-pro',
                    seed = 0,
                    beta = 1,
                    alpha = 0,
                    get_answer=0,
                    max_workers=10,
                    sample_ratio = 0.5,
                ):
        super().__init__(
                model_list = model_list,
                    max_budget=max_budget,
                    parallel_eval=parallel_eval,
                    max_worker=max_worker,
                    backtracker = backtracker,
                    allocator = allocator,
                    critic = critic,
                    verbose = verbose)
        self.judge_model = judge_model
        self.diag = Diagnoser(diag_model)
        random.seed(seed)
        self.beta = beta
        self.alpha = alpha
        self.get_answer = get_answer
        self.max_workers = max_workers
        self.sample_ratio = sample_ratio
        self.seed = seed
        self.model_list = model_list
        pass
    
    def optimize(self,
                 training_df,
                 metric,
                 compoundaisystem,
                 show_progress=False,
                 get_answer=0,
                 max_iter=1000,
                 init_mi=[],
                ):
        # 0. initialize
        setup_logger(show_progress=show_progress)
        init_mi = self.get_init(compoundaisystem=compoundaisystem,init_mi=init_mi)
        # 1. sample from the training data
        train_df_sample = training_df.sample(frac=self.sample_ratio, random_state=self.seed)
        # 2. initialize the starting point by LLMs
        init_mi_by_diagnoser = self.initialize_diagose(
            training_df = train_df_sample,
            compoundaisystem=compoundaisystem,
            init_mi=init_mi,
            metric=metric,
            )
        logger.debug(f"init_mi by diag: {init_mi_by_diagnoser}")
        '''
        score_current = self.get_diag_score_current(compoundaisystem=compoundaisystem,
                                                    metric=metric,
                                                    df = train_df_sample,
            init_mi_by_diagnoser=init_mi_by_diagnoser)
        logger.critical(f"score current by diag: {score_current}")
        '''
        direction_score = self.get_direction_score(

            compoundaisystem=compoundaisystem,
                                                    metric=metric,
                                                    df = train_df_sample,
            init_mi_by_diagnoser=init_mi_by_diagnoser
        )
        logger.critical(f"score direction: {direction_score}")
        highest = max(direction_score, key=lambda x: x[0])
        init_mi_by_diagnoser = highest[1]
        logger.critical(f"chosen one by the score dierction: {init_mi_by_diagnoser}")

        # 3. greedy search again
        self.set_models(compoundaisystem=compoundaisystem,combo=init_mi_by_diagnoser)
        self.greedy_search(training_df=training_df,
                           compoundaisystem=compoundaisystem,
                           init_mi=init_mi_by_diagnoser,
                           metric=metric)
        
        return self.get_score_history()

    def get_direction_score(self,
                               df,
                                compoundaisystem,
                                metric,
                                init_mi_by_diagnoser,
                               ):
        # for each module, update one and get the improved scores and models
        L = len(compoundaisystem.get_pipeline())-1
        improvement = [self.get_improvement(
                                df=df,
                                compoundaisystem=compoundaisystem,
                                metric=metric,
                                init_mi_by_diagnoser=init_mi_by_diagnoser,
                                index=i,

        ) for i in range(L)]
        return improvement

    def get_improvement(self,
                               df,
                                compoundaisystem,
                                metric,
                                init_mi_by_diagnoser,
                                index=0,
                               ):
        # for each allocation with varying index position, 
        allocator = init_mi_by_diagnoser
        module_idx = index
        allocator_list = [allocator[:module_idx] + [model] + allocator[module_idx + 1:] for model in self.models]
        improve = [-100,allocator_list[0]]
        for a in allocator_list:
            self.set_models(compoundaisystem=compoundaisystem,combo=a)
            score = self.eval(data_df=df,AIAgent=compoundaisystem,M=metric)
            if(score>improve[0]):
                improve = [score, a]
        return improve
         
    def get_diag_score_current(self,
                               df,
                                compoundaisystem,
                                metric,
                                init_mi_by_diagnoser,
                               ):
        L = len(compoundaisystem.get_pipeline())-1
        # compute the score for each 
        error_list = []
        for iter in range(L): 
            error_list.append(0)
            for i in range(len(df)):
                self.set_models(compoundaisystem=compoundaisystem,combo=init_mi_by_diagnoser)
                module_idx = self.update_module(iter=iter,L=L)
                error, analysis = self.get_score_LLM_onequery(df.iloc[i],compoundaisystem,module_idx=module_idx)
                error_list[iter]+=error
        return error_list

    def initialize_diagose(self,
                           training_df,
                           compoundaisystem,
                           metric,
                           init_mi=[],
                           ):
        # for each query, find one allocation by diagnose
        #logger.debug(f"training set query example {training_df.iloc[0]}")
        logging.critical(f"initial the response by diagnose")
        allocation_list = [self.get_one_query_allocation(
            df=training_df.iloc[[i]].reset_index(drop=True),
                                 compoundaisystem=compoundaisystem,
                                 metric=metric,
                                 init_mi=init_mi,
        ) 
                    for i in tqdm(range(len(training_df)))
                           ]
        coordinatewise_choice =[a[1] for a in allocation_list]
        mode, frequency = compute_modes_and_frequencies(coordinatewise_choice)
        logger.debug(f"coordinatewise_choice before filtering {coordinatewise_choice}")
        logger.critical(f"coordinatewise mode and frequency before filtering {mode} {frequency}")
        return mode
        
        allocation_list =[a[0] for a in allocation_list]
        # remove allocations that cannot find the correct final response
        logger.debug(f"allocation before filtering {allocation_list}")
        allocation_new = self.filter(allocation_list)
        logger.debug(f"full allocation new {allocation_new}")
        # take the mode of these outputs
        allocation_new = self.find_mode(allocation_new)
        # choose between the two options
        logger.critical(f"extracted initial allocation {allocation_new}")

        return allocation_new

    def get_one_query_allocation(self,
                                 df,
                                 compoundaisystem,
                                 metric,
                                 init_mi=[],
                                 ):
        #  for each module, choose one model based on an LLM diagnoser
        L = len(compoundaisystem.get_pipeline())-1
        allocator = init_mi
        final = allocator
        all_models = {}
        for iter in range(L): 
            # 1. correct. exit. 
            self.set_models(compoundaisystem=compoundaisystem,combo=allocator)
            score = self.eval(data_df=df,AIAgent=compoundaisystem,M=metric)
            logger.debug(f"allocator is {allocator} and score is {score}")
            if(score==1):
                final = allocator
            all_models[iter] = []
            # 2. incorrect, compute per module score
            module_idx = self.update_module(iter=iter,L=L)
            allocator_list = [allocator[:module_idx] + [model] + allocator[module_idx + 1:] for model in self.models]
            scores = []
            next_combo = copy.deepcopy(allocator)
            for a in allocator_list: 
                self.set_models(compoundaisystem=compoundaisystem,combo=a)  
                #logger.debug(f"the df is  {df.iloc[0]}")

                score = self.get_score_LLM_onequery(df.iloc[0],compoundaisystem,module_idx=module_idx)
                logger.debug(f"---allocate: {a}\n index: {module_idx}\n error: {score[0]}\n analysis: {score[1]}")
                scores.append(score)
                if(score[0]==0):
                    next_combo = a
                    all_models[iter].append(a[module_idx])
            allocator = next_combo

        self.set_models(compoundaisystem=compoundaisystem,combo=allocator)
        s1 = self.eval(data_df=df,AIAgent=compoundaisystem,M=metric)
        logger.debug(f"--final allocation for this query is {allocator} with score {s1}")
        if(s1<0):
            return "None"
        return final, all_models
    
    def find_mode(self, data):
        # Convert inner lists to tuples so they are hashable
        tuple_data = [tuple(item) for item in data]
        counter = Counter(tuple_data)
        logger.critical(f" frequency of the learned traces {counter}")
        mode_tuple, _ = counter.most_common(1)[0]  # Get most common tuple
        return list(mode_tuple)  # Convert back to list if needed
    
    def filter(self,data_list:list,item="None"):
        new_list = []
        for a in data_list:
            if (a!="None"):
                new_list.append(a)
        return new_list
    
    def get_init(self,compoundaisystem,init_mi=[]):
        L = len(compoundaisystem.get_pipeline())-1
        allocator = list(tuple(random.choices(self.models, k=L)))
        if(len(init_mi)==L):
            allocator = init_mi
        return  allocator


    def greedy_search(self,
                     training_df,
                     compoundaisystem,
                     metric,
                     init_mi=[],
                     ):
        
        Opt1 = OptimizerLLMDiagnoser(model_list=self.model_list)
        Opt1.optimize(init_mi=init_mi,
                      training_df=training_df,
                      metric=metric,
                      compoundaisystem=compoundaisystem,
                      )
        return

def compute_modes_and_frequencies(coordinate_list):
    if not coordinate_list:
        return [], []

    keys = coordinate_list[0].keys()
    mode_list = []
    frequency_list = []

    for key in keys:
        # Flatten all values for this key across all dictionaries
        all_values = []
        for d in coordinate_list:
            all_values.extend(d[key])

        # Count frequencies of all values
        counts = Counter(all_values)
        
        # Find the mode (most common value)
        if counts:
            mode_val, _ = counts.most_common(1)[0]
        else:
            mode_val = None
        
        mode_list.append(mode_val)
        frequency_list.append(dict(counts))  # convert Counter to dict for cleaner output

    return mode_list, frequency_list


def setup_logger(show_progress: bool = True):
    logger.setLevel(logging.DEBUG if show_progress else logging.CRITICAL)



class OptimizerLLMtrait(OptimizerLLMDiagnoser):        
    pass