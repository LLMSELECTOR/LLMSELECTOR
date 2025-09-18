import os, json, time, copy
from .optimizer import Optimizer
from .engine_module import BacktrackerFull, AllocatorNaive, CriticNaive
from .llm import Get_Generate

DEFAULT_MODEL = "gpt-4o-2024-05-13"


DESCRIPTION = '''Placeholder.
'''

DIAGNOSEPROMPT = '''Placeholder.
'''

MAX_NUMBER_MODULES = 100000000

class CompoundAI(object):
    def __init__(self,
                allocator = None,
                backtracker = BacktrackerFull(),
                critic = CriticNaive(),
                eps = 0.5,
                max_search=100,
                default_model = 'claude-3-5-sonnet-20240620',
                diagnoseprompt = DIAGNOSEPROMPT,
                description = DESCRIPTION,
                save_history_flag=True,
                ):
        self.allocator = allocator
        if(allocator is None):
            self.allocator = AllocatorNaive(default_model)
        self.backtracker = backtracker
        self.critic = critic
        self.eps = eps
        self.max_search = max_search
        self.default_model = default_model
        self.diagnoseprompt = diagnoseprompt
        self.description = description
        self.return_model_i = MAX_NUMBER_MODULES
        self.save_history_flag = save_history_flag
        return
    
    def set_module_output(self, module_index=MAX_NUMBER_MODULES):
        self.return_model_i = module_index
        return
    
    def get_diagnoseprompt(self,):
        return self.diagnoseprompt

    def get_description(self):
        return self.description
        
    def create_pipeline(self,
                        pipeline=[],
                        early_stop=None,
                        postprocess=None,
                       ):
        self.pipeline = pipeline
        self.early_stop=early_stop
        self.postprocess = postprocess

    def setup_ABC(self,
                 allocator = AllocatorNaive(),
                 backtracker = BacktrackerFull(),
                critic = CriticNaive(),
                ):
        self.allocator = allocator
        self.backtracker = backtracker
        self.critic = critic
        return
        
    def generate(self, query="hello", metric="",ans=""):
        #time1 = time.time()
        c = 1
        H = []
        h = []
        eps = self.eps
        q = query
        iter = 0
        max_search = self.max_search
        model_usage = []
        last_full = []
        last_res = ""
        response_list = [query]
        bp=False
        max_index = min(len(self.pipeline)-1,self.return_model_i)

        while(1<=c and c<=max_index and max_search>iter):
            params = [response_list[t] for t in self.pipeline[c][1]]
            # 1. Select model
            model = self.allocator.get_model(q,h,H)
            model_usage.append(model)
            '''
            if(c==1):
                print(f"parm is {params}")
            '''
            '''
            if(len(H)!=0):
                print(f"lens of H {len(H)}",c,model)
            '''
            self.pipeline[c][0].setmodel(model)
            #time1 = time.time()
            res = self.pipeline[c][0].get_response(*params)
            response_list.append(res)
            #print(f"the time to generate in compound ai system {time.time()-time1}")
            h.append([model,res])
            # 2. Verify the quality
            score = self.critic.judge(q,h,H,metric,ans)
            #print(f"pos and score {c} {score}")
            if(score>eps):
                c += 1 # high quality, continue
                bp=False
            else:
                #print(f"backtrack")
                #H.append(tuple([t for t in h]))
#                H.append(tuple(copy.deepcopy(t) for t in h))
                if(c==len(self.pipeline)-1):
                    last_full = model_usage
                    last_res = res
#                H.append(tuple(copy.deepcopy(t) for t in h))
                H.append(h)

                #print(f"history lens {len(H)}")
                #c = self.backtracker.track(q,h,H)
                #h = h[:c-1]
                h = []
                response_list=[query]
                model_usage = []
                c = 1
                # TDOO: We need to reset the pipeline env
                #self.pipeline[c][0].setmodel(model)
                #print(f"c and h {c} {h}")
                bp=True
            iter+=1
            if(not(self.early_stop is None)):
                finish = self.early_stop(query,res)
                if(bp==True):
                    finish = 0
                ''' 
                if(len(H)>0):
                    print(f'listroty length {len(H)} and finish {finish}')
                '''
                if(finish==1):
                    if(not(self.postprocess is None)):
                        res = self.postprocess(query,res)
                    return res
            
        #H_unique = [list(item) for item in set(tuple(sublist) for sublist in H)]
        #print(f"History length {len(H)}")
        #print(f"c is {c} and return_model_i is {self.return_model_i}")
        if(max_search<=iter):
            model_usage = last_full
            res = res
        #print("model_usage",model_usage,c)
        if(not(self.postprocess is None)):
            res = self.postprocess(query,res)
        self.save_history(q=query,t=h,h=H,response_list=response_list)
        
        #print(f"the end-to-end time to compoundaisystem.generate {time.time()-time1}")
            
        return res

    def generate_naive(self, query="hello"):
        response_list = [query]
        for c in range(1,len(self.pipeline)):
            params = [response_list[t] for t in self.pipeline[c][1]]
            #print(f"params are::: {params}")
            res = self.pipeline[c][0].get_response(*params)
            response_list.append(res)
            print(response_list)
        return res

    def generate_onestep(self, query="hello",step=0):
        response_list = [query]
        for c in range(1,min(len(self.pipeline),step)):
            params = [response_list[t] for t in self.pipeline[c][1]]
            res = self.pipeline[c][0].get_response(*params)
            response_list.append(res)
        return res

    def get_pipeline(self,
                    ):
        return self.pipeline

    def save_history(self,q,t,h,response_list):
        if(self.save_history_flag==False):
            return
        self.q = q
        self.t = t
        self.h = h
        self.response_list = response_list
        return
    
    def load_history(self,):
        
        return {'query':self.q,'trace':self.t,'history':self.h,
               'response_list':self.response_list,
               }
