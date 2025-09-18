from .module import Module, DEFAULT_MODEL
from ..llm import Get_Generate
from ..compoundai import CompoundAI
import re
from collections import Counter
import copy, time
import math
from PIL import Image

def resize_to_max_pixels(image:dict, max_pixels=10000000):
    width, height = image['raw'].size
    total_pixels = width * height

    if total_pixels <= max_pixels:
        return image
    #print(f"total pixel is {width} * {height} = {width*height}")
    # Calculate the scaling factor
    scale = math.sqrt(max_pixels / total_pixels)

    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Resize image
    img_resize = image['raw'].resize((new_width, new_height), Image.Resampling.LANCZOS)
    return {'raw':img_resize,'hash':image['hash']}

class Generator(Module):
    def __init__(self,
                add_space=0,
                gen_prompt = '''{query}''',
                max_tokens = 1000,
                model = DEFAULT_MODEL,
                ):
        self.add_space = add_space
        self.gen_prompt = gen_prompt
        super().__init__(max_tokens=max_tokens,model=model)
        pass

    def format_text(self,text="",):
        newtext = self.gen_prompt.format(query=text)
        newtext = " "*self.add_space + newtext
        if(self.add_space>10):
            print(f"add_space is: {self.add_space}")
            print(f"query is: {text}")
            print(f"newtext is: {newtext}")
        return newtext
    
    def get_response(self,query):
        #time1 = time.time()
        query = copy.copy(query)  # Make a deep copy to avoid mutating original
        #print(f'copy time is {time.time()-time1}')
        if(type(query)==str):
            query = self.format_text(query)
        else:
            query['text'] = self.format_text(query['text'])
            query['query_images'] = [resize_to_max_pixels(img) for img in query['query_images']]
        #time1 = time.time()
        r1 = Get_Generate(query,self.model,
                            max_tokens=self.max_tokens,
                           )
        #print(f'gen time is {time.time()-time1}')
        return r1
    
class DirectGen(CompoundAI):
    def __init__(self,
                 description="This directly generates answers from the prompt",
                 save_history_flag=True,
                ):
        super().__init__(description=description,save_history_flag=save_history_flag)
        self.create_pipeline(pipeline= self._get_pipeline())
        pass
        
    def _get_pipeline(self):
        pipeline = [["query",0],
                    [Generator(),[0]],
                    ]
        return pipeline