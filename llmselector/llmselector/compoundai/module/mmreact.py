from .module import Module, DEFAULT_MODEL
from ..llm import Get_Generate
from ..compoundai import CompoundAI
import re
from collections import Counter
from .directgen import Generator, resize_to_max_pixels
import copy

DESCRIPTION_MMReact = '''This compound AI system uses 3 modules to solve a question.
'''


PROMPT_ACTION = '''Answer the following user task. In addition to the image(s), you may leverage the given OCR and image captions as needed. 
[User Task]: {query}
[OCR] : {ocr_result}
[Image Caption]: {caption_result}
[Your Response]:
'''

PROMPT_ACTION_V2 = '''Answer the following user task. In addition to the image(s), you may leverage the given OCR, image captions, and detected objects as needed. 
[User Task]: {query}
[OCR] : {ocr_result}
[Image Caption]: {caption_result}
[Detected Objects]: {obj_detect_result}
[Your Response]:
'''

PROMPT_OCR = '''Extract all texts from the given image. Do not generate anything else.'''

PROMPT_CAPTION = '''Generate a detailed describption of the given image.''' 

PROMPT_OBJDETECTOR = '''Describe all objects in the image in detail as much as possible. For each object, your description should include at least names, locations, bounding boxes, shapes, and colors. You should also include the relationships between different objects if applicable.'''

class Get_Answer(Generator):
    def format_text(self,text="",ocr_result="",caption_result=""):
        newtext = self.gen_prompt.format(query=text,ocr_result=ocr_result,caption_result=caption_result)
        newtext = " "*self.add_space + newtext
        if(self.add_space>10):
            print(f"add_space is: {self.add_space}")
            print(f"query is: {text}")
            print(f"newtext is: {newtext}")
        return newtext
    
    def get_response(self,query:dict, ocr_result:str, caption_result:str):
        #time1 = time.time()
        query = copy.copy(query)  # Make a deep copy to avoid mutating original
        #print(f'copy time is {time.time()-time1}')
        if(type(query)==str):
            query = self.format_text(query,ocr_result,caption_result)
        else:
            query['text'] = self.format_text(query['text'],ocr_result,caption_result)
            query['query_images'] = [resize_to_max_pixels(img) for img in query['query_images']]
        #time1 = time.time()
        r1 = Get_Generate(query,self.model,
                            max_tokens=self.max_tokens,
                           )
        #print(f'gen time is {time.time()-time1}')
        return r1

class Get_AnswerV2(Generator):
    def format_text(self,
                    text="",
                    ocr_result="",
                    caption_result="",
                    obj_detect_result=""):
        newtext = self.gen_prompt.format(query=text,
                                         ocr_result=ocr_result,
                                         caption_result=caption_result,
                                         obj_detect_result=obj_detect_result)
        newtext = " "*self.add_space + newtext
        if(self.add_space>10):
            print(f"add_space is: {self.add_space}")
            print(f"query is: {text}")
            print(f"newtext is: {newtext}")
        return newtext
    
    def get_response(self,query:dict, ocr_result:str, caption_result:str,obj_detect_result:str):
        #time1 = time.time()
        query = copy.copy(query)  # Make a deep copy to avoid mutating original
        #print(f'copy time is {time.time()-time1}')
        if(type(query)==str):
            query = self.format_text(text=query,ocr_result=ocr_result,
                                     caption_result=caption_result,obj_detect_result=obj_detect_result)
        else:
            query['text'] = self.format_text(text=query['text'],
                                             ocr_result=ocr_result,
                                             caption_result=caption_result,
                                             obj_detect_result=obj_detect_result)
            query['query_images'] = [resize_to_max_pixels(img) for img in query['query_images']]
        #time1 = time.time()
        r1 = Get_Generate(query,self.model,
                            max_tokens=self.max_tokens,
                           )
        #print(f'gen time is {time.time()-time1}')
        return r1
    
class Get_OCR(Generator):
    pass

class Get_Caption(Generator):
    pass

class Detect_Object(Generator):
    pass

class MMReact(CompoundAI):
    def __init__(self,
                 description = DESCRIPTION_MMReact,
                 prompt_ocr = PROMPT_OCR,
                 prompt_caption = PROMPT_CAPTION,
                 prompt_action = PROMPT_ACTION,
                 merge_regex= r'\A(.*)\Z',
                ):
        super().__init__(description=description)
        self.prompt_ocr = prompt_ocr
        self.prompt_cation = prompt_caption
        self.prompt_action = prompt_action
        self.merge_regex = merge_regex
        self.create_pipeline(pipeline= self._get_pipeline())
        pass
        
    def _get_pipeline(self):
        pipeline = [["query",0],
                           [Get_OCR(add_space=0,gen_prompt=self.prompt_ocr),[0]],
                           [Get_Caption(add_space=0,gen_prompt=self.prompt_cation),[0]],
                           [Get_Answer(add_space=0,gen_prompt=self.prompt_action),[0,1,2]],
                           ]
        return pipeline

class MMReactV2(CompoundAI):
    def __init__(self,
                 description = DESCRIPTION_MMReact,
                 prompt_ocr = PROMPT_OCR,
                 prompt_caption = PROMPT_CAPTION,
                 prompt_objdetect = PROMPT_OBJDETECTOR,
                 prompt_action = PROMPT_ACTION_V2,
                 merge_regex= r'\A(.*)\Z',
                ):
        super().__init__(description=description)
        self.prompt_ocr = prompt_ocr
        self.prompt_cation = prompt_caption
        self.prompt_action = prompt_action
        self.prompt_objdetect = prompt_objdetect
        self.merge_regex = merge_regex
        self.create_pipeline(pipeline= self._get_pipeline())
        pass
        
    def _get_pipeline(self):
        pipeline = [["query",0],
                           [Get_OCR(add_space=0,gen_prompt=self.prompt_ocr),[0]],
                           [Get_Caption(add_space=0,gen_prompt=self.prompt_cation),[0]],
                           [Detect_Object(add_space=0,gen_prompt=self.prompt_objdetect),[0]],
                           [Get_AnswerV2(add_space=0,gen_prompt=self.prompt_action),[0,1,2,3]],
                           ]
        return pipeline
