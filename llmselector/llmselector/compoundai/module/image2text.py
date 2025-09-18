from .module import Module, DEFAULT_MODEL
from ..llm import Get_Generate
from ..compoundai import CompoundAI
import re
from collections import Counter


DESCRIPTION_Image2Text = '''This compound AI system uses 2 modules to solve a question involving an image. Module 0 extracts textual information from the image, and module 1 solves the task given the textual info and the image, and returns the final answer.
'''

PROMPT_TEMPLATE_GETIMAGETEXT= '''Extract all texts from the image. Seperate each word by a space.
'''

PROMPT_TEMPLATE_GETIMAGETEXT_BOUNDINGBOX="""Extract all texts (both print and handwritten) in the image with their own bounding box locations. Do not generate any other texts.
The bounding box format: The coordinate system is a 2D normalized Cartesian system where the image’s width and height are scaled from 0 to 1000. The origin (0, 0) is at the top-left corner, with the x-axis increasing rightward and the y-axis increasing downward. Bounding boxes are defined as [x_min, y_min, x_max, y_max], where x_min and y_min are the top-left coordinates, and x_max and y_max are the bottom-right coordinates, all normalized relative to the image dimensions.
"""

PROMPT_TEMPLATE_RAW = """
{query}
"""

PROMPT_TEMPLATE_SOLVESUBTASK='''
{query} 
The text extracted from the image: {text}.
Your response:
'''

PROMPT_TEMPLATE_SOLVESUBTASK_BOX="""Answer the following question using the given image and the extracted text with bounding boxes. Do not generate extract content.
question: {query}
extracted texted with bounding boxes: {text} 
"""

# Classes for Table task
class GetImageText(Module):
    def __init__(self,
                 prompt_template=PROMPT_TEMPLATE_GETIMAGETEXT_BOUNDINGBOX,
                 max_tokens = 1000,
                ):
        super().__init__(max_tokens=max_tokens)
        self.prompt_template = prompt_template
    def get_response(self,query):
        return Get_Generate(self.prompt_template.format(query=query['text']),self.model,
                           max_tokens=self.max_tokens,
                           query_images=query['query_images'],
                           )

class SolveSubTaskwithText(Module):
    def __init__(self,
                 prompt_template_solvesubtask=PROMPT_TEMPLATE_SOLVESUBTASK_BOX,
                                 max_tokens = 1000,

                ):
        super().__init__(max_tokens=max_tokens)
        self.prompt_template_solvesubtask = prompt_template_solvesubtask
    def get_response(self,
                     query,
                     text,
                     ):
        return Get_Generate(self.prompt_template_solvesubtask.format(query=query['text'],
                                                                    text=text,
                                                                    ),
                            self.model,
                            max_tokens=self.max_tokens,
                            query_images=query['query_images'],
                           )

class Image2Text(CompoundAI):
    def __init__(self,
                 description=DESCRIPTION_Image2Text,
                ):
        super().__init__(description=DESCRIPTION_Image2Text)
        self.create_pipeline(pipeline= self._get_pipeline())
        pass
        
    def _get_pipeline(self):
        pipeline = [["query",0],
               [GetImageText(),[0]],
               [SolveSubTaskwithText(), [0,1]],
               ]
        return pipeline
    
class GenRaw(Module):
    def __init__(self,
                 prompt_template=PROMPT_TEMPLATE_RAW,
                 max_tokens = 1000,
                ):
        super().__init__(max_tokens=max_tokens)
        self.prompt_template = prompt_template
    def get_response(self,query):
        return Get_Generate(self.prompt_template.format(query=query['text']),self.model,
                           max_tokens=self.max_tokens,
                           query_images=query['query_images'],
                           )
    
    
class Image2TextRaw(CompoundAI):
    def __init__(self,
                 description=DESCRIPTION_Image2Text,
                ):
        super().__init__(description=DESCRIPTION_Image2Text)
        self.create_pipeline(pipeline= self._get_pipeline())
        pass
        
    def _get_pipeline(self):
        pipeline = [["query",0],
               [GenRaw(),[0]],
               ]
        return pipeline