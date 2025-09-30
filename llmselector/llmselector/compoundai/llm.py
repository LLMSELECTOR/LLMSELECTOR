from itertools import product
from tqdm import tqdm
import random
#from .compoundai import Get_Generate
import re
from tqdm import tqdm
import concurrent.futures

import os, json
from sqlitedict import SqliteDict
import google.generativeai as genai
import time
from google import genai
from google.genai import types
from PIL import Image
import io
from io import BytesIO
import base64
import hashlib, time
import numpy as np
from PIL import Image
import time

# Make tqdm work with pandas apply
tqdm.pandas()

os.environ['TASK_NAME'] = 'test'
os.environ['DB_PATH'] = 'test.sqlite'

os.environ['OPENAI_API_KEY'] = "openai_api_key"
os.environ['ANTHROPIC_API_KEY'] = "anthropic_api_key" 
os.environ['TOGETHER_API_KEY'] = "together_ai_api_key"
os.environ['GEMINI_API_KEY'] = "gemini_api_key"
os.environ['DB_EXTRACT'] = 'False'

InMemCache = {}

sleep_time = 10
def Get_Generate(prompt, model_gen,stop=None,
                max_tokens=1000,
                 temperature=0.1,
                 query_images=None,
                ):
    max_attempts = 3
    attempt = 0
    #print(f'Get_Generate::: model is {model_gen}')
    #print(f"--models and stop {model_gen} and {stop}-- ")
    if model_gen in model_provider_mapper:
        Provider = model_provider_mapper[model_gen]
    else:
        Provider = MyAPI_OpenAI

    '''
    #print(f"prompt is {prompt}")
    return Provider.get_response(text=prompt, model=model_gen,
                                         stop=stop,
                                         max_tokens=max_tokens)
    '''
    if(type(prompt)==dict):
        query_images = prompt['query_images']
        prompt = prompt['text']
    #print(f"prompt is {prompt}")
    #print(f"query iamge is {query_images}")
    while attempt < max_attempts:
        try:
            r1 = Provider.get_response(text=prompt, 
                                         model=model_gen,
                                         stop=stop,
                                        max_tokens=max_tokens,
                                         temperature=temperature,
                                         query_images=query_images,
                                        )
            if(r1==None):
                r1 = ""
            return r1
        except Exception as e:
            attempt += 1
            if attempt < max_attempts:
                print(f"Attempt {attempt} failed with error: {e}. Query is {prompt}. Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print(f"Attempt {attempt} failed with error: {e}. No more retries left.")
                return f"cannot answer due to {e}"
                #raise


## Lower level LLM services
from openai import OpenAI

import zlib, pickle, sqlite3
def my_encode(obj):
    return sqlite3.Binary(zlib.compress(pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)))

def my_decode(obj):
    return pickle.loads(zlib.decompress(bytes(obj)))

tentative_db = SqliteDict("temp.sqlite",encode=my_encode, decode=my_decode)

def extract_db(key, value):
    if(eval(os.getenv('DB_EXTRACT'))==False):
#    if(1):
        return # remove this if you want to extract the cache
    tentative_db[key] = value
    tentative_db.commit()
    return


class VLMService(object):
    def __init__(self, db_name='all'):
        dbname = os.getenv('DB_PATH').format(name=db_name)
        #print("dbname",dbname)
        self.db = SqliteDict(dbname,encode=my_encode, decode=my_decode)
#        self.db = SqliteDict(os.getenv('DB_PATH').format(name=db_name))

        return
    
    def setup_db():
        dbname = os.getenv('DB_PATH').format(name=db_name)
        #print("dbname",dbname)
        self.db = SqliteDict(dbname,encode=my_encode, decode=my_decode)
        pass

    def get_response(self, text, model='gpt-4o'):
        return

    def get_cache_key(self,
                     text,
                     model='gpt-4o',
                    max_tokens=1000,
                    temperature=0.1,
                    stop=None,
                    query_images=None,
                     ):
        if(stop is None):
            key = f"{model}_{text}_{max_tokens}_{temperature}"
        else:
            key = f"{model}_{text}_{max_tokens}_{temperature}_{stop}"
        if(query_images):
            for img in query_images:
                key+= f"_{img['hash']}"
        #print(f"key is __{key}__")
        return key

    def get_image_bytes(self, pil_image):
        buffer = BytesIO()
        pil_image.save(buffer, format='PNG')
        return buffer.getvalue()

    def get_image_hash_fast(self, img):
        img = img.convert("RGB").tobytes()  # Normalize mode
        return img

    def get_image_hash_fast2(self, img, size=(16, 16)):
        #print(f"img mode is {img.mode}")
        time1 = time.time()
        if img.mode != "RGB":
            img = img.convert("RGB")
        print(f"time to RGB {time.time()-time1}")
        time1 = time.time()
        img_bytes = img.tobytes()
        print(f"time to convert {time.time()-time1}")

        return img_bytes  # Returns a 32-bit int

    def load_response(self,
                      text,
                     model='gpt-4o',
                    max_tokens=1000,
                    temperature=0.1,
                      stop=None,
                    query_images=None,
                     ):
        #time1 = time.time()
        key = self.get_cache_key(text=text,model=model,
                                 max_tokens=max_tokens,
                                 temperature=temperature,
                                 stop=stop,query_images=query_images)
        #print(f"time to get the key {time.time()-time1}")
        
        #print("key length", len(key))

        #if(key not in self.db):
            #print("key is not found!")
            #print(f"key is _{key}_")
        #time1 = time.time()
        #'''
        try:
            result = InMemCache[key]
        except:
            result = self.db[key]
            InMemCache[key] = result
        #'''
        #result = self.db[key]

        #print(f"time to load from db {time.time()-time1}")
        #time1 = time.time()
        extract_db(key,result)
        #print(f"time to load {model} response  {time.time()-time1}")
        return result

    def save_response(self,
                      text,
                     model='gpt-4o',
                    max_tokens=1000,
                    temperature=0.1,
                      stop=None,
                      response ="",
                      query_images=None,
                     ):
        key = self.get_cache_key(text=text,model=model,
                                 max_tokens=max_tokens,
                                 temperature=temperature,
                                 stop=stop,
                                 query_images=query_images)
        self.db[key] = response
        self.db.commit()
        result = self.db[key]
        return True
        
    def base64encode(self, query_images):
        import base64
        from io import BytesIO

        encoded_images = []

        for query_image in query_images:

            buffered = BytesIO()
            query_image.save(buffered, format="png")
            base64_bytes = base64.b64encode(buffered.getvalue())
            base64_string = base64_bytes.decode("utf-8")
            encoded_images.append(base64_string)

        return encoded_images

    def compress_and_base64encode(self, original_image: Image.Image, max_base64_size=5 * 1024 * 1024):
        #print("start copying the image")
        image = original_image.copy()
        def get_base64(img: Image.Image, quality: int) -> tuple[str, int]:
            buffer = BytesIO()
            img.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=True)
            raw_bytes = buffer.getvalue()
            b64_str = base64.b64encode(raw_bytes).decode("utf-8")
            return b64_str, len(b64_str)

        quality = 95
        min_quality = 1
        resize_factor = 0.9  # Reduce by 10% if compression isn't enough
        min_size = 128  # Don't shrink below 64x64
        width, height = image.size
        i = 1
        while (width > min_size and height > min_size) or (i==1):
            current_quality = quality
            while current_quality >= min_quality:
                b64_str, b64_len = get_base64(image, current_quality)
                if b64_len <= max_base64_size:
                    #print(f"convert to length {b64_len}")
                    return b64_str
                current_quality -= 5
            i+=1

            # Compression failed, so resize the image and try again
            width = int(width * resize_factor)
            height = int(height * resize_factor)
            image = image.resize((width, height), Image.LANCZOS)

        raise ValueError(f"Failed to compress image under base64 5MB limit with size {b64_len}")
    
    def compress_image_to_under_5mb_pil(sekf,
                                        image: Image.Image, target_size=1 * 512 * 512) -> Image.Image:
        quality = 95
        step = 5

        while quality > 10:
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format='JPEG', quality=quality, optimize=True)
            size = buffer.tell()
            if size <= target_size:
                buffer.seek(0)
                return Image.open(buffer).copy()  # Return a PIL.Image instance
            quality -= step

        raise ValueError("Failed to compress image under 5MB")



class OpenAILLMService(VLMService):
    def __init__(self,db_name='all'):
        super(OpenAILLMService, self).__init__(db_name)        
        client = OpenAI()
        self.client =client
        return
        
    def get_response(self, text, 
                     model='gpt-4o',
                    max_tokens=1000,
                    temperature=0.1,
                     stop=None,
                     query_images=None,
                    ):
        #print(f'model is {model}')
        #print(f"--stop is {stop}--")
        try: 
            #time1 = time.time()
            res = self.load_response(text=text,model=model,max_tokens=max_tokens,temperature=temperature,
                                     stop=stop,query_images=query_images)
            #print(f"time to load openAI response {time.time()-time1}")
            return res
        except:
            #print(f"load failed on {model} with text _{text} , directly run")
            pass
        user_content = [
            {"type": "input_text", 
             "text": f"{text}"},
          ]

        if(query_images):
            query_img_list = [img['raw'] for img in query_images]
            encoded_images = self.base64encode(query_img_list)
            for img1 in encoded_images:
                user_content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{img1}",
                })

        if(not(stop is None)):
            response = self.client.responses.create(
          model=model,
          stop=stop,
        input=[
            {
          "role": "user",
          "content": user_content,
        }
      ],          
      max_output_tokens=max_tokens,
      #max_completion_tokens=max_tokens,

        temperature=temperature,
    )
        else:
            response = self.client.responses.create(
          model=model,         
            input=[
            {
          "role": "user",
          "content": user_content,
        }
      ],
      max_output_tokens=max_tokens,
      #max_completion_tokens=max_tokens,

    temperature=temperature,
    )
        res = response.model_dump_json()
        res = self.get_text(res) 
        self.save_response(text=text,
                           model=model,max_tokens=max_tokens,
                           temperature=temperature,
                           stop=stop,
                           response=res,
                           query_images=query_images)
        return res #response.choices[0].message.content

    def get_text(self,res):
        data = json.loads(res)
        text = data['output'][0]['content'][0]['text']
        return text


import re

def split_with_level(s: str):
    match = re.search(r'(low|minimal|medium|high)$', s)
    if match:
        return s[:match.start()], match.group()
    else:
        return s, "high"


class OpenAILLMOService(OpenAILLMService):
    def __init__(self,db_name='all'):
        super(OpenAILLMOService, self).__init__(db_name)        
        client = OpenAI()
        self.client =client
        return
        
    def get_response(self, text, 
                     model='gpt-4o',
                    max_tokens=1000,
                    temperature=0.1,
                     stop=None,
                     query_images=None,
                    ):
        #print(f'model is {model}')
        #print(f"--stop is {stop}--")
        model, effort = split_with_level(model)
        try: 
            return self.load_response(text=text,
                                      model=model+effort,
                                      max_tokens=max_tokens,temperature=temperature,
                                     stop=stop,query_images=query_images)
        except:
            #print(f"load failed on {model} with text _{text} , directly run")
            pass
        user_content = [
            {"type": "input_text", 
             "text": f"{text}"},
          ]

        if(query_images):
            query_img_list = [img['raw'] for img in query_images]
            encoded_images = self.base64encode(query_img_list)
            for img1 in encoded_images:
                user_content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{img1}",
                })

        if(not(stop is None)):
            response = self.client.responses.create(
          model=model,
          stop=stop,
            reasoning={
    "effort": effort
  },
        input=[
            {
          "role": "user",
          "content": user_content,
        }
      ],          
      #max_completion_tokens=max_tokens,

    )
        else:
            response = self.client.responses.create(
          model=model,         
            input=[
            {
          "role": "user",
          "content": user_content,
        }
      ],
        reasoning={
    "effort": effort
  },
      #max_completion_tokens=max_tokens,

    )
        res = response.model_dump_json()
        res = self.get_text(res) 
        self.save_response(text=text,
                           model=model+effort,
                           max_tokens=max_tokens,
                           temperature=temperature,
                           stop=stop,
                           response=res,
                           query_images=query_images)
        return res #response.choices[0].message.content

    def get_text(self,res):
        data = json.loads(res)
        #print("data is--",data['output'])
        text = data['output'][1]['content'][0]['text']
        return text
    
import anthropic

class AnthropicLLMService(VLMService):
    def __init__(self,db_name='all'):
        self.max_allowed_tokens = 5000
        super(AnthropicLLMService, self).__init__(db_name)        
        self.client = anthropic.Anthropic(            
            timeout=1000.0,    
)
        return
        
    def get_response(self, text, 
                     model='claude-3-opus-20240229',
                    max_tokens=1000,
                    temperature=0.1,
                     stop=None,
                     query_images=None,

                    ):
        '''
        #print("loading keys...")
        res =  self.load_response(text=text,model=model,max_tokens=max_tokens,
                                      temperature=temperature,
                                      stop=stop,
                                     )
        #print("load key success")
        '''
        #print(f"stop is --{stop}--")
        #print("text is",text)
        if(max_tokens>self.max_allowed_tokens):
            max_tokens = self.max_allowed_tokens
        try: 
            res =  self.load_response(text=text,model=model,max_tokens=max_tokens,
                                      temperature=temperature,
                                      stop=stop,
                                      query_images=query_images,
                                     )
            #print("load cached!")
            return res
        except:
            #print(f"load cache failed on {model}. Generate new")
            #iled on {model} with text _{text}__, directly run-")
            user_content = [{
                        "type": "text",
                        "text": f"{text}"
                    }]
            if(query_images):
                #query_images = self.compress_and_base64encode(query_images) 
                #print("too large; resize...")
                #image1_data = self.base64encode(query_images=query_images)
                query_img_list = [img['raw'] for img in query_images]

                image1_data = [self.compress_and_base64encode(img) for img in query_img_list]
                for img1 in image1_data:
                    user_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": img1,
                        },
                    })

            try:
                response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_sequences=stop,
            messages=[
            {
                "role": "user",
                "content":user_content,
            }
        ],
            )
            except:
                response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
            {
                "role": "user",
                "content": user_content,
            }
        ],
            )
                
            self.save_response(text=text,
                               model=model,
                               max_tokens=max_tokens,
                               stop=stop,
                               query_images=query_images,
                               temperature=temperature,response=response.content[0].text)
            return response.content[0].text


class AnthropicLLMThinkService(VLMService):
    def __init__(self,db_name='all'):
        super(AnthropicLLMThinkService, self).__init__(db_name)        
        self.client = anthropic.Anthropic(    
            timeout=10000.0,
            )
        return
        
    def get_response(self, text, 
                     model='claude-3-opus-20240229',
                    max_tokens=1000,
                    temperature=0.1,
                     stop=None,
                     query_images=None,

                    ):
        '''
        #print("loading keys...")
        res =  self.load_response(text=text,model=model,max_tokens=max_tokens,
                                      temperature=temperature,
                                      stop=stop,
                                     )
        #print("load key success")
        '''
        #print(f"stop is --{stop}--")
        #print("text is",text)
        model,effort = split_with_level(model)
        try: 
            res =  self.load_response(text=text,model=model+effort,max_tokens=max_tokens,
                                      temperature=temperature,
                                      stop=stop,
                                      query_images=query_images,
                                     )
            #print("load cached!")
            return res
        except:
            #print(f"load cache failed on {model}. Generate new")
            #iled on {model} with text _{text}__, directly run-")
            user_content = [{
                        "type": "text",
                        "text": f"{text}"
                    }]
            if(query_images):
                #query_images = self.compress_and_base64encode(query_images) 
                #print("too large; resize...")
                #image1_data = self.base64encode(query_images=query_images)
                query_img_list = [img['raw'] for img in query_images]

                image1_data = [self.compress_and_base64encode(img) for img in query_img_list]
                for img1 in image1_data:
                    user_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": img1,
                        },
                    })

            if(1):
            #try:
                thinking={
            "type": "enabled",
            "budget_tokens": int(max_tokens*0.8)
                }
                
                if(effort=='minimal'):
                    thinking={
                    "type": "disabled",
                    #"budget_tokens": 0
                    }
                    # using minimal with thinking ::: {thinking}")
                    response = self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    thinking=thinking,
                    #temperature=temperature,
                    stop_sequences=stop,
                    messages=[
                    {
                        "role": "user",
                        "content":user_content,
                    }
                    ],
                    )
                else:
                    thinking={
                    "type": "enabled",
                    "budget_tokens": int(max_tokens*0.8)
                    }
                    response = self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    thinking=thinking,
                    #temperature=temperature,
                    stop_sequences=stop,
                    messages=[
                    {
                        "role": "user",
                        "content":user_content,
                    }
                    ],
                    )
                #print(f"complete the try with {thinking}")
            #except:
            else:
                response = self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    #temperature=temperature,
                
                    messages=[
                    {
                        "role": "user",
                        "content": user_content,
                    }
                ],
                )

            #print(f"the response is {response.content[0]}")
            if(effort == 'minimal'):
                #print(f"use content[0], ")
                #print(f"response is {response}")
                if(len(response.content)==0):
                    text_output = ""
                else:
                    text_output = response.content[0].text
    #            print(f"end of use content[0], ")

            else:
                text_output = response.content[1].text

            self.save_response(text=text,
                               model=model+effort,
                               max_tokens=max_tokens,
                               stop=stop,
                               query_images=query_images,
                               temperature=temperature,response=text_output)
            #print(f"the response is {response.content[0]}")
            return text_output




from together import Together


class TogetherAILLMService(VLMService):
    def __init__(self,db_name='all'):
        super(TogetherAILLMService, self).__init__(db_name)        
        self.client = Together(api_key=os.environ.get('TOGETHER_API_KEY'))
        return
        
    def get_response(self, text, 
                     model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
                    max_tokens=1000,
                    temperature=0.1,
                     stop=None,
                     query_images=None,

                    ):
        '''
        res =  self.load_response(text=text,model=model,max_tokens=max_tokens,
                                      temperature=temperature,
                                      stop=stop,
                                     )
        '''
        try: 
            res =  self.load_response(text=text,model=model,max_tokens=max_tokens,
                                      temperature=temperature,
                                      stop=stop,
                                     )
            #print("load cached!")
            return res
        except:
            #print(f"load failed on {model} with text _{text}__, directly run-")

            client = self.client

            # Prepare the message context
            messages = [{"role": "user", "content": text}]
        
            # Make the API call to the LLaMA model
            response = client.chat.completions.create(
            model=model,
            #model = 'meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo',
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
            stop=stop,
            stream=False
            )
            res = response.choices[0].message.content
                
            self.save_response(text=text,
                               model=model,
                               max_tokens=max_tokens,
                               stop=stop,
                               temperature=temperature,response=res)
            return res


class GeminiLLMService(VLMService):
    def __init__(self,db_name='all'):
        super(GeminiLLMService, self).__init__(db_name)        
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        #genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.client =client
        return
    




        
    def get_response(self, text, 
                     model='gpt-4o',
                    max_tokens=1000,
                    temperature=0.1,
                     stop=None,
                     query_images=None,

                    ):
        #print(f"--stop is {stop}--")
        try: 
            return self.load_response(text=text,model=model,max_tokens=max_tokens,temperature=temperature,
                                     stop=stop,
                                     query_images=query_images)
        except:
            #print(f"load failed on {model} with text _{text} , directly run")
            pass
        
        '''
        generation_config = {
            "temperature": temperature,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": max_tokens,
            "response_mime_type": "text/plain",
            }
        model_instance = genai.GenerativeModel(
            model_name=model,
        generation_config=generation_config,
        )
        chat_session = model_instance.start_chat(
          history=[
            ]
        )
        response = chat_session.send_message(text)
        output_text = response.candidates[0].content.parts[0].text
        '''
        user_content=[]
        if(query_images):
            for im_dict in query_images:
                img = im_dict['raw']
#            img = query_images[0]['raw'] # self.compress_image_to_under_5mb_pil(image=query_images[0]) 
                user_content.append(img)
        user_content.append(text)
        response = self.client.models.generate_content(
                model=model,
                contents=user_content,
            config=types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        temperature=temperature,
    )

                    )
        output_text = response.text
        self.save_response(text=text,
                           model=model,max_tokens=max_tokens,
                           temperature=temperature,
                           stop=stop,
                           response=output_text,
                           query_images=query_images,
                           )
        return output_text



class GeminiLLMThinkService(VLMService):
    def __init__(self,db_name='all'):
        super(GeminiLLMThinkService, self).__init__(db_name)        
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                            http_options=types.HttpOptions(timeout=10000_000), # timeout is in milliseconds
     
                              )
        #genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.client =client
        return
    




        
    def get_response(self, text, 
                     model='gpt-4o',
                    max_tokens=1000,
                    temperature=0.1,
                     stop=None,
                     query_images=None,

                    ):
        model, effort = split_with_level(model)
        #print(f"--stop is {stop}--")
        try: 
            return self.load_response(text=text,model=model+effort,
                                      max_tokens=max_tokens,temperature=temperature,
                                     stop=stop,
                                     query_images=query_images)
        except:
            #print(f"load failed on {model} with text _{text} , directly run")
            pass
        
        '''
        generation_config = {
            "temperature": temperature,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": max_tokens,
            "response_mime_type": "text/plain",
            }
        model_instance = genai.GenerativeModel(
            model_name=model,
        generation_config=generation_config,
        )
        chat_session = model_instance.start_chat(
          history=[
            ]
        )
        response = chat_session.send_message(text)
        output_text = response.candidates[0].content.parts[0].text
        '''
        user_content=[]
        if(query_images):
            for im_dict in query_images:
                img = im_dict['raw']
#            img = query_images[0]['raw'] # self.compress_image_to_under_5mb_pil(image=query_images[0]) 
                user_content.append(img)
        user_content.append(text)
        #print(f"Gemini max token is {max_tokens}")
        thinking_budget = min(int(max_tokens*0.8),32767)
        if(effort=='minimal'):
            thinking_budget = 128
            if(model=='gemini-2.5-flash-lite'):
                thinking_budget=512
        response = self.client.models.generate_content(
                model=model,
                contents=user_content,
            config=types.GenerateContentConfig(
            
        thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),

        max_output_tokens=max_tokens,
        temperature=temperature,
    )

                    )
        output_text = response.text
        self.save_response(text=text,
                           model=model+effort,
                           max_tokens=max_tokens,
                           temperature=temperature,
                           stop=stop,
                           response=output_text,
                           query_images=query_images,
                           )
        return output_text
    
'''
MyAPI_OpenAI = OpenAILLMService(db_name=os.getenv('TASK_NAME'))
MyAPI_Anthropic = AnthropicLLMService(db_name=os.getenv('TASK_NAME'))
MyAPI_Together = TogetherAILLMService(db_name=os.getenv('TASK_NAME'))
MyAPI_Google = GeminiLLMService(db_name=os.getenv('TASK_NAME'))

model_provider_mapper={
    "claude-3-opus-20240229":MyAPI_Anthropic,
    "gpt-4-turbo-2024-04-09":MyAPI_OpenAI,
    "gpt-4o-mini-2024-07-18":MyAPI_OpenAI,
    'gpt-4o-2024-05-13':MyAPI_OpenAI,
    'gpt-4o-2024-08-06':MyAPI_OpenAI,

    "o1-mini-2024-09-12":MyAPI_OpenAI,
    'claude-3-5-sonnet-20240620':MyAPI_Anthropic,
    'claude-3-haiku-20240307':MyAPI_Anthropic,
    'meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo':MyAPI_Together,
    'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo':MyAPI_Together,
    'databricks/dbrx-instruct':MyAPI_Together,
    'Qwen/Qwen2.5-72B-Instruct-Turbo':MyAPI_Together,
    'mistralai/Mixtral-8x22B-Instruct-v0.1':MyAPI_Together,
    'gemini-1.5-pro':MyAPI_Google,
    'gemini-1.5-flash':MyAPI_Google, 
    'gemini-2.0-flash-exp':MyAPI_Google,
    'gemini-1.5-flash-8b':MyAPI_Google,
}
'''

MyAPI_OpenAI = None
MyAPI_Anthropic = None
MyAPI_Together = None
MyAPI_Google = None
model_provider_mapper = {}

# Define the model_provider_mapper
def initialize_services(task_name="test"):
    global MyAPI_OpenAI, MyAPI_Anthropic, MyAPI_Together, MyAPI_Google, model_provider_mapper

    # Initialize LLM services based on task_name
    MyAPI_OpenAI = OpenAILLMService(db_name=task_name)
    MyAPI_OpenAIO = OpenAILLMOService(db_name=task_name)
    MyAPI_Anthropic = AnthropicLLMService(db_name=task_name)
    MyAPI_AnthropicThink = AnthropicLLMThinkService(db_name=task_name)

    MyAPI_Together = TogetherAILLMService(db_name=task_name)
    MyAPI_Google = GeminiLLMService(db_name=task_name)
    MyAPI_GoogleThink = GeminiLLMThinkService(db_name=task_name)


    # Update model_provider_mapper
    model_provider_mapper = {
        "claude-3-opus-20240229": MyAPI_Anthropic,
        "claude-3-7-sonnet-20250219":MyAPI_Anthropic,
        "gpt-4-turbo-2024-04-09": MyAPI_OpenAI,
        "gpt-4o-mini-2024-07-18": MyAPI_OpenAI,
        'gpt-4o-2024-05-13': MyAPI_OpenAI,
        'gpt-4o-2024-08-06': MyAPI_OpenAI,
        'gpt-5-2025-08-07':MyAPI_OpenAIO,
        "o3-2025-04-16":MyAPI_OpenAIO,
        'gpt-5-2025-08-07low':MyAPI_OpenAIO,
        'gpt-5-2025-08-07minimal':MyAPI_OpenAIO,
        'gpt-5-mini-2025-08-07minimal':MyAPI_OpenAIO,
        'gpt-5-nano-2025-08-07minimal':MyAPI_OpenAIO,
        "o3-2025-04-16low":MyAPI_OpenAIO,
        'gpt-5-2025-08-07high':MyAPI_OpenAIO,
        "o3-2025-04-16high":MyAPI_OpenAIO,
        
        "o1-mini-2024-09-12": MyAPI_OpenAIO,
        "o3-mini-2025-01-31":MyAPI_OpenAIO,
        "o1-2024-12-17":MyAPI_OpenAIO,
        'claude-3-5-sonnet-20240620': MyAPI_Anthropic,
        'claude-3-haiku-20240307': MyAPI_Anthropic,
        'claude-opus-4-1-20250805':MyAPI_AnthropicThink,
        'claude-opus-4-1-20250805minimal':MyAPI_AnthropicThink,
        'claude-opus-4-1-20250805high':MyAPI_AnthropicThink,
        'claude-sonnet-4-20250514minimal':MyAPI_AnthropicThink,
        'claude-3-5-haiku-20241022':MyAPI_Anthropic,
        'meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo': MyAPI_Together,
        'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo': MyAPI_Together,
        "deepseek-ai/DeepSeek-R1":MyAPI_Together,
        'databricks/dbrx-instruct': MyAPI_Together,
        'Qwen/Qwen2.5-72B-Instruct-Turbo': MyAPI_Together,
        "Qwen/QwQ-32B":MyAPI_Together,
        'mistralai/Mixtral-8x22B-Instruct-v0.1': MyAPI_Together,
        'gemini-1.5-pro': MyAPI_Google,
        'gemini-1.5-flash': MyAPI_Google, 
        'gemini-2.0-flash-exp': MyAPI_Google,
        'gemini-1.5-flash-8b': MyAPI_Google,
        'gemini-2.0-flash':MyAPI_Google,
        'gemini-2.5-pro': MyAPI_GoogleThink,
        'gemini-2.5-prolow': MyAPI_GoogleThink,
        'gemini-2.5-prominimal': MyAPI_GoogleThink,
        'gemini-2.5-prohigh': MyAPI_GoogleThink,
        'gemini-2.5-flashminimal':MyAPI_GoogleThink,
        'gemini-2.5-flash-liteminimal':MyAPI_GoogleThink,
    }

# Initialize services when the module is loaded
initialize_services()


 