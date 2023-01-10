import json
from fastapi import FastAPI, Query, HTTPException, Path
from pydantic import BaseModel
import uvicorn

from transformers import pipeline
"""
divert me : melakukan klasifikasi text twitter apakah merupakan side issue atau tidak.
Input : Text 
output : divert me : False / True
Package yang dibutuhkan :


"""
#from extractive_summarizer import Word2VecSummarizer

path_model = '/setneg-dir-02/saas-socmed/dev-nlp/divertme/model_divertme_final'
classifier = pipeline('sentiment-analysis', model = path_model)

import re
import fasttext
from string import punctuation
import unicodedata2 as unicodedata
import pandas as pd
def lowercase(text):
    return text.lower()

#casefolding
def casefolding(s):
    new_str = s.lower()  
    return new_str

def masking_entity(str):
    new_url =  re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}     /)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',"", str)
    return new_url

def cleaning(str):
    #remove digit from string
    str = re.sub("\S*\d\S*", "", str).strip()
    #removeHashtag
    str = re.sub('#[^\s]+','',str)
    #remove mention
    str = re.sub("@([a-zA-Z0-9_]{1,50})","",str)
    #remove non-ascii
    str = unicodedata.normalize('NFKD', str).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    #remove_punctuation
    str = str.translate(str.maketrans("", "", punctuation))
    #to lowercase
    str = str.lower()
    #Remove additional white spaces
    str = re.sub('[\s]+', ' ', str)
    
    return str

#slang word
def normalize_slang_word(str):
    text_list = str.split(' ')
    slang_words_raw = pd.read_csv('/setneg-dir-02/saas-socmed/dev-nlp/dev-divertme/slang_word_list.csv', sep=',', header=None)
    slang_word_dict = {}
    
    for item in slang_words_raw.values:
        slang_word_dict[item[0]] = item[1]
        
        for index in range(len(text_list)):
            if text_list[index] in slang_word_dict.keys():
                text_list[index] = slang_word_dict[text_list[index]]
    
    return ' '.join(text_list)

def preprocess_text(text):
    text = casefolding(text)
    text = cleaning (text)
    text = masking_entity (text)
    text = "".join(text)
    text = normalize_slang_word(text)
    return text

def predict_class(tweet):
    tweet_prep = preprocess_text(tweet)
    cek_class = classifier(tweet_prep)

    if cek_class[0]['label'] == 'LABEL_0':
        return('False')
    else:
        return('True')
    

app = FastAPI()

class divertme(BaseModel):
#    id: Optional[int] = None
    tweet: str

@app.post("/predict_divertme", status_code = 200)
async def predict_result(Items : divertme):
    label = predict_class(Items.tweet)
    return { "Side Info" : label}

if __name__ == '__main__':
    uvicorn.run("api_divertme:app", host='172.17.61.220', port=2013
)


