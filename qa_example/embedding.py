import os
from openai import OpenAI
import numpy as np

proxy = 'http://localhost:8001'
os.environ['HTTP_PROXY'] = proxy
os.environ['HTTPS_PROXY'] = proxy

with open(".key","r") as f:
    key=f.read()

client = OpenAI(
    api_key=key,
)
engine='text-embedding-ada-002'

def get_embedding(text, model="text-embedding-ada-002"):
   resp=client.embeddings.create(input = [text], model=model)
   return resp.data[0].embedding

def embedding_df(df,process_dir):

    df['embeddings'] = df.text.apply(lambda x: get_embedding(x, model=engine))

    df.to_csv(process_dir+'embeddings.csv')
    print(df.head())


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))