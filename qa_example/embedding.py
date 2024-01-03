import os
from openai import OpenAI
import numpy as np

class OpenAIEmbedding(object):

    def __init__(self,key,model='text-embedding-ada-002'):

        self.client = OpenAI(api_key=key)
        self.model=model

    def get_embedding(self,text):
        resp=self.client.embeddings.create(input = [text], model=self.model)
        return resp.data[0].embedding

    def add_embedding_for_df(self,df,out_path=None):

        df['embeddings'] = df.text.apply(lambda x: self.get_embedding(x))
        
        if out_path:
            df.to_csv(out_path)

        print(df.head())

        return df


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))