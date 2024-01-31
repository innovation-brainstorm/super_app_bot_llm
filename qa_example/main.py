
import os
from pathlib import Path
import pandas as pd
import numpy as np

from openai import OpenAI

from crawler import crawl
from preprocess import preprocess_text
from embedding import OpenAIEmbedding,cosine_similarity

# Configure proxy settings of environment varialbes
# proxy = 'http://172.25.208.1:10811'
# os.environ['HTTP_PROXY'] = proxy
# os.environ['HTTPS_PROXY'] = proxy

def prepare(openai_key):
    domain = "citigroup.com" # <- put your domain to be crawled
    full_url = "https://www.citigroup.com/global" # <- put your domain to be crawled with https or http

    text_dir=Path("qa_example/text/"+domain+"/")
    process_dir=Path("qa_example/processed/")


    text_dir.mkdir(parents=True,exist_ok=True)
    process_dir.mkdir(parents=True,exist_ok=True)
    

    crawl(full_url,domain,text_dir)
    preprocess_text(text_dir,process_dir)
    df=pd.read_csv(process_dir / 'chunked.csv', index_col=0)

    embedding=OpenAIEmbedding(openai_key)
    embedding.add_embedding_for_df(df,process_dir/'embeddings.csv')


class QaBot(object):
     
    def __init__(self,api_key,embedding_path):
        self.client = OpenAI(api_key=api_key)

        self.embedding=OpenAIEmbedding(api_key)

        self.messages=[ 
                    #{"role": "system", "content": "Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\n"},
                    #{"role": "system", "content": "Answer the question based on the context below with reference to the actual context, and if the question can't be answered based on the context, say \"I don't know\"\n\n"},
                    {"role": "system", "content": "Answer the question only based on the context below\n\n"}
                    ]

        self.df=pd.read_csv(embedding_path, index_col=0)
        self.df['embeddings'] = self.df['embeddings'].apply(eval).apply(np.array)


    def create_context(self,question, max_len=1800):
        """
        Create a context for a question by finding the most similar context from the dataframe
        """

        # Get the embeddings for the question
        q_embeddings = self.embedding.get_embedding(question)

        # Get the distances from the embeddings
        self.df['similarity'] = self.df.embeddings.apply(lambda row:cosine_similarity(q_embeddings, row))


        returns = []
        cur_len = 0

        # Sort by distance and add the text to the context until the context is too long
        for i, row in self.df.sort_values('similarity', ascending=False).iterrows():

            # Add the length of the text to the current length
            cur_len += row['n_tokens'] + 4

            # If the context is too long, break
            if cur_len > max_len:
                break

            # Else add it to the text that is being returned
            returns.append(row["text"])

        # Return the context
        return "\n\n###\n\n".join(returns)
    
    def answer_question(self,
        question="Am I allowed to publish model outputs to Twitter, without a human review?",
        max_len=1800,
        debug=False,
        max_tokens=256,
        stop_sequence=None
    ):
        """
        Answer a question based on the most similar context from the dataframe texts
        """
        context = self.create_context(question,max_len=max_len)
        # If debug, print the raw model response
        if debug:
            print("Context:\n" + context)
            print("\n\n")

        try:
            
            # opt1. remove context from history messages opt2. make a summary for previous context
            self.messages.append({"role": "user", "content": f"Context: {context}\n\n---\n\nQuestion: {question}\nAnswer:"})
            # Create a chat completion using the question and context
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",#gpt-4-1106-preview,gpt-3.5-turbo
                messages=self.messages,
                temperature=0,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=stop_sequence,
            )

            response_message = response.choices[0].message

            self.messages.append(response_message)

            return response_message.content.strip()
        except Exception as e:
            print(e)
            return ""


if __name__=="__main__":

    with open(".key","r") as f:
        openai_key=f.read()
    
    #prepare(openai_key)



    bot=QaBot(openai_key,"qa_example/processed/embeddings.csv")


    # print(bot.answer_question(question="What day is it today?", debug=True))
    # print(bot.answer_question(question="What is our newest embeddings model?", debug=True))
    # print(bot.answer_question(question="What is the goal of city in 2024?", debug=True))
    while 1:
        user_query=input("please input your question:")
        print(bot.answer_question(user_query, debug=True))
        print()

