
import os
import pandas as pd
import numpy as np

from openai import OpenAI

from crawler import crawl
from preprocess import preprocess_text
from embedding import embedding_df,get_embedding,cosine_similarity

def prepare():
    domain = "openai.com" # <- put your domain to be crawled
    full_url = "https://openai.com/" # <- put your domain to be crawled with https or http

    text_dir="qa_example/text/"+domain+"/"
    process_dir="qa_example/processed/"

       # Create a directory to store the text files
    if not os.path.exists("text/"):
        os.mkdir("text/")

    if not os.path.exists(text_dir):
        os.mkdir(text_dir)

    # Create a directory to store the csv files
    if not os.path.exists(process_dir):
            os.mkdir(process_dir)

    crawl(full_url,domain,text_dir)
    preprocess_text(text_dir,process_dir)
    df=pd.read_csv(process_dir+'chunked.csv', index_col=0)
    embedding_df(df,process_dir)


class QaBot(object):
     
    def __init__(self,api_key,embedding_path):
        self.client = OpenAI(api_key=api_key)

        self.df=pd.read_csv(embedding_path, index_col=0)
        self.df['embeddings'] = self.df['embeddings'].apply(eval).apply(np.array)


    def create_context(self,question, max_len=1800):
        """
        Create a context for a question by finding the most similar context from the dataframe
        """

        # Get the embeddings for the question
        q_embeddings = get_embedding(question)

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
        max_tokens=150,
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
            # Create a chat completion using the question and context
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\n"},
                    #{"role": "system", "content": "Answer the question based on the context below\n\n"},
                    {"role": "user", "content": f"Context: {context}\n\n---\n\nQuestion: {question}\nAnswer:"}
                ],
                temperature=0,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=stop_sequence,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(e)
            return ""


if __name__=="__main__":
    
    #prepare()

    with open(".key","r") as f:
        key=f.read()
    bot=QaBot(key,"qa_example/processed/embeddings.csv")

    print(bot.answer_question(question="What day is it today?", debug=True))
    print(bot.answer_question(question="What is our newest embeddings model?", debug=True))
    print(bot.answer_question(question="What is ChatGPT?", debug=True))
