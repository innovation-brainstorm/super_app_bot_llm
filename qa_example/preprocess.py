import os
import pandas as pd
import tiktoken
import magic

def identify_file_type(file_name):
    mime = magic.Magic(mime=True)
    file_type = mime.from_file(file_name)
    return file_type

def is_text_file(file_name):
    file_type=identify_file_type(file_name)
    if file_type=='text/plain':
        return True
    return False

def remove_newlines(serie):
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\\n', ' ')
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('  ', ' ')
    return serie


def preprocess_text(text_dir,process_dir):
    # Create a list to store the text files
    texts=[]

    # Get all the text files in the text directory
    for file in os.listdir(text_dir):
        file_path=text_dir / file

        if is_text_file(file_path):

            with open(file_path, "r", encoding="UTF-8") as f:
                text = f.read()

                # clean file name then add with text
                texts.append((file[14:-4].replace('-',' ').replace('_', ' '), text))

    # Create a dataframe from the list of texts
    df = pd.DataFrame(texts, columns = ['fname', 'text'])

    # Set the text column to be the raw text with the newlines removed
    df['text'] = df.fname + ". " + remove_newlines(df.text)
    df.to_csv(process_dir /'scraped.csv')

    print(df.head())

    chunked_df=chunk(df)

    chunked_df.to_csv(process_dir /'chunked.csv')


def chunk(df):
    # Load the cl100k_base tokenizer which is designed to work with the ada-002 model
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Tokenize the text and save the number of tokens to a new column
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

    max_tokens = 500

    # Function to split the text into chunks of a maximum number of tokens
    def split_into_many(text, max_tokens = max_tokens):

        # Split the text into sentences
        sentences = text.split('. ')

        # Get the number of tokens for each sentence
        n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]

        chunks = []
        tokens_so_far = 0
        chunk = []

        # Loop through the sentences and tokens joined together in a tuple
        for sentence, token in zip(sentences, n_tokens):

            # If the number of tokens so far plus the number of tokens in the current sentence is greater
            # than the max number of tokens, then add the chunk to the list of chunks and reset
            # the chunk and tokens so far
            if tokens_so_far + token > max_tokens:
                chunks.append(". ".join(chunk) + ".")
                chunk = []
                tokens_so_far = 0

            # If the number of tokens in the current sentence is greater than the max number of
            # tokens, go to the next sentence
            if token > max_tokens:
                continue

            # Otherwise, add the sentence to the chunk and add the number of tokens to the total
            chunk.append(sentence)
            tokens_so_far += token + 1

        return chunks


    shortened = []

    # Loop through the dataframe
    for row in df.iterrows():
        try:
            # If the text is None, go to the next row
            if row[1]['text'] is None:
                continue

            # If the number of tokens is greater than the max number of tokens, split the text into chunks
            if row[1]['n_tokens'] > max_tokens:
                shortened += split_into_many(row[1]['text'])

            # Otherwise, add the text to the list of shortened texts
            else:
                shortened.append( row[1]['text'] )
        except Exception as e:
            print(f"chunk line no. {row[0]} failed.",e)

    chunked_df = pd.DataFrame(shortened, columns = ['text'])
    chunked_df['n_tokens'] = chunked_df.text.apply(lambda x: len(tokenizer.encode(x)))

    return chunked_df
