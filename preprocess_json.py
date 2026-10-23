# import requests
# import os 
# import json

# def create_embedding(text):
#     r = requests.post("http://localhost:11434/api/embeddings", json={
#         "model": "bge-m3",
#         "prompt": text
#     })

#     # embedding = r.json()['embeddings']              # This prints embeddings in the sentence 
#     # print(embedding)
#     print(r.json())





# jsons = os.listdir("jsons")
# #print(jsons)
# my_dicts = []                # appending that all in the list
# chunk_id = 0                 # giving each of them id 

# for json_file in jsons:
#     with open(f"jsons/{json_file}") as f :         # As the json is just the file name, so we are opening those and reading the file
#         content = json.load(f)

#         for chunk in content['chunks']:
#             print(chunk)
#             chunk['chunk_id'] = chunk_id                                  # giving id to the chunk 
#             chunk['embedding'] = create_embedding(chunk['text'])          # giving the embeddings to the text present in the chunk
#             chunk_id += 1
#             my_dicts.append(chunk)
    
#     break

# print(my_dicts)

#a = create_embedding("Cat sat on the mat")
#print(a)



####### THE ABOVE PROCESS IS VERY SLOW AS IT WAS PROCESSING 1 SENTENCE FOR 1 Second therefore we converted it, into such a 
####### way that it takes list of sentences and then process it at once.
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings



# import requests
# import os 
# import json
# import pandas as pd 


# def create_embedding(text_list):
#     r = requests.post("http://localhost:11434/api/embed", json={
#         "model": "bge-m3",
#         "input": text_list
#     })

#     embedding = r.json()['embeddings']              # This prints embeddings in the sentence 
#     return embedding

# jsons = os.listdir("jsons")       # List all the jsons 
# #print(jsons)
# my_dicts = []                # appending that all in the list
# chunk_id = 0                 # giving each of them id 

# for json_file in jsons:
#     with open(f"jsons/{json_file}") as f :         # As the json is just the file name, so we are opening those and reading the file
#         content = json.load(f)
#     print(f"Creating Embeddings for {json_file}")

#     embeddings = create_embedding([c['text'] for c in content['chunks']])


    
#     for i, chunk in enumerate(content['chunks']):
#         #print(chunk)
#         chunk['chunk_id'] = chunk_id                                  # giving id to the chunk 
#         chunk['embedding'] = embeddings[i]          # giving the embeddings to the text present in the chunk
#         chunk_id += 1
#         my_dicts.append(chunk)

    

#print(my_dicts)

# a = create_embedding(["Cat sat on the mat", "Harry dances on a mat"])
# print(a)









############# As there are some places where there is the error or there is a this type of error in the json file: 
# "}{
#    "chunks": [
# which causing that file to be broke, there fore we need to handle it in the code only,


import requests
import os
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib



######################## EMBEDDING FUNCTION
def create_embedding(text_list): 

    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    data = r.json()
    if "embeddings" not in data:
        print("Error:", data)
        return []
    
    return data["embeddings"]


# ---------- BATCH EMBEDDING FUNCTION (FIXES NaN ERROR) ----------
def batch_embed(texts, batch_size = 35):
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        embeddings = create_embedding(batch)

        if not embeddings:
            print(f"Skipping bad batch starting at index {i}")

            # Insert placeholder embeddings to maintain alignment
            for _ in batch:
                all_embeddings.append(None)

            continue

        all_embeddings.extend(embeddings)

    return all_embeddings


# Here starts main script : 

jsons = os.listdir("jsons")
my_dicts = []
chunk_id = 0

for json_file in jsons:


    with open(f"jsons/{json_file}", encoding="utf-8" ) as f :
        raw = f.read()

    # Fix broken JSON formatting 

    raw = raw.replace("}{", "},{")

    if not raw.startswith('['):
        raw = '[' + raw + ']'

    content = json.loads(raw)

    # Fix broken JSON formatting
    # to check whether the input is list or not,Handle both single object {"chunks": [...]} and array [{"chunks": [...]}, ...]
    # it takes the variable and its type so that we can verify

    if isinstance(content, list):
        all_chunks = []
        for item in content:
            all_chunks.extend(item.get("chunks", []))
    else:
        all_chunks = content.get("chunks", [])

    print(f"Creating Embeddings for {json_file}, ({len(all_chunks)} chunks)")



    texts = [c['text'] for c in all_chunks]
    embeddings = batch_embed(texts)

    if not embeddings:
        print(f"ERROR: No embeddings returned for {json_file}")
        continue

    for i, chunk in enumerate(all_chunks):

        # if (chunk_id == 6):    # Read only 5 chunks for now 
        #     break

        if embeddings[i] is None:
            continue

        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embeddings[i]
        chunk_id += 1
        my_dicts.append(chunk)




# print("Embedding process completed successfully.")

df = pd.DataFrame.from_records(my_dicts)
# print(df)

# Save this DataFrame 
joblib.dump(df, 'embeddings.joblib')

