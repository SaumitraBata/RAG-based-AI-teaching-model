import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import requests


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


def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        # "model": "deepseek-r1",
        "model": "llama3.2",
        "prompt": prompt,
        "stream" : False    
    })
    
    respomse = r.json()
    return respomse



df = joblib.load("embeddings.joblib")

incoming_query = input("Ask a Question: ")
question_embedding = create_embedding([incoming_query])[0]
#print(question_embedding)


# Cosine similarity        (ranges from 0 to 1)
# cos theta =   A.B / |A| . |B|
# A and B ka dot product and then under root, A and B ke weights ka square and then unka multiply (Mod finding)




# Find similarities of question_embedding with other embeddings 


# print(np.vstack(df['embedding'].values))            # df['embedding'].values: Prints array
# print(np.vstack(df['embedding'].shape))             # df['embedding'].shape:  Prints Shape of the Array 

# basically np.vstack() converts it into the 2D array     

 
similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten() 
# .flatten() will give us flat array 
                                

# df['embedding'].values this will give us Array
# here [a] becuase we have to make it array of embeddings because df['embedding'].values this is also array of embeddings 
# cosine_similarity takes a 2D vector, therefore we need to convert it into the 2D array 

# print(similarities)
top_results = 10
max_indx = similarities.argsort()[::-1][0:top_results]
# np.argmax()  : gives index of the largest value
# np.argsort() : gives indices that would sort the array, smallest to largest order indices
# print(max_indx)

new_df = df.loc[max_indx]
# print(new_df[["title", "number", "text"]])

prompt = f'''I am teaching web development in my Sigma web Development course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time: 

{new_df[["title", "number", "start", "end", "text"]].to_json(orient = "records")}
--------------------------------------------------------------------------------------------------------
"{incoming_query}"
User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format it is just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course 


'''

# (orient = "records"), orient decides the shape/layout of the json, because of "records" we get list of dictionaries

# with open("prompt.txt", "w") as f:
#     f.write(prompt)

response = inference(prompt)["response"]
print(response)

with open("response.txt", "w") as f:
    f.write(response)

for index, item in new_df.iterrows():
    print(index, item["title"], item["number"], item["text"], item["start"], item["end"])

# .iterrows() lets you loop through each row of a DataFrame one by one.
# .iterrows() is slow for large datasets. Works fine for: 100 - 1000 rows
# Not ideal for: 100k+ rows
# Better alternative later:     .itertuples()  which is faster.




