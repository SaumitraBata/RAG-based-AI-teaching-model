
######################################## Renamed the files

# import os
# folder_path = r"C:\Saumitra\Data Science Course\RAG based AI teaching model\videos"

# # Get all files in folder
# files = os.listdir(folder_path)

# for file in files:
#     if file.endswith(".mp4"):
#         old_path = os.path.join(folder_path, file)
        
#         # Replace "_" with space
#         new_name = file.replace("_", " ")
#         new_path = os.path.join(folder_path, new_name)
        
#         # Rename file
#         os.rename(old_path, new_path)

#         print(f"Renamed: {file} -> {new_name}")



##########    as the FFMPEG accepts mp3 
################################### Converting the Videos into the mp3 

import os
import subprocess

files = os.listdir("videos")
for file in files:
    #print(file)
    tutorial_number = file.split("- ")[1].split(" ")[1]
    #print(tutorial_number)
    file_name = file.split(" -")[0].split(" Sigma")[0]
    print(tutorial_number, file_name)

    subprocess.run(["ffmpeg", "-i", f"videos/{file}", f"audios/{tutorial_number}_{file_name}.mp3"])






