import os
import pandas as pd
import pickle
path = r'C:\Users\HP\Desktop\ATML\Gutenberg_English_Fiction_1k\Gutenberg_English_Fiction_1k\Genre_Text_Books'
fileslist=[]
paracount=[]
d={}
for subdir, dirs, files in os.walk(path):
    for file in files:
        # print(file)
        with open(os.path.join(path,subdir,file),encoding='UTF-8') as f:
            for i, l in enumerate(f):
                pass
        fileslist.append(file)
        paracount.append(i+1)


d['file']=fileslist
d['para_count'] = paracount

df = pd.DataFrame(data=d)
df['para_count'] = (df['para_count']-df['para_count'].mean())/(df['para_count'].max()-df['para_count'].min())
print(df.head())
with open('para_count_dict.pickle', 'wb') as handle:
    pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
