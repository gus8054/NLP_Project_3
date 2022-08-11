# %%
path = '/content/drive/MyDrive/GoormProject/GoormProject3'

# %%
import json
import pandas as pd

# %%
% cd /content/drive/MyDrive/GoormProject/GoormProject3

# %%
data = 'data/구어체_이진분류_dataset.json'

with open(data, 'r') as fd :
    data = json.load(fd) 
    
enko = pd.DataFrame()
en = []
ko = []

for i in range(len(data['data'])) :
    if data['data'][i]['style'] == "해요체" :
        en.append(data['data'][i]['en'])
        ko.append(data['data'][i]['ko'])


enko['en'] = en
enko['ko'] = ko

# %%
enko

# %%
enko.to_csv('data/en-ko.csv', encoding='utf-8')
# %%
