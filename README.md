# NLP_Project_3
자연어처리 세 번째 프로젝트 협업 공간입니다.

# How To Use
## KLHdataset
```Python
train_data_file = '/content/drive/MyDrive/GoormProject/GoormProject3/data/문어체_해요체_끝어절.csv'
train_data_file_added = '/content/drive/MyDrive/GoormProject/GoormProject3/data/문어체_해요체_끝어절_added.csv'

# csv 파일에서 데이터 불러오기
data = KLHdataset.load_from_csv(train_data_file, recover = True)
# 다른 csv 파일에서 데이터를 불러와서 합침
data.add(KLHdataset.load_from_csv(train_data_file_added, recover=True))
# Train, Validation, Test 데이터로 쪼갬
train_data, val_data, test_data = KLHdataset.split(data)
```