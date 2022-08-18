# NLP_Project_3
자연어처리 세 번째 프로젝트 협업 공간입니다.

# Install
## KoGPT-2 어체 변환 모델 사용 시
토큰은 GitHub에 접속 후 Settings - Developer settings - Personal access tokens에서 발급받아야 한다.
```Python
!git clone -b feature/style_transfer https://토큰@github.com/gus8054/NLP_Project_3.git
```
```Python
pip install -r NLP_Project_3/kogpt2-requirements.txt
```

# How To Use
## Drive Mount
Google Colab에서 아래 코드를 실행하기 위해서는 G드라이브에 연동해야 합니다.
```Python
from google.colab import drive
drive.mount('/content/drive')
```
## KLHdataset
KoGPT-2 어체 변환 모델에 쓰일 데이터를 불러옵니다.
```Python
from NLP_Project_3.KLHdataset import KLHdataset

# 학습 데이터 파일 1
train_data_file = '/content/drive/MyDrive/GoormProject/GoormProject3/data/문어체_해요체_끝어절.csv'
# 학습 데이터 파일 2
train_data_file_added = '/content/drive/MyDrive/GoormProject/GoormProject3/data/문어체_해요체_끝어절_added.csv'
# Data Augmentation된 학습 데이터 파일
train_data_file_added_2 = '/content/drive/MyDrive/GoormProject/GoormProject3/data/문어체_해요체_added_2.csv'

# csv 파일에서 데이터 불러오기
data = KLHdataset.load_from_csv(train_data_file, recover = True)
# 다른 csv 파일에서 데이터를 불러와서 합침
data.add(KLHdataset.load_from_csv(train_data_file_added, recover=True))
data.add(KLHdataset.load_from_csv(train_data_file_added_2, recover=True))
# Train, Validation, Test 데이터로 쪼갬
train_data, val_data, test_data = KLHdataset.split(data)
```

## KoGPT2StyleTransfer
KoGPT-2 어체 변환 모델입니다.
### 모델 학습 / 저장 / 불러오기
```Python
from NLP_Project_3.KoGPT2StyleTransfer import KoGPT2StyleTransfer

#model_path = '/content/drive/MyDrive/GoormProject/GoormProject3/kogpt2_style_transfer/kogpt2_style_transfer.ckpt'
# Data Augmentation된 데이터까지 학습한 모델을 사용하고자 한다면 아래 모델을 사용한다.
model_path = '/content/drive/MyDrive/GoormProject/GoormProject3/kogpt2_style_transfer/kogpt2_style_transfer_data_aug.ckpt'

style_transfer_model = KoGPT2StyleTransfer()
# 모델 학습
# train_loss, val_loss = style_transfer_model.train(train_data, val_data)
# 모델 저장
# style_transfer_model.save_model(model_path)

# 이미 학습된 모델 호출하기
style_transfer_model.load_model(model_path)
```
### 어절 어체 변환하기
```Python
# 어체 변환
>>> sentence = '현재 NLP 기술은 급격히 발전하고 있다.'
>>> print(style_transfer_model.style_transfer(sentence))
현재 NLP 기술은 급속히 발전해요.
```
모델 출력에 따라서 가끔 위 코드에서 오류가 나는 경우가 있다. 이를 방지하기 위해 아래와 같이 오류를 무시하도록 코드를 짜주면 좋다.
```Python
try:
    print(style_transfer_mode.style_transfer(sentence))
except:
    print(sentence) # 오류날 시 원래 문장대로 내보냄
```
### 변환 결과를 원래 문장에 적용하기
```Python
from NLP_Project_3.KoGPT2StyleTransfer import replace_sent
segment = '아침까지 안개가 끼는 곳이 있어요.'
sent = '오늘 중부지방의 하늘에는 구름이 많이 지나겠고, 중부 내륙에는 아침까지 안개가 끼는 곳이 있다.'

recover_line = replace_sent(segment, sent)
print(recover_line)
```