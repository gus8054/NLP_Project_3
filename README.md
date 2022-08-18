# 실행 환경
* 구글 코랩 Pro+ 에서 테스트했습니다.

# 설치 패키지
* !pip install flask_ngrok
* !pip install pyngrok==4.1.1
* !pip install transformers
* !pip install sentencepiece
* !pip install torch
* !pip install kss
* !pip install nltk
* !pip install bert_score

# ngrok에서 가입후 토큰을 받아온다.
* 주소 : https://ngrok.com/
* !ngrok authtoken "토큰" # 쌍따옴표를 남긴다.

# 학습모델은 팀 프로젝트 드라이브에 올림, 위치는 다음과 같이 세팅한다.
* enko_transfer 폴더 안에 best_model폴더 안에 config.json, pytorch_model.bin
* style_transfer 폴더 안에 kogpt2_style_transfer 폴더와 kogpt2_style_transfer.tf 폴더

# Usage
* server.py 코드를 복사 후 코랩에 붙여넣기 해서 실행한다.
