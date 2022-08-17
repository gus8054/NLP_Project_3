import csv
import random
from transformers import AutoTokenizer

def recover_sent(line, recover_len=5):
  # line 안에는 [문어체의 끝 어절, 해요체의 끝 어절, 문어체 원 문장]이 들어가 있다.
  # 끝 어절을 추출하던 방법을 이용하여 [문어체 문장, 해요체 문장, 문어체 원 문장]으로 만들어 반환한다.
  # recover_len은 마지막 몇 어절까지 살릴 것인가를 정하며, 기본값으로 마지막 5어절만 사용한다.
  split_sentence = line[2].split()
  start = max(0,len(split_sentence)-recover_len)
  haeyo_sentence = ' '.join(split_sentence[start:-1]) + ' ' + line[1] if len(split_sentence[-1])>2 else ' '.join(split_sentence[start:-2]) + ' ' + line[1]
  return [' '.join(split_sentence[start:]), haeyo_sentence, line[2]]

class KLHdataset():
  def __init__(self, data):
    # [문어체의 끝 어절, 해요체의 끝 어절, 문어체 원 문장, 인코딩된 문어체+해요체 문장]
    self.data = data

  @classmethod
  def load_from_csv(cls, file_path:str, recover=False):
    # csv 파일에 있는 데이터를 불러온다.
    encoded_data = []
    tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2',
                                                    bos_token='</s>',
                                                    eos_token='</s>',
                                                    pad_token='<pad>')
    bos_token = [tokenizer.bos_token_id]
    eos_token = [tokenizer.eos_token_id]

    with open(file_path, 'r') as f:
      rdr = csv.reader(f)
      for line in rdr:
        if line[1]: # 해요체 변환 데이터가 없는 경우를 제외한다 (보통 문장에 문제가 있던 경우임)
          sent = recover_sent(line) if recover else line
          encoded_data.append(sent +[bos_token + tokenizer.encode('<usr>' + sent[0] + '<sys>' + sent[1]) + eos_token])

    return cls(encoded_data[1:])

  @classmethod
  def load(cls, data, recover=False):
    # load_from_csv와 비슷하나, csv파일 대신 list로 들어오는 데이터를 불러온다.
    encoded_data = []
    tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2',
                                                    bos_token='</s>',
                                                    eos_token='</s>',
                                                    pad_token='<pad>')
    bos_token = [tokenizer.bos_token_id]
    eos_token = [tokenizer.eos_token_id]

    for line in data:
      if line[1]: # 해요체 변환 데이터가 없는 경우를 제외한다 (보통 문장에 문제가 있던 경우임)
        sent = recover_sent(line) if recover else line
        encoded_data.append(sent +[bos_token + tokenizer.encode('<usr>' + sent[0] + '<sys>' + sent[1]) + eos_token])
    return cls(encoded_data)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index: int):
    # [문어체의 끝 어절, 해요체의 끝 어절, 문어체 원 문장, 인코딩된 문어체+해요체 문장] 반환
    return self.data[index]

  @classmethod
  def split(cls, dataset, train_ratio=0.8, test_ratio=0.1, seed=42):
    # Train, Validation, Test 데이터를 쪼개준다.
    # 기본 값 0.8 / 0.1 / 0.1
    rand_dataset = dataset.data.copy()
    random.seed(seed)
    random.shuffle(rand_dataset)
    train_idx = int(len(dataset)*train_ratio)
    test_idx = int(len(dataset)*(1-test_ratio))
    # train, eval, test
    return cls(rand_dataset[:train_idx]), cls(rand_dataset[train_idx:test_idx]), cls(rand_dataset[test_idx:])
  
  def add(self, other):
    # 다른 데이터셋이랑 합쳐준다
    self.data += other.data