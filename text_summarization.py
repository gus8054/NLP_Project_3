import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import numpy as np
import os
from urllib.request import urlretrieve
import zipfile

glove_dict = dict()

  
nltk.download('stopwords')
nltk.download('punkt')

if not(os.path.isfile('glove.6B.zip')):
    urlretrieve("http://nlp.stanford.edu/data/glove.6B.zip", filename="glove.6B.zip")
    zf = zipfile.ZipFile('glove.6B.zip')
    zf.extractall() 
    zf.close()

with open('glove.6B.100d.txt', encoding="utf8") as f: # 100차원의 GloVe 벡터를 사용 
    for line in f:
        word_vector = line.split()
        word = word_vector[0]
        word_vector_arr = np.asarray(word_vector[1:], dtype='float32') # 100개의 값을 가지는 array로 변환
        glove_dict[word] = word_vector_arr

def summarize(eng_text:list)->list:
  text = " ".join(eng_text)
  # 토큰화 함수
  def tokenization(sentences):
      return [word_tokenize(sentence) for sentence in sentences]

  # 전처리 함수
  def preprocess_sentence(sentence):
    # 영어를 제외한 숫자, 특수 문자 등은 전부 제거. 모든 알파벳은 소문자화
    sentence = [re.sub(r'[^a-zA-z\s]', '', word).lower() for word in sentence]
    # 불용어가 아니면서 단어가 실제로 존재해야 한다.
    return [word for word in sentence if word not in stop_words and word]

  # 위 전처리 함수를 모든 문장에 대해서 수행. 이 함수를 호출하면 모든 행에 대해서 수행.
  def preprocess_sentences(sentences):
      return [preprocess_sentence(sentence) for sentence in sentences]

  # 단어 벡터의 평균으로부터 문장 벡터를 얻는다.
  def calculate_sentence_vector(sentence):
    if len(sentence) != 0:
      return sum([glove_dict.get(word, zero_vector) 
                    for word in sentence])/len(sentence)
    else:
      return zero_vector

  # 각 문장에 대해서 문장 벡터를 반환
  def sentences_to_vectors(sentences):
      return [calculate_sentence_vector(sentence) 
                for sentence in sentences]

  def similarity_matrix(sentence_embedding):
    sim_mat = np.zeros([len(sentence_embedding), len(sentence_embedding)])
    for i in range(len(sentence_embedding)):
        for j in range(len(sentence_embedding)):
          sim_mat[i][j] = cosine_similarity(sentence_embedding[i].reshape(1, embedding_dim),
                                            sentence_embedding[j].reshape(1, embedding_dim))[0,0]
    return sim_mat

  def calculate_score(sim_matrix):
      nx_graph = nx.from_numpy_array(sim_matrix)
      scores = nx.pagerank(nx_graph, max_iter = 100000, tol=1.0e-3)  #오류때문에 max_iter 추가
      return scores

  def ranked_sentences(sentences, scores, n=5):   # 요약 5 문장으로 출력
      top_scores = sorted(((scores[i], s, order) for order, (i,s) in enumerate(enumerate(sentences))), key=lambda x: x[0], reverse=True)
      top_n_sentences = [sentence for score, sentence, order in sorted(top_scores[:n], key=lambda x: x[2])]
      return " ".join(top_n_sentences)

  # glove_dict = init.get_glove_dict()
  stop_words = stopwords.words('english')
  
  data = {}
  data['content'] = text
  data['sentences'] = sent_tokenize(data['content'])
  data['tokenized_sentences'] = tokenization(data['sentences'])
  data['tokenized_sentences'] = preprocess_sentences(data['tokenized_sentences'])
  embedding_dim = 100
  zero_vector = np.zeros(embedding_dim)
  data['SentenceEmbedding'] = sentences_to_vectors(data['tokenized_sentences'])
  data['SimMatrix'] = similarity_matrix(data['SentenceEmbedding'])
  data['score'] = calculate_score(data['SimMatrix'])
  data['summary'] = ranked_sentences(data['sentences'], data['score'])
  result = sent_tokenize(data['summary'])
  del data

  return result