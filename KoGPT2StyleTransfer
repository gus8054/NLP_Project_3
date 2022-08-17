import tensorflow as tf
from transformers import AutoTokenizer
from transformers import TFGPT2LMHeadModel

import pandas as pd
from tqdm.notebook import tqdm

class KoGPT2StyleTransfer():
  def __init__(self, model_name = 'skt/kogpt2-base-v2'):
    self.model = TFGPT2LMHeadModel.from_pretrained(model_name, from_pt=True)
    self._tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                    bos_token='</s>',
                                                    eos_token='</s>',
                                                    pad_token='<pad>')
    

  def train(self, train_data, val_data, batch_size = 8, epochs = 1, learning_rate = 3e-5, epslion = 1e-08):
    # train_data와 val_data는 KLHdataset이라 가정한다.
    # 어체 변환 모델을 학습하며, train_loss와 val_loss를 각각 list로 반환한다.
    self._adam = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=epslion)
    train_steps = len(train_data) // batch_size + 1
    val_steps = len(val_data) // batch_size + 1

    def get_train_data():
      for i in train_data: yield i[3]
    def get_val_data():
      for i in val_data: yield i[3]

    try:
      train_dataset = tf.data.Dataset.from_generator(get_train_data, output_types=tf.int32)
    except:
      print('train_data와 val_data가 KHLdataset인지 확인해주세요.')
      return [], []
    train_dataset = train_dataset.padded_batch(batch_size=batch_size, padded_shapes=(None,), padding_values=self._tokenizer.pad_token_id)

    val_dataset = tf.data.Dataset.from_generator(get_val_data, output_types=tf.int32)
    val_dataset = val_dataset.padded_batch(batch_size=batch_size, padded_shapes=(None,), padding_values=self._tokenizer.pad_token_id)

    train_loss_history = []
    val_loss_history = []
    for epoch in range(epochs):
      epoch_loss = 0
      val_loss = 0

      # Train
      for batch in tqdm(train_dataset, total=train_steps, desc=f'Training Epoch {epoch}...'):
          with tf.GradientTape() as tape:
              result = self.model(batch, labels=batch)
              loss = result[0]
              batch_loss = tf.reduce_mean(loss)
              
          grads = tape.gradient(batch_loss, self.model.trainable_variables)
          self._adam.apply_gradients(zip(grads, self.model.trainable_variables))
          epoch_loss += batch_loss

      # Eval
      for batch in tqdm(val_dataset, total=val_steps, desc=f'Evaulating Epoch {epoch}...'):
          with tf.GradientTape() as tape:
              result = self.model(batch, labels=batch)
              loss = result[0]
              batch_loss = tf.reduce_mean(loss)
          val_loss += batch_loss
      epoch_loss /= train_steps; val_loss /= val_steps

      print('[Epoch: {:>4}] Train loss = {:>.9} Val loss = {:>.9}'.format(epoch + 1, epoch_loss, val_loss))
      train_loss_history.append(float(epoch_loss)); val_loss_history.append(float(val_loss))
      return train_loss_history, val_loss_history

  def style_transfer(self, user_text: str, max_length = 50, top_k = 20):
      # 주어진 문어체 문장을 모델을 통해서 해요체로 변환한다.
      sent = '<usr>' + user_text + '<sys>'
      input_ids = [self._tokenizer.bos_token_id] + self._tokenizer.encode(sent)
      input_ids = tf.convert_to_tensor([input_ids])
      output = self.model.generate(input_ids, max_length=max_length, do_sample=True, top_k=top_k)
      sentence = self._tokenizer.decode(output[0].numpy().tolist())
      # 모델 출력에 따라서 아래 부분에서 간혹 오류가 날 수 있다.
      response = sentence.split('<sys> ')[1].replace('</s>', '')
      return response

  def load_model(self, file_path: str):
    # 모델 weights를 불러온다.
    self.model.load_weights(file_path)

  def save_model(self, file_path: str):
    # 현재 모델 weights를 저장한다
    self.model.save_weights(file_path)