# %%
from text_style_transfer import KoGPT2StyleTransfer, replace_sent
# %%
# print(os.path.isfile('./model/kogpt2_style_transfer/kogpt2_style_transfer.ckpt'))
model_path = './model/kogpt2_style_transfer/kogpt2_style_transfer.ckpt'
style_transfer_model = KoGPT2StyleTransfer()
style_transfer_model.load_model(model_path)

# %%
# sentence = '현재 NLP 기술은 급격히 발전하고 있다.'
sentence = '있다.'
print(style_transfer_model.style_transfer(sentence))
# %%
segment = '아침까지 안개가 끼는 곳이 있어요.'
sent = '오늘 중부지방의 하늘에는 구름이 많이 지나겠고, 중부 내륙에는 아침까지 안개가 끼는 곳이 있다.'

recover_line = replace_sent(segment, sent)
print(recover_line)
# # 모델 학습
# train_loss, val_loss = style_transfer_model.train(train_data, val_data)
# # 모델 저장
# style_transfer_model.save_model(model_path)
# # 이미 학습된 모델 호출하기
# %%
