from nltk.tokenize import sent_tokenize
from transformers import (
    PreTrainedTokenizerFast,
    DistilBertTokenizerFast,
)
from transformers.models.encoder_decoder.modeling_encoder_decoder import EncoderDecoderModel
from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration
from kss import split_sentences
import torch
import text_style_transfer

###########추가된 코드#####################
# cuda memory error 피하기
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import gc
gc.collect()
torch.cuda.empty_cache()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
########################################

# src_tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')
# trg_tokenizer = PreTrainedTokenizerFast.from_pretrained('skt/kogpt2-base-v2')
# # special tokens 설정
# special_tokens_dict = {'bos_token': '<BOS>', 'eos_token': '<EOS>', 'pad_token': '<PAD>'}
# num_added_toks = trg_tokenizer.add_special_tokens(special_tokens_dict)

# model = EncoderDecoderModel.from_pretrained('./model/enko_transfer/best_model')
# model.eval()
# model.cuda()
# model.config.decoder_start_token_id = trg_tokenizer.bos_token_id
# split_sentences("")

model_path = './model/enko_transfer/checkpoints/checkpoint-1878'
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", src_lang="en", tgt_lang="ko")
model = M2M100ForConditionalGeneration.from_pretrained(model_path)

tokenizer.src_lang = "en"
tokenizer.tgt_lang = "ko"

model.eval()
model.cuda()

# def translate_title(eng_title:str) :
#     text = eng_title
#     embeddings = src_tokenizer(text, return_attention_mask=False, return_token_type_ids=False, return_tensors='pt')
#     embeddings = {k: v.cuda() for k, v in embeddings.items()}
#     output = model.generate(**embeddings, 
#                         max_length=30,
#                         num_return_sequences=1,
#                         no_repeat_ngram_size=2,
#                         num_beams=3,
#                         length_penalty=10
#                         )[0,1:-1]
#     ko_title = trg_tokenizer.decode(output.cpu(), skip_special_tokens=True)
#     ko_title = split_sentences(ko_title)[0]
#     changed_ko_title = text_style_transfer.translate_text_style(ko_title)
#     return changed_ko_title

def translate_title(eng_title:str) :
    text = eng_title
    embeddings = tokenizer.encode(text, return_tensors='pt').cuda()
    output = model.generate(embeddings, 
                            num_beams=2,
                            num_return_sequences=1,
                            max_length=30,
                            no_repeat_ngram_size = 2,
                            length_penalty = 10,
                            forced_bos_token_id=tokenizer.get_lang_id("ko"))[0]

    ko_title = tokenizer.decode(output, skip_special_tokens=True)        
    # ko_title = split_sentences(ko_title)[0]
    changed_ko_title = text_style_transfer.translate_text_style(ko_title)
    return changed_ko_title

# def translate_text(eng_text:list) :
#     ko_texts = []
#     for text in eng_text :
#         embeddings = src_tokenizer(text, return_attention_mask=False, return_token_type_ids=False, return_tensors='pt')
#         embeddings = {k: v.cuda() for k, v in embeddings.items()}
#         output = model.generate(**embeddings, 
#                             max_length=50,
#                             num_return_sequences=1,
#                             no_repeat_ngram_size=3,
#                             num_beams=7,
#                             length_penalty=5
#                             )[0,1:-1]
#         ko_text = trg_tokenizer.decode(output.cpu(), skip_special_tokens=True)
#         ko_text = split_sentences(ko_text, num_workers=-1)[0]
#         changed_ko_text = text_style_transfer.translate_text_style(ko_text)
#         ko_texts.append(changed_ko_text)
#     return ko_texts

def translate_text(eng_text:list) :
    ko_texts = []
    for text in eng_text :
        embeddings = tokenizer.encode(text, return_tensors='pt').cuda()
        output = model.generate(embeddings, 
                                num_beams=5,
                                num_return_sequences=1,
                                max_length=512,
                                no_repeat_ngram_size = 2,
                                forced_bos_token_id=tokenizer.get_lang_id("ko"))[0]
        
        ko_text = tokenizer.decode(output.cpu(), skip_special_tokens=True)
        ko_text = split_sentences(ko_text, num_workers=-1)[0]
        changed_ko_text = text_style_transfer.translate_text_style(ko_text)
        ko_texts.append(changed_ko_text)
    return ko_texts
