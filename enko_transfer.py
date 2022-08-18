from nltk.tokenize import sent_tokenize
from transformers import (
    PreTrainedTokenizerFast,
    DistilBertTokenizerFast,
)
from transformers.models.encoder_decoder.modeling_encoder_decoder import EncoderDecoderModel
from kss import split_sentences
# import os

# os.environ["TOKENIZERS_PARALLELISM"] = "false"

src_tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')
trg_tokenizer = PreTrainedTokenizerFast.from_pretrained('skt/kogpt2-base-v2')
# special tokens 설정
special_tokens_dict = {'bos_token': '<BOS>', 'eos_token': '<EOS>', 'pad_token': '<PAD>'}
num_added_toks = trg_tokenizer.add_special_tokens(special_tokens_dict)

model = EncoderDecoderModel.from_pretrained('./model/enko_transfer/best_model')
model.eval()
# model.cuda()
model.config.decoder_start_token_id = trg_tokenizer.bos_token_id
split_sentences('a')

def translate_title(eng_title:str) :
    text = eng_title
    embeddings = src_tokenizer(text, return_attention_mask=False, return_token_type_ids=False, return_tensors='pt')
    embeddings = {k: v for k, v in embeddings.items()}
    output = model.generate(**embeddings, 
                        max_length=30,
                        num_return_sequences=1,
                        no_repeat_ngram_size=2,
                        num_beams=3,
                        length_penalty=10
                        )[0,1:-1]
    ko_title = trg_tokenizer.decode(output.cpu(), skip_special_tokens=True)
    ko_title = split_sentences(ko_title)[0]

    return ko_title

def translate_text(eng_text:str) :
    ko_texts = []
    texts = sent_tokenize(eng_text)
    for text in texts :
        embeddings = src_tokenizer(text, return_attention_mask=False, return_token_type_ids=False, return_tensors='pt')
        embeddings = {k: v for k, v in embeddings.items()}
        output = model.generate(**embeddings, 
                            max_length=50,
                            num_return_sequences=1,
                            no_repeat_ngram_size=3,
                            num_beams=7,
                            length_penalty=5
                            )[0,1:-1]
        ko_text = trg_tokenizer.decode(output.cpu(), skip_special_tokens=True)
        ko_text = split_sentences(ko_text, num_workers=-1)[0]
        ko_texts.append(ko_text)
    return " ".join(ko_texts)


if __name__ == '__main__':
    print("시작")
    a = '''TUNIS (Reuters) - Tunisia will submit a national reform programme to the International Monetary Fund (IMF) in September, the economy minister said on Wednesday, part of efforts to secure a bailout to rescue strained public finances. Economy Minister Samir Saied was quoted by state news agency TAP as saying the programme would be submitted to the IMF board. Tunisia has said it wants $4 billion in loans from the IMF, though diplomats have said it is more likely to get $2 billion $3 billion. The programme is expected to include measures to reduce the public sector wage bill, cut subsidies and outline a path to restructuring indebted state-owned companies. Last week the government signed an agreement with the major labour union, a longstanding opponent of state spending cuts that could hit its members, and with the major business union, to hold talks on reforms.'''
    print(translate_text(a))
