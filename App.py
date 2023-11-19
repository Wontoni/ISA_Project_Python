# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def index():
    return "ISA Project Flask Server"


@app.post('/translate')
def translate():
    article_en = request.form['original_text']
    translate_code = request.form['translate_code']
    tokenizer = AutoTokenizer.from_pretrained("SnypzZz/Llama2-13b-Language-translate")
    model = AutoModelForSeq2SeqLM.from_pretrained("SnypzZz/Llama2-13b-Language-translate")

    model_inputs = tokenizer(article_en, return_tensors="pt")

    # translate from English
    generated_tokens = model.generate(
        **model_inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[translate_code]
    )
    translated_sentence = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return translated_sentence[0]

    # English (en_XX), 
    # Spanish (es_XX), 
    # French (fr_XX), 
    # Japanese (ja_XX),
    # Korean (ko_KR),
    # Russian (ru_RU)
    # Vietnamese (vi_VN), 
    # Chinese (zh_CN),
    # Mongolian (mn_MN),
    # Urdu (ur_PK)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)