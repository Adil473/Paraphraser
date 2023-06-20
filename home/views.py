from django.http import HttpResponse
from django.shortcuts import render, HttpResponse
from parrot import Parrot
import torch
import warnings

warnings.filterwarnings("ignore")
# Create your views here.

# def paraphr(request):
#     text = request.GET.get('ip_text')
#     return render(request, "paraphr.html" , {'variable2' : para})

def index(request):
    return render(request, 'frontend.html')

def paraphr(request):
    text = request.GET.get('ip_text')
    def random_state(seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    random_state(1234)

    parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=False)

    phrases = text.split(".")
    for phrase in phrases:
        para_phrases = parrot.augment(input_phrase=phrase, do_diverse=True, diversity_ranker="levenshtein")
    
    return render(request , 'paraphr.html' , {"variable2" : para_phrases[0][0]})


def paraphrase(request):
    text = request.GET.get("ip_text")
    import torch
    from transformers import PegasusForConditionalGeneration, PegasusTokenizer
    model_name = 'tuner007/pegasus_paraphrase'
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = PegasusTokenizer.from_pretrained('tuner007/pegasus_paraphrase')
    model = PegasusForConditionalGeneration.from_pretrained('tuner007/pegasus_paraphrase').to(torch_device)

    def get_response(input_text,num_return_sequences,num_beams):
        batch = tokenizer([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
        translated = model.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
        tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
        return tgt_text

    # text="A major drawback of statistical methods is that they require elaborate feature engineering. Since 2015,[19] the field has thus largely abandoned statistical methods and shifted to neural networks for machine learning. Popular techniques include the use of word embeddings to capture semantic properties of words, and an increase in end-to-end learning of a higher-level task (e.g., question answering) instead of relying on a pipeline of separate intermediate tasks (e.g., part-of-speech tagging and dependency parsing). In some areas, this shift has entailed substantial changes in how NLP systems are designed, such that deep neural network-based approaches may be viewed as a new paradigm distinct from statistical natural language processing. For instance, the term neural machine translation (NMT) emphasizes the fact that deep learning-based approaches to machine translation directly learn sequence-to-sequence transformations, obviating the need for intermediate steps such as word alignment and language modeling that was used in statistical machine translation"
    get_response(text,1,1) 

    paraphrase = []
    sentence_list = text.split('.')
    for phrase in sentence_list:
        a = get_response(phrase, 1, 1)
        paraphrase.append(a)

    paraphrase2 = [" ".join(x) for x in paraphrase]
    paraphrase3 = [" ".join(x for x in paraphrase2)]
    paraphrased_text = str(paraphrase3).strip('[]').strip('"')
    
    return render(request, "paraphr.html" , {'variable2' : paraphrased_text , 'variable1' : text})