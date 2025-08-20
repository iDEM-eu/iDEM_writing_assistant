import pathlib

import numpy as np
import sys
print(sys.argv, file=sys.stderr)
import torch
import sys,os,time
import pathlib
sys.path.append(os.getcwd()) #Add the current directory to the places where python looks for libraries
from core.correctifier import Correctifier
from datareaders.read_cowsl import read_cowsl
import spacy
import argparse
from langdetect import DetectorFactory
from pathlib import Path

DetectorFactory.seed = 0

tagsubstitutions = {
	"*FIRST_N*FIRST_NAME*ME*" : "John",
	"**STATE*" : "Florida",
	"*PLACE*-*PLACE*" : "Seramia",
	"*(name)*" : "Namame",
	"*FIRS_NAME*" : "Maria",
	"*FIRST_*PLACE*AME*" : "Maria",
	"*PLA*PLACE*E*" : "Seramia",
	"*FIRST_NAME*" : "Maria",
	"*LAST_NAME*" : "Smith",
	"*AGE*" : "veinticinco",
	"*UNIVERSITY*" : "Cambridge",
	"*PLACE*" : "Seramia",
	"*CITY*" : "Chicago",
	"*STATE*" : "California",
	"*NUMBER*" : "4",
	"*BIRTH_DATE*" : "5",
	"*NAME*" : "Namame",
	"*BIRTH_DATE**" : "Febrero",
	"*EMAIL*" : "abc@acme.com",
	"*MONTH*" : "Febrero",
	"*YEAR*" : "1984",
}

def replace_anonym_tags(data):
    for i in range(len(data)):
        doc = data[i]
        for j in range(len(doc)):
            newstr = doc[j]
            for tag in tagsubstitutions.keys():
                newstr = newstr.replace(tag, tagsubstitutions[tag])
            data[i][j] = newstr
    return data


# def parse_responses(sentence, output_text, lang):
#     response = output_text  # [output_text.find('OUTPUT:\n'):]
#     # for end_marker in ["<eos>", "<|eot_id|>", "```<|end_of_text|>", "<end_of_turn>", "<|endoftext|>"]:
#     #    if response.endswith(end_marker):
#     #        response = response[0:(-(len(end_marker)))]
#
#     responses = response.split("\n")
#     responses = [response for response in responses if not (
#             response.strip() in ['', 'OUTPUT:', 'INPUT:', sentence] or response.startswith(
#         'Rephrasing ') or response.startswith('Here are ') or response.startswith('Rewrite '))]
#     responses = [response.lstrip('1234567890-').lstrip('.').lstrip() for response in responses]
#     if (Path(langdetect.__file__).parents[0] / 'profiles' / lang).exists():
#         responses = [response for response in responses if len(response) < 10 or detect(response) == lang]
#     if len(responses) > 0:
#         return responses[0]
#     else:
#         return ''


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default="LLAMA1B", help="The list of currently supported models "
                                    "can be seen in core/correctifier.py. At present, this "
                                    "defaults to LLAMA1B and the list of supported models is LLAMA1B, LLAMA3B, GEMMA4B, "
                                    "OLMO7B, GEMMA12B and GEMMA27B. More models are planned to be added, especially "
                                    "salamandra models.")
parser.add_argument('--access_token', type=str, default='es', help="The access token for Hugging face")
parser.add_argument('--lang', type=str, default='es', help="The language which is used. Currently only Spanish "
                                                           "is supported. Default is 'es'.")
parser.add_argument('--max_docs', type=str, help="The maximum number of documents to be processed. This "
                                    "can be reduced for test runs, to cut run down time.")
args = parser.parse_args()

lang = args.lang
pretrained_model = args.model_name

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print("Device " + str(i) + ' : ' + torch.cuda.get_device_properties(i).name)
        print("Total memory: " + str(torch.cuda.get_device_properties(i).total_memory))
        print("Reserved: " + str(torch.cuda.memory_reserved(i)))
        print("Allocated: " + str(torch.cuda.memory_allocated(i)))
else:
    print("CUDA unavailable")

script_path = os.path.dirname(os.path.realpath(sys.argv[0]))
print("script_path", script_path, type(script_path))
path = Path(script_path)
# parent_path = path.parent.absolute()
# print("parent_path", parent_path, type(parent_path))

data_path = pathlib.Path.home() / "data" / "iDEM" / "writing_assistant"
cowsl_path = str(data_path) + "/cowsl2h-master/"
print("cowsl_path", cowsl_path, type(cowsl_path))

data = read_cowsl(cowsl_path)
data = [doc for doc in data if doc[2] != ""]
data = replace_anonym_tags(data)

if len(data) == 0:
    print("There does not seem to be data in: ", cowsl_path, file=sys.stderr)
    exit(1)

# If there is a cutoff given cut of the data
if args.max_docs:
    data = data[:int(args.max_docs)]

prompt = ("Por favor, reescribe la siguiente oración para corregir errores tipográficos, gramaticales y estilísticos "
          "cometidos por hablantes no nativos del español. No generes ningún texto excepto la reformulacion.")


device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
device = torch.device("mps") if torch.backends.mps.is_available() else device
print(device)

access_token = args.

print("Running experiments with:  ", pretrained_model)
print("Language:                  ", lang)
print("Data size:                 ", len(data))
t0 = time.time()
corrector = Correctifier(pretrained_model, access_token, device, lang)
t1 = time.time()
print("Time to load model:        ", t1-t0)

nlp = spacy.load("es_core_news_sm")

os.makedirs("outputs", exist_ok=True)  # succeeds even if directory exists.

model_name = pretrained_model
model_name = model_name.replace("/", "_")
outfile_name = f"outputs/{model_name}_{lang}_{str(device)}.tsv"
with open(outfile_name, "w") as outputfile: #iniciate empty file, overwrite if already existing. Prepare for appending
    outputfile.write("Original\tPrediction\tAnnotation\tFirst Reference\tdocId\n")

numsentences = 0

# print(len(data))
# print(data[1][0])
# print(data[1][1])
# print(data[1][2])

eval_per_doc = list()
for doc in data:
    # print("LEN ANNOTATIONS",  len(doc[2]), file=sys.stderr)
    if doc[2] == "":
        continue
    proc_doc = nlp(doc[0])
    proc_ref = nlp(doc[1])
    annot_doc_references = nlp(doc[2])
    docid = doc[3]
    sentences = [sent.text for sent in proc_doc.sents]
    ref_sentences = [sent.text for sent in proc_ref.sents]
    annot_sentences = [sent.text for sent in annot_doc_references.sents]
    numsentences += len(sentences)

    print(sentences[:3])
    corrections = [corrector.correct(sent, force_prompt=prompt, force_raw_output=True) for sent in sentences]
    # corrections_corr = [parse_responses(i[0], i[1], lang) for i in zip(sentences, corrections)]
    # score = eval_on_annotated(corrections, annot_doc_references)
    # eval_per_doc.append(score)
    corrections = ["<NEWLINE>".join(x.split("\n")) for x in corrections]

    #print(docid, len(corrections), len(corrections_corr), len(doc), len(sentences), len(proc_doc), len(annot_sentences), len(annot_doc_references))
    # for i in (zip(corrections, corrections_corr)):
    #     print("--------------------------------------")
    #     print(i[0])
    #     print(i[1])

    # evaluation_data = [(doc[i][1],simplifications[i]) for i in range(len(doc))]
    # similarity = SimilarityBLEURT(device)
    # results = similarity.assess(evaluation_data)

    with open(outfile_name, "a") as outputfile:
        for i in range(len(sentences)):
            annsent = annot_sentences[i] if i < len(annot_sentences) else ""
            refsent = ref_sentences[i] if i < len(ref_sentences) else ""
            outline = f"{str(sentences[i])}\t{str(corrections[i])}\t{annsent}\t{refsent}\t{docid}\n"
            outline = outline.replace("\n", "")
            outputfile.write(outline + "\n")

    t3 = time.time()


t2 = time.time()

print("Time to run experiment:    ", t2-t1)
print("Average time per sentence: ", float(t2-t1)/numsentences)
# print("Corrections in line with annotations:", statistics.mean(eval_per_doc))

# from evaluate import load
# sari = load("sari")
# sources=["About 95 species are currently accepted."]
# predictions=["About 95 you now get in."]
# references=[["About 95 species are currently known.","About 95 species are now accepted.","95 species are now accepted."]]
# sari_score = sari.compute(sources=sources, predictions=predictions, references=references)
#
# bleu = load("bleu")
# from nltk.tokenize import word_tokenize
# predictions = [["hello there general kenobi"], ["foo bar foobar"]]
# references = [[["hello there general kenobi"], ["hello there!"]], [["foo bar foobar"]]]
# results = bleu.compute(predictions=predictions, references=references, tokenizer=word_tokenize)
# print(results['bleu'])

# print("Time to evaluate:          ", t3-t2)
# print("Similarity BLEURT:         ", np.mean(results))

# print()
# print(evaluation_data[0])
# print()
# print(data[0])
# print()
#
# for i in evaluation_data:
#     print(i[1])
#     print(i[0])
#     print()

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print("Device " + str(i) + ' : ' + torch.cuda.get_device_properties(i).name)
        print("Total memory: " + str(torch.cuda.get_device_properties(i).total_memory))
        print("Reserved: " + str(torch.cuda.memory_reserved(i)))
        print("Allocated: " + str(torch.cuda.memory_allocated(i)))
else:
    print("CUDA unavailable")
