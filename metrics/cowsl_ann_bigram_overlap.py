import glob
import os
import pathlib
import re
import sys
import statistics
# import spacy
# nlp = spacy.load("es_core_news_sm")
from collections import defaultdict

substitution_tags = {
	"John" : "*FIRST_NAME*",
	"Florida" : "*STATE*",
	"Maria" : "*FIRST_NAME*",
	"María" : "*FIRST_NAME*",
	"Smith" : "*LAST_NAME*",
	"veinticinco" : "*AGE*",
	"veintiséis" : "*AGE*",
	"Cambridge" : "*UNIVERSITY*",
	"Seramia" : "*PLACE*",
	"Chicago" : "*CITY*",
	"California" : "*STATE*",
	"Namame" : "*NAME*",
	"abc@acme.com" : "*EMAIL*",
	"Febrero" : "*MONTH*",
	"1984" : "*YEAR*",
}

def dates2tags(sentence):
    for i in substitution_tags:
        sentence = sentence.replace(i, substitution_tags[i])
    return sentence

indicatorcnt = defaultdict(lambda: 0)

def parse_responses(sentence, output_text):
    bad_sentence_indicators = ["estoy aquí para ayudarte", "mejorar tu ortografía",
                               "a frase que proporcionaste contiene varios errores",
                               "o puedo crear contenido que promueva la discriminación",
                               "o puedo crear ni compartir contenido que promueva",
                               "i nombre es Salamandra",
                               "reescribe la oración para corregir errores tipográficos",
                               "corregir errores tipográficos",
                               "reescribir la oración",
                               ". . . . . . . . ",
                               "contiene varios errores tipográficos",
                               "or favor, reescrib", "Explanation: ",
                               "or favor, rectifique",
                               "e utiliza el diacrítico",
                               "n qué puedo ayudarte o en qué podemos conversar juntos",
                               "o puedo proporcionar ayuda en la creación de contenido",
                               "roporciona la oración que",
                               "oy Salamandra",
                               "hay errores tipográficos",
                               "una oración para corregir",
                               "e reescribo para corregir",
                               "o entiendo la oración que",
                               "contiene una errata ortográfica",
                               "realizado una revisión ortográfica",
                               "que contenga errores tipográficos",
                               "el error tipográfico",
                               "e corrigió a",
                               "xplicación de los cambios",
                               "para reflejar la ortografía",
                               "e corrigieron los siguientes errores",
                               "e agregó un",
                               "e eliminó el",
                               "por los errores tipográficos",
                               "oy un modelo lingüístico",
                               "La oración corregida es:",
                               "No se ha realizado ningún cambio",
                               "corrigió el error estilístico",
                               "los errores ortográficos",
                               "información para reescribir",
                               "algún error en el formato",
                               "oración que contiene errores tipográficos",
                               "proporciona la oración completa"
                               ]
    unnecessary_bits_in_output = ["La respuesta correcta sería ", "Output: ", "OUTPUT: ",
                                  "Correct answer: ", "La oración corregida sería: ", "La oración corregida sería: ",
                                  "La respuesta correcta es: ", "La frase correcta sería", "La oración corregida es: ",
                                  "La respuesta corregida sería: "
                                  ]
    global indicatorcnt
    response = output_text
    # response = output_text[output_text.find('OUTPUT:\n'):]
    for end_marker in ["<eos>", "<|eot_id|>", "```<|end_of_text|>", "<end_of_turn>", "<|endoftext|>"]:
        if response.endswith(end_marker):
            response = response[0:(-(len(end_marker)))]
    # print()
    # print(f">>>{sentence}<<<")
    #
    responses = response.split("\n")
    # print("--",responses)
    responses = [response for response in responses if not (
            response.strip() in ['', 'OUTPUT:', 'INPUT:'] or response.startswith(
        'Rephrasing ') or response.startswith('Here are '))]
    new = list()
    for x in responses:
        for string in unnecessary_bits_in_output:
            x = x.replace(string, "")
        if x.startswith("OUTPUT:"):
            new.append(x.replace("OUTPUT: ",""))
        elif True in [indicator in x for indicator in bad_sentence_indicators]:
            for indicator in bad_sentence_indicators:
                if indicator in x:
                    indicatorcnt[indicator] += 1
            pass
        else:
            new.append(x)
    responses = new
    responses = [x.rstrip("\"\'") for x in responses]
    responses = [x.lstrip("\"\'") for x in responses]

    responses = [response.lstrip('1234567890-').lstrip('.').lstrip() for response in responses]
    responses = [x for x in responses if x != '']
    # print(responses)
    # print()
    if len(responses) > 0:
        return responses[-1]
    else:
        return ''


class cowsl_evaluator(object):

    def __init__(self):
        # pattern_annotation = re.compile(r"[\[\]\{\}<>\W]*\[([^\]]*)\]\{([^\}]*)\}<([^>]*)>")
        self.pattern_annotation = re.compile(r"\[([^\]]*)\]\{([^\}]*)\}<([^>]*)>")
        pattern_annotation_pure = re.compile(r"\[[^\]]*\]\{[^\}]*\}<[^>]*>")
        pattern_word = re.compile(r"\b[\w\*]+\b")
        pattern_punctuation = re.compile(r"[^\w\s]+")
        pattern_substitued_word = re.compile(r" \*[\w]+\* ")
        self.pattern_word_or_annotation = re.compile(pattern_annotation_pure.pattern
                                                + "|" + pattern_substitued_word.pattern
                                                + "|" + pattern_word.pattern
                                                + "|" + pattern_punctuation.pattern)
        # pattern_2_annotations = re.compile(pattern_annotation.pattern + " " + pattern_annotation.pattern)
        self.ngramHits = {
            1: {"yes": 0, "no": 0},
            2: {"yes": 0, "no": 0},
            3: {"yes": 0, "no": 0},
            4: {"yes": 0, "no": 0},
            5: {"yes": 0, "no": 0},
            6: {"yes": 0, "no": 0},
            7: {"yes": 0, "no": 0},
        }

    def _tokenizeWords(self, text):
        return self.pattern_word_or_annotation.findall(text)

    def tokenize_sentence(self, sentence):
        return self._tokenizeWords(sentence)


    def annotated2lists(self, annotated_sent):
        wordlist_annotated = self._tokenizeWords(annotated_sent)
        # print("wordlist_annotated", wordlist_annotated)
        # print()
        # print(corrected_doc)
        # print()
        # print(annotated_sent)
        # print()
        # split_text = annotated_sent.split(" ")
        # print()
        # print()
        # print("spacy")
        # print(type(nlp(annotated_doc)))
        # print(list(nlp(annotated_doc)))
        # print()
        # print(wordlist)

        words_orig_str_annotated = []
        words_uncorrected_annotated = []
        words_corrected_annotated = []
        annotation_tags_annotated = []
        for word in wordlist_annotated:
            word = word.strip()
            match = self.pattern_annotation.search(word)
            if match:
                words_orig_str_annotated.append(word)
                words_uncorrected_annotated.append(match.group(1))
                words_corrected_annotated.append(match.group(2))
                annotation_tags_annotated.append(match.group(3))
            else:
                words_orig_str_annotated.append(word)
                words_uncorrected_annotated.append(word)
                words_corrected_annotated.append(word)
                annotation_tags_annotated.append(None)

        return (words_orig_str_annotated, words_uncorrected_annotated, words_corrected_annotated,
                annotation_tags_annotated)

    def eval_on_annotated(self, predicted, annotated):
        print("Predicted:", predicted)
        print("Annotated:", annotated)
        returnval = [0, 0]
        for i in zip(predicted, annotated):

            predicted_sent = i[0]
            annotated_sent = i[1]
            print(annotated_sent)
            if annotated_sent:
                (words_orig_str_annotated, words_uncorrected_annotated, words_corrected_annotated,
                 annotation_tags_annotated) = self.annotated2lists(annotated_sent)
                words_predicted = self.tokenize_sentence(predicted_sent)
                words_predicted = [i.lower() for i in words_predicted]
                unigrams_predicted = []
                bigrams_predicted = []
                trigrams_predicted = []
                for idx in range(len(words_predicted)):
                    unigrams_predicted.append(tuple([words_predicted[idx]]))
                    if idx >= 1:
                        bigrams_predicted.append((words_predicted[idx - 1], words_predicted[idx]))
                    if idx >= 2:
                        trigrams_predicted.append((words_predicted[idx - 2], words_predicted[idx - 1],
                                                   words_predicted[idx]))

                unigrams_predicted = set(unigrams_predicted)
                bigrams_predicted = set(bigrams_predicted)
                trigrams_predicted = set(trigrams_predicted)

                ngrams_predicted = (unigrams_predicted.union(bigrams_predicted)).union(trigrams_predicted)

                # unigrams_annotated = set(unigrams_annotated)
                # bigrams_annotated = set(bigrams_annotated)
                # trigrams_annotated = set(trigrams_annotated)

                print("words_corrected_annotated", words_corrected_annotated)

                unigrams_ann_only_including_annotated = []
                bigrams_ann_only_including_annotated = []
                trigrams_ann_only_including_annotated = []
                for idx in range(len(words_corrected_annotated)):
                    if annotation_tags_annotated[idx] is not None:
                        # print(words_corrected_annotated[idx], annotation_tags_annotated[idx])
                        unigrams_ann_only_including_annotated.append(tuple([words_corrected_annotated[idx]]))
                        if idx >= 1:
                            bigrams_ann_only_including_annotated.append((words_corrected_annotated[idx - 1], words_corrected_annotated[idx]))
                            if idx < len(words_corrected_annotated) - 1:
                                print( "A",
                                    (words_corrected_annotated[idx-1], words_corrected_annotated[idx],
                                     words_corrected_annotated[idx+1]))
                                trigrams_ann_only_including_annotated.append(
                                    (words_corrected_annotated[idx-1], words_corrected_annotated[idx],
                                     words_corrected_annotated[idx+1]))
                        if idx < len(words_corrected_annotated)-1:
                            bigrams_ann_only_including_annotated.append(
                                (words_corrected_annotated[idx], words_corrected_annotated[idx + 1]))
                        if idx >= 2:
                            print("B", (words_corrected_annotated[idx - 2], words_corrected_annotated[idx - 1],
                                                       words_corrected_annotated[idx]))
                            trigrams_ann_only_including_annotated.append((words_corrected_annotated[idx - 2], words_corrected_annotated[idx - 1],
                                                       words_corrected_annotated[idx]))
                        if idx < len(words_corrected_annotated)-2:
                            trigrams_ann_only_including_annotated.append(
                                (words_corrected_annotated[idx], words_corrected_annotated[idx+1], words_corrected_annotated[idx+2]))

                unigrams_ann_only_including_annotated = set(unigrams_ann_only_including_annotated)
                bigrams_ann_only_including_annotated = set(bigrams_ann_only_including_annotated)
                trigrams_ann_only_including_annotated = set(trigrams_ann_only_including_annotated)


                # print("unigrams", len(unigrams_ann_only_including_annotated), unigrams_ann_only_including_annotated)
                # print("bigrams", len(bigrams_ann_only_including_annotated), bigrams_ann_only_including_annotated)
                # print("trigrams", len(trigrams_ann_only_including_annotated), trigrams_ann_only_including_annotated)

                ngrams_including_annotated_element = ((unigrams_ann_only_including_annotated
                                                      .union(bigrams_ann_only_including_annotated))
                                                      .union(trigrams_ann_only_including_annotated))

                print("ngrams_including_annotated_element", ngrams_including_annotated_element)

                #Eliminate empty elements in the solution "Yo había" -> "había"
                print("Correct ngrams for empty elements")
                newset = set()
                for ngram in list(ngrams_including_annotated_element):
                    print(type(ngram), ngram)
                    if "" in list(ngram):
                        # print(ngram)
                        newNgramAsList =[i for i in list(ngram) if i != ""]
                        if len(newNgramAsList) > 0:
                            newset.add(tuple(newNgramAsList))
                    else:
                        newset.add(ngram)
                ngrams_including_annotated_element = newset


                #correcte cases where a word is changed into more than one in the correction annotation like permitió - > había permitido
                for ngram in list(ngrams_including_annotated_element):
                    newngram = []
                    changed = False
                    for word in ngram:
                        word = word.strip()
                        if " " in word:
                            if ngram in ngrams_including_annotated_element:
                                ngrams_including_annotated_element.remove(ngram)
                                # print(word.split(" "))
                            newngram += word.split(" ")
                            changed = True
                        else:
                            newngram.append(word)
                    if changed:
                        ngrams_including_annotated_element.add(tuple(newngram))

                #############

                print("ngrams_including_annotated_element", ngrams_including_annotated_element)
                print("ngrams_predicted", ngrams_predicted)
                for elm in ngrams_including_annotated_element:
                    if len(elm) <= 7:
                        print(len(elm), elm, elm in ngrams_predicted)
                        if elm in ngrams_predicted:
                            self.ngramHits[len(elm)]["yes"] += 1
                        else:
                            self.ngramHits[len(elm)]["no"] += 1

                # match = pattern_annotation.findall(annotated_sent)
                # # print(">>>",match2)
                # for m in match:
                #     print(type(m), m)
                #     # searchstring = f"{m[0]} {m[2]} {m[4]}"
                #     searchstring = f"{m[1]}"
                #     print(searchstring)
                #     # if re.search(searchstring, predicted_sent):
                #     if searchstring in corrected_doc:
                #         returnval[0] += 1
                #         print("YES")
                #     else:
                #         returnval[1] += 1
                #         print("NO")
                #
                #     # match2 = pattern_annotation.search(m)
                #     # print(type(match2), match2)
                print()

        print(returnval)
        # return(returnval[0]/(returnval[1]+returnval[0])*100)

    def get_weighted_ngram_precision(self):
        return statistics.mean([
                self.ngramHits[1]["yes"] / (self.ngramHits[1]["yes"] + self.ngramHits[1]["no"]),
                self.ngramHits[2]["yes"] / (self.ngramHits[2]["yes"] + self.ngramHits[2]["no"]),
                self.ngramHits[3]["yes"] / (self.ngramHits[3]["yes"] + self.ngramHits[3]["no"]),
        ])


if __name__ == '__main__':
    evaluator = cowsl_evaluator()
    sys.path.append(os.getcwd())  # Add the current directory to the places where python looks for libraries
    from datareaders.read_cowsl import read_cowsl
    path = "/Users/stefan/data/iDEM/writing_assistant/cowsl2h-master/"
    data = read_cowsl(path)
    print(len(data))
    [print(x, "\n") for x in data]
    for i in data:
        if len(i) < 3:
            print(i)
    annotated=[x[2] for x in data]
    corrected=[x[1] for x in data]
    evaluator.eval_on_annotated(predicted=annotated, annotated=annotated)
