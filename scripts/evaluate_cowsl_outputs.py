import statistics
import sys, os, csv
sys.path.append(os.getcwd()) #Add the current directory to the places where python looks for libraries
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from metrics.wagnerfischer import levenshtein
from statistics import mean

# from starling.core.similarity import SimilarityBLEURT
from datareaders.read_cowsl import read_cowsl
from metrics.cowsl_ann_bigram_overlap import cowsl_evaluator, parse_responses, dates2tags

LLM_rubbish=['Aquí te dejo cinco reformulaciones diferentes para la oración dada',
             'No puedo proporcionar ayuda en la creación de contenido que promueva la explotación sexual de menores',
             'Lo siento, pero no puedo cumplir con esa solicitud.',
             r'\|',
             'Por favor, te proporciono las siguientes reformulaciones para la oración dada',
             'No puedo proporcionar ayuda en este tema.',
             'Lo siento, pero no puedo proporcionar ayuda',
             'No proporcioné información para reescribir.']

tag_backtranslations = {'*ciudad*' : "*CITY*",
                        '[CITY]' : "*CITY*",
                        "*nombre completo*" : "*FIRST_NAME",
                        "*apellidos*" : "*LAST_NAME",
                        "*institución educativa*" : "*UNIVERSITY*",
                        "*Estado*" : "*STATE*",
                        }

edit_d_vals = list()
wer_vals = list()

class eval_cowsl(object):
    def __init__(self):
        self.docID2Info = dict()
        self.sent2Info = list()
        self.evaluator = cowsl_evaluator()

    def read_cowsl_out_file(self,filename, debug=False):
        with open(filename, newline='', encoding="utf-8") as file:
            tsv_reader = csv.reader(file, delimiter='\t')
            # tsv_reader = csv.reader(codecs.open(file, "r", "utf-8"), delimiter='\t')
            for row in tsv_reader:
                if len(row) == 5:
                    orig = dates2tags(row[0])
                    sys_out = row[1]
                    sys_out = "\n".join(sys_out.split("<NEWLINE>"))
                    sys_out = dates2tags(parse_responses(orig,sys_out))
                    ref_text = dates2tags(row[3])
                    annotated_text = dates2tags(row[2])
                    textID = row[4]
                    if debug:
                        print("____")
                    wordListSysOut = self.evaluator.tokenize_sentence(sys_out)
                    wordListSysOut = [i.lower() for i in wordListSysOut]
                    wordListRef = self.evaluator.tokenize_sentence(ref_text)
                    wordListRef = [i.lower() for i in wordListRef]
                    wordListOrig = self.evaluator.tokenize_sentence(orig)
                    (_, _, words_corrected_from_annotated, _) = self.evaluator.annotated2lists(annotated_text)
                    words_corrected_from_annotated = [i.lower() for i in words_corrected_from_annotated]
                    if debug:
                        print(int(levenshtein(sys_out, ref_text)))
                    levenstein_sys2ref = int(levenshtein(wordListSysOut,
                                          wordListRef))
                    levenstein_sys2ann = int(levenshtein(wordListSysOut,
                                          words_corrected_from_annotated))
                    if debug:
                        print(orig)
                        print("sys_out",sys_out)
                        print("ref_text", ref_text)
                        print(wordListRef)
                        print("annotated_text", annotated_text)
                        print(words_corrected_from_annotated)
                    self.sent2Info.append({"orig": orig, "sys_out": sys_out, "orig_list": wordListOrig,
                                        "annotated_text_list" : words_corrected_from_annotated,
                                        "ref_text": ref_text, "ref_text_list": wordListRef, "textID": textID,
                                        "annotated_text": annotated_text, "sys_out_list": wordListSysOut})
                    if len(wordListRef) > 0:
                        if debug:
                            print("LEVENSTEIN SYS2REF", levenstein_sys2ref)
                            print("WER SYS2REF:", levenstein_sys2ref/len(wordListRef))
                        edit_d_vals.append(levenstein_sys2ref)
                        wer_vals.append(levenstein_sys2ref/len(wordListRef))
                    if len(words_corrected_from_annotated) > 0:
                        if debug:
                            print("LEVENSTEIN SYS2ANN", levenstein_sys2ann)
                            print("WER SYS2ANN:", levenstein_sys2ann/len(words_corrected_from_annotated))
                        edit_d_vals.append(levenstein_sys2ann)
                        wer_vals.append(levenstein_sys2ann/len(words_corrected_from_annotated))

    def filter_LMM_rubbish(self):
        newSent2Info = []
        for i in self.sent2Info:
            isOK = True
            for j in LLM_rubbish:
                if j in i["sys_out"]:
                    isOK = False
            if isOK:
                newSent2Info.append(i)
        self.sent2Info = newSent2Info

    def backtranslate_tags(self):
        for i in self.sent2Info:
            for j in tag_backtranslations.keys():
                if j in i["sys_out"]:
                    i["sys_out"] = i["sys_out"].replace(j, tag_backtranslations[j])
                    # print(">>>", j, "-----", i["sys_out"])
                    pass

    def reconstruct_docs(self):
        self.docID2Info = dict()
        current_id = None
        for i in self.sent2Info:
            if current_id != i["textID"]:
                if current_id in self.docID2Info:
                    pass
                    # print(self.docID2Info[current_id], "looks like a duplicate ID", file=sys.stderr)
            if i["textID"] not in self.docID2Info:
                self.docID2Info[i["textID"]] = {"orig": "",
                                                "orig_list": [],
                                               "sys_out": "",
                                                "sys_out_list": [],
                                               "ref_text": "",
                                                "ref_text_list": [],
                                               "annotated_text": "",
                                                "reference_extracted_from_annotated_list": []}
            self.docID2Info[i["textID"]]["orig"] += " " + i["orig"]
            self.docID2Info[i["textID"]]["orig_list"] += i["orig_list"]
            self.docID2Info[i["textID"]]["sys_out"] += " " + i["sys_out"]
            self.docID2Info[i["textID"]]["sys_out_list"] += i["sys_out_list"]
            self.docID2Info[i["textID"]]["ref_text"] += " " + i["ref_text"]
            self.docID2Info[i["textID"]]["ref_text_list"] += i["ref_text_list"]
            self.docID2Info[i["textID"]]["annotated_text"] += " " + i["annotated_text"]
            extracted_ref_from_annotated = i["annotated_text_list"]
            self.docID2Info[i["textID"]]["reference_extracted_from_annotated_list"] += extracted_ref_from_annotated
            current_id = i["textID"]

    def eval_on_annotated(self):
        list_sys_out = []
        list_ref_texts = []
        list_ref_texts_from_annotated = []

        print("Start sents")

        for i in self.sent2Info:
            print(i)
            sys_out = i["sys_out"]
            ref_text = i["ref_text"]
            ref_text_wordlist = i["ref_text_list"]
            annotated_text = i["annotated_text"]
            print("annotated_text", annotated_text)
            words_corrected_from_annotated = i["annotated_text_list"]
            wordListSysOut = self.evaluator.tokenize_sentence(sys_out)
            print(wordListSysOut)
            print(words_corrected_from_annotated)
            list_sys_out.append(wordListSysOut)
            list_ref_texts_from_annotated.append([words_corrected_from_annotated])
            list_ref_texts.append([ref_text_wordlist])
            # bleu_score = sentence_bleu([words_corrected_from_annotated], wordListSysOut)
            print("__________")
            # print("SBLEU:", bleu_score)
            self.evaluator.eval_on_annotated([sys_out], [annotated_text])
        print(self.evaluator.ngramHits)
        print("Average ngram precision (3):", self.evaluator.get_weighted_ngram_precision())
        print("Corpus BLEU to Annotated:", corpus_bleu(list_ref_texts_from_annotated, list_sys_out))
        print("Corpus BLEU to Refs (sent):", corpus_bleu(list_ref_texts, list_sys_out))
        print("Average levensthein distance per sent (both):", mean(edit_d_vals))
        print("Average WER values per sent (both)", mean(wer_vals))

        # print("Start doc")

        list_sys_out_doc = []
        list_ref_texts_doc = []
        list_ref_texts_from_annotated_doc = []
        list_both_refs = []
        for i in self.docID2Info.keys():
            list_sys_out_doc.append(self.docID2Info[i]["sys_out_list"])
            list_ref_texts_doc.append([self.docID2Info[i]["ref_text_list"]])
            list_ref_texts_from_annotated_doc.append([self.docID2Info[i]["reference_extracted_from_annotated_list"]])
            list_both_refs.append([self.docID2Info[i]["ref_text_list"], self.docID2Info[i]["reference_extracted_from_annotated_list"]])
        #     print(i)
        #     print(self.docID2Info[i]["reference_extracted_from_annotated_list"])
        #     print(self.docID2Info[i]["orig"])
        #     print(self.docID2Info[i]["sys_out"])
        #     print(self.docID2Info[i]["ref_text"])
        #     print(self.docID2Info[i]["annotated_text"])
        #     self.evaluator.eval_on_annotated([self.docID2Info[i]["sys_out"]],
        #                                      [self.docID2Info[i]["annotated_text"]])
        # print(self.evaluator.ngramHits)
        print("Corpus Bleu to Annotated, doc level:",  corpus_bleu(list_ref_texts_from_annotated_doc, list_sys_out_doc))
        print("Corpus Bleu to Reference, doc level:",  corpus_bleu(list_ref_texts_doc, list_sys_out_doc))
        print("Corpus Bleu to both, doc level:",  corpus_bleu(list_both_refs, list_sys_out_doc))

    def av_n_gram_annotated_vs_references(self):
        print("Calculating Ceiling")
        eval2 = cowsl_evaluator()
        for i in self.sent2Info:
            print(i)
            sys_out = i["sys_out"]
            ref_text = i["ref_text"]
            ref_text_wordlist = i["ref_text_list"]
            annotated_text = i["annotated_text"]
            print("annotated_text", annotated_text)
            words_corrected_from_annotated = i["annotated_text_list"]
            wordListSysOut = self.evaluator.tokenize_sentence(sys_out)
            print(wordListSysOut)
            print(words_corrected_from_annotated)
            # bleu_score = sentence_bleu([words_corrected_from_annotated], wordListSysOut)
            print("__________")
            # print("SBLEU:", bleu_score)
            eval2.eval_on_annotated([ref_text], [annotated_text])
        print("Ceiling average ngram precision (3):", eval2.get_weighted_ngram_precision())



if __name__ == "__main__":
    cowsl = eval_cowsl()
    cowsl.read_cowsl_out_file(sys.argv[1])
    cowsl.filter_LMM_rubbish()
    cowsl.backtranslate_tags()
    cowsl.reconstruct_docs()
    cowsl.eval_on_annotated()
    # cowsl.av_n_gram_annotated_vs_references()
