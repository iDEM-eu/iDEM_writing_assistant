import sys, os
import codecs
import torch
sys.path.append(os.getcwd()) #Add the current directory to the places where python looks for libraries
from core.similarity import SimilarityBLEURT
from evaluate_cowsl_outputs import eval_cowsl
import numpy as np

device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
for file in sys.argv[1:]:
    print(file)
    cowsl = eval_cowsl()
    cowsl.read_cowsl_out_file(file)
    cowsl.filter_LMM_rubbish()
    cowsl.backtranslate_tags()
    cowsl.reconstruct_docs()

    list_sys_out_doc = []
    list_ref_texts_doc = []
    list_ref_texts_from_annotated_doc = []
    list_both_refs = []
    for i in cowsl.docID2Info.keys():
        if cowsl.docID2Info[i]["sys_out"].lstrip() == "Prediction":
            continue
        list_sys_out_doc.append(cowsl.docID2Info[i]["sys_out"])
        list_ref_texts_doc.append(cowsl.docID2Info[i]["ref_text"])
        # print("reference_extracted_from_annotated_list", cowsl.docID2Info[i]["reference_extracted_from_annotated_list"])
        list_ref_texts_from_annotated_doc.append(" ".join(cowsl.docID2Info[i]["reference_extracted_from_annotated_list"]))
        list_both_refs.append(
            [cowsl.docID2Info[i]["ref_text"], " ".join(cowsl.docID2Info[i]["reference_extracted_from_annotated_list"])])

    # print(len(list_sys_out_doc))
    # print("list_ref_texts_doc[1]", list_ref_texts_doc[1])
    # print("len(list_ref_texts_doc)",len(list_ref_texts_doc))
    # print("list_ref_texts_doc[1]", list_ref_texts_doc[1])
    # print("len(list_sys_out_doc)",len(list_sys_out_doc))
    evaluation_data = [(list_ref_texts_doc[i],list_sys_out_doc[i]) for i in range(len(list_sys_out_doc))]
    # print("len(evaluation_data):", len(evaluation_data))
    # for c,i in enumerate(evaluation_data):
    #     print(c)
    #     for j in i:
    #         print("\t", j)
    similarity = SimilarityBLEURT(device)
    results = similarity.assess(evaluation_data)

    print(f"BLEURT for {file}")
    print("BLEURT over references (doc lev)", np.mean(results))

    evaluation_data = [(list_ref_texts_from_annotated_doc[i],list_sys_out_doc[i]) for i in range(len(list_sys_out_doc))]
    similarity = SimilarityBLEURT(device)
    results = similarity.assess(evaluation_data)

    print("BLEURT over annotated (doc lev)", np.mean(results))

    # evaluation_data = [(list_both_refs[i],list_sys_out_doc[i]) for i in range(len(list_sys_out_doc))]
    # similarity = SimilarityBLEURT(device)
    # results = similarity.assess(evaluation_data)
    #
    # print("BLEURT over annotated and references (doc lev)", np.mean(results))

    # import torch
    # from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
    #
    # config = BleurtConfig.from_pretrained('lucadiliello/BLEURT-20-D12')
    # model = BleurtForSequenceClassification.from_pretrained('lucadiliello/BLEURT-20-D12')
    # tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20-D12')
    #
    # references = ["a bird chirps by the window", "this is a random sentence"]
    # candidates = ["a bird chirps by the window", "this looks like a random sentence"]
    #
    # model.eval()
    # with torch.no_grad():
    #     inputs = tokenizer(references, candidates, padding='longest', return_tensors='pt')
    #     res = model(**inputs).logits.flatten().tolist()
    # print(res)
    #
    # config = BleurtConfig.from_pretrained('lucadiliello/BLEURT-20-D12')
    # model = BleurtForSequenceClassification.from_pretrained('lucadiliello/BLEURT-20-D12')
    # tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20-D12')
    #
    #
    # references = ["a bird chirps by the window", "this is a random sentence"]
    # candidates = ["a bird chirps by the window", "this looks like a random sentence"]
    #
    # model.eval()
    # with torch.no_grad():
    #     inputs = tokenizer(references, candidates, padding='longest', return_tensors='pt')
    #     res = model(**inputs).logits.flatten().tolist()
    # print(res)

    list_sys_out = []
    list_ref_texts = []
    list_ref_texts_from_annotated = []
    list_both_refs = []
    similarity = SimilarityBLEURT(device)
    # bleurttestdata= [("a bird chirps by the window","a bird chirps by the window"),
    #                  ("this is a random sentence","this looks like a random sentence"),
    #                  ("this is a random sentence.", "this looks like a random sentence."),
    #                  ("He estado en once países y treinta estados.","He estado en once países y treinta estados."),
    #                  ("Mi lugar favorito es Madrid, España.","Mi lugar favorito es Madrid, en España.")]
    #
    # references = [x[0] for x in bleurttestdata]
    # candidates = [x[1] for x in bleurttestdata]
    #
    # model.eval()
    # with torch.no_grad():
    #     inputs = tokenizer(references, candidates, padding='longest', return_tensors='pt')
    #     res = model(**inputs).logits.flatten().tolist()
    # print(1, res)
    #
    # model.eval()
    # with torch.no_grad():
    #     inputs = tokenizer(references[0:1], candidates[0:1], padding='longest', return_tensors='pt')
    #     res = model(**inputs).logits.flatten().tolist()
    # print(2, res)

    # references = [x[0][0] for x in bleurttestdata]
    # candidates = [x[1] for x in bleurttestdata]
    #
    # model.eval()
    # with torch.no_grad():
    #     inputs = tokenizer(references, candidates, padding='longest', return_tensors='pt')
    #     res = model(**inputs).logits.flatten().tolist()
    # print(3, res)

    # for i  in bleurttestdata:
    #     print(i)
    #     print("bleurt trough module:", similarity.assess([i]))
    #     with torch.no_grad():
    #         print(i)
    #         print("[i[0]][0]", [i[0][0]])
    #         inputs = tokenizer([i[0][0]], [i[1]], padding='longest', return_tensors='pt')
    #         res = model(**inputs).logits.flatten().tolist()
    #         print(res)

    for i in cowsl.sent2Info:
        if i["sys_out"].lstrip() == "Prediction":
            continue
        list_sys_out.append(i["sys_out"])
        list_ref_texts.append(i["ref_text"])
        # evaluation_data = [(list_ref_texts[i], list_sys_out[i]) for i in range(len(list_sys_out))]
        # with torch.no_grad():
        #     datapoint = evaluation_data[-1]
        #     inputs = tokenizer([datapoint[0]], [datapoint[1]], padding='longest', return_tensors='pt')
        #     res = model(**inputs).logits.flatten().tolist()
        # print(similarity.assess([([i["sys_out"]], [[i["ref_text"]]])]))
        # print("reference_extracted_from_annotated_list", i["annotated_text_list"])
        list_ref_texts_from_annotated.append(" ".join(i["annotated_text_list"]))
        list_both_refs.append([i["ref_text"], " ".join(i["annotated_text_list"])])

    evaluation_data = [(list_ref_texts[i],list_sys_out[i]) for i in range(len(list_sys_out))]
    print("len(evaluation_data):", len(evaluation_data))
    similarity = SimilarityBLEURT(device)
    results = similarity.assess(evaluation_data)

    print(f"BLEURT for {file}")
    print("BLEURT over references (sent lev)", np.mean(results))

    evaluation_data = [(list_both_refs[i],list_sys_out[i]) for i in range(len(list_sys_out))]
    similarity = SimilarityBLEURT(device)
    results = similarity.assess(evaluation_data)

    print("BLEURT over annotated and references (sent lev)", np.mean(results))
