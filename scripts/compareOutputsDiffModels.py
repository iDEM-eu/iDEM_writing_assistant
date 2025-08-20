import os, sys
import statistics

import pandas as pd

# from functools import reduce
sys.path.append(os.getcwd()) #Add the current directory to the places where python looks for libraries
from metrics.wagnerfischer import levenshtein
from metrics.cowsl_ann_bigram_overlap import cowsl_evaluator, parse_responses, dates2tags, indicatorcnt


evaluator = cowsl_evaluator()
dataframes = list()

df_support = None

firstFrame = True
for i in sys.argv[1:]:
    # print()
    df = pd.read_csv(i, sep='\t', on_bad_lines='skip')
    df_support = df.copy()
    df_support = df_support.drop(columns=['Prediction'])
    df = df.rename(columns={"Prediction": i.replace("outputs/", "").replace("_cuda.tsv", "")})
    if firstFrame:
        df = df.rename(columns={"First Reference": "Reference"})
    else:
        df = df.drop(["Annotation", "First Reference", 'docId'], axis=1)
    firstFrame = False
    df.memory_usage(index=True).sum()
    dataframes.append(df)


# maindf = reduce(lambda left, right: pd.merge(left, right, how='left', on='Original'), [maindf, i])

all_data = dict()
for df in dataframes:
    for index, row in df.iterrows():
        original_sent = row['Original']
        if original_sent not in all_data:
            all_data[original_sent] = dict()
        for colname in df.columns:
            all_data[original_sent][colname] = row[colname]

NonSysOutputs = ["Annotation", "docId"]
for i in all_data.keys():
    if str(i) != "nan":
        print("Original", dates2tags(i), sep="\t")
        distances = list()
        wordcnt = 0
        for colname in all_data[i]:
            if colname != "Original":
                content = all_data[i][colname]
                # print(f"<{content}>")
                if str(content) != "nan":
                    if colname not in NonSysOutputs:
                        content = content.replace("<NEWLINE>", "\n")
                        content = "\n".join(content.split("<NEWLINE>"))
                        content = parse_responses(i, content)
                        content = dates2tags(content)
                        print(colname, content, end="", sep="\t")
                        print("\t", end="")
                        leng = len(evaluator.tokenize_sentence(str(content)))
                        editdist = levenshtein(str(content), str(all_data[i]["Original"]))
                        if len(content) > 0:
                            if colname != "Reference":
                                distances.append(editdist)
                                wordcnt += leng
                            print("EditD:", editdist, end="\t")
                            if wordcnt > 0:
                                print("WER:", editdist/wordcnt, end="")
                    else:
                        print(colname, dates2tags(str(content)), end="", sep="\t")
                    print()
        if len(distances) > 0 and wordcnt > 0:
            print("AV EditD:", statistics.mean(distances))
            print("AV WER:", statistics.mean(distances)/ wordcnt)
        print()

for i in indicatorcnt.keys():
    print(i, indicatorcnt[i], sep="\t")
# print("Start Merging")
#
# maindf = dataframes.pop(0)
# print(maindf.columns)

# for i in dataframes:
#     print(i.memory_usage(index=True).sum())
#     print(i.columns)
#     print(maindf.memory_usage(index=True).sum())
#     print(maindf.columns)
#     print("\ta",maindf.columns)
#     print("\t\tb",i.columns)
#     for j in maindf.columns:
#         maindf = maindf[maindf[j].notnull()]
#     maindf = maindf.merge(i, how='left', on='Original',
#                       copy=False)
#     for j in maindf.columns:
#         maindf = maindf[maindf[j].notnull()]
#     print("\t\t\tc",maindf.columns)
#     maindf = maindf.reset_index(drop=True)
#
#
# # maindf = maindf.merge(df_support, how='left', on='Original', copy=False)
# print("maindf.columns after merges", maindf.columns)
#
# maindf = maindf.reset_index(drop=True)
#
#
# # print(maindf.columns)
# # print(maindf.head())
# cols = maindf.columns

# for index, row in maindf.iterrows():
#     for i in cols:
#         print(i, row[i])
#     print()

