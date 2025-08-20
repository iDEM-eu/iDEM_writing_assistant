import glob
import os
import pathlib
import re
import sys
# import spacy
# nlp = spacy.load("es_core_news_sm")


def get_text_from_file(filename):
    with open(filename) as f:
        return f.read()

def read_cowsl(path):
    print("reading cowsl", path)
    all_essays = glob.glob(path + "**/essays/**/*.txt", recursive=True)
    print("number of essays: ", len(all_essays))

    corrected_files = glob.glob(path + "*/*/*/*corrected.txt", recursive=True)
    annotated_files = glob.glob(path + "*/*/*/*annotated_verbs.txt", recursive=True)

    id2files = {}
    for f in corrected_files:
        # print(f)
        path_parts = pathlib.Path(f).parts
        fileId = os.path.basename(f).split(".")[0]
        related_files = [x for x in all_essays if fileId in x]
        essay_file = None
        for g in related_files:
            # print("\t",g)
            if "essay" in g:
                if path_parts[-3] in g:
                    if path_parts[-4] in g:
                        # print("\t\t",g)
                        essay_file = g
                        break
        if essay_file is not None:
            id2files[fileId] = {}
            id2files[fileId]["essay"] = essay_file
            id2files[fileId]["corrected"] = f
        else:
            print("No essay file found for", fileId, file=sys.stderr)
            print(related_files)

    for f in annotated_files:
        path_parts = pathlib.Path(f).parts
        fileId = os.path.basename(f).split(".")[0]
        related_files = [x for x in all_essays if fileId in x]
        ann_file = None
        for g in related_files:
            if "essay" in g:
                if path_parts[-3] in g:
                    if path_parts[-4] in g:
                        ann_file = g
                        break
        if ann_file is not None:
            if fileId not in id2files:
                id2files[fileId] = {}
                id2files[fileId]["essay"] = ann_file
            id2files[fileId]["annotated"] = f
        else:
            print("No essay file found for annotated file", fileId, file=sys.stderr)
            print(related_files)

    return_data = []
    for i in id2files:
        entry = ["", "", "", i]
        entry[0] = get_text_from_file(id2files[i]['essay'])
        entry[1] = get_text_from_file(id2files[i]['corrected'])
        if "annotated" in id2files[i]:
            entry[2] = get_text_from_file(id2files[i]['annotated'])
        return_data.append(entry)
    return return_data


if __name__ == "__main__":
    path = "~/data/iDEM/writing_assistant/cowsl2h-master"
    data = read_cowsl(path)
    print(len(data))
    # [print(x, "\n") for x in data]
    for i in data:
        if len(i) < 3:
            print(i)
    annotated=[x[2] for x in data]
    corrected=[x[1] for x in data]
