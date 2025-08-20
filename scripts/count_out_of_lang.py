import sys, os
sys.path.append(os.getcwd()) #Add the current directory to the places where python looks for libraries
from evaluate_cowsl_outputs import eval_cowsl
from langdetect import detect
from collections import defaultdict



for file in sys.argv[1:]:
    print(file)
    cowsl = eval_cowsl()
    cowsl.read_cowsl_out_file(file)
    cowsl.filter_LMM_rubbish()
    cowsl.backtranslate_tags()
    cowsl.reconstruct_docs()

    langcnt = defaultdict(lambda: 0)

    for i in cowsl.sent2Info:
        if i["sys_out"].lstrip() == "Prediction":
            continue
        if i["sys_out"].lstrip().rstrip() != "":
            try:
                detected_lang = detect(i["sys_out"])
                langcnt[detected_lang] += 1
                if detected_lang != "es":
                    print(detected_lang, i["sys_out"])
            except:
                pass
    print("COUNT: ", file, langcnt)


