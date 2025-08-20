# Prompt based on https://aclanthology.org/2023.emnlp-main.821/
prompts = {"en-multi":
               "Please rewrite the following complex sentence in order to make it easier to understand by non-native "
               "speakers of the language. You can do so by replacing complex words with simpler synonyms (i.e. paraphrasing), deleting "
               "unimportant information (i.e. compression), and/or splitting a long complex sentence into several simpler ones. "
               "The final simplified sentence needs to be grammatical, fluent, and retain the main ideas of its original counterpart "
               "without altering its meaning. Make sure the output is in the same language as the original.\n"
               "Return five different rephrasings, separated by newline. Do not generate any text except the reformulations.",
           "en":
               "Please rewrite the following complex sentence in order to make it easier to understand by non-native "
               "speakers of English. You can do so by replacing complex words with simpler synonyms (i.e. paraphrasing), "
               "deleting unimportant information (i.e. compression), and/or splitting a long complex sentence into "
               "several simpler ones. The final simplified sentence needs to be grammatical, fluent, and retain the "
               "main ideas of its original counterpart without altering its meaning. \n"
               "Return five different rephrasings, separated by newline. Do not generate any text except the reformulations.",
           "es":
               "Por favor, reescriba la siguiente oración compleja para que sea más fácil de entender para quienes no "
               "hablan español como lengua materna. Puede hacerlo reemplazando palabras complejas con sinónimos más "
               "simples (es decir, parafraseando), eliminando información irrelevante (es decir, comprimiendo) o "
               "dividiendo una oración compleja larga en varias más simples. La oración simplificada final debe ser "
               "gramaticalmente correcta, fluida y conservar las ideas principales de su contraparte original sin "
               "alterar su significado. \n"
               "Da cinco reformulaciones diferentes, separadas por saltos de línea. No genere ningún texto "
               "excepto las reformulaciones.",
           "ca":
               "Si us plau, reescriviu la següent frase complexa per tal que sigui més fàcil d'entendre per a parlants "
               "no nadius del català. Podeu fer-ho substituint paraules complexes per sinònims més simples (és a dir, "
               "parafrasejant), eliminant informació no important (és a dir, compressió) i/o dividint una frase llarga "
               "i complexa en diverses de més simples. La frase simplificada final ha de ser gramatical, fluida i "
               "conservar les idees principals de la seva contrapart original sense alterar-ne el significat. \n"
               "Dona cinc reformulacions diferents, separades per salt de línia. No genereu cap text excepte les "
               "reformulacions.",
           "it":
               "Riscrivi la seguente frase complessa per renderla più comprensibile a chi non è madrelingua italiano. È "
               "possibile farlo sostituendo le parole complesse con sinonimi più semplici (un modo di farlo potrebbe "
               "essere parafrasando), eliminando le informazioni non importanti (ad esempio, riducendo il testo) e/o "
               "suddividendo una frase complessa lunga in più frasi più semplici. La frase semplificata finale deve"
               " essere grammaticalmente corretta, scorrevole e mantenere le idee principali della sua controparte"
               " originale senza alterarne il significato.\nRestituisci cinque diverse riformulazioni, separate da una"
               " nuova riga. Non generare altro testo ad eccezione delle riformulazioni.",
           "gr":
               "Παρακαλώ γράψε ξανά την ακόλουθη σύνθετη πρόταση για να την κατανοήσουν πιο εύκολα οι ομιλητές των "
               "ελληνικών που δεν έχουν μητρική γλώσσα την ελληνική. Μπορείς να το κάνεις αυτό αντικαθιστώντας σύνθετες "
               "λέξεις με απλούστερα συνώνυμα (παράφραση), διαγράφοντας ασήμαντες πληροφορίες (συμπίεση) ή/και "
               "χωρίζοντας μια μεγάλη σύνθετη πρόταση σε περισσότερες από μία απλούστερες προτάσεις. Η τελική "
               "απλοποιημένη πρόταση πρέπει να είναι γραμματικά σωστή, εύγλωττη και να διατηρεί τις κύριες ιδέες της "
               "αρχικής της πρότασης χωρίς να αλλοιώνει το νόημά της.\nΕπέστρεψε πέντε διαφορετικές αναδιατυπώσεις, "
               "διαχωρισμένες με νέα γραμμή (κάθε αναδιατύπωση σε ξεχωριστή γραμμή). Μην γράψεις κανένα άλλο κείμενο "
               "εκτός από τις αναδιατυπώσεις.",
           "fa":
               "جمله پیچیده زیر را طوری بازنویسی کن تا برای افراد غیرفارسی زبان قابل فهم‌تر باشد. می‌توانی این کار را با جایگزینی کلمات پیچیده با مترادف‌های ساده‌تر (ساده‌سازی)، حذف اطلاعات بی‌اهمیت (فشرده‌سازی) و یا تقسیم یک جمله پیچیده طولانی به چندین جمله ساده‌تر انجام دهی. جمله ساده‌شده نهایی باید از نظر دستوری، درست و روان باشد و ایده‌های اصلی جمله اصلی را بدون تغییر معنی حفظ کند."
               "پنج بازنویسی مختلف را که با خط جدید از هم جدا شده‌اند، ارائه کن و هیچ متن دیگری به جز بازنویسی‌ها ایجاد نکن.",
           "fr":
               "Veuillez réécrire la phrase complexe suivante afin de la rendre plus compréhensible pour les "
               "non-francophones. Vous pouvez le faire en remplaçant les mots complexes par des synonymes plus simples "
               "(c’est-à-dire, en paraphrasant des mots), en supprimant les informations superflues (c’est-à-dire, en "
               "compressant le texte) et/ou en divisant une phrase longue et complexe en plusieurs phrases plus simples. "
               "La phrase simplifiée finale doit être grammaticalement correcte, fluide et conserver les idées "
               "principales de la phrase originale sans en altérer le sens.\nRenvoyez cinq reformulations différentes, "
               "séparées par un retour à la ligne. Ne générez aucun texte autre que les reformulations.",
           "ar":  # manually checked, but still needs to find a proper word for "compression"
                "يرجى إعادة كتابة الجملة المعقدة التالية لتسهيل فهمها على غير الناطقين بالعربية. يمكنك القيام بذلك عن طريق استبدال الكلمات المعقدة بمرادفات أبسط (مثل إعادة الصياغة)، وحذف المعلومات غير المهمة (مثل التلخيص )، و/أو تقسيم جملة معقدة طويلة إلى عدة جمل أبسط. يجب أن تكون الجملة المبسطة النهائية سليمة نحويًا، وسلسة، وتحتفظ بالأفكار الرئيسية  للجملة  الأصلية دون تغيير معناها." + "\n" + " أرجو إعادة صياغتها بخمس طرق مختلفة، كل واحدة في سطر منفصل. لا تُنتِج أي نص آخر باستثناء الصياغات الجديدة.",
           "de":
               "Bitte formuliere den folgenden komplexen Satz um, um ihn für Nicht-Muttersprachler verständlicher "
               "zu machen. Dies erreichst du, indem Sie komplexe Wörter durch einfachere Synonyme ersetzt "
               "(Paraphrasierung), unwichtige Informationen entfernst (Komprimierung) und/oder einen langen, komplexen "
               "Satz in mehrere einfachere Sätze aufteilst. Der endgültige vereinfachte Satz muss grammatikalisch "
               "korrekt und flüssig sein und die Kernaussagen des ursprünglichen Satzes unverändert beibehalten.\nGib "
               "fünf verschiedene Umformulierungen zurück, getrennt durch Zeilenumbrüche. Generiere außer den "
               "Umformulierungen keinen weiteren Text.",
               
           }
