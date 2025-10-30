import os
import json


class Conll2003:
    def __init__(self):
        self.lines = []  # Intended to consist for tuples (word, label)
        self.labels = set()  # Intended to contain all labels (with B- I- tags)

    def load_from_conllfile(self, filepath):
        with open(filepath, "r", encoding="utf-8") as inf:
            for line in inf.readlines():
                try:
                    data = line.split()
                    token = data[0]
                    label = data[-1]  # NER label must be last, if >2 elems in data
                except IndexError:
                    token = ""
                    label = ""
                self.lines.append((token, label))

    def write_output(self, filepath):
        with open(filepath, "w", encoding="utf-8") as outf:
            for line in self.lines:
                line2write = "{} {}".format(line[0], line[1])
                line2write = line2write.strip()
                if (line2write != line[1]) or (len(line[1]) == 0):
                    outf.write("{}\n".format(line2write))

    def update_labels_file(self, filepath):
        # Intended to be used when writing new file to update labels.txt if
        # there are new labels in current conll2003 file.
        def load_labels(label_path):
            with open(label_path, "r", encoding="utf-8") as labelfile:
                for line in labelfile.readlines():
                    line = line.strip()
                    self.labels.add(line)

        def extract_unseen_labels_from_current_file():
            new_labels = set()
            for line in self.lines:
                label = line[1]
                if label and label not in self.labels:
                    new_labels.add(label)
            return new_labels

        def save_labels(label_path):
            with open(label_path, "w", encoding="utf-8") as labelfile:
                for label in self.labels:
                    if not label.strip():
                        continue
                    labelfile.write("{}\n".format(label))

        data_dir = os.path.dirname(filepath)
        label_path = os.path.join(data_dir, "labels.txt")
        try:
            load_labels(label_path)
        except Exception as e:
            print(e)
            self.labels = set()
        new_labels = extract_unseen_labels_from_current_file()
        if new_labels:
            self.labels = self.labels.union(new_labels)
            save_labels(label_path)


class JSONLines:
    """
    JSONLines
    format is as follows:
    a sentence of text in each line, with annotations

    {"text":"Google was founded on September 4, 1998, by Larry Page and Sergey Brin.","entities":[[0, 6, "ORG"],[22, 39, "DATE"],[44, 54, "PERSON"],[59, 70, "PERSON"]]}
    {"text":"Another sentence", "entities":[]}
    This is somewhat similar to WebAnno tsv format
    """

    def __init__(self):
        # Intended to consist of tuples (line, (labels))
        # Where each label is a dict {"start_offset":int,"end_offset":int,"label":"str"}
        self.lines = []

    def load_from_jsonlines(self, filepath):
        with open(filepath, "r", encoding="utf-8") as inf:
            for line in inf.readlines():
                try:
                    data = json.loads(line)
                    self.lines.append(data)
                except Exception as e:
                    print(f"Exception in file {filepath}")
                    print(e)

    def write_output(self, filepath):
        with open(filepath, "w", encoding="utf-8") as outf:
            for line in self.lines:
                line = json.dumps(line)
                outf.write("{}\n".format(line))


def convert(sourcefile, trgfile):
    lines = JSONLines()
    lines.load_from_jsonlines(sourcefile)
    conllfile = Conll2003()

    for line in lines.lines:
        if len(conllfile.lines) > 0:
            conllfile.lines.append(("", ""))
        text = line["text"]
        ents = line["label"]
        curtok = ""
        inent = False
        label = "O"
        biolabel = "O"
        for pos in range(len(text)):
            for start, end, ent_label in ents:
                if end == pos:
                    if not inent:
                        print(f"File {sourcefile}")
                        print(
                            f"Found end of entity without start. Possible overlap? Line was {line}"
                        )
                    # curtok += text[pos]
                    if curtok:
                        conllfile.lines.append((curtok, biolabel))
                    inent = False
                    curtok = ""
                    label = "O"
                    biolabel = "O"
                if start == pos:
                    if inent:  # No overlap
                        continue
                    if curtok:
                        conllfile.lines.append((curtok, biolabel))
                    curtok = ""
                    label = ent_label
                    biolabel = "B-" + label
                    inent = True

            if text[pos].isspace():
                if curtok:
                    conllfile.lines.append((curtok, biolabel))
                if label != "O":
                    biolabel = "I-" + label
                curtok = ""
                continue
            curtok += text[pos]
        conllfile.lines.append((curtok, biolabel))
    conllfile.write_output(trgfile)

    labelfile = os.path.join(os.path.dirname(trgfile), "labels.txt")
    with open(labelfile, "w") as f:
        f.write("")
    conllfile.update_labels_file(labelfile)
