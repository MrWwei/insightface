#! encoding: utf-8

import os
import random

class GeneratePairs:
    """
    Generate the pairs.txt file that is used for training face classifier when calling python `src/train_softmax.py`.
    Or others' python scripts that needs the file of pairs.txt.
    Doc Reference: http://vis-www.cs.umass.edu/lfw/README.txt
    """

    def __init__(self, data_dir, pairs_filepath, img_ext):
        """
        Parameter data_dir, is your data directory.
        Parameter pairs_filepath, where is the pairs.txt that belongs to.
        Parameter img_ext, is the image data extension for all of your image data.
        """
        self.data_dir = data_dir
        self.pairs_filepath = pairs_filepath
        self.img_ext = img_ext


    def generate(self):
        self._generate_matches_pairs()
        self._generate_mismatches_pairs()


    def _generate_matches_pairs(self):
        """
        Generate all matches pairs
        """
        test = os.listdir(self.data_dir)
        for name in os.listdir(self.data_dir):
            suffix = os.path.splitext(name)[1].lower()
            if suffix == ".xml":
                continue

            a = []
            for file in os.listdir(self.data_dir + name):
                suffix = os.path.splitext(file)[1].lower()
                if suffix == ".png":
                    a.append(file)

            with open(self.pairs_filepath, "a") as f:
                for i in range(2):
                    temp = random.choice(a).split("_") # This line may vary depending on how your images are named.
                    w = temp[0]
                    l = data_dir + w + '/' + random.choice(a)

                    if l=='':
                        l='0'
                    r = data_dir + w + '/' + random.choice(a)
                    if r == '':
                        r = '0'
                    #f.write(w + "\t" + str(l) + "\t" + str(r) + "\n")
                    f.write(l + '\t' + r + '\t' + str(1) + '\n')


    def _generate_mismatches_pairs(self):
        """
        Generate all mismatches pairs
        """
        for i, name in enumerate(os.listdir(self.data_dir)):
            suffix = os.path.splitext(name)[1].lower()
            if suffix == ".xml":
                continue
            a = []
            for file in os.listdir(self.data_dir + name):
                suffix = os.path.splitext(file)[1].lower()
                if suffix == ".png":
                    a.append(file)
            remaining = os.listdir(self.data_dir)
            remaining = [f_n for f_n in remaining if f_n != ".xml"]
            # del remaining[i] # deletes the file from the list, so that it is not chosen again
            other_dir = random.choice(remaining)
            b = []
            for file in os.listdir(self.data_dir + other_dir):
                suffix = os.path.splitext(file)[1].lower()
                if suffix == ".png":
                    b.append(file)
            with open(self.pairs_filepath, "a") as f:
                for i in range(1):
                    file1 = random.choice(a)
                    file2 = random.choice(b)
                    picid1 = data_dir + name + '/' + file1
                    if picid1=='':
                        picid1 = '0'
                    picid2 = data_dir + other_dir + '/' + file2
                    if picid2 == '':
                        picid2 = '0'
                    f.write(picid1 + "\t" + picid2 + "\t" + str(0) + '\n')
                #f.write("\n")


if __name__ == '__main__':
    data_dir = "/home/heisai/Pictures/output/"
    pairs_filepath = "pairs.txt"
    img_ext = ".png"
    generatePairs = GeneratePairs(data_dir, pairs_filepath, img_ext)
    generatePairs.generate()