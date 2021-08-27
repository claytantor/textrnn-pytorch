
import string
import os, sys
import argparse

if __name__ == "__main__":
    

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--in", action="store",
        required=True, dest="infile", help="Input file") 


    parser.add_argument("-o", "--out", action="store",
        required=False, dest="outfile", help="Out file") 

    args = parser.parse_args()    

    # load text
    file = open(args.infile, 'rt')
    text = file.read()
    file.close()

    tokens = text.split()
    # remove punctuation from each word

    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]


    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]


    #value_if_true if condition else value_if_false
    tokens = ["friend" if word.lower() == "jew" else word for word in tokens]
    
	# make lower case
    tokens = [word.lower() for word in tokens]

    tokens = ["I" if word.lower() == "i" else word for word in tokens]
    tokens = ["you will" if word.lower() == "youll" else word for word in tokens]
    


    print(' '.join(tokens))