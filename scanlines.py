import sys
import re 

def main(argv):
    print(argv)
    lines = []
    with open(argv[0], newline='') as textfile:
        lines = textfile.readlines()

    for line in lines:
        result = re.match('\D\w', line)
        if(result is not None and line != '\n'):
            print(line.replace('\n',''))


if __name__ == "__main__":
    main(sys.argv[1:])     