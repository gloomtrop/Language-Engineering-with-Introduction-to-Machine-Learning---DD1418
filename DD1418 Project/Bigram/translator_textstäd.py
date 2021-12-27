import os
import string


class Cleaner(object):
    def __init__(self, filenames):
        self.__sources = filenames

    def clean_files(self):
        """
        Öppnar alla filer och låter clean_line rensa rader.
        Slutligen sparas (ersätter gamla) filen med rensad text.
        """

        files_read = 0
        for fname in self.__sources:
            with open(fname, encoding='utf8', errors='ignore') as f:
                t = []
                for line in f:
                    t.append(self.clean_line(line))
            f.close()
            t.remove("\n")
            t[:] = [x for x in t if x != "\n"]
            out = open(fname, "w")
            out.writelines(t)
            files_read += 1

    def clean_line(self, line):
        """
        Samma typ av enkla textrensning som implementerades i Random Indexing-laborationen.
        En mindre skillnad är att denna inte returnerar en sträng och inte en lista
        """
        junk = string.punctuation
        digits = string.digits
        for char in line:
            if char in junk or char in digits:
                line = line.replace(char, "").strip()
        cleaned_line = line.lower()
        return cleaned_line


if __name__ == '__main__':
    dir_name = "textfiler (kopia)"
    filenames = [os.path.join(dir_name, fn) for fn in os.listdir(
        dir_name)]  # Lägger alla filer från mappen "textfiler (kopia)" i en lista som kan itereras genom senare.
    cleaner = Cleaner(filenames)  # Skapar cleaner-objekter och lägger till filerna som attribut
    cleaner.clean_files()  # Enkel rensning av filer.
