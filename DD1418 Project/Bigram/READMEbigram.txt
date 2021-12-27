1. Lägg mappen "textfiler" (eller en kopia ifall man vill behålla ursprungliga datan) tillsammans med python-filerna
translator_textstäd.py, translator_bitrainer.py, translator_npbidecoder.py i en mapp.


Det behövs inga parametrar när man kör någon av koderna.

2. Kör programmet translator_textstäd.py. Den kommer rengöra .txt-filerna i mappen textfiler.



3. Kör programmet translator_bitrainer.py som kommer spara två filer: word_stats_bi.txt och bigram_probs.txt.

Om man vill kan man ändra hur många av filerna som används som träningskorpus genom att ändra variabeln "n_files" i main.
Vid inlämning är n_files=200 av totalt 6517 txt-filer.



4. Kör programmet translator_npbidecoder.py som kommer läsa in filerna från steg 3 och därefter initiera A- och B-matriser.
OBS! När A- och B-matriser initieras första gången så kommer programmet spara matriserna som .npy-filer. n_files = 200
ger .npy-filer som tar cirka 7GB minne vardera. Om detta inte är önskvärt är det enkelt att ta bort den kod-biten längst
ner i konstruktion och endast behålla raderna "self.init_a(filename)", "self.init_b()".

Efter en kort stund är programmet färdig-initierat och ber om en mening på engelska. Det tar väldigt lång tid att köra
så korta meningar är att rekommendera. För varje ord i meningen ber programmet därefter om alternativa översättningar för
det ordet. Observationssannolikheter i B-matrisen för ordet och dess alternativ kommer därefter att uppdateras och slutligen
sker viterbi-avkodning. Efter att resultatet skrivits ut ber programmet om en ny mening men man bör ha i åtanke att
B-matrisen inte nollställs mellan körningar så därför bör man starta om programmet.

Exempel-inpput:
I.
"Enter the english sentence": and he are
II.
Enter other words for "and": also including
Enter other words for "he": him
Enter other words for "are": is
Output:
and he is
