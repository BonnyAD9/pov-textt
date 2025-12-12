Název projektu: Automatic text transcription (OCR)

Našim úkolem je natrénovat a zhodnotit neuronovou síť, která bude rozpoznávat
text. Natrénovaná neuronová síť bude použitelná z jednoduché CLI aplikace,
která na vstup dostane obrázek obsahující jednořádkový text a na výstupu bude
rozpoznaný text.

Implementačním jazykem bude Python s využitím knihovny PyTorch.

Pro implementaci využijeme CRNN, kdy pro základ využijeme předtrénovanou síť
ResNet 18. Jako dataset využijeme dataset z projektu PERO, konkrétně
"Handwriting Adaptation Dataset". Část datasetu vymezíme pro validaci a zbylou
část pro trénování neuronové sítě. Jako Loss funkci využijeme CTC. Nakonec náš
dosažený výsledek porovnáme s existujícími OCR knihovnami, například Tesseract.