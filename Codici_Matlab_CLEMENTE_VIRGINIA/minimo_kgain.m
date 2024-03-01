clc
clear all

T = readtable('errore_kgain_senza_funzionale_matrix1.csv');
Kg=table2array(T);
[indice, K]=mink(Kg, 10);