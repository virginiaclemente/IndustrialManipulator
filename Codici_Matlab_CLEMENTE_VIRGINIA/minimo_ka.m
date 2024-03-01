clc
clear all

T = readtable('errore_funzionale_matrix1.csv');
A=table2array(T);
[indice, minimo]=mink(A, 10);