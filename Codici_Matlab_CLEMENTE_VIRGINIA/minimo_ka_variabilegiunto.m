clc
clear all

T = readtable('distanza_dal_centrocorsa_matrix1.csv');
A=table2array(T);
[indice, minimo]=mink(A, 10);