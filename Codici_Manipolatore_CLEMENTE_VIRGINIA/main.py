#!/usr/bin/env python3
from ev3dev2.motor import LargeMotor, MediumMotor, OUTPUT_A,  OUTPUT_B,  OUTPUT_C,  OUTPUT_D #, SpeedPercent
from ev3dev2.power import PowerSupply
import numpy as np
from time import sleep

m1 = LargeMotor(OUTPUT_A) # motore 1 porta A
m2 = LargeMotor(OUTPUT_B) # motore 2 porta B
m3 = LargeMotor(OUTPUT_C) # motore 3 porta C
m4 = MediumMotor(OUTPUT_D) # motore 4 porta D

m1.reset 
m2.reset
m3.reset
m4.reset

a= PowerSupply() 
Qd = np.loadtxt('Qd_funzionale.txt') #carico file con i Qd calcolati in simulazione con funzionale
print(len(Qd)) #201

angles = np.zeros((len(Qd),4)) #inizializzo la matrice degli angoli
length = len(Qd)

for i in range(0, length, 1): #cambiare il counter per velocizzare
    Qd[i, 0] = np.degrees(Qd[i, 0])
    Qd[i, 1] = np.degrees(Qd[i, 1])
    Qd[i, 2] = np.degrees(Qd[i, 2])
    Qd[i, 3] = np.degrees(Qd[i, 3])

for i in range(0, length, 1):
    m1.on_to_position(10, Qd[i,0], brake=True, block=True) # Gira fino all'angolo desiderato con velocitá pari al 10% della velocitá del motore e con brake sto andando a 'frenare' elettronicamente i motori
    angles[i,0] = m1.position

    m2.on_to_position(10, Qd[i,1],brake=True, block=True)
    angles[i,1] = m2.position

    m3.on_to_position(10, Qd[i,2],brake=True, block=True)
    angles[i,2] = m3.position 

    m4.on_to_position(10, Qd[i,3],brake=True, block=True)
    angles[i,3] = m4.position
    #sleep(.1) 

np.savetxt('angoli.txt', angles)