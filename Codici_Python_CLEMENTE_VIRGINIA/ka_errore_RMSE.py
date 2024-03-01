import numpy as np
import matplotlib.pyplot as plt
import math

# Matrice di trasformazione omogenea
def DH_computation(d, a, alpha, theta):
    T = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                 [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                 [0, np.sin(alpha), np.cos(alpha), d],
                 [0, 0, 0, 1]])
    return T

# Cinematica diretta con quaternione unitario
def DirectKinematics(T):

    # Estraggo dalla trasformazione omogenea il vettore traslazione
    x = T[0:3,3]

    # Estraggo dalla trasformazione omogenea il quaternione unitario, potrei anche non farlo?
    sgn_1 = np.sign(T[2,1]-T[1,2])
    sgn_2 = np.sign(T[0,2]-T[2,0])
    sgn_3 = np.sign(T[1,0]-T[0,1])

    eta = np.sqrt(T[0,0]+T[1,1]+T[2,2]+1)/2
    eps = 0.5*np.array([sgn_1*np.sqrt(T[0,0]-T[1,1]-T[2,2]+1),
                        sgn_2*np.sqrt(-T[0,0]+T[1,1]-T[2,2]+1),
                        sgn_3*np.sqrt(-T[0,0]-T[1,1]+T[2,2]+1)])

    Q = np.hstack([eta, eps]) # Q = [eta, epsilon]

    return x, Q 

#Pianificazione di traiettoria in SO di due segmenti consecutivi 
def CartesianPlanner(pi, pf, ti, tf, t): # richiamare nel main la funzione due volte
    s = len(pi) # contiene 3 elementi x, y e z

    if t<ti: # tempi precedenti all'istante iniziale
        pd = pi # #posizione assunta dal manipolatore all'istante del tempo i-esimo che mi coincide con posizione iniziale pi
        pdot = np.zeros(s) # velocitá al ti é nulla, derivata di una cost = 0

    elif t<tf: # tempi inferiori al tempo finale
        A = np.array([[ti**3, ti**2, ti, 1], # calcolo la matrice A usando il polinmio cubico in cui specifico, non imposto vincoli su accelerazione
                      [tf**3, tf**2, tf, 1], # abbiamo quindi 4 righe
                      [3*ti**2, 2*ti, 1, 0],
                      [3*tf**2, 2*tf, 1, 0]])
        
        b = np.array([[0], [np.linalg.norm(pf-pi)], [0], [0]]) #vettore b
        
        x = np.linalg.solve(A, b) #coefficienti del polinomio
        
        s = x[0]*t**3+x[1]*t**2+x[2]*t+x[3] # polinomio di 3 grado
        sdot = 3*x[0]*t**2+2*x[1]*t+x[2] #derivata di s = velocitá

        pd = pi + (s/np.linalg.norm(pf-pi))*(pf-pi) # definizione di segmento nello spazio p(s) 
        pdot = (sdot/np.linalg.norm(pf-pi))*(pf-pi) # velocitá e quindi derivata pd

    else: # si verifica per t = tf 
        pd = pf             # posizione é uguale alla posizione desiderata
        pdot = np.zeros(s)  # velocitá torna ad essere 0 -> manipolatore fermo

    return pd, pdot

# Calcolo dello Jacobiano Geometrico perché a noi interessa solo la posizione e quindi in termini di posizione coincide con analitico
def Jacobian(q, a, d, alpha):

    T01 = DH_computation(d[0], a[0], alpha[0], q[0])
    T12 = DH_computation(d[1], a[1], alpha[1], q[1])
    T22p = DH_computation(d[2], a[2], alpha[2], q[2]) 
    T2p3 = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]) #rotazione fittizia 2 -> 2'
    T34 = DH_computation(d[3], a[3], alpha[3], q[3])

    T02 = np.matmul(T01, T12) # La funzione numpy.matmul() restituisce il prodotto matrice di due array, quindi in questo caso restituisce il prodotto delle due matrici di rototraslazione tra parentesi dandomi la matrice di rototraslazione risultante
    T02p = np.matmul(T02, T22p)
    T03 = np.matmul(T02p, T2p3)
    T04 = np.matmul(T03, T34)

    # Estraggo dal primo blocco della matrice di rototraslazione i vettori z_i-1 (versore asse di rotazione) -> vettori riga
    z0 = np.array([0,0,1])
    z1 = T01[0:3,2]                            
    z2 = T02[0:3,2]
    z3 = T03[0:3,2]

    # Estraggo dall'ultima cololnna i vettori p_i-1 (vettore posizione) 
    p0 = np.array([0,0,0])
    p1 = T01[0:3,3]
    p2 = T02[0:3,3]
    p3 = T03[0:3,3]
    pe = T04[0:3,3] #posizione End Effector

    # Considero solo la parte di posizionamento dello Jacobiano (Jp) 
    J1 = np.cross(z0,pe-p0)
    J2 = np.cross(z1,pe-p1)
    J3 = np.cross(z2,pe-p2)
    J4 = np.cross(z3,pe-p3)

    J = np.vstack((J1, J2, J3, J4)) # 4x3 (J1 nella prima riga, J2 nella seconda etc.)

    return J.T # lo jacobiano é una matrice 3x4 -> J1, J2, J3 e J4 sono vettori colonna.

# Inversione cinematica SENZA funzionale di costo (centro corsa)
def InversioneCinematica(q, a, d, alpha, pos_d, pos_d_dot, k, algorithm):

    # Calcolo della posa attuale
    T01 = DH_computation(d[0], a[0], alpha[0], q[0])
    T12 = DH_computation(d[1], a[1], alpha[1], q[1])
    T22p = DH_computation(d[2], a[2], alpha[2], q[2])
    T2p3 = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    T34 = DH_computation(d[3], a[3], alpha[3], q[3])

    T02 = np.matmul(T01, T12) 
    T02p = np.matmul(T02, T22p)
    T03 = np.matmul(T02p, T2p3)
    T04 = np.matmul(T03, T34)

    # Calcolo della posa corrente
    P,_ = DirectKinematics(T04)  # DirectKinematics resistituisce P e Q ma ci interessa solo P (posizione organo terminale) quindi metto _ al posto di Q

    # Calcolo dell'errore in posizione nello SO 1x3
    err = pos_d - P   # posizione desiderata - posizione corrente      

    # Matrice dei guadagni 3x3
    K = k*np.eye(3)         # k é uno scalare (guadagno) e lo moltiplico per una matrice diagonale

    # Errore x guadagno proporzionale
    Err = np.matmul(K,err)
   
    # Calcolo dello Jacobiano geometrico 
    J = Jacobian(q, a, d, alpha)

    if algorithm == "t":
        # Inversione con la trasposta: q_dot = JT*K*e
        Q_dot = np.matmul(J.T,Err)
    else:
        # Inversione con l'inversa: q_dot = Jinv*(pos_d_dot + K*e)
        Jinv = np.linalg.pinv(J) # calcolo la pseudo inversa
        Err2 = pos_d_dot + Err #  velocità desiderata + errore, che rappresenta la formula dell'inversione con l'inversa dello jacobiano, senza il termine di ridondanza
        Q_dot = np.matmul(Jinv,Err2) #calcolo della velocità desiderata S.G. come prodotto tra Jinv e Err2

    return Q_dot, err

# Calcolo del differenziale della funzione obiettivo
def d_W_q_distanza_centrocorsa(q, q_min, q_max): #da minimizzare -> derivate = 0
    #vado già a considerare le derivate rispetto a q--> svolgiamo il quadrato di binomio e andiamo a derivare dalla definizione della funzione obiettivo omega
    a11=((1/(q_min[0]-q_max[0])**2)*(q[0]-2+q_min[0]+q_max[0]))/8
    a22=((1/(q_min[1]-q_max[1])**2)*(q[1]-2+q_min[1]+q_max[1]))/8
    a33=((1/(q_min[2]-q_max[2])**2)*(q[2]-2+q_min[2]+q_max[2]))/8
    a44=((1/(q_min[3]-q_max[3])**2)*(q[3]-2+q_min[3]+q_max[3]))/8
    d_W_q=np.array([a11,a22,a33,a44]) #vettore riga
    return d_W_q


# Inversione cinematica CON funzionale di costo (centro corsa)
def InversioneCinematica_centrocorsa(q, a, d, alpha, pos_d, pos_d_dot, k, ka, algorithm):

    # Calcolo della posa attuale
    T01 = DH_computation(d[0], a[0], alpha[0], q[0])
    T12 = DH_computation(d[1], a[1], alpha[1], q[1])
    T22p = DH_computation(d[2], a[2], alpha[2], q[2])
    T2p3 = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    T34 = DH_computation(d[3], a[3], alpha[3], q[3])

    T02 = np.matmul(T01, T12) # La funzione numpy.matmul() restituisce il prodotto matrice di due array, quindi in questo caso restituisce il prodotto delle due matrici di rototraslazione tra parentesi dandomi la matrice di rototraslazione risultante
    T02p = np.matmul(T02, T22p)
    T03 = np.matmul(T02p, T2p3)
    T04 = np.matmul(T03, T34)

    P,_ = DirectKinematics(T04) # ,_ prende solo la posizione e non Q

    # Calcolo dell'errore 1x3
    err = pos_d - P        

    # Matrici dei guadagni
    K = k*np.eye(3)         # 3x3 (perché moltiplica err)
    Err = np.matmul(K,err)  # errore x guadagno proporzionale
    Ka = ka*np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])      # 4x4 (perché moltiplica qa)

    dW = d_W_q_distanza_centrocorsa(q, q_m, q_M)
    qa_dot = np.matmul(Ka, dW)
    # Calcolo dello Jacobiano geometrico == analitico
    J = Jacobian(q, a, d, alpha)   # 3x4
    Jpseudinv = np.linalg.pinv(J) #4x3 pseudoinversa destra
    Jprod=np.matmul(Jpseudinv,J) # 4x4 pseudoinversa destra per J 
    Proiettore = (np.eye(4)-Jprod) # 4x4 PROIETTORE nel NULLO 
    # utilizzo l'inversa, seppur con un onere computazionale maggiore rispetto alla trasposta, riesco ad ottenere un errore più piccolo
    
    # Calcolo delle velocità desiderate nello SG
    if algorithm == "t":
        # Inversione con la trasposta: q_dot = JT*K*e
        Q_dot = np.matmul(J.T,Err) + np.matmul(Proiettore,qa_dot)                                            #gestione ridondanza anche qui?????

    else:
        # Inversione con la pseudoinversa dello Jacobiano: q_dot = Jpinv*(pos_d_dot + K*e)+(I-JpinvJ)*qa_dot
        Err_ = pos_d_dot + Err
        Q_dot = np.matmul(Jpseudinv,Err_) + np.matmul(Proiettore,qa_dot)
    return Q_dot, err

if __name__ == '__main__':

    #definisco i tempi simulazione
    t0 = 0    
    dt = 0.1
    t1 = 10
    t2 = 20

    #vettori tempo
    time1 = np.arange(t0, t1, dt)
    time2 = np.arange(t1, t2, dt)
    time = np.concatenate((time1,time2), axis = 0)

    #lunghezza link e parametri DH
    l1 = 0.145
    l2 = 0.12
    l3 = 0.06
    l4 = 0.025
    l5 = 0.045
    lmax = l1+l2+l3+l4
    a = [l1, l2, l3, l4]
    d = [0, 0, 0, l5]
    alpha = np.array([-np.pi/2, np.pi/2, 0, 0])

    # Posizione iniziale manipolatore = punto A che coincide con la configurazione a riposo
    qi=np.array([np.pi/2, 0,-np.pi*3/4,0]) # configurazione a riposo

    T01 = DH_computation(0, l1, -np.pi/2, qi[0])
    T12 = DH_computation(0, l2, np.pi/2, qi[1])
    T22p = DH_computation(0, l3, 0, qi[2]) 
    T2p3 = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]) #rotazione fittizia 2 -> 2'
    T34 = DH_computation(l5, l4, 0, qi[3])

    T02 = np.matmul(T01, T12) 
    T02p = np.matmul(T02, T22p)
    T03 = np.matmul(T02p, T2p3)
    T04 = np.matmul(T03, T34)

    XY1, _ = DirectKinematics(T01)
    XY2, _ = DirectKinematics(T02)
    XY3, _ = DirectKinematics(T03)
    pA, _ = DirectKinematics(T04) #punto di partenza = punto iniziale A
    print('pA:', pA)

    #punto centrale B [m]
    pB = np.array([0.163, 0.215, 0.035])
    # punto finale C [m] 
    pC = np.array([-0.010, 0.269, 0.035]) 

    # Inizializzo posizione e velocità nello spazio dei giunti = sto andando a creare le matrici vuote
    # senza funzionale
    Qd = np.zeros((len(time)+1,4))    # posizione nello spazio dei giunti -> ho 4 variabili di giunto e quindi 4 colonne
    Qdot = np.zeros((len(time)+1,4))  # velocitá nello spazio dei giunti
    # con funzionale
    Qd_f =np.zeros((len(time)+1, 4))
    Qdot_f =np.zeros((len(time)+1, 4))
    # valori variabili di giunto e derivate all'istante iniziale
    Qd[0,:] = qi                      # posizione iniziale = alla configurazione di partenza -> riempo prima riga
    Qdot[0,:] = np.array([0,0,0,0])   # velocitá iniziale impostata a 0
    Qd_f[0,:]= np.array([np.pi/2, 0,-np.pi*3/4,0])
    Qdot_f[0,:]=np.array([0,0,0,0])
    
    # Inizializzazione delle variabili nello spazio operativo.
    Pd = np.zeros((len(time)+1,3))      # posizione desiderata
    Pddot = np.zeros((len(time)+1,3))   # velocitá desiderata
    P_ee = np.zeros((len(time)+1,3))    # Posizione End Effector
    P_ee_f = np.zeros((len(time)+1,3))  # Posizione End Effectori con funzionale
    
    # Valori posizione e velocitá istante iniziale
    Pd[0,:] = pA
    Pddot [0, :] = np.array([0, 0, 0]) # velocitá nulla perché istante iniziale
    P_ee[0, :] = pA
    P_ee_f[0, :] = pA

    # Posizione dei giunti senza funzionale
    P_1 = np.zeros((len(time)+1,3))
    P_2 = np.zeros((len(time)+1,3))
    P_3 = np.zeros((len(time)+1,3))

    # Posizione dei giunti con funzionale
    P_1_f = np.zeros((len(time)+1,3))
    P_2_f = np.zeros((len(time)+1,3))
    P_3_f = np.zeros((len(time)+1,3))

    # Matrici degli errori 
    errore=np.zeros((len(time)+1, 3))
    errore_f = np.zeros((len(time)+1,3))

    errore[0,:]=np.array([0,0,0]) #istante iniziale errore é nullo
    errore_f[0,:] = np.array([0, 0, 0])

    errore_medio = np.zeros((len(time)+1,3))       #senza funzionale
    errore_medio_f = np.zeros((len(time)+1,2))

    #Limiti Spazio di Lavoro, variazione angolare min e max di ogni giunto
    q_m=np.array([-np.pi/6, -np.pi/2, -np.pi*3/2, -np.pi])
    q_M=np.array([np.pi*3/4, np.pi/2, np.pi/3, np.pi])  

    # Definisco i guadagni
    kgain = 19
    ka_vec = np.linspace(1,500,500)

    errore_funzionale_matrix = np.zeros((len(ka_vec),2))       #senza funzionale
    errore_funzionale_matrix[0,:] = np.array([0, 0])


    # Pongo il contatore a 1 (perchè dopo le condizioni iniziali)
    counter_ka=0
    for ka in ka_vec:

        counter = 1

        for t in time:
            # devo leggere la posizione attuale dei giunti Q e salvarla in un vettore

            if t <= t1:
                # Pianificazione prima segmento (da A a B)
                pd, pdot = CartesianPlanner(pA, pB, t0, t1, t)
            else:
                # Pianificazione secondo segmento (da B a C)
                pd, pdot = CartesianPlanner(pB, pC, t1, t2, t)

            # Inversione cinematica senza gestione ridondanza
            Qdot[counter,:], errore[counter,:] = InversioneCinematica(Qd[counter-1,:], a, d, alpha, pd, pdot, kgain, "i")

            # Inversione cinematica con gestione ridondanza--> ricavo la qdot da integrare per trovare Q
            Qdot_f[counter,:], errore_f[counter,:] = InversioneCinematica_centrocorsa(Qd_f[counter-1,:], a, d, alpha, pd, pdot, kgain, ka, "i")
            
            # Integrazione numerica per ottenere Q da qdot
            Qd[counter,:] = Qd[counter-1,:] + dt*Qdot[counter,:]       # questa è la Qd da mandare ai motori per muoverli
            Qd_f[counter,:] = Qd_f[counter-1,:] + dt*Qdot_f[counter,:]

            # Salvo un posizioni per graficare
            Pd[counter,:] = pd            # counter - 1 perché parte da 0 non da 1
            Pddot[counter,:] = pdot

            #Calcolo cinematica della simulazione senza funzionale
            T01 = DH_computation(d[0], a[0], alpha[0], Qd[counter,0])
            T12 = DH_computation(d[1], a[1], alpha[1], Qd[counter,1])
            T22p = DH_computation(d[2], a[2], alpha[2], Qd[counter,2])
            T2p3 = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
            T34 = DH_computation(d[3], a[3], alpha[3], Qd[counter,3])        
            T02 = np.matmul(T01, T12)
            T02p = np.matmul(T02, T22p)
            T03 = np.matmul(T02p, T2p3)
            T04 = np.matmul(T03, T34)

            # Posizione dell'end effector e dei giunti SENZA funzionale
            P_1[counter,:],_ = DirectKinematics(T01)
            P_2[counter,:],_ = DirectKinematics(T02)
            P_3[counter,:],_ = DirectKinematics(T03)
            P_ee[counter,:],_ = DirectKinematics(T04)
            
            #Calcolo cinematica della simulazione CON funzionale
            T01_f = DH_computation(d[0], a[0], alpha[0], Qd_f[counter,0])
            T12_f = DH_computation(d[1], a[1], alpha[1], Qd_f[counter,1])
            T22p_f = DH_computation(d[2], a[2], alpha[2], Qd_f[counter,2])
            T2p3_f = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
            T34_f = DH_computation(d[3], a[3], alpha[3], Qd_f[counter,3])        
            T02_f = np.matmul(T01_f, T12_f)
            T02p_f = np.matmul(T02_f, T22p_f)
            T03_f = np.matmul(T02p_f, T2p3_f)
            T04_f = np.matmul(T03_f, T34_f)
            

            # Posizione giunti e dell'end effector CON funzionale
            P_1_f[counter,:],_ = DirectKinematics(T01_f)
            P_2_f[counter,:],_ = DirectKinematics(T02_f)
            P_3_f[counter,:],_ = DirectKinematics(T03_f)
            P_ee_f[counter,:],_ = DirectKinematics(T04_f)
        
            
            counter += 1



        # Errore Inversione Cinematica con funzionale
        erroref_x=0
        erroref_y=0
        erroref_z=0

        erroref_x= errore_f.T[0,:]#componente x mediata per tutti gli istanti di tempo
        erroref_y= errore_f.T[1,:]
        erroref_z= errore_f.T[2,:]

        MSE_1_f = np.square(np.subtract(erroref_x, 0)).mean()
        RMSE_1_f = math.sqrt(MSE_1_f)
        MSE_2_f = np.square(np.subtract(erroref_y, 0)).mean()
        RMSE_2_f = math.sqrt(MSE_2_f)
        MSE_3_f = np.square(np.subtract(erroref_z, 0)).mean()
        RMSE_3_f = math.sqrt(MSE_3_f)
        errore_medio_f=[ka,(RMSE_1_f + RMSE_2_f + RMSE_3_f)]

        errore_funzionale_matrix[counter_ka,:]= errore_medio_f
        counter_ka+=1


        print(counter_ka)
    print("la matrice di errore con funzionale è: ", errore_funzionale_matrix)
    np.savetxt("errore_funzionale_matrix1.csv", errore_funzionale_matrix, delimiter=",")

    print(" ")