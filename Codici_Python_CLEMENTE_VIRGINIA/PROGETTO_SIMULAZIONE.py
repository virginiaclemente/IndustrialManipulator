import numpy as np
import matplotlib.pyplot as plt

#DEFINISCO LE FUNZIONI
def DH_computation(d, a, alpha, theta): # Matrice di rotazione con i parametri di DH
    T = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                 [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                 [0, np.sin(alpha), np.cos(alpha), d],
                 [0, 0, 0, 1]])
    return T

# Cinematica Diretta
def DirectKinematics(T):
    
    # Estraggo dalla trasformazione omogenea il vettore traslazione
    x = T[0:3,3] 

    # Estraggo dalla trasformazione omogenea il quaternione unitario (no problemi di singolaritá)
    sgn_1 = np.sign(T[2,1]-T[1,2]) #sgn(x) = 1 con x>=0 o 0 con x<0
    sgn_2 = np.sign(T[0,2]-T[2,0])
    sgn_3 = np.sign(T[1,0]-T[0,1])

    eta = np.sqrt(T[0,0]+T[1,1]+T[2,2]+1)/2
    eps = 0.5*np.array([sgn_1*np.sqrt(T[0,0]-T[1,1]-T[2,2]+1),
                        sgn_2*np.sqrt(-T[0,0]+T[1,1]-T[2,2]+1),
                        sgn_3*np.sqrt(-T[0,0]-T[1,1]+T[2,2]+1)])

    Q = np.hstack([eta, eps]) #hstack unisce due parametri in un unico vettore

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
    qa_dot = np.matmul(Ka, dW) #espressa attraverso il funzionale di costo ed é pari a Ka*dW(derivata di W(q) rispetto q)
    # Calcolo dello Jacobiano geometrico == analitico
    J = Jacobian(q, a, d, alpha)   # 3x4
    Jpseudinv = np.linalg.pinv(J) #4x3 pseudoinversa destra
    Jprod=np.matmul(Jpseudinv,J) # 4x4 pseudoinversa destra per J 
    Proiettore = (np.eye(4)-Jprod) # 4x4 PROIETTORE nel NULLO P = I - Jpseudoinversadx * J
        
    # Calcolo delle velocità desiderate nello SG
    if algorithm == "t":
        # Inversione con la trasposta: q_dot = JT*K*e
        Q_dot = np.matmul(J.T,Err) + np.matmul(Proiettore,qa_dot)

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
    XY4, _ = DirectKinematics(T04) #punto di partenza = punto iniziale A
    
    pA = XY4
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

    Qd_f_6 =np.zeros((len(time)+1, 4))
    Qdot_f_6 =np.zeros((len(time)+1, 4))
    Qd_f_6[0,:]= np.array([np.pi/2, 0,-np.pi*3/4,0])
    Qdot_f_6[0,:]=np.array([0,0,0,0])
    Qd_f_3 =np.zeros((len(time)+1, 4))
    Qdot_f_3 =np.zeros((len(time)+1, 4))
    Qd_f_3[0,:]= np.array([np.pi/2, 0,-np.pi*3/4,0])
    Qdot_f_3[0,:]=np.array([0,0,0,0])
    Qd_f_300 =np.zeros((len(time)+1, 4))
    Qdot_f_300 =np.zeros((len(time)+1, 4))
    Qd_f_300[0,:]= np.array([np.pi/2, 0,-np.pi*3/4,0])
    Qdot_f_300[0,:]=np.array([0,0,0,0])

    # Inizializzazione delle variabili nello spazio operativo.
    Pd = np.zeros((len(time)+1,3))      # posizione desiderata
    Pddot = np.zeros((len(time)+1,3))   # velocitá desiderata
    P_ee = np.zeros((len(time)+1,3))    # Posizione End Effector
    P_ee_f = np.zeros((len(time)+1,3))  # Posizione End Effectori con funzionale
    P_ee_f_3 = np.zeros((len(time)+1,3))
    P_ee_f_6= np.zeros((len(time)+1,3))
    P_ee_f_300 = np.zeros((len(time)+1,3))
    
    # Valori posizione e velocitá istante iniziale
    Pd[0,:] = pA
    Pddot [0, :] = np.array([0, 0, 0]) # velocitá nulla perché istante iniziale
    P_ee[0, :] = pA
    P_ee_f[0, :] = pA
    P_ee_f_3[0, :] = pA
    P_ee_f_6[0, :] = pA
    P_ee_f_300[0, :] = pA


    # Posizione dei giunti senza funzionale
    P_1 = np.zeros((len(time)+1,3))
    P_2 = np.zeros((len(time)+1,3))
    P_3 = np.zeros((len(time)+1,3))

    # Posizione dei giunti con funzionale
    P_1_f = np.zeros((len(time)+1,3))
    P_2_f = np.zeros((len(time)+1,3))
    P_3_f = np.zeros((len(time)+1,3))
    P_1_f_3 = np.zeros((len(time)+1,3))
    P_2_f_3 = np.zeros((len(time)+1,3))
    P_3_f_3 = np.zeros((len(time)+1,3))
    P_1_f_6 = np.zeros((len(time)+1,3))
    P_2_f_6= np.zeros((len(time)+1,3))
    P_3_f_6= np.zeros((len(time)+1,3))
    P_1_f_300 = np.zeros((len(time)+1,3))
    P_2_f_300 = np.zeros((len(time)+1,3))
    P_3_f_300 = np.zeros((len(time)+1,3))
    
    # Matrici degli errori 
    errore=np.zeros((len(time)+1, 3))
    errore_f = np.zeros((len(time)+1,3))
    errore_f_6 = np.zeros((len(time)+1,3))
    errore_f_3 = np.zeros((len(time)+1,3))
    errore_f_300 = np.zeros((len(time)+1,3))


    errore[0,:]=np.array([0,0,0]) #istante iniziale errore é nullo
    errore_f[0,:] = np.array([0, 0, 0])
    errore_f_6[0,:] = np.array([0, 0, 0])
    errore_f_3[0,:] = np.array([0, 0, 0])
    errore_f_300[0,:] = np.array([0, 0, 0])

    #PLOT DELLA CONFIGURAZIONE INIZIALE
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot([0,XY1[0]],[0,XY1[1]],[0,XY1[2]], lw=5, color = 'royalblue')
    ax.scatter([0],[0],[0],color = 'black')
    ax.plot([XY1[0],XY2[0]],[XY1[1],XY2[1]],[XY1[2],XY2[2]], lw=5, color = 'green')
    ax.scatter(XY1[0],XY1[1],XY1[2],color = 'black')
    ax.plot([XY2[0],XY3[0]],[XY2[1],XY3[1]],[XY2[2],XY3[2]], lw = 5, color ='violet')
    ax.scatter(XY2[0],XY2[1],XY2[2], color = 'black')
    ax.plot([XY3[0],XY4[0]],[XY3[1],XY4[1]],[XY3[2],XY4[2]], lw=5, color = 'gold') 
    ax.scatter(XY3[0],XY3[1],XY3[2], color = 'black')
    ax.scatter(XY4[0],XY4[1],XY4[2], color = 'black')
    ax.set_xlabel('x [m]',fontsize=12)
    ax.set_ylabel('y [m]',fontsize=12)
    ax.set_zlabel('z [m]',fontsize=12)
    ax.set_xlim([-lmax, lmax])
    ax.set_ylim([-lmax, lmax])
    ax.set_zlim([-lmax, lmax])
    ax.set_title("Configurazione del Manipolatore nel punto iniziale A", fontsize=20)
    plt.show()

    #Limiti Spazio di Lavoro, variazione angolare min e max di ogni giunto
    q_m=np.array([-np.pi/6, -np.pi/2, -np.pi*3/2, -np.pi])
    q_M=np.array([np.pi*3/4, np.pi/2, np.pi/3, np.pi])     
    
    # guadagni
    k= 19 # scelto il k che mi da errore piú piccolo nel codice kgain
    ka= 4 # in comune tra migliore errore e miglior centro corsa
    # Pianificazione della traiettoria
    counter = 1 # stiamo dopo la configuazione iniziale e quindi le prime righe giá piene

    for t in time:
        # devo leggere la posizione attuale dei giunti Q e salvarla in un vettore

        if t <= t1:
            # Pianificazione prima segmento (da A a B)
            pd, pdot = CartesianPlanner(pA, pB, t0, t1, t)
        else:
            # Pianificazione secondo segmento (da B a C)
            pd, pdot = CartesianPlanner(pB, pC, t1, t2, t)

        # Inversione cinematica senza gestione ridondanza
        Qdot[counter,:], errore[counter,:]= InversioneCinematica(Qd[counter-1,:], a, d, alpha, pd, pdot, k, "i")

        # Inversione cinematica con gestione ridondanza--> ricavo la qdot da integrare per trovare Q
        Qdot_f[counter,:], errore_f[counter,:]= InversioneCinematica_centrocorsa(Qd_f[counter-1,:],a, d, alpha, pd, pdot, k, ka, "i")

        # Integrazione numerica per ottenere Q da qdot
        Qd[counter,:] = Qd[counter-1,:] + dt*Qdot[counter,:]       
        Qd_f[counter,:] = Qd_f[counter-1,:] + dt*Qdot_f[counter,:] # questa è la Qd da mandare ai motori per muoverli

        # Salvo un posizioni per graficare
        Pd[counter,:] = pd
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

    k= 19 # scelto il k che mi da errore piú piccolo nel codice kgain
    ka_errore= 6  # ka con migliore errore
    # Pianificazione della traiettoria
    counter = 1 # stiamo dopo la configuazione iniziale e quindi le prime righe giá piene

    for t in time:
        # devo leggere la posizione attuale dei giunti Q e salvarla in un vettore

        if t <= t1:
            # Pianificazione prima segmento (da A a B)
            pd, pdot = CartesianPlanner(pA, pB, t0, t1, t)
        else:
            # Pianificazione secondo segmento (da B a C)
            pd, pdot = CartesianPlanner(pB, pC, t1, t2, t)


        # Inversione cinematica con gestione ridondanza--> ricavo la qdot da integrare per trovare Q
        Qdot_f_6[counter,:], errore_f_6[counter,:]= InversioneCinematica_centrocorsa(Qd_f_6[counter-1,:],a, d, alpha, pd, pdot, k, ka_errore, "i")

        # Integrazione numerica per ottenere Q da qdot      
        Qd_f_6[counter,:] = Qd_f_6[counter-1,:] + dt*Qdot_f_6[counter,:] # questa è la Qd da mandare ai motori per muoverli

        # Salvo un posizioni per graficare
        Pd[counter,:] = pd
        Pddot[counter,:] = pdot
        
        #Calcolo cinematica della simulazione CON funzionale
        T01_f_6 = DH_computation(d[0], a[0], alpha[0], Qd_f_6[counter,0])
        T12_f_6 = DH_computation(d[1], a[1], alpha[1], Qd_f_6[counter,1])
        T22p_f_6 = DH_computation(d[2], a[2], alpha[2], Qd_f_6[counter,2])
        T2p3_f_6 = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        T34_f_6 = DH_computation(d[3], a[3], alpha[3], Qd_f_6[counter,3])        
        T02_f_6 = np.matmul(T01_f_6, T12_f_6)
        T02p_f_6 = np.matmul(T02_f_6, T22p_f_6)
        T03_f_6 = np.matmul(T02p_f_6, T2p3_f_6)
        T04_f_6 = np.matmul(T03_f_6, T34_f_6)
        

        # Posizione giunti e dell'end effector CON funzionale
        P_1_f_6[counter,:],_ = DirectKinematics(T01_f_6)
        P_2_f_6[counter,:],_ = DirectKinematics(T02_f_6)
        P_3_f_6[counter,:],_ = DirectKinematics(T03_f_6)
        P_ee_f_6[counter,:],_ = DirectKinematics(T04_f_6)


        counter += 1

    k= 19 # scelto il k che mi da errore piú piccolo nel codice kgain
    ka_giunto= 3  #ka con il miglior funzionale
    # Pianificazione della traiettoria
    counter = 1 # stiamo dopo la configuazione iniziale e quindi le prime righe giá piene

    for t in time:
        # devo leggere la posizione attuale dei giunti Q e salvarla in un vettore

        if t <= t1:
            # Pianificazione prima segmento (da A a B)
            pd, pdot = CartesianPlanner(pA, pB, t0, t1, t)
        else:
            # Pianificazione secondo segmento (da B a C)
            pd, pdot = CartesianPlanner(pB, pC, t1, t2, t)


        # Inversione cinematica con gestione ridondanza--> ricavo la qdot da integrare per trovare Q
        Qdot_f_3[counter,:], errore_f_3[counter,:]= InversioneCinematica_centrocorsa(Qd_f_3[counter-1,:],a, d, alpha, pd, pdot, k, ka_giunto, "i")

        # Integrazione numerica per ottenere Q da qdot      
        Qd_f_3[counter,:] = Qd_f_3[counter-1,:] + dt*Qdot_f_3[counter,:] # questa è la Qd da mandare ai motori per muoverli

        # Salvo un posizioni per graficare
        Pd[counter,:] = pd
        Pddot[counter,:] = pdot
        
        #Calcolo cinematica della simulazione CON funzionale
        T01_f_3 = DH_computation(d[0], a[0], alpha[0], Qd_f_3[counter,0])
        T12_f_3 = DH_computation(d[1], a[1], alpha[1], Qd_f_3[counter,1])
        T22p_f_3 = DH_computation(d[2], a[2], alpha[2], Qd_f_3[counter,2])
        T2p3_f_3 = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        T34_f_3 = DH_computation(d[3], a[3], alpha[3], Qd_f_3[counter,3])        
        T02_f_3 = np.matmul(T01_f_3, T12_f_3)
        T02p_f_3 = np.matmul(T02_f_3, T22p_f_3)
        T03_f_3 = np.matmul(T02p_f, T2p3_f_3)
        T04_f_3 = np.matmul(T03_f, T34_f_3)
        

        # Posizione giunti e dell'end effector CON funzionale
        P_1_f_3[counter,:],_ = DirectKinematics(T01_f_3)
        P_2_f_3[counter,:],_ = DirectKinematics(T02_f_3)
        P_3_f_3[counter,:],_ = DirectKinematics(T03_f_3)
        P_ee_f_3[counter,:],_ = DirectKinematics(T04_f_3)


        counter += 1
    
    k= 19 # scelto il k che mi da errore piú piccolo nel codice kgain
    ka_giunto= 300  
    # Pianificazione della traiettoria
    counter = 1 # stiamo dopo la configuazione iniziale e quindi le prime righe giá piene

    for t in time:
        # devo leggere la posizione attuale dei giunti Q e salvarla in un vettore

        if t <= t1:
            # Pianificazione prima segmento (da A a B)
            pd, pdot = CartesianPlanner(pA, pB, t0, t1, t)
        else:
            # Pianificazione secondo segmento (da B a C)
            pd, pdot = CartesianPlanner(pB, pC, t1, t2, t)

        # Inversione cinematica con gestione ridondanza--> ricavo la qdot da integrare per trovare Q
        Qdot_f_300[counter,:], errore_f_300[counter,:]= InversioneCinematica_centrocorsa(Qd_f_300[counter-1,:],a, d, alpha, pd, pdot, k, ka_giunto, "i")

        # Integrazione numerica per ottenere Q da qdot      
        Qd_f_300[counter,:] = Qd_f_300[counter-1,:] + dt*Qdot_f_300[counter,:] # questa è la Qd da mandare ai motori per muoverli

        # Salvo un posizioni per graficare
        Pd[counter,:] = pd
        Pddot[counter,:] = pdot
        
        #Calcolo cinematica della simulazione CON funzionale
        T01_f_300 = DH_computation(d[0], a[0], alpha[0], Qd_f_300[counter,0])
        T12_f_300 = DH_computation(d[1], a[1], alpha[1], Qd_f_300[counter,1])
        T22p_f_300 = DH_computation(d[2], a[2], alpha[2], Qd_f_300[counter,2])
        T2p3_f_300 = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        T34_f_300 = DH_computation(d[3], a[3], alpha[3], Qd_f_300[counter,3])        
        T02_f_300 = np.matmul(T01_f_300, T12_f_300)
        T02p_f_300 = np.matmul(T02_f_300, T22p_f_300)
        T03_f_300 = np.matmul(T02p_f_300, T2p3_f_300)
        T04_f_300 = np.matmul(T03_f_300, T34_f_300)
        

        # Posizione giunti e dell'end effector CON funzionale
        P_1_f_300[counter,:],_ = DirectKinematics(T01_f_300)
        P_2_f_300[counter,:],_ = DirectKinematics(T02_f_300)
        P_3_f_300[counter,:],_ = DirectKinematics(T03_f_300)
        P_ee_f_300[counter,:],_ = DirectKinematics(T04_f_300)


        counter += 1
    
    print("Qd_funzionale", Qd_f)
    print("Qdot_funzionale",Qdot_f)
    print('Matrice di rototraslazione:',T04)
    print('Matrice di rototraslazione:',T04_f)

    np.savetxt("Qd_funzionale.txt", Qd_f) #salvo le Qd con funzionale in un file txt per fare in offline il controllo del manipolatore

    centrocorsa_1 = (q_m[0]+q_M[0])/2
    centrocorsa_2 = (q_m[1]+q_M[1])/2
    centrocorsa_3 = (q_m[2]+q_M[2])/2
    centrocorsa_4 = (q_m[3]+q_M[3])/2
    print("Centro corsa giunto 1:", np.rad2deg(centrocorsa_1))
    print("   ")

    print("Centro corsa giunto 2:", np.rad2deg(centrocorsa_2))
    print("   ")

    print("Centro corsa giunto 3:", np.rad2deg(centrocorsa_3))
    print("   ")

    print("Centro corsa giunto 4:", np.rad2deg(centrocorsa_4))
    print("   ")
    #Media variabile di giunto senza funzionale
    Qd1 = sum(Qd.T[0,:])/len(Qd.T[0,:])
    Qd2 = sum(Qd.T[1,:])/len(Qd.T[1,:])
    Qd3 = sum(Qd.T[2,:])/len(Qd.T[2,:])
    Qd4 = sum(Qd.T[3,:])/len(Qd.T[3,:])
    print("Media variabile di giunto 1 senza funzionale: ", np.rad2deg(Qd1))
    print("   ")

    print("Media variabile di giunto 2 senza funzionale: ", np.rad2deg(Qd2))
    print("   ")

    print("Media variabile di giunto 3 senza funzionale: ", np.rad2deg(Qd3))
    print("   ")

    print("Media variabile di giunto 4 senza funzionale: ", np.rad2deg(Qd4))
    print("   ")

    #Media variabile di giunto con funzionale
    Qd1_f = sum(Qd_f.T[0,:])/len(Qd_f.T[0,:])
    Qd2_f = sum(Qd_f.T[1,:])/len(Qd_f.T[1,:])
    Qd3_f = sum(Qd_f.T[2,:])/len(Qd_f.T[2,:])
    Qd4_f = sum(Qd_f.T[3,:])/len(Qd_f.T[3,:])
    print("Media variabile di giunto 1 con funzionale: ", np.rad2deg(Qd1_f))
    print("   ")

    print("Media variabile di giunto 2 con funzionale: ", np.rad2deg(Qd2_f))
    print("   ")

    print("Media variabile di giunto 3 con funzionale: ", np.rad2deg(Qd3_f))
    print("   ")

    print("Media variabile di giunto 4 con funzionale: ", np.rad2deg(Qd4_f))
    print("   ")

    #Media variabile di giunto con funzionale
    Qd1_f_3 = sum(Qd_f_3.T[0,:])/len(Qd_f_3.T[0,:])
    Qd2_f_3 = sum(Qd_f_3.T[1,:])/len(Qd_f_3.T[1,:])
    Qd3_f_3 = sum(Qd_f_3.T[2,:])/len(Qd_f_3.T[2,:])
    Qd4_f_3 = sum(Qd_f_3.T[3,:])/len(Qd_f_3.T[3,:])
    print("Media variabile di giunto 1 con funzionale ka = 3: ", np.rad2deg(Qd1_f_3))
    print("   ")

    print("Media variabile di giunto 2 con funzionale ka = 3: ", np.rad2deg(Qd2_f_3))
    print("   ")

    print("Media variabile di giunto 3 con funzionale ka = 3: ", np.rad2deg(Qd3_f_3))
    print("   ")

    print("Media variabile di giunto 4 con funzionale ka = 3: ", np.rad2deg(Qd4_f_3))
    print("   ")

    # Errore Inversione Cinematica senza funzionale
    errore_x=0
    errore_y=0
    errore_z=0

    errore_x=sum(errore.T[0,:])/len(errore.T[0,:])
    errore_y=sum(errore.T[1,:])/len(errore.T[1,:])
    errore_z=sum(errore.T[2,:])/len(errore.T[2,:])

    print("Media errore su x metri: ", np.abs(errore_x))
    print("   ")

    print("Media errore su y metri: ", np.abs(errore_y))
    print("   ")

    print("Media errore su z metri: ", np.abs(errore_z))
    print("   ")

    # Errore Inversione Cinematica con funzionale scelto ka = 4
    erroref_x=0
    erroref_y=0
    erroref_z=0

    erroref_x=sum(errore_f.T[0,:])/len(errore_f.T[0,:]) #componente x mediata per tutti gli istanti di tempo
    erroref_y=sum(errore_f.T[1,:])/len(errore_f.T[1,:])
    erroref_z=sum(errore_f.T[2,:])/len(errore_f.T[2,:])

    print("Media errore con funzionale su x metri: ", np.abs(erroref_x))
    print("   ")

    print("Media errore con funzionale su y metri: ", np.abs(erroref_y))
    print("   ")

    print("Media errore con funzionale su z metri: ", np.abs(erroref_z))
    print("   ")

    # Errore Inversione Cinematica con funzionale e con:
    # ka = 6
    erroref_6_x=0
    erroref_6_y=0
    erroref_6_z=0

    erroref_6_x=sum(errore_f_6.T[0,:])/len(errore_f_6.T[0,:]) #componente x mediata per tutti gli istanti di tempo
    erroref_6_y=sum(errore_f_6.T[1,:])/len(errore_f_6.T[1,:])
    erroref_6_z=sum(errore_f_6.T[2,:])/len(errore_f_6.T[2,:])

    print("Media errore con funzionale su x metri (ka = 6): ", np.abs(erroref_6_x))
    print("   ")

    print("Media errore con funzionale su y metri (ka = 6): ", np.abs(erroref_6_y))
    print("   ")

    print("Media errore con funzionale su z metri (ka = 6): ", np.abs(erroref_6_z))
    print("   ")

    # ka = 3
    erroref_3_x=0
    erroref_3_y=0
    erroref_3_z=0

    erroref_3_x=sum(errore_f_3.T[0,:])/len(errore_f_3.T[0,:]) #componente x mediata per tutti gli istanti di tempo
    erroref_3_y=sum(errore_f_3.T[1,:])/len(errore_f_3.T[1,:])
    erroref_3_z=sum(errore_f_3.T[2,:])/len(errore_f_3.T[2,:])

    print("Media errore con funzionale su x metri (ka = 3): ", np.abs(erroref_3_x))
    print("   ")

    print("Media errore con funzionale su y metri (ka = 3): ", np.abs(erroref_3_y))
    print("   ")

    print("Media errore con funzionale su z metri (ka = 3): ", np.abs(erroref_3_z))
    print("   ")

    # ka = 300
    erroref_300_x=0
    erroref_300_y=0
    erroref_300_z=0

    erroref_300_x=sum(errore_f_300.T[0,:])/len(errore_f_300.T[0,:]) #componente x mediata per tutti gli istanti di tempo
    erroref_300_y=sum(errore_f_300.T[1,:])/len(errore_f_300.T[1,:])
    erroref_300_z=sum(errore_f_300.T[2,:])/len(errore_f_300.T[2,:])

    print("Media errore con funzionale su x metri (ka = 300): ", np.abs(erroref_300_x))
    print("   ")

    print("Media errore con funzionale su y metri (ka = 300): ", np.abs(erroref_300_y))
    print("   ")

    print("Media errore con funzionale su z metri (ka = 300): ", np.abs(erroref_300_z))
    print("   ")


    #PLOT ERRORE CON E SENZA FUNZIONALE con i vari Ka
    fig = plt.figure()
    plt.subplot(3,1,1)
    plt.plot(time, np.abs(errore[0:len(time),0]), lw = 3, color='orange', label = 'Senza funzionale')
    plt.plot(time, np.abs(errore_f_6[0:len(time),0]), lw = 2, color='green',  label = 'ka = 6')
    plt.plot(time, np.abs(errore_f_3[0:len(time),0]), lw = 2, color='violet',  label = 'ka = 3')
    plt.plot(time, np.abs(errore_f_300[0:len(time),0]), lw = 2, color='gold',  label = 'ka = 300')
    plt.plot(time, np.abs(errore_f[0:len(time),0]), lw = 2, color='royalblue',  label = 'ka = 4')
    plt.xlim([t0, t2])
    plt.legend()
    plt.xlabel("t [s]")
    plt.ylabel('Errore asse x [m]',fontsize=10)

    plt.subplot(3,1,2)
    plt.plot(time, np.abs(errore[0:len(time),1]), lw = 3, color='orange',  label = 'Senza funzionale')
    plt.plot(time, np.abs(errore_f_6[0:len(time),1]), lw = 2, color='green',  label = 'ka = 6')
    plt.plot(time, np.abs(errore_f_3[0:len(time),1]), lw = 2, color='violet',  label = 'ka = 3')
    plt.plot(time, np.abs(errore_f_300[0:len(time),1]), lw = 2, color='gold',  label = 'ka = 300')
    plt.plot(time, np.abs(errore_f[0:len(time),1]), lw = 2, color='royalblue',  label = 'ka = 4')
    plt.xlim([t0, t2])
    plt.legend()
    plt.xlabel("t [s]")
    plt.ylabel('Errore asse y [m]',fontsize=10)

    plt.subplot(3,1,3)
    plt.plot(time, np.abs(errore[0:len(time),2]), lw = 3, color='orange',  label = 'Senza funzionale')
    plt.plot(time, np.abs(errore_f_6[0:len(time),2]), lw = 2, color='green', label = 'ka = 6')
    plt.plot(time, np.abs(errore_f_3[0:len(time),2]), lw = 2, color='violet',  label = 'ka = 3')
    plt.plot(time, np.abs(errore_f_300[0:len(time),2]), lw = 2, color='gold',  label = 'ka = 300')
    plt.plot(time, np.abs(errore_f[0:len(time),2]), lw = 2, color='royalblue',  label = 'ka = 4')
    plt.xlim([t0, t2])
    plt.legend()
    plt.xlabel('t [s]',fontsize=16)
    plt.ylabel('Errore asse z [m]',fontsize=10)

    fig.suptitle('Errore nel tempo con vari ka',fontsize=20)

    #PLOT ERRORE CON E SENZA FUNZIONALE con i vari Ka senza ka = 300
    fig = plt.figure()
    plt.subplot(3,1,1)
    plt.plot(time, np.abs(errore[0:len(time),0]), lw = 3, color='orange', label = 'Senza funzionale')
    plt.plot(time, np.abs(errore_f_6[0:len(time),0]), lw = 2, color='green',  label = 'ka = 6')
    plt.plot(time, np.abs(errore_f_3[0:len(time),0]), lw = 2, color='violet',  label = 'ka = 3')
    plt.plot(time, np.abs(errore_f[0:len(time),0]), lw = 2, color='royalblue',  label = 'ka = 4')
    plt.xlim([t0, t2])
    plt.legend()
    plt.xlabel("t [s]")
    plt.ylabel('Errore asse x [m]',fontsize=10)

    plt.subplot(3,1,2)
    plt.plot(time, np.abs(errore[0:len(time),1]), lw = 3, color='orange',  label = 'Senza funzionale')
    plt.plot(time, np.abs(errore_f_6[0:len(time),1]), lw = 2, color='green',  label = 'ka = 6')
    plt.plot(time, np.abs(errore_f_3[0:len(time),1]), lw = 2, color='violet',  label = 'ka = 3')
    plt.plot(time, np.abs(errore_f[0:len(time),1]), lw = 2, color='royalblue',  label = 'ka = 4')
    plt.xlim([t0, t2])
    plt.legend()
    plt.xlabel("t [s]")
    plt.ylabel('Errore asse y [m]',fontsize=10)

    plt.subplot(3,1,3)
    plt.plot(time, np.abs(errore[0:len(time),2]), lw = 3, color='orange',  label = 'Senza funzionale')
    plt.plot(time, np.abs(errore_f_6[0:len(time),2]), lw = 2, color='green', label = 'ka = 6')
    plt.plot(time, np.abs(errore_f_3[0:len(time),2]), lw = 2, color='violet',  label = 'ka = 3')
    plt.plot(time, np.abs(errore_f[0:len(time),2]), lw = 2, color='royalblue',  label = 'ka = 4')
    plt.xlim([t0, t2])
    plt.legend()
    plt.xlabel('t [s]',fontsize=16)
    plt.ylabel('Errore asse z [m]',fontsize=10)

    fig.suptitle('Errore nel tempo',fontsize=20)

    #PLOT ERRORE SENZA FUNZIONALE e CON FUNZIONALE SCELTO
    fig = plt.figure()
    plt.subplot(3,1,1)
    plt.plot(time, np.abs(errore[0:len(time),0]), lw = 3, color='orange', label = 'NO funzionale',)
    plt.plot(time, np.abs(errore_f[0:len(time),0]), lw = 2, color='royalblue',  label = 'CON funzionale')
    plt.xlim([t0, t2])
    plt.legend()
    plt.xlabel("t [s]")
    plt.ylabel('Errore asse x [m]',fontsize=10)

    plt.subplot(3,1,2)
    plt.plot(time, np.abs(errore[0:len(time),1]), lw = 3, color='orange',  label = 'NO funzionale',)
    plt.plot(time, np.abs(errore_f[0:len(time),1]), lw = 2, color='royalblue',  label = 'CON funzionale' )
    plt.xlim([t0, t2])
    plt.legend()
    plt.xlabel("t [s]")
    plt.ylabel('Errore asse y [m]',fontsize=10)

    plt.subplot(3,1,3)
    plt.plot(time, np.abs(errore[0:len(time),2]), lw = 3, color='orange',  label = 'NO funzionale')
    plt.plot(time, np.abs(errore_f[0:len(time),2]), lw = 2, color='royalblue', label = 'CON funzionale')
    plt.xlim([t0, t2])
    plt.legend()
    plt.xlabel('t [s]',fontsize=16)
    plt.ylabel('Errore asse z [m]',fontsize=10)

    fig.suptitle('Errore nel tempo con funzionale ka = 4',fontsize=18)

    #PLOT PIANIFICAZIONE
    fig = plt.figure() 
    fig.suptitle('Pianificazione di traiettoria')
    plt.subplot(2,2,1) #TRAIETTORIA
    plt.plot(time, Pd[0:len(time),0], lw = 3, color = 'royalblue', label = 'desired x')
    plt.plot(time, Pd[0:len(time),1], lw = 3, color = 'gold', label = 'desired y')
    plt.plot(time, Pd[0:len(time),2], lw = 3, color = 'firebrick',label = 'desired z')    
    #marker 1
    plt.plot(time[0],Pd[0,0],marker='.',markersize=15, color = 'black')
    plt.plot(time[0],Pd[0,1],marker='.',markersize=15, color = 'black')
    plt.plot(time[0],Pd[0,2],marker='.',markersize=15, color = 'black')
    #marker 2
    plt.plot(time[len(time1)],Pd[len(time1),0],marker='.',markersize=15, color = 'black')
    plt.plot(time[len(time1)],Pd[len(time1),1],marker='.',markersize=15, color = 'black')
    plt.plot(time[len(time1)],Pd[len(time1),2],marker='.',markersize=15, color = 'black')
    #marker 3
    plt.plot(time[-1],Pd[-1,0],marker='.',markersize=15, color = 'black')
    plt.plot(time[-1],Pd[-1,1],marker='.',markersize=15, color = 'black')
    plt.plot(time[-1],Pd[-1,2],marker='.',markersize=15, color = 'black')
    #assi
    plt.xlim([t0, t2])
    plt.legend()
    plt.ylabel('Posizione end-effector [m]',fontsize=12)
    plt.xlabel('Time [s]',fontsize=12)
#PLOT VELOCITÀ
    plt.subplot(2,2,3) 
    plt.plot(time, Pddot[0:len(time),0], lw = 3, color = 'royalblue', label = 'desired vx')
    plt.plot(time, Pddot[0:len(time),1], lw = 3, color = 'gold', label = 'desired vy')
    plt.plot(time, Pddot[0:len(time),2], lw = 3, color = 'firebrick',label = 'desired vz')    
    #marker 1
    plt.plot(time[0],Pddot[0,0],marker='.',markersize=15, color = 'black')
    plt.plot(time[0],Pddot[0,1],marker='.',markersize=15, color = 'black')
    plt.plot(time[0],Pddot[0,2],marker='.',markersize=15, color = 'black')
    #marker 2
    plt.plot(time[len(time1)],Pddot[len(time1),0],marker='.',markersize=15, color = 'black')
    plt.plot(time[len(time1)],Pddot[len(time1),1],marker='.',markersize=15, color = 'black')
    plt.plot(time[len(time1)],Pddot[len(time1),2],marker='.',markersize=15, color = 'black')
    #marker 3
    plt.plot(time[-1],Pddot[-1,0],marker='.',markersize=15, color = 'black')
    plt.plot(time[-1],Pddot[-1,1],marker='.',markersize=15, color = 'black')
    plt.plot(time[-1],Pddot[-1,2],marker='.',markersize=15, color = 'black')
    #assi
    plt.xlim([t0, t2])
    plt.legend()
    plt.ylabel('Velocità end-effector [m/s]',fontsize=12)
    plt.xlabel('Time [s]',fontsize=12)
    #Traiettoria in 3D
    ax = plt.subplot(2, 2, (2, 4), projection = '3d') 
    fig.show()
    ax.plot(Pd[0:len(time),0],Pd[0:len(time),1],Pd[0:len(time),2],lw=5, color = 'royalblue')
    ax.plot(P_ee_f[0:len(time),0],P_ee_f[0:len(time),1],P_ee_f[0:len(time),2],lw=3, color = 'firebrick')
    ax.scatter(pA[0],pA[1],pA[2],color = 'black', label='B')
    ax.scatter(pB[0],pB[1],pB[2],color = 'black', label='C')
    ax.scatter(pC[0],pC[1],pC[2],color = 'black', label='D')
    ax.set_xlabel('X [m]',fontsize=15)
    ax.set_ylabel('Y [m]',fontsize=15)
    ax.set_zlabel('Z [m]',fontsize=15)
    ax.set_xlim([-lmax, lmax])
    ax.set_ylim([-lmax, lmax])
    ax.set_zlim([-lmax, lmax])
    ax.set_title("Pianificazione della Traiettoria 3D", fontsize=25)
    plt.show()

    # Plot x,y,z posizioni desiderate ed effettive nel tempo (end-effector in S.O.)
    fig = plt.plot
    fig = plt.figure()
    # x nel tempo
    plt.subplot(3,1,1)
    plt.plot(time, Pd[0:len(time),0], lw = 4, color='orange', label = 'Desiderata') # pianificata
    plt.plot(time, P_ee_f[0:len(time),0], lw = 4, color='royalblue',label = 'Reale') # reale
    plt.xlim([t0, t2])
    plt.legend()
    plt.xlabel("t [s]")
    plt.ylabel('X [m]',fontsize=18)
    # y nel tempo
    plt.subplot(3,1,2)
    plt.plot(time, Pd[0:len(time),1], lw = 4, color='orange', label = 'Desiderata')
    plt.plot(time, P_ee_f[0:len(time),1], lw = 4, color='royalblue', label = 'Reale')
    plt.xlim([t0, t2])
    plt.legend()
    plt.xlabel("t [s]")
    plt.ylabel('Y [m]',fontsize=18)
    # z nel tempo
    plt.subplot(3,1,3)
    plt.plot(time, Pd[0:len(time),2], lw = 4, color='orange', label = 'Desiderata')
    plt.plot(time, P_ee_f[0:len(time),2], lw = 4, color='royalblue', label = 'Reale')
    plt.xlim([t0, t2])
    plt.legend()
    plt.xlabel('t [s]',fontsize=18)
    plt.ylabel('Z [m]',fontsize=18)
    fig.suptitle('Posizione desiderata VS reale con funzionale',fontsize=20)
    plt.show()

    # Plot x,y,z posizioni desiderate ed effettive nel tempo senza funzionale (end-effector in S.O.)
    fig = plt.plot
    fig = plt.figure()
    # x nel tempo
    plt.subplot(3,1,1)
    plt.plot(time, Pd[0:len(time),0], lw = 4, color='orange', label = 'Desiderata') # pianificata
    plt.plot(time, P_ee[0:len(time),0], lw = 4, color='royalblue',label = 'Reale') # reale
    plt.xlim([t0, t2])
    plt.legend()
    plt.xlabel("t [s]")
    plt.ylabel('X [m]',fontsize=18)
    # y nel tempo
    plt.subplot(3,1,2)
    plt.plot(time, Pd[0:len(time),1], lw = 4, color='orange', label = 'Desiderata')
    plt.plot(time, P_ee[0:len(time),1], lw = 4, color='royalblue', label = 'Reale')
    plt.xlim([t0, t2])
    plt.legend()
    plt.xlabel("t [s]")
    plt.ylabel('Y [m]',fontsize=18)
    # z nel tempo
    plt.subplot(3,1,3)
    plt.plot(time, Pd[0:len(time),2], lw = 4, color='orange', label = 'Desiderata')
    plt.plot(time, P_ee[0:len(time),2], lw = 4, color='royalblue', label = 'Reale')
    plt.xlim([t0, t2])
    plt.legend()
    plt.xlabel('t [s]',fontsize=18)
    plt.ylabel('Z [m]',fontsize=18)
    fig.suptitle('Posizione desiderata VS reale senza funzionale',fontsize=20)
    plt.show()

    #PLOT VARIABILI DI GIUNTO CON E SENZA FUNZIONALE
    fig = plt.figure(figsize=(11,5.5))
    plt.subplot(2,2,1)
    plt.plot(time, (Qd[0:len(time),0]), lw = 1, color='orange', label = 'Senza funzionale')
    plt.plot(time, (Qd_f[0:len(time),0]), lw = 1, color='royalblue', label = 'ka = 4')
    plt.plot(time, (Qd_f_3[0:len(time),0]), lw = 2, color='violet', label = 'ka = 3')
    plt.plot(time, (Qd_f_6[0:len(time),0]), lw = 2, color='green', label = 'ka = 6')
    plt.plot(time, (Qd_f_300[0:len(time),0]), lw = 2, color='gold', label = 'ka = 300')
    plt.xlim([t0, t2])
    plt.ylim([-np.pi*2, np.pi*2])
    plt.legend()
    plt.ylabel('Q1 [rad]',fontsize=12)
    plt.xlabel('t [s]',fontsize=12)
    plt.plot(time, q_m[0]*np.ones(len(time)), lw=2, color='red') # plot delle linee orizzonatali rosse
    plt.plot(time, q_M[0]*np.ones(len(time)), lw=2, color='red')
    plt.yticks([-np.pi*2, -np.pi, -np.pi/6, 0, np.pi*3/4, np.pi, np.pi*2],
               [r'$-2pi$',r'$-pi$',r'$-pi/6$',r'$0$',r'$pi3/4$', r'$pi$',r'$+2pi$'])


    plt.subplot(2,2,2)
    plt.plot(time, (Qd[0:len(time),1]), lw = 1, color='orange', label = 'Senza funzionale')
    plt.plot(time, (Qd_f[0:len(time),1]), lw = 1, color='royalblue', label = 'ka = 4')
    plt.plot(time, (Qd_f_3[0:len(time),1]), lw = 2, color='violet', label = 'ka = 3')
    plt.plot(time, (Qd_f_6[0:len(time),1]), lw = 2, color='green', label = 'ka = 6')
    plt.plot(time, (Qd_f_300[0:len(time),1]), lw = 2, color='gold', label = 'ka = 300')
    plt.xlim([t0, t2])
    plt.xlim([t0, t2])
    plt.ylim([-np.pi*2, np.pi*2])
    plt.legend()
    plt.ylabel('Q2 [rad]',fontsize=12)
    plt.xlabel('t [s]',fontsize=12)
    plt.plot(time, q_m[1]*np.ones(len(time)), lw=2, color='red')
    plt.plot(time, q_M[1]*np.ones(len(time)), lw=2, color='red')
    plt.yticks([-np.pi*2,-np.pi, -np.pi/2, 0, np.pi/2, np.pi, np.pi*2],
            [r'$-2pi$',r'$-pi$',r'$-pi/2$',r'$0$',r'$pi/2$', r'$pi$',r'$+2pi$'])

    plt.subplot(2,2,3)
    plt.plot(time, (Qd[0:len(time),2]), lw = 1, color='orange', label = 'Senza funzionale')
    plt.plot(time, (Qd_f[0:len(time),2]), lw = 1, color='royalblue', label = 'ka = 4')
    plt.plot(time, (Qd_f_3[0:len(time),2]), lw = 2, color='violet', label = 'ka = 3')
    plt.plot(time, (Qd_f_6[0:len(time),2]), lw = 2, color='green', label = 'ka = 6')
    plt.plot(time, (Qd_f_300[0:len(time),2]), lw = 2, color='gold', label = 'ka = 300')
    plt.xlim([t0, t2])
    plt.xlim([t0, t2])
    plt.ylim([-np.pi*2, np.pi*2])
    plt.legend()
    plt.xlabel('t [s]',fontsize=12)
    plt.ylabel('Q3 [rad]',fontsize=12)
    plt.plot(time, q_m[2]*np.ones(len(time)), lw=2, color='red')
    plt.plot(time, q_M[2]*np.ones(len(time)), lw=2, color='red')
    plt.yticks([-np.pi*2,-np.pi*3/2, -np.pi/2, 0, np.pi/3, np.pi/2, np.pi*2],
            [r'$-2pi$',r'$-pi*3/2$',r'$-pi/2$',r'$0$',r'$pi/3$', r'$pi/2$',r'$+2pi$'])


    plt.subplot(2,2,4)
    plt.plot(time, (Qd[0:len(time),3]), lw = 1, color='orange', label = 'Senza funzionale')
    plt.plot(time, (Qd_f[0:len(time),3]), lw = 1, color='royalblue', label = 'ka = 4')
    plt.plot(time, (Qd_f_3[0:len(time),3]), lw = 2, color='violet', label = 'ka = 3')
    plt.plot(time, (Qd_f_6[0:len(time),3]), lw = 2, color='green', label = 'ka = 6')
    plt.plot(time, (Qd_f_300[0:len(time),3]), lw = 2, color='gold', label = 'ka = 300')
    plt.xlim([t0, t2])
    plt.xlim([t0, t2])
    plt.ylim([-np.pi*2, np.pi*2])
    plt.legend()
    plt.xlabel('t [s]',fontsize=12)
    plt.ylabel('Q4 [rad]',fontsize=12)
    plt.plot(time, q_m[3]*np.ones(len(time)), lw=2, color='red')
    plt.plot(time, q_M[3]*np.ones(len(time)), lw=2, color='red')
    plt.yticks([-np.pi*2,-np.pi, -np.pi/2, 0, np.pi/2, np.pi, np.pi*2],
            [r'$-2pi$',r'$-pi$',r'$-pi/2$',r'$0$',r'$pi/2$', r'$pi$',r'$+2pi$'])
    fig.suptitle('Variabili di giunto',fontsize=25)
    plt.show()

    #PLOT VARIABILI DI GIUNTO CON FUNZIONALE SCELTO E SENZA FUNZIONALE
    fig = plt.figure()
    plt.subplot(2,2,1)
    plt.plot(time, (Qd[0:len(time),0]), lw = 1, color='orange', label = 'Senza funzionale')
    plt.plot(time, (Qd_f[0:len(time),0]), lw = 1, color='royalblue', label = 'Con funzionale')
    plt.xlim([t0, t2])
    plt.ylim([-np.pi*2, np.pi*2])
    plt.legend()
    plt.ylabel('Q1 [rad]',fontsize=10)
    plt.xlabel('t [s]',fontsize=10)
    plt.plot(time, q_m[0]*np.ones(len(time)), lw=2, color='red') # plot delle linee di limite orizzonatali rosse
    plt.plot(time, q_M[0]*np.ones(len(time)), lw=2, color='red')
    plt.yticks([-np.pi*2, -np.pi, -np.pi/6, 0, np.pi*3/4, np.pi, np.pi*2],
               [r'$-2pi$',r'$-pi$',r'$-pi/6$',r'$0$',r'$pi3/4$', r'$pi$',r'$+2pi$'])


    plt.subplot(2,2,2)
    plt.plot(time, (Qd[0:len(time),1]), lw = 1, color='orange', label = 'Senza funzionale')
    plt.plot(time, (Qd_f[0:len(time),1]), lw = 1, color='royalblue', label = 'Con funzionale')
    plt.xlim([t0, t2])
    plt.xlim([t0, t2])
    plt.ylim([-np.pi*2, np.pi*2])
    plt.legend()
    plt.ylabel('Q2 [rad]',fontsize=10)
    plt.xlabel('t [s]',fontsize=10)
    plt.plot(time, q_m[1]*np.ones(len(time)), lw=2, color='red')
    plt.plot(time, q_M[1]*np.ones(len(time)), lw=2, color='red')
    plt.yticks([-np.pi*2,-np.pi, -np.pi/2, 0, np.pi/2, np.pi, np.pi*2],
            [r'$-2pi$',r'$-pi$',r'$-pi/2$',r'$0$',r'$pi/2$', r'$pi$',r'$+2pi$'])

    plt.subplot(2,2,3)
    plt.plot(time, (Qd[0:len(time),2]), lw = 1, color='orange', label = 'Senza funzionale')
    plt.plot(time, (Qd_f[0:len(time),2]), lw = 1, color='royalblue', label = 'Con funzionale')
    plt.xlim([t0, t2])
    plt.xlim([t0, t2])
    plt.ylim([-np.pi*2, np.pi*2])
    plt.legend()
    plt.xlabel('t [s]',fontsize=10)
    plt.ylabel('Q3 [rad]',fontsize=10)
    plt.plot(time, q_m[2]*np.ones(len(time)), lw=2, color='red')
    plt.plot(time, q_M[2]*np.ones(len(time)), lw=2, color='red')
    plt.yticks([-np.pi*2,-np.pi*3/2, -np.pi/2, 0, np.pi/3, np.pi/2, np.pi*2],
            [r'$-2pi$',r'$-pi*3/2$',r'$-pi/2$',r'$0$',r'$pi/3$', r'$pi/2$',r'$+2pi$'])


    plt.subplot(2,2,4)
    plt.plot(time, (Qd[0:len(time),3]), lw = 1, color='orange', label = 'Senza funzionale')
    plt.plot(time, (Qd_f[0:len(time),3]), lw = 1, color='royalblue', label = 'Con funzionale')
    plt.xlim([t0, t2])
    plt.xlim([t0, t2])
    plt.ylim([-np.pi*2, np.pi*2])
    plt.legend()
    plt.xlabel('t [s]',fontsize=10)
    plt.ylabel('Q4 [rad]',fontsize=10)
    plt.plot(time, q_m[3]*np.ones(len(time)), lw=2, color='red')
    plt.plot(time, q_M[3]*np.ones(len(time)), lw=2, color='red')
    plt.yticks([-np.pi*2,-np.pi, -np.pi/2, 0, np.pi/2, np.pi, np.pi*2],
            [r'$-2pi$',r'$-pi$',r'$-pi/2$',r'$0$',r'$pi/2$', r'$pi$',r'$+2pi$'])
    fig.suptitle('Variabili di giunto',fontsize=20)
    plt.show()

    plt.pause(10)

   
#PLOT 3D DINAMICO con FUNZIONALE (SIMULAZIONE)
    fig,ax = plt.subplots()
    ax = plt.axes(projection = '3d')
    viewer = fig.add_subplot(111,projection = '3d')
    fig.show()
    valore = 10
    for counter in range(0,len(time)+1, valore):
      viewer.clear()

      plt.plot([0,P_1_f[counter,0]], [0,P_1_f[counter,1]], [0,P_1_f[counter,2]], lw = 5, color = 'royalblue')
      plt.plot([0], [0], [0], marker = '.', markersize = 15, color = 'black')

      plt.plot([P_1_f[counter,0],P_2_f[counter,0]], [P_1_f[counter,1],P_2_f[counter,1]], [P_1_f[counter,2],P_2_f[counter,2]], lw = 5, color = 'green')
      plt.plot(P_1_f[counter,0], P_1_f[counter,1], P_1_f[counter,2], marker = '.', markersize = 15, color = 'black')

      plt.plot([P_2_f[counter,0],P_3_f[counter,0]], [P_2_f[counter,1],P_3_f[counter,1]], [P_2_f[counter,2],P_3_f[counter,2]], lw = 5, color ='violet')
      plt.plot(P_2_f[counter,0], P_2_f[counter,1], P_2_f[counter,2], marker = '.', markersize = 15, color ='black' )

      plt.plot([P_3_f[counter,0],P_ee_f[counter,0]], [P_3_f[counter,1],P_ee_f[counter,1]], [P_3_f[counter,2],P_ee_f[counter,2]], lw = 5, color = 'gold')
      plt.plot(P_3_f[counter,0], P_3_f[counter,1], P_3_f[counter,2], marker = '.', markersize = 15, color = 'black')
      plt.plot(P_ee_f[counter,0], P_ee_f[counter,1], P_ee_f[counter,2], marker = '.', markersize = 10, color = 'black')
      plt.plot(Pd[:,0], Pd[:,1], Pd[:,2], color = 'orange')
      plt.plot(P_ee_f[0:counter,0], P_ee_f[0:counter,1], P_ee_f[0:counter,2], lw = 2, c='firebrick')


      viewer.set_xlim([-lmax, lmax])
      viewer.set_ylim([-lmax, lmax])
      viewer.set_zlim([-lmax, lmax])
      viewer.set_xlabel('x [m]', fontsize = 12)
      viewer.set_ylabel('y [m]', fontsize = 12)
      viewer.set_zlabel('z [m]', fontsize = 12)
      viewer.set_aspect('auto','box')
      plt.title("time: "+"{:.2f}".format(counter*1))
      plt.pause(0.01)
      fig.canvas.draw()
    
    plt.pause(100)

# MANIPOLATORE REALE

angoli = np.loadtxt("angoli.txt")

#Correggo i rapporti di trasmissione, i segni delle rotazioni e converto la misura in radianti

angoli[0:len(time),0] = -(angoli[0:len(time),0])*24/40
angoli[0:len(time),1] = -angoli[0:len(time),1]
angoli[0:len(time),2] = angoli[0:len(time),2]
angoli[0:len(time),3] = -angoli[0:len(time),3]

Qreale = np.deg2rad(angoli)

fig = plt.figure()
plt.subplot(1,4,1)
plt.plot(time, Qd_f[0:len(time),0], lw = 2, color = 'orange',label = "q0 desiderata")
plt.plot(time, Qreale[0:len(time),0], lw = 2, color = 'royalblue', label = "q0 reale")
plt.title('Qd e Qreale del giunto 1')
plt.ylabel('rad[s]')
plt.xlabel('t [s]')
plt.legend()
plt.xlim([t0,t2])

plt.subplot(1,4,2)
plt.plot(time, Qd_f[0:len(time),1], lw = 2, color = 'orange', label = "q1 desiderata")
plt.plot(time, Qreale[0:len(time),1], lw = 2, color = 'royalblue',label = "q1 reale")
plt.title('Qd e Qreale del giunto 2')
plt.ylabel('rad[s]')
plt.xlabel('t [s]')
plt.legend()
plt.xlim([t0,t2])

plt.subplot(1,4,3)
plt.plot(time, Qd_f[0:len(time),2], lw = 2, color = 'orange', label = "q2 desiderata")
plt.plot(time, Qreale[0:len(time),2], lw = 2, color = 'royalblue', label = "q2 reale")
plt.title('Qd e Qreale del giunto 3')
plt.ylabel('rad[s]')
plt.xlabel('t [s]')
plt.legend()
plt.xlim([t0,t2])

plt.subplot(1,4,4)
plt.plot(time, Qd_f[0:len(time),3], lw = 2, color = 'orange', label = "q3 desiderata")
plt.plot(time, Qreale[0:len(time),3], lw = 2, color = 'royalblue', label = "q3 reale")
plt.title('Qd e Qreale del giunto 4')
plt.ylabel('rad[s]')
plt.xlabel('t [s]')
plt.legend()
plt.xlim([t0,t2])
plt.show()