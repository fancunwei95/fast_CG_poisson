import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splin
import matplotlib.pyplot as plt
import itertools
import time

def construct_A(N,d,h):
    diags = [-1.0*np.ones(N-1) , 2.0*np.ones(N), -1.0*np.ones(N-1) ]
    A = sp.diags(diags, [-1,0,1])
    I = sp.diags(np.ones(N), 0)
    if d==1 :
        answer = A
    elif d==2:
        answer = sp.kron(A,I) + sp.kron(I,A)
    elif d==3:
        answer = sp.kron(sp.kron(A,I),I) +  sp.kron(sp.kron(I,A),I) +  sp.kron(sp.kron(I,I),A)
    else:
        answer = None
    return answer/(h*h)

def Mesh(N,d):
    X = np.linspace(0.0,1.0,N+2)
    h = X[1] - X[0]
    mesh = np.array([np.array(index) for index in itertools.product(range(1,N+1),repeat = d) ])
    # [u_11,u_12,u_13......
    return h, mesh*h, X

def CG(A_input,b,tol = 1.0e-8):
    x = np.ones(len(b))
    A = sp.csc_matrix(A_input)
    r = b - A.dot(x)
    p = b - A.dot(x)
    diff = np.linalg.norm(r,2)
    count =0
    while diff > tol and count < 1000 :
        alpha = diff*diff/(p.dot(A.dot(p)))
        x = x + alpha*p
        r_new = r - alpha*A.dot(p)
        p = r_new + r_new.dot(r_new)/diff/diff*p
        count +=1
        r = r_new
        diff = np.linalg.norm(r,2)
    return x,count

def fast(N,d, X, b):
    h = X[1] -X [0]
    W,S = np.linalg.eigh(construct_A(N,1,h).toarray())
    S_trans = S.transpose() #np.array( [ np.sin(k*np.pi*X) for k in range(1,N+1) ] )
    I = np.identity(N)
    if d ==1 :
        A_inverse = (S.dot(np.diag(1.0/W))).dot(S_trans)
        answer = A_inverse.dot(b)
    elif d==2 :
        b = np.reshape(b,(N,N),'F')
        D = np.tile(W,(N,1))
        D = D.transpose()+D
        D = 1.0/D
        #W = sp.diags(W) 
        #D = sp.kron(I,W)+sp.kron(W,I)
        #D = 1.0/D.diagonal()
        #D = np.reshape(D,(N,N),'F')
        my_b = np.reshape(b,(N,N),'F')
        answer = S.dot((D*(S_trans.dot(my_b.dot(S)))).dot(S_trans))
        answer = answer.flatten('F')
    elif d==3 :
        my_b = np.reshape(b,(N,N,N),'F')
        my_b = np.tensordot(S_trans,my_b,axes=([1],[0]))
        my_b = np.tensordot(S_trans,my_b, axes=([1],[1]))
        my_b = np.tensordot(S_trans,my_b, axes=([1],[2]))
        A2 = np.tile(W,(N,1))
        A2 = A2.transpose()+A2 # the two tensor product of A  
        D = np.tile(A2,(N,1,1))
        A3 = np.repeat ( (W[np.newaxis]).transpose(), N*N, axis=1)      
        A3 = np.reshape(A3,(N,N,N))
        D =1.0/( D + A3) 
        answer = my_b*D
        answer = np.tensordot(S,answer, axes=([1],[0]))
        answer = np.tensordot(S,answer, axes=([1],[1]))
        answer = np.tensordot(S,answer, axes=([1],[2]))
        
        answer = answer.flatten('F')
        

    return answer


def solver(N,d, solver_method):
    h, mesh, X = Mesh(N,d)
    A = construct_A(N,d,h)
    b = construct_b(mesh,N)
    start = time.time()
    count = None
    method = ['direct','CG','fast']
    if solver_method == 0:
        # direct solver
        answer = splin.spsolve(sp.csr_matrix(A),b)
    elif solver_method == 1:
        # CG solver
        answer,count = splin.cg(A,b,tol=1.0e-10)
        #answer,count = CG(A,b)
    elif solver_method == 2 :
        answer = fast(N,d,X,b)
    end = time.time()
    time_taken = end -start
    print (method[solver_method], 'time: '+ str(time_taken), 'iter: '+str( count)) 
    return answer, mesh,time_taken,count

def construct_b(mesh,grid_per_dim):
    a = 1.0
    if len(mesh[0]) == 1:
        b = np.sin(a*np.pi*mesh.flatten())
    elif len(mesh[0]) ==2:
        N = len(mesh)
        b = np.zeros(N)
        for i in range(N):
            b[i] = np.sin(a*mesh[i][0]*np.pi)*np.sin(a*mesh[i][1]*np.pi)
        b = np.reshape(b,(grid_per_dim,grid_per_dim),'F')
        b = b.flatten()
    else :
        N = len(mesh)
        b = np.zeros(N)
        for i in range(N):
            b[i] = np.sin(a*mesh[i][0]*np.pi)*np.sin(a*mesh[i][1]*np.pi)*np.sin(a*mesh[i][2]*np.pi)
        b = np.reshape(b,(grid_per_dim,grid_per_dim,grid_per_dim),'F')
        b = b.flatten()
    return b

def construct_x(mesh,grid_per_dim):
    if len(mesh[0]) == 1:
        x = np.sin(np.pi*mesh.flatten())/(np.pi*np.pi)
    elif len(mesh[0]) ==2:
        N = len(mesh)
        x = np.zeros(N)
        for i in range(N):
            x[i] = np.sin(mesh[i][0]*np.pi)*np.sin(mesh[i][1]*np.pi)/2.0/np.pi/np.pi
        x = np.reshape(x,(grid_per_dim,grid_per_dim),'F')
        x = x.flatten()
    else :
        N = len(mesh)
        x = np.zeros(N)
        for i in range(N):
            x[i] = np.sin(mesh[i][0]*np.pi)*np.sin(mesh[i][1]*np.pi)*np.sin(mesh[i][2]*np.pi)/(np.pi*np.pi*3.0)
        x = np.reshape(x,(grid_per_dim,grid_per_dim,grid_per_dim),'F')
        x = x.flatten()
    return x


def test_accuracy(d,N_list):
    #direct_error = test_accuracy_method(d,0,N_list[0])
    CG_error = test_accuracy_method(d,1,N_list[1])
    fast_error = test_accuracy_method(d,2,N_list[2])
    plt.figure()
    plt.title("accuracy, dimension = "+str(d),fontsize=20)
    #plt.loglog(N_list[0],direct_error,'r--',label="direct")
    plt.loglog(N_list[1],CG_error,'^',label="CG")
    plt.loglog(N_list[2],fast_error,'g:',label="fast")
    plt.xlabel("N",fontsize=20)
    plt.ylabel("error",fontsize=20)
    x = np.linspace(N_list[2][0], N_list[2][-1],100)
    plt.loglog(x,1.0/x/x,label="1/x^2")
    plt.legend(loc="best")
    plt.show()

def test_accuracy_method(d,method,N_list):
    err_list = np.zeros(len(N_list))
    for i in range(len(N_list)):
        N = N_list[i]
        answer, mesh,time_taken, count = solver(N,d,method)
        exact = construct_x(mesh,N)
        err_list[i] = np.linalg.norm(np.abs((exact-answer)/answer),ord = np.inf)
    return err_list

def timing_method(d,method,N_list):
    time_list = np.zeros(len(N_list))
    for i in range(len(N_list)):
        N = N_list[i]
        answer, mesh, time_taken, count = solver(N,d,method)
        time_list[i] = time_taken
    return time_list

def timing(N_list):
    fo = open("timing_2.txt",'w')
    fo.write("method\t\tdim\t\tN\n")
    for method in range(3):
        for d in range(3):
            line_str = str(method)+'\t\t'+str(d+1)+'\t\t'
            if d== 0:
                time_list = timing_method(d+1,method,N_list)
            elif d ==1:
                if method == 0:
                    time_list = timing_method(d+1,method,N_list[:8])
                if method == 1:
                    time_list = timing_method(d+1,method,N_list[:9])
                if method == 2:
                    time_list = timing_method(d+1,method,N_list)
            else:
                if method == 0:
                    time_list = timing_method(d+1,method,N_list[:5])
                if method == 1:
                    time_list = timing_method(d+1,method,N_list[:8])
                if method == 2:
                    time_list = timing_method(d+1,method,N_list[:8])
            for i in range(len(time_list)):
                line_str+=str(time_list[i])+'\t\t'
            fo.write(line_str+'\n')
    fo.close()
            
def CG_count(N_list):
    count_list = np.zeros(len(N_list))
    plt.figure()
    for d in range(3):
        for i in range(len(N_list)):
            N = N_list[i]
            if d==3 and N>64:
                continue
            answer,mesh,time_taken,count = solver(N,d+1,1)
            count_list[i] = count
        plt.plot(N_list,count_list,label="dim="+str(d+1))
    plt.legend(loc="best")
    plt.show()
#------------------------------------------------------------------------------------------------------------------------------------#


#test_accuracy(3, [ [4,8,16,32],[4,8,16,32,64],[4,8,16,32,64,128]] )

timing([4,8,16,24,32,48,64,128,256,512,786,1024,1536])

#CG_count([4,8,16,32,64,128])

#N_list= [4,8,16,24,32,48,64,128,256,512,786,1024,1536]
#N_list= [2**12,2**14,2**16]
#time_list_1 = timing_method(1,0,N_list[:9])
#time_list_2 = timing_method(1,1,N_list)

#plt.loglog(N_list[:9],time_list_1,'^--', label="CG")
#plt.loglog(N_list,time_list_2,'o--',label="fast")
#plt.legend(loc="best")
#plt.show()
