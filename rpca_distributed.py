import numpy as np
import torch
import math
# from rpca import RPCA
import argparse
from scipy.sparse import random as srandom
import matplotlib.pyplot as plt
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--method",type=str,default="DCF",choices=["DCF","APGM","CF"])
parser.add_argument("--num_clients", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--tau", type=int, default=10)
parser.add_argument("--epoch", type=int, default=200)
parser.add_argument("--M",type=int,default=500)
parser.add_argument("--N",type=int,default=500)
parser.add_argument("--R",type=float,default=0.05)
parser.add_argument("--sparsity",type=float,default=0.05)
parser.add_argument("--output",type=str,default="data.pkl")
parser.add_argument("--mu0",type=float,default=1)
args = parser.parse_args()

class rpca_distributed():
    def __init__(self,args):
        self.M = args.M
        self.N = args.N
        self.rank_gt = int(args.R*min(self.M,self.N))
        self.rank = int(self.rank_gt)
        self.sparsity = args.sparsity
        self.L_gt = self._random_low_rank_matrix() 
        self.max_entry = np.max(np.abs(self.L_gt))
        # self.S_gt = self._random_sparse_matrix("Gaussian") * math.sqrt(self.rank_gt)
        self.S_gt = self._random_sparse_matrix("Impulsive")
        self.target = self.L_gt + self.S_gt

        self.E = args.num_clients
        self.lr0 = args.lr
        self.tau = args.tau
        self.local_matrix = self._split_matrix()
        self.U = np.random.randn(self.M, self.rank)
        self.n_epoch = args.epoch
        self.local_V, self.local_S = self.setup()
        self.rho1, self.rho2 = 1/math.sqrt(min(self.M,self.N)), args.mu0

    def _random_low_rank_matrix(self):
        U = np.random.randn(self.M,self.rank_gt)
        V = np.random.randn(self.rank_gt, self.N)
        return U @ V

    def _random_sparse_matrix(self,type = "Impulsive"):
        if type == "Impulsive":
            r = srandom(self.M,self.N,density=self.sparsity).A
            random_M = np.random.randn(self.M,self.N)
            return np.sign(random_M)*r*np.sqrt(self.M*self.N)
        elif type == "Gaussian":
            return srandom(self.M,self.N,density=self.sparsity).A
        else:
            raise NotImplementedError

    def _split_matrix(self):
        columns_per_client = self.target.shape[1] // self.E
        local_matrix = []
        for i in range(self.E - 1):
            local_matrix.append(self.target[:,columns_per_client*i:columns_per_client*(i+1)])
        local_matrix.append(self.target[:, columns_per_client*(self.E-1):])
        return local_matrix
    
    def setup(self):
        localV, localS = [], []
        for i in range(self.E):
            ni = self.local_matrix[i].shape[1]
            localV.append(np.zeros((ni,self.rank)))
            localS.append(np.zeros((self.M,ni)))
        return localV, localS

    def train(self):
        err = []
        err.append(self.evaluate())
        for epoch in range(self.n_epoch):
            # print(epoch)
            self.lr = self.lr0/np.sqrt(epoch+1)
            # self.lr = self.lr0
            updated_U = []
            for client in range(self.E):
                localU = self.U.copy()
                V = self.local_V[client].copy()
                S = self.local_S[client].copy()
                M = self.local_matrix[client]
                for t in range(self.tau):
                    loss = self.loss(localU,V,S,M)
                    # print(loss)
                    V = self.argminV(localU,S,M)
                    # loss = self.loss(localU,V,S,M)
                    S = self.argminS(localU,V,M)
                    self.local_S[client] = S
                    self.local_V[client] = V
                    localU = localU - self.lr * self.gradientU(localU, V,S,M)
                updated_U.append(localU)
            U1 = self.U.copy()
            self.aggregate(updated_U)
            # print(epoch)
            if np.linalg.norm(U1 - self.U,ord='fro')/np.linalg.norm(U1,ord='fro') < 1e-5:
                break
            err.append(self.evaluate())

        totalV = np.concatenate(self.local_V,axis=0)
        totalU = self.U
        totalS = np.concatenate(self.local_S,axis=1)

        return totalU@totalV.T, totalS, err

    def argminV(self, U,S,M):
        D = (U.T @ U + self.rho1*np.eye(self.rank))
        return (np.linalg.inv(D) @ U.T @ (M-S)).T
    
    def argminS(self, U,V,M):
        # print(U.shape,V.shape)
        P = M - U@(V.T)
        return np.sign(P) * np.maximum(np.abs(P)-self.rho2, 0)

    def loss(self, U,V,S,M):
        return np.linalg.norm(U@V.T + S - M,ord='fro')**2 + self.rho1*np.linalg.norm(V,ord='fro')**2 + 2*self.rho2*np.abs(S).sum()

    def gradientU(self, U,V,S,M):
        return (U@V.T + S - M)@V + self.rho1*V.shape[0]*U/self.N
    
    def aggregate(self, updated_U):
        newU = np.zeros_like(self.U)
        for client in range(self.E):
            newU += self.local_matrix[client].shape[1] * updated_U[client]
        self.U = newU / self.target.shape[1]

    def getL(self):
        return self.L_gt

    def evaluate(self):
        totalV = np.concatenate(self.local_V,axis=0).copy()
        totalU = self.U.copy()
        totalS = np.concatenate(self.local_S,axis=1).copy()
        L_pred = totalU@totalV.T
        S_pred = totalS
        err_norm = np.linalg.norm(L_pred - self.L_gt,ord='fro')**2 + np.linalg.norm(S_pred - self.S_gt,ord='fro')**2
        ori_norm = np.linalg.norm(self.L_gt,ord='fro')**2 + np.linalg.norm(self.S_gt,ord='fro')**2
        return err_norm/ori_norm

    def APGM(self,mu0=1):
        '''
        This function implements an accelerated proximal gradient method
        [1] Z. Lin, A. Ganesh, J. Wright, L. Wu, M. Chen, and Y. Ma, “Fast convex 
        optimization algorithms for exact recovery of a corrupted low-rank matrix”.
        '''
        m,n = self.target.shape
        mu = np.sqrt(2*max(self.M,self.N))*mu0
        lam = 1/np.sqrt(max(self.M,self.N))
        L0 = np.zeros((m,n)) 
        L1 = np.zeros((m,n)) 
        S0 = np.zeros((m,n)) 
        S1 = np.zeros((m,n)) 
        t0 = 1
        t1 = 1
        mu_iter = mu
        k = 1
        errs = []
        err_norm = np.linalg.norm(L1-self.L_gt,ord='fro')**2 + np.linalg.norm(S1-self.S_gt,ord='fro')**2
        ori_norm = np.linalg.norm(self.L_gt,ord='fro')**2 + np.linalg.norm(self.S_gt,ord='fro')**2
        errs.append(err_norm/ori_norm)
        while 1:
            Y_L = L1 + (t0-1)/t1*(L1-L0)
            Y_S = S1 + (t0-1)/t1*(S1-S0)
            G_L = Y_L - 0.5*(Y_L + Y_S - self.target)
            U, sigmas, V = np.linalg.svd(G_L,full_matrices=False)
            rank = (sigmas > mu_iter/2).sum()
            Sigma = np.diag(sigmas[0:rank] - mu_iter/2)
            L0 = L1
            L1 = U[:,0:rank] @ Sigma @ V[0:rank,:]
            G_S = Y_S - 0.5*(Y_L + Y_S - self.target)
            S0 = S1
            S1 = (G_S - lam*mu_iter/2) * (G_S - lam*mu_iter/2 > 0)
            S1 = S1 + (G_S + lam*mu_iter/2) * (G_S + lam*mu_iter/2 < 0)
            t1, t0 = (np.sqrt(t1**2+1) + 1)/2, t1
            
            # stop the algorithm when converge
            E_L =2*(Y_L - L1) + (L1 + S1 - Y_L - Y_S)
            E_S =2*(Y_S - S1) + (L1 + S1 - Y_L - Y_S) 
            dist = np.sqrt(np.linalg.norm(E_L, ord='fro')**2 + np.linalg.norm(E_S, ord='fro')**2)
            err_norm = np.linalg.norm(L1-self.L_gt,ord='fro')**2 + np.linalg.norm(S1-self.S_gt,ord='fro')**2
            ori_norm = np.linalg.norm(self.L_gt,ord='fro')**2 + np.linalg.norm(self.S_gt,ord='fro')**2
            errs.append(err_norm/ori_norm)
            if k > self.n_epoch or dist < 1e-8:
                break
            else:
                k += 1
        return L1,S1,errs

    def PCP(self):
        m,n = self.target.shape
        S = np.zeros((m,n)) 
        Y = np.zeros((m,n))
        mu = 0.01*m*n/(4*np.abs(self.target).sum())
        lam = 1/np.sqrt(max(self.M,self.N))
        while 1:
            U,sigmas, V = np.linalg.svd(self.target - S - Y/mu,full_matrices=False)
            rank = (sigmas > mu).sum()
            Sigma = np.diag(sigmas[0:rank] - mu)
            L = U[:,0:rank]@Sigma@V[0:rank,:]
            temp = (self.target - L + Y/mu)
            S = (temp - lam*mu)*(temp - lam*mu > 0) + (temp + lam*mu)*(temp+lam*mu<0)
            Y = Y + mu*(self.target - L - S)
            if np.linalg.norm(self.target-L-S,ord='fro') < 1e-8*np.linalg.norm(self.target,ord='fro'):
                break
        errs = []
        err_norm = np.linalg.norm(L-self.L_gt,ord='fro')**2 + np.linalg.norm(S-self.S_gt,ord='fro')**2
        ori_norm = np.linalg.norm(self.L_gt,ord='fro')**2 + np.linalg.norm(self.S_gt,ord='fro')**2
        errs.append(err_norm/ori_norm)
        return L,S,errs

# errs = 0
# for i in range(5):
system = rpca_distributed(args)
if args.method =='DCF':
    L,S,err = system.train()
    a_file = open(args.output, "rb")
    dict = pickle.load(a_file)
    a_file.close()
    data = {"stat": err, "num_clients":args.num_clients, "tau": args.tau, "lambda":args.mu0, "lr":args.lr}
    dict["DCF_"+str(args.M)+"_null"] = data
    a_file = open(args.output, "wb")
    pickle.dump(dict, a_file)
    a_file.close()

elif args.method =='CF':
    args.tau = 1
    args.num_clients = 1
    system = rpca_distributed(args)
    L,S,err = system.train()
    a_file = open(args.output, "rb")
    dict = pickle.load(a_file)
    a_file.close()
    data = {"stat": err, "lr":args.lr}
    dict["CF_"+str(args.M)] = data
    a_file = open(args.output, "wb")
    pickle.dump(dict, a_file)
    a_file.close()

elif args.method =='APGM':
    L,S,err = system.APGM(args.mu0)
    a_file = open(args.output, "rb")
    dict = pickle.load(a_file)
    a_file.close()
    data = {"stat": err, "mu0":args.mu0}
    dict["APGM_"+str(args.M)] = data
    a_file = open(args.output, "wb")
    pickle.dump(dict, a_file)
    a_file.close()
else:
    raise NotImplementedError

print("Relative err:",err[-1])
# errs += err[-1]
# errs += np.linalg.norm(L - system.getL(),ord='fro')**2/np.linalg.norm(system.getL(),ord='fro')**2
# print(L-system.getL())
# print(errs/5)


# system = rpca_distributed(args)
# L,S,err = system.train()
# print(err)
# s,v,d = np.linalg.svd(L)
# s0,v0,d0 = np.linalg.svd(system.getL())
# rank = int(args.M*args.R)
# np.set_printoptions(precision=3)

# if not os.path.exists("data/M{}_r{}_p{}".format(args.M,rank,5+rank)):
#     os.makedirs("data/M{}_r{}_p{}".format(args.M,rank,5+rank))
# np.save("data/M{}_r{}_p{}/singular_value".format(args.M,rank,5+rank),np.array(v[:5+rank]))
# np.save("data/M{}_r{}_p{}/singular_value_gt".format(args.M,rank,5+rank),np.array(v0[:5+rank]))



# errs.append(np.linalg.norm(L - system.getL(),ord='fro')**2/np.linalg.norm(system.getL(),ord='fro')**2)
# print(L-system.getL())



# print(errs)