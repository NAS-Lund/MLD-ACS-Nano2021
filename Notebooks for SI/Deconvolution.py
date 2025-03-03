import numpy as np
from scipy import optimize
from scipy import signal
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


class XRD_decon():
    
    def __init__(self,filename,x0=[5.0,10.0,1.0,10.0,1.0],q_zero = 0.01106768,inst_broadening = 0.007,C = 4):
        '''
        Initializing a XRD_decon object requires a csv file with the following structure: 
        1st column: q values
        2nd column: measured pXRD intensities
        
        filename: string, name of the csv file containing data in the above structure
        
        A parameter guess can be supplied at this point, as well as a zero point correction to the q values. 
        If no guess or correction are supplied, a default guess or correction will be used.
        '''
        self.filename = filename 
        self.data = pd.read_csv(self.filename)
        self.zpc = q_zero
        self.x0 = x0
        self.q_vals = self.data.iloc[:,0].values
        self.intens = self.data.iloc[:,1].values/max(self.data.iloc[:,1].values)
        self.simulated = np.zeros(len(self.q_vals))
        self.old_x = self.x0
        self.q_vals_list = []
        self.intens_list = []
        self.HWHM = inst_broadening
        self.N = C
        
    def error(self,intens = None,a = 0.0, b = 0.03):
        if type(intens)==type(None):
            intens = self.intens
        return  intens*a + b*max(intens)
        
    def sim(self,q, x0):
        '''
        Core function creating the simulation of XRD patterns
        Parameters 
        q_vals: np array of q values to simulate PXRD pattern over
        x0 =  d, L, sigmaL, N, S, sigmaS: scalar variables used in simulation
        (S = Size of crystals, sigmaS = standard deviation of size, d = lattice spacing, 
        L = inter-crystal space, sigmaL = standard deviation of intra-crystal space, N = min. length of superlattice, zpc = zero point correction to q-values)
        if only 5 or 6 values are passed, N is not fitted and defined separately over the N keyword option. If 6 values are passed, the 6th value is used as a zero point correction.
        HWHM: scalar variable, creates a gaussian convolution for each point with the HWHM of said gaussian.
        
        returns: np.array : self.simulated, a simulated XRD pattern based on the parameters given of length q_vals
        this simulation is based on the following paper: 
        Eric E. Fullerton, Ivan K. Schuller, H. Vanderstraeten, and Y. Bruynseraede
        Phys. Rev. B 45, 9292 – Published 15 April 1992 https://doi.org/10.1103/PhysRevB.45.9292
        '''
        # creates a discrete distribution of integer sizes based on a mean and standard deviation (scalar variables)
        # return two np arrays integer sizes, distribution
        
        
        
        if type(x0) == type(None):
            x0 = self.x0
            
        if len(x0)==5:
            d,L,sigmaL,S,sigmaS = x0
            N = self.N
            zpc = self.zpc
        elif len(x0)==6:
            d,L,sigmaL,S,sigmaS,zpc = x0
            N = self.N
        elif len(x0)==7:
            d,L,sigmaL,S,sigmaS,zpc,N = x0 
        else:
            raise ValueError ('There should be 5, 6 or 7 starting parameters for the simulation: d, L, sigmaL, (N), S, sigmaS, (zpc)')
        
        if type(q) == type(None):
            q = self.q_vals
        q = q + zpc
        
        dist_x = np.arange(np.floor(S - 3*sigmaS),np.ceil(S + 3*sigmaS),1)
        dist_y = np.exp(-0.5*(np.square((dist_x-S)/sigmaS)))
        dist_y = dist_y/sum(dist_y)

        Icalc = np.zeros(len(q))
        for i in range(len(dist_x)):

            # GENERATE NC SIZE RELATED AVERAGED QUANTITIES

            F = (1 - np.exp(1j*q*dist_x[i]*d))/(1 - np.exp(1j*q*d))
            FconjF = F*np.conj(F)
            PHI = np.exp(1j*(dist_x[i]-1)*q*d)*np.conj(F)
            T = np.exp(1j*(dist_x[i]-1)*q*d)

            # SIMULATE DIFFRACTION PROFILE

            XI = 1j*q*L - q**2*sigmaL**2/2

            IA = N * (2 * FconjF + 2*np.real(np.exp(XI)*PHI*F))
            IB = np.exp(-XI)* PHI * F / T**2 + 2* PHI * F / T + np.exp(XI) *PHI * F
            IC = ((N - (N+1)*np.exp(2*XI)*(T**2) + (np.exp(2*XI)*(T**2))**(N+1)) / (1 - np.exp(2*XI)* T*T)**2 - N)

            Ij = IA + 2*np.real (IB* IC)
            Ij = np.real (Ij)

            Icalc = Icalc + dist_y[i] * Ij

        Iconv = np.zeros(len(q))
        for i in range(len(q)):
            Iconv = Iconv + Icalc[i]*np.exp(-(q[i]-q)**2/(2*self.HWHM**2))
            
        self.simulated = Iconv/max(Iconv)

        return self.simulated
    
    def Residuals(self,x0,q_vals,intenses,fit_derivs = True,weights = None,fit_errors = False):
        # implements a function that calculates residuals for multiple q-ranges for non-linear least squares fitting
        res = np.array([])
        self.sim_list = []
        if weights == None:
            weights = [1.0 for i in range(len(q_vals))]
        if fit_derivs:
            self.dsim_list = []
            self.dintens_list = []
            for i in range(len(q_vals)):
                sims = self.sim(q_vals[i],x0)
                dsims = signal.savgol_filter(sims,15,3,deriv=1)
                dsims = dsims/max(dsims)
                dintens = signal.savgol_filter(intenses[i],15,3,deriv=1)
                dintens = dintens/max(dintens)
                self.sim_list.append(sims)
                self.dsim_list.append(dsims)
                self.dintens_list.append(dintens)
                res = np.hstack((res,weights[i]*np.array(list(sims-intenses[i]) + list(dsims-dintens))))
            return res
        else:
            for i in range(len(q_vals)):
                sims = self.sim(q_vals[i],x0)
                self.sim_list.append(sims)
                res = np.hstack((res,weights[i]*(sims-intenses[i])))
            return res
    
    def MSE(self,x0,fit_derivs = True):
        # implements a function that calculates ChiSquared for multiple q-ranges (right now, MSE, since we don't have error bars)
        res = np.array([])
        self.sim_list = []
        intenses = self.intens_list
        q_vals = self.q_vals_list
        if fit_derivs:
            self.dsim_list = []
            self.dintens_list = []
            for i in range(len(q_vals)):
                sims = self.sim(q_vals[i],x0)
                dsims = signal.savgol_filter(sims,15,3,deriv=1)
                dsims = dsims/max(dsims)
                dintens = signal.savgol_filter(intenses[i],15,3,deriv=1)
                dintens = dintens/max(dintens)
                self.sim_list.append(sims)
                self.dsim_list.append(dsims)
                self.dintens_list.append(dintens)
                res = np.hstack((res,np.array(list(sims-intenses[i]) + list(dsims-dintens))))
            return np.sum(res**2)/len(res)
        else:
            for i in range(len(q_vals)):
                sims = self.sim(q_vals[i],x0)
                self.sim_list.append(sims)
                res = np.hstack((res,sims-intenses[i]))
            return np.sum(res**2)/len(res)
        
    def ChiSquared(self,x0,fit_derivs = True,error_func = None,err_kw = {'a' : 0.0, 'b' : 0.03}):
        # implements a function that calculates ChiSquared for multiple q-ranges 
        res = np.array([])
        self.sim_list = []
        intenses = self.intens_list
        q_vals = self.q_vals_list
        if type(error_func)==type(None):
            error_func = self.error

        errors = np.array([])
        if fit_derivs:
            self.dsim_list = []
            self.dintens_list = []
            for i in range(len(q_vals)):
                sims = self.sim(q_vals[i],x0)
                dsims = signal.savgol_filter(sims,15,3,deriv=1)
                dsims = dsims/max(dsims)
                dintens = signal.savgol_filter(intenses[i],15,3,deriv=1)
                dintens = dintens/max(dintens)
                self.sim_list.append(sims)
                self.dsim_list.append(dsims)
                self.dintens_list.append(dintens)
                res = np.hstack((res,np.array(list(sims-intenses[i]) + list(dsims-dintens))))
                errors = np.hstack((errors,np.array(list(error_func(intens = intenses[i],**err_kw)) + list(np.sqrt(2)*error_func(intens = intenses[i],**err_kw)/np.gradient(q_vals[i])))))
        else:
            for i in range(len(q_vals)):
                sims = self.sim(q_vals[i],x0)
                self.sim_list.append(sims)
                res = np.hstack((res,sims-intenses[i]))
                errors = np.hstack((errors,np.array(error_func(intens = intenses[i],**err_kw))))

        return np.sum(res**2/errors**2)/(len(res)-len(x0))
    
    def load_data(self,filename,zero_point_correction = 0.01106768):
        # loads a new data file. Ideally, this function will not to be used, as a corect file can be supplied at initialization.
        # this function will overwrite the data from the initial file and change the data to the new file supplied.
        # the data structure of the file should be the same, with qvalues in the first column and measured X-ray intensities in the second
        self.filename = filename 
        self.data = pd.read_csv(self.filename)
        self.zpc = zero_point_correction
        self.q_vals = self.data.iloc[:,0].values
        self.intens = self.data.iloc[:,1].values/max(self.data.iloc[:,1].values)
        self.simulated = np.zeros(len(self.q_vals))
        return(self.q_vals,self.intens)
    
    def fit(self, q_ranges,x0 = None,bounds = None,fit_derivs = True,weights = None):
        q_vals = []
        intenses = []
        for q_range in q_ranges:
            print(q_range)
            data_sel = self.data[(self.data.iloc[:,0]>q_range[0]) & (self.data.iloc[:,0]<q_range[1])]
            q_vals.append(data_sel.iloc[:,0].values)
            self.q_vals = q_vals[-1]
            intenses.append(data_sel.iloc[:,1].values/max(data_sel.iloc[:,1].values))
            self.intens = intenses[-1]

        self.intens_list = intenses
        self.q_vals_list = q_vals
        res = optimize.least_squares(self.Residuals,x0=x0,bounds = bounds,args = (q_vals,intenses),kwargs = {'fit_derivs' : fit_derivs,'weights' : weights},ftol = 10**(-12), gtol = None,xtol=None)
        self.x0 = res.x
        
        return res
    
    
    def fit_bootstrap(self, q_ranges,x0 = None,bounds = None,fit_derivs = True,n_random = 100,error_func = None,err_kw = {'a' : 0.0, 'b' : 0.03},weights = None,verbose = False):
        results = []
        i = 0
        n_failed = 0
        if type(error_func)==type(None):
            error_func = self.error

        while i < n_random:
            
            q_vals = []
            intenses = []
            for q_range in q_ranges:
                data_sel = self.data[(self.data.iloc[:,0]>q_range[0]) & (self.data.iloc[:,0]<q_range[1])]
                q_vals.append(data_sel.iloc[:,0].values)
                self.q_vals = q_vals[-1]
                error_dat = np.random.normal(loc = data_sel.iloc[:,1].values,scale = error_func(intens = data_sel.iloc[:,1].values,**err_kw),size = len(self.q_vals) )
                intenses.append(error_dat/max(error_dat))
                self.intens = intenses[-1]

            self.intens_list = intenses
            self.q_vals_list = q_vals
            if verbose:
                print("Starting Iteration " + str(i))
            try:
                res = optimize.least_squares(self.Residuals,x0=x0,bounds = bounds,args = (q_vals,intenses),kwargs = {'fit_derivs' : fit_derivs, 'weights' : weights},ftol = 10**(-12), gtol = None,xtol=None)
            except np.linalg.LinAlgError:
                n_failed +=1
                continue
            results.append([])
            self.x0 = res.x
            results[i].append(list(res.x) + list(res.fun))
            if verbose:
                print('Iteration ' + str(i) + ' results: ' + str(self.x0))
            if n_failed > 100:
                print('This fitting procedure has failed too often. Try using a more reasonable starting guess, or a more reasonable error approximation')
                break
                
            i+=1

        return results
    
    def get_lin_uncertainties(self,bounds,q_ranges = None,x0 = None,n_points=100,error_func = None,err_kw = {'a' : 0.0, 'b' : 0.03}):
        '''
        

        '''

        if type(x0)==type(None):
            x = self.x0
        else:
            x = x0

        params = []
        results = []
        if not type(q_ranges)==type(None):
            q_vals = []
            intenses = []
            for q_range in q_ranges:
                print(q_range)
                data_sel = self.data[(self.data.iloc[:,0]>q_range[0]) & (self.data.iloc[:,0]<q_range[1])]
                q_vals.append(data_sel.iloc[:,0].values)
                self.q_vals = q_vals[-1]
                intenses.append(data_sel.iloc[:,1].values/max(data_sel.iloc[:,1].values))
                self.intens = intenses[-1]

            self.intens_list = intenses
            self.q_vals_list = q_vals


        for i in range(len(x)):

            results.append([])
            xi = np.array([x for m in range(n_points)])
            # Sweep parameter i over the bounds supplied to the model 
            # while keeping all other parameters at optimum values
            xi[:,i] = np.linspace(bounds[0][i],bounds[1][i],n_points)
            # predict error metric for each point
            for j in range(len(xi)):
                results[i].append(self.ChiSquared(xi[j,:],error_func = error_func, err_kw = err_kw))
            params.append(xi[:,i])

        return params,results
    
        
        