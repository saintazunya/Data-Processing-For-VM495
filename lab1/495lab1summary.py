import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sympy import *
import scipy.stats
def find_std(x):
    #x = np.linspace(15, 50, 500)
    #E=np.empty((5,x.shape[0]))
    #ERR=np.empty(x.shape[0])
    E=np.empty(5)# pressure std
    E[0] = 0.0946351550329 * x + 0.770132957906
    E[1] = 0.0853694646877 * x + 4.03822156582
    E[2] = 0.0845234089432 * x + 2.42978154566
    E[3] = 0.175470966994 * x + -1.97749171132
    E[4]= 0.129940121323 * x + 0.550779010841
    return np.std(E)
class data_plot:
    def __init__(self,a):
        '''
        t = 1e-4
        r = 33.2e-3
        S = 2
        E = 69e10
        v = 0.35
        R2 = 119
        R4 = 119
        Vs = 5
        A = 100
        '''
        self.a=a
        self.const = 1.93e1
        if a == 1:
            self.d = np.load('a.npy')
            self.refV1 = self.d[-500, 2]
            self.refV2 = self.d[-500, 3]
        elif a == 2:
            self.d = np.load('a2.npy')
            self.refV1 = self.d[-500, 2]
            self.refV2 = self.d[-500, 3]
        elif a == 3:
            self.d = np.load('a3.npy')
            self.refV1 = self.d[-500, 2]-0.3
            self.refV2 = self.d[-500, 3]
        elif a == 4:
            self.d = np.load('a4.npy')
            self.refV1 = self.d[-500, 2]-0.5
            self.refV2 = self.d[-500, 3]
        elif a == 5:
            self.d = np.load('a5.npy')
            self.refV1 = self.d[-500, 2]+0.15
            self.refV2 = self.d[-500, 3]
        self.e = np.empty([int(self.d.shape[0] / 50), 5], dtype="float64")
        self.totallen=int(self.d.shape[0] / 50)-3
        for i in range(self.totallen ):
            self.e[i, 0] = self.d[i * 50, 1]  # temperature
            self.e[i, 1] = self.const * ((self.d[i * 50, 2] - self.refV1) + 0.35 * (self.d[i * 50, 3] - self.refV2))  # pressure
            self.e[i, 2] = self.d[i * 50, 0]  # time
            self.e[i, 3] = self.d[i * 50, 2]  # v1
            self.e[i, 4] = self.d[i * 50, 3]  # v2
    def find_frequency(self):
        self.freq=np.fft.fft(self.d[:,2])
        self.freq_axis = np.linspace(0, 44100, len(self.d))
        plt.plot(self.freq_axis[0:int(len(self.d)/2)],self.freq[0:int(len(self.d)/2)])
        plt.xlabel('Frequency')
        plt.ylabel("Amplitude")
        plt.title("FFT of Pressure Data")
        plt.axis([0, 20000, 0, 3000])
        plt.savefig('FFT of Pressure Data'+str(self.a))
        #plt.show()
        plt.close()
    def linear_fit(self,startpoint,lenofplot):
        self.coeff = np.polyfit(self.e[startpoint:lenofplot, 0], self.e[startpoint:lenofplot, 1], 1)
        self.predict=self.coeff[0] * self.e[0:lenofplot, 0] + self.coeff[1]
        print('regression:', self.coeff[0], '*x+', self.coeff[1])
    def plot_pressure_temperature(self,lenofplot,startpoint=0):
        plt.figure('pressure_temperature'+str(self.a))
        plt.scatter(self.e[startpoint:lenofplot, 0], self.e[startpoint:lenofplot, 1], s=0.01,c='r',label='Original Data')
        plt.axis([15, 45, 0, 10])
        plt.plot(self.e[startpoint:lenofplot, 0],self.predict,c='b',label='Fitted Data')
        self.confidence_interval()
        plt.plot(self.e[startpoint:lenofplot, 0],self.upbb,c='y',label='95% CI')
        plt.plot(self.e[startpoint:lenofplot, 0], self.lobb, c='y')
        aa=r'^{\circ}C'
        plt.xlabel("Temperature("+'$%s$' %aa+')')
        plt.ylabel("Pressure(atm)")
        plt.title("Temperature-Pressure")
        plt.legend(scatterpoints=1,loc=4)
        plt.savefig('pressure_temperature'+str(self.a))
        plt.close()
        #plt.show()
    def plot_time_v(self,startpoint=0):
        plt.figure('v-time'+str(self.a))
        lenofplot = self.totallen
        plt.scatter(self.e[startpoint:lenofplot, 2]*500, self.e[startpoint:lenofplot, 3],c='b', s=0.01,label='Time-V1')
        plt.scatter(self.e[startpoint:lenofplot, 2]*500, self.e[startpoint:lenofplot, 4],c='r', s=0.01,label='Time-V2')
        plt.axis([0,4000*10,0,4])
        plt.xlabel("Time(s)")
        plt.ylabel("Voltage(V)")
        plt.title("Time-Voltage")
        plt.legend(scatterpoints=1,loc=4)
        plt.savefig('Time-Voltage' + str(self.a))
        plt.close()
        #plt.show()
        return None
    def plot_temperture_v(self,startpoint=0):
        plt.figure('Temperature-Voltage'+str(self.a))
        lenofplot = self.totallen
        plt.scatter(self.e[startpoint:lenofplot, 0], self.e[startpoint:lenofplot, 3],c='b', s=0.01,label='Temperature-V1')
        plt.scatter(self.e[startpoint:lenofplot, 0], self.e[startpoint:lenofplot, 4],c='r', s=0.01,label='Temperature-V2')
        plt.axis([0, 55,0,4])
        aa=r'^{\circ}C'
        plt.xlabel("Temperature("+'$%s$' %aa+')')
        plt.ylabel("Voltage(V)")
        plt.title("Temperature-Voltage")
        plt.legend(scatterpoints=1,loc=4)
        plt.savefig('Temperature-Voltage' + str(self.a))
        plt.close()
        #plt.show()
        return None
    def PCC(self,lenofplot,startpoint):
        with open("pcc%d.txt"%self.a, "w") as f:
            f.write('Peasonor Coefficient is: '+str(pearsonr(self.e[startpoint:lenofplot, 0], self.e[startpoint:lenofplot, 1]))+'\n'
                    '95%confidence interval=')

        return None
    def error_analysis(self,lenofplot,startpoint):
        self.mse=mean_squared_error(self.e[startpoint:lenofplot, 1], self.coeff[0] * self.e[startpoint:lenofplot, 0] + self.coeff[1])
        self.se=np.sqrt(self.mse)
        print('self.se is ',self.se,'self.mse is',self.mse)
    def plot_std_residual(self,a,b):
        plt.scatter(a,b,s=0.01,label='Standardlized Errors')
        aa=r'^{\circ}C'
        plt.xlabel("Temperature("+'$%s$' %aa+')')
        plt.ylabel("std error(V)")
        plt.title("Residual")
        plt.axis([15,45,-4,4])
        plt.legend(scatterpoints=1,loc=4)
        plt.savefig('Standardized Errors' + str(self.a))
        plt.close()
        return None
    def confidence_interval(self):
        #95% CI
        z=1.96
        self.upbb=self.predict+z*self.se
        self.lobb=self.predict-z*self.se
    def exclude_outliers(self,startpoint,lenofplot,a):
        self.residual=self.predict-self.e[0:lenofplot,1]
        self.plot_std_residual(self.e[0:lenofplot,0],self.residual)
        out_index=[]
        for idx,val in enumerate(self.residual):
            if abs(val)>2*self.se:
                out_index.append(idx)
        self.e=np.delete(self.e,out_index,0)
        return None
    def make_error_bar(self):
        ut=5e-5
        uD=5e-4
        uV=1e-6
        uS=2*0.01
        uVs=27.5e-3
        t=Symbol('t')
        E=69e9
        D=Symbol('D')
        v1=Symbol('v1')
        v2=Symbol('v2')
        refv1=Symbol('refv1')
        refv2=Symbol('refv2')
        vs=Symbol('vs')
        A=98.8
        S=2
        eq=(16*E*t*(v1-refv1+0.35*v2-refv2))/(A*D*S*vs*0.8775)
        self.uncertainty=(diff(eq,t)*ut)**2+(diff(eq,D)*uD)**2+(diff(eq,v1)*uV)**2+(diff(eq,v2)*uV)**2+(diff(eq,refv1)*uV)**2+(diff(eq,refv2)*uV)**2+(diff(eq,vs)*uVs)**2
        print(self.uncertainty)
    def plot_error_bar(self,startpoint,lenofplot):
        t=1e-4
        D=66.4e-3
        vs=5
        refv1=self.refV1
        refv2=self.refV2
        v1=self.e[startpoint:lenofplot,2]
        v2=self.e[startpoint:lenofplot,3]
        error_bar=np.empty(v1.shape[0])
        error_random=np.empty(v1.shape[0])
        relative_error=np.empty(v1.shape[0])
        relative_random=np.empty(v1.shape[0])
        real_realtive=np.empty(v1.shape[0])
        ERR=np.load('ERR.npy')
        aa = r'^{\circ}C'
        for i in range(v1.shape[0]):
            error_bar[i]=1e-6*sqrt(12658214858.7309 * t ** 2 / (D ** 2 * vs ** 2) + 3.0657405882835e+18 * t ** 2 * (
                                                                                            -refv1 - refv2 + v1[i] + 0.35 * v2[i]) ** 2 / (
                                                             D ** 2 * vs ** 4) + 10134679630689.3 * (
                                                                                                    -refv1 - refv2 + v1[i] + 0.35 * v2[i]) ** 2 / (
                                                                                 D ** 2 * vs ** 2) + 1.01346796306893e+15 * t ** 2 * (
                                                                                                                                     -refv1 - refv2 + v1[i] + 0.35 * v2[i]) ** 2 / (
                                                                                                     D ** 4 * vs ** 2))+self.predict[i]*0.3*100000
            error_random[i]=error_bar[i]-self.predict[i]*0.3*100000# random error, theroy
            relative_error[i]=(error_bar[i]/self.predict[i])/10000# not used
            relative_random[i]=(error_random[i]/self.predict[i])/10000# random error, relative, theory
            real_realtive[i]=1.96*find_std(self.e[i,0])/self.predict[i]# real error based on 4 trials, 95%CI
        '''plt.figure('Absolute Uncertainty Distribution' + str(self.a))
        plt.scatter(self.e[startpoint:lenofplot,0],error_bar/100000,s=0.01)
        plt.xlabel("Temperature(" + '$%s$' % aa + ')')
        plt.ylabel("Absolute Uncertainty (atm)")
        plt.title("Absolute Uncertainty Distribution")
        plt.savefig('absolute error distribution' + str(self.a))
        plt.close()

        #------------------------------------
        plt.figure('Relative Uncertainty Distribution'+str(self.a))
        plt.scatter(self.e[startpoint:lenofplot,0],relative_error*100,s=0.01)
        aa = r'^{\circ}C'
        plt.xlabel("Temperature(" + '$%s$' % aa + ')')
        plt.ylabel("Relative Uncertainty bar(%)")
        plt.title("Relative Uncertainty Distribution")
        plt.savefig('relative error distribution' + str(self.a))
        plt.close()
        '''
        #---------------------------
        plt.figure('Theoretical Absolute Uncertainty Distribution')
        plt.scatter(self.e[startpoint:lenofplot,0],(error_random)/100000,s=0.01)
        plt.xlabel("Temperature(" + '$%s$' % aa + ')')
        plt.ylabel("Absolute Uncertainty Error bar(atm)")
        plt.title("Theoretical Absolute Uncertainty Distribution")
        plt.savefig('Theoretical Absolute Uncertainty Distribution' + str(self.a))
        plt.close()
        #-----------------------
        plt.figure('Theoretical Relative Uncertainty Error Distribution')
        plt.scatter(self.e[startpoint:lenofplot,0],(relative_random)*100,s=0.01)
        plt.xlabel("Temperature(" + '$%s$' % aa + ')')
        plt.ylabel("Theoretical Relative Uncertainty Error bar(%)")
        plt.title("Theoretical Relative Uncertainty Error Distribution")
        plt.savefig('Theoretical Relative Uncertainty Error Distribution' + str(self.a))
        plt.close()
        #-----------------
        '''
        plt.figure('random only relative')
        plt.scatter(self.e[startpoint:lenofplot,0],(relative_random)*100,s=0.01)
        plt.xlabel("Temperature(" + '$%s$' % aa + ')')
        plt.ylabel("Relative Uncertainty Error bar(%)")
        plt.title("Relative Uncertainty Error Distribution")
        plt.savefig('Relative Uncertainty error distribution' + str(self.a))
        plt.close()
        '''
        #----------------------
        plt.figure('Real Relative Uncertainty Error Distribution')
        plt.scatter(self.e[startpoint:lenofplot,0],real_realtive*100,s=0.01)
        plt.xlabel("Temperature(" + '$%s$' % aa + ')')
        plt.ylabel("Real Relative Uncertainty Error Distribution(%)")
        plt.title("Real Relative Uncertainty Error Distribution")
        plt.savefig('Real Relative Uncertainty Error Distribution' + str(self.a))
        plt.close()
def linear_regression(x, y, prob):
    """
    Return the linear regression parameters and their <prob> confidence intervals.
    ex:
    #>>> linear_regression([.1,.2,.3],[10,11,11.5],0.95)
    """
    x = np.array(x)
    y = np.array(y)
    n = len(x)
    xy = x * y
    xx = x * x

    # estimates

    b1 = (xy.mean() - x.mean() * y.mean()) / (xx.mean() - x.mean() ** 2)
    b0 = y.mean() - b1 * x.mean()
    s2 = 1. / n * sum([(y[i] - b0 - b1 * x[i]) ** 2 for i in xrange(n)])
    print(    'b0 = ', b0)
    print('b1 = ', b1)
    print('s2 = ', s2)

    # confidence intervals

    alpha = 1 - prob
    c1 = scipy.stats.chi2.ppf(alpha / 2., n - 2)
    c2 = scipy.stats.chi2.ppf(1 - alpha / 2., n - 2)
    print(
    'the confidence interval of s2 is: ', [n * s2 / c2, n * s2 / c1])

    c = -1 * scipy.stats.t.ppf(alpha / 2., n - 2)
    bb1 = c * (s2 / ((n - 2) * (xx.mean() - (x.mean()) ** 2))) ** .5
    print(
    'the confidence interval of b1 is: ', [b1 - bb1, b1 + bb1])

    bb0 = c * ((s2 / (n - 2)) * (1 + (x.mean()) ** 2 / (xx.mean() - (x.mean()) ** 2))) ** .5
    print    ('the confidence interval of b0 is: ', [b0 - bb0, b0 + bb0])
    return None
def analysis(a,lenofplot):
    print('This is the ',a,'th trial')
    if a==3:
        startpoint=4000
    elif a==4:
        startpoint=3000
    elif a==5:
        startpoint=2000
    else:
        startpoint=1000
    trial=data_plot(a)
    #-----initial fit
    trial.linear_fit(startpoint,lenofplot)
    trial.PCC(lenofplot,startpoint)
    trial.error_analysis(lenofplot,startpoint)
    trial.exclude_outliers(startpoint,lenofplot,a)
    if a==2:
        trial.linear_fit(startpoint, lenofplot)
        trial.PCC(lenofplot, startpoint)
        trial.error_analysis(lenofplot, startpoint)
        trial.exclude_outliers(startpoint, lenofplot, a)
    #-------------------------
    trial.linear_fit(0, lenofplot)
    trial.plot_pressure_temperature(lenofplot)
    trial.plot_temperture_v()
    trial.plot_time_v()
    trial.plot_error_bar(startpoint,lenofplot)
    return None
def analysis_not_killed(a,lenofplot):
    print('Outliers not killed')
    print('This is the ', a, 'th trial')
    startpoint=0
    trial = data_plot(a)
    trial.linear_fit(startpoint, lenofplot)
    trial.PCC(lenofplot, startpoint)
    trial.error_analysis(lenofplot, startpoint)
    #trial.plot_pressure_temperature(lenofplot)
    #trial.plot_temperture_v()
    #trial.plot_time_v()
    #trial.exclude_outliers(startpoint, lenofplot, a)
def combine(a):

    x=np.linspace(15,50,500)
    y1=0.0946351550329 *x+ 0.770132957906
    y2=0.0853694646877 *x+ 4.03822156582
    y3=0.0845234089432 *x+ 2.42978154566
    y4=0.175470966994 *x+ -1.97749171132
    y5=0.129940121323 *x+ 0.550779010841
    plt.figure('combine'+str(a))
    if a==1:
        plt.plot(x,y1,c='r',label='trail1')
        plt.plot(x, y2, c='g', label='trail2')
        plt.plot(x, y3, c='b', label='trail3')
        plt.plot(x, y5, c='k', label='trail5')
    if a==2:
        plt.plot(x, y4, c='y', label='trail4')
        plt.plot(x, y5, c='k', label='trail5')
    aa = r'^{\circ}C'
    plt.xlabel("Temperature(" + '$%s$' % aa + ')')
    plt.ylabel("Pressure(atm)")
    plt.title("Temperature-Pressure")
    plt.legend(scatterpoints=1, loc=4)
    plt.savefig('combined'+str(a))
    plt.close()

for i in range(1,6):
   analysis(i,12000)

for i in range(1,6):
   trial = data_plot(i)
   trial.find_frequency()
