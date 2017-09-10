import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
def find_std(x):
    #x = np.linspace(15, 50, 500)
    #E=np.empty((5,x.shape[0]))
    #ERR=np.empty(x.shape[0])
    E=np.empty(5)# pressure std
    E[0] = 0.0946351550329 * x + 0.770132957906
    E[1] = 0.0853694646877 * x + 4.03822156582
    E[2] = 0.0845234089432 * x + 2.42978154566
    #E[3] = 0.175470966994 * x + -1.97749171132
    E[4]= 0.129940121323 * x + 0.550779010841
    return np.std(E)
def find_mean(x):
    E=np.empty(4)# pressure std
    E[0] = 0.0946351550329 * x + 0.770132957906
    E[1] = 0.0853694646877 * x + 4.03822156582
    E[2] = 0.0845234089432 * x + 2.42978154566
    #E[3] = 0.175470966994 * x + -1.97749171132
    E[3]= 0.129940121323 * x + 0.550779010841
    return E
x = np.linspace(15, 50, 5000)
E=np.empty((5,x.shape[0]))
#ERR=np.empty(x.shape[0])
E[0] = 0.0946351550329 * x + 0.770132957906
E[1] = 0.0853694646877 * x + 4.03822156582
E[2] = 0.0845234089432 * x + 2.42978154566
#E[3] = 0.175470966994 * x + -1.97749171132
E[4]= 0.129940121323 * x + 0.550779010841
ERR=np.std(E,0)
np.save('ERR',ERR)
plt.scatter(x,1.96*ERR,s=0.01)
aa = r'^{\circ}C'
plt.xlabel("Temperature(" + '$%s$' % aa + ')')
plt.ylabel("Pressure(atm)")
plt.title("Absolute Error Distribution with 95%CI")
plt.savefig('Absolute Error Distribution with 95CI' )
plt.close()
E=find_mean(20)
print(ERR)
print(scipy.stats.ttest_1samp(E,np.mean(E)+0.1*np.random.rand()))
P=np.array([2.3,2.6,3.8,5.3])
Q=abs(P-np.mean(P))/np.std(P)
print(np.std(P))
print(2*abs((scipy.stats.norm(np.mean(P), np.std(P)).cdf(P))-0.5))
print(0.6903711* 0.55394045*0.20050464*0.87249167)
pass
