import numpy as np
import matplotlib.pyplot as plt
def data_plot(filename,stp,filenum):
    path=r"D:\Learning\Undergraduate\vm495\lab2\lingdu"
    file=path+filename+'.txt'
    data_raw=np.loadtxt(file,delimiter=',')
    data=np.empty([int(data_raw.shape[0]/10+1),int(data_raw.shape[1])])
    len = data.shape[0]
    for j in range(data.shape[0]):
        for k in range(data.shape[1]):
            data[j,k]=data_raw[10*j,k]
    acc = np.empty([len])
    x_ref=data[5,1]
    y_ref=data[5,2]
    data[:,1]-=x_ref
    data[:,2]-=y_ref
    accref = np.sqrt(np.square(data[5, 1]) + np.square(data[5, 2]))
    for i in range(data.shape[0]):
        acc[i]=np.sqrt(np.square(data[i,1])+np.square(data[i,2]))-accref
    plt.figure(1)
    plt.plot(data[stp:500,0]/2000,acc[stp:500])
    plt.xlabel('time(s)')
    plt.ylabel('a/g ratio')
    plt.savefig('combined acc-time '+filenum+'.jpg')
    plt.figure(2)
    plt.plot(data[stp:,0],data[stp:,1])
    plt.plot(data[stp:,0],data[stp:,2])
    plt.savefig('x-y acc-time ' + filenum+'.jpg')
    fft_test(acc[stp:],3)
    plt.show()
def fft_test(data,fig_idx):
    plt.figure(fig_idx)
    freq=np.fft.fft(data)
    N = 44100
    freq_axis = np.linspace(0, N, len(data))
    plt.plot(freq_axis[0:int(len(data)/2)],abs(freq)[0:int(len(data)/2)])
    plt.savefig('frequency ' + filenum+'.jpg')
if __name__ == "__main__":
    filename=r'\DATALOG'
    filenum='4'
    data_plot(filename,20,filenum)