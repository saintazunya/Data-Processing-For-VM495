import numpy as np,pandas as pd
import matplotlib.pyplot as plt
def data_plotting(filepath,filename):
    position = np.loadtxt(filepath)
    for i in range(position.shape[0]):
        if position[i,1]>0.5:
            stpt=i
            break
    #print(stpt)
    position= position[stpt:,:]
    stp=70#how many set of data points used to find speed
    speed=np.empty([position.shape[0]-1,2],'float64')#distance/time
    for j in range(position.shape[0]-stp):
        speed[j,1]=(position[j+stp,1]-position[j,1])/(position[j+stp,0]-position[j,0])
        speed[j,0]=position[j,0]
    stp2=10
    acc=np.empty([speed.shape[0]-1,2],'float64')
    for k in range(speed.shape[0]-stp2):
        if (speed[k+stp2,0]-speed[k,0])==0:
            acc[k, 1]=1e3
            pass
        acc[k,1]=(speed[k+stp2,1]-speed[k,1])/(speed[k+stp2,0]-speed[k,0])
        acc[k,0]=speed[k,0]
    acceleraction=np.average(acc[2000:3000,1])
    print(speed[100,1])
    print(acceleraction)
    plt.figure(1)
    plt.scatter(position[:,0],position[:,1],s=0.01)
    #plt.show()
    plt.xlabel('Time(s)')
    plt.ylabel("Position(mm)")
    plt.title("Time-position")
    plt.savefig('Time-position_' + filename+".jpg")
    plt.figure(2)
    plt.plot(speed[:,0],speed[:,1],c='r')
    plt.xlabel('Time(s)')
    plt.ylabel("Speed(mm/s)")
    plt.title("Time-Speed")
    plt.savefig('Time-Speed_' + filename+".jpg")
    #plt.show()
    plt.figure(3)
    plt.plot(acc[:,0],acc[:,1],c='r')
    plt.xlabel('Time(s)')
    plt.ylabel("Acceleration(mm/s^2)")
    plt.title("Time-Acceleration")
    plt.savefig('Time-Acceleration_' + filename+".jpg")
    fft_test(acc[1000:5000,1], 4)
    freq = np.fft.fft(acc[1000:5000,1])
    fft_test(freq,5)
    plt.show()

def fft_test(data, fig_idx):
    plt.figure(fig_idx)
    freq = np.fft.fft(data)
    N = 44100
    freq_axis = np.linspace(0, N, len(data))
    plt.plot(freq_axis[0:int(len(data) / 2)], abs(freq)[0:int(len(data) / 2)])
    plt.axis([0, 1000, 0, 3000])
    plt.savefig('frequency ' + str(fig_idx) + '.jpg')


if __name__ == "__main__":
    filepath=r"D:\Learning\Undergraduate\vm495\lab2\data7.19\test___s.lvm"
    filename=r"test___s"
    data_plotting(filepath,filename)
    #114.768833212
    #-9.35989324436