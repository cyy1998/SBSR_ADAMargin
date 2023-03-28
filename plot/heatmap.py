import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def main():
    def norm_softmax(theta,m=0.5):
        p=np.exp(64*np.cos(theta))/(np.exp(64*np.cos(theta))+np.exp(64*np.cos(np.pi/2-theta)))
        return (p-1)*64

    def norm_softmax_alter(theta,m=0.5):
        p=np.exp(64*np.cos(theta))/(np.exp(64*np.cos(theta))+np.exp(64*np.cos(np.pi/2-theta)))
        return p*64

    def cosface(theta,m=0.5):
        p=np.exp(64*(np.cos(theta)-0.5))/(np.exp(64*(np.cos(theta)-0.5))+np.exp(64*(np.cos(np.pi/2-theta)+0.5)))
        return (p-1)*64

    def cosface_alter(theta,m=0.5):
        p = np.exp(64 * np.cos(theta))/ (np.exp(64 * (np.cos(np.pi/2-theta)-0.5)) + np.exp(64 *np.cos(theta)))
        return p * 64

    def arcface(theta,m=0.5):
        p=np.exp(64*np.cos(theta+0.5))/(np.exp(64*np.cos(theta+0.5))+np.exp(64*np.cos(np.pi/2-theta)))
        return (p-1)*64*(np.cos(0.5)+np.sin(0.5)/np.tan(theta+0.1))

    def arcface_alter(theta, m=0.5):
        p = np.exp(64 * np.cos(theta)) / (
                    np.exp(64 * np.cos(theta)) + np.exp(64 * np.cos(np.pi / 2 - theta+0.5)))
        return p * 64

    def negative_arcface(theta,m=0.5):
        p=np.exp(64*np.cos(theta-0.5))/(np.exp(64*np.cos(theta-0.5))+np.exp(64*np.cos(np.pi/2-theta)))
        return (p-1)*64*(np.cos(-0.5)+np.sin(-0.5)/np.tan(theta+0.1))

    def negative_arcface_alter(theta, m=0.5):
        p = np.exp(64 * np.cos(theta)) / (
                    np.exp(64 * np.cos(theta)) + np.exp(64 * np.cos(np.pi / 2 - theta-0.5)))
        return p * 64

    def adamargin(theta,m=0.5):
        p=np.exp(64*(np.cos(theta-0.7*m)-m))/(np.exp(64*(np.cos(theta-0.7*m)-m))+np.exp(64*(np.cos(np.pi/2-theta)+0.4)))
        return (p-1)*64*(np.cos(-0.7*m)+np.sin(-0.7*m)/np.tan(theta+0.01))
        #return -64 * (np.cos(-m) + np.sin(-m) / np.tan(theta + 0.1))

    def adamargin_alter(theta, m=0.5):
        p = np.exp(64 * (np.cos(theta) - 0.4)) / (
                    np.exp(64 * (np.cos(np.pi / 2 - theta - 0.7*m) - m)) + np.exp(64 * (np.cos(theta) - 0.4)))
        return p* 64



    rad=np.linspace(-0.4,0.4, 1000)
    azm=np.linspace(0,np.pi/2,1000)
    r, th = np.meshgrid(rad, azm)
    #z = (r ** 2.0) / 4.0
    softmax_result=np.abs(norm_softmax(th,r))
    softmax_alter_result=np.abs(norm_softmax_alter(th,r))
    cosface_result=np.abs(cosface(th,r))
    cosface_alter_result=np.abs(cosface_alter(th,r))
    arcface_result=np.abs(arcface(th,r))
    arcface_alter_result = np.abs(arcface_alter(th, r))
    negative_arcface_result=np.abs(negative_arcface(th,r))
    negative_arcface_alter_result=np.abs(negative_arcface_alter(th,r))
    adamargin_result=np.abs(adamargin(th,r))
    adamargin_alter_result=np.abs(adamargin_alter(th,r))
    result=np.clip(negative_arcface_alter_result,0,500)
    print(np.max(result))
    print(np.min(result))
    #print(adamargin(0,0.5))
    # for i,theta in enumerate(azm):
    #     softmax_result[i]=norm_softmax(theta)
    #     cosface_result[i]=cosface(theta)
    #     arcface_result[i]=arcface(theta+0.001)

    plt.subplot(projection="polar")
    plt.pcolormesh(th, r, result,cmap=mpl.colormaps['jet'])
    plt.axis('off')
    #plt.colorbar()
    #plt.imshow(res,interpolation="quadric",cmap=mpl.colormaps['RdYlBu'])
    #plt.colorbar()
    plt.show()


if __name__ == '__main__':
    main()