

# Import libraries 
import math
import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt 
import scipy.signal.windows as window
from scipy.fft import fft, fftfreq
# from numpy.fft import fft


# Reading data from file
data0 = np.loadtxt('data/HHG_SSFM.txt')
t_au = data0[:,0]
t_fs = data0[:,1]
E_l = data0[:,2]
a_dipol_SSFM = data0[:,3]
label0='HHG_SSFM'

# data1 = np.loadtxt('data/HHG-no-opt.txt')
data1 = np.loadtxt('data/HHG.txt')
a_dipol = data1[1:]
label1='HHG_SPD'

norm_FFT='forward' # 'backward' - 'ortho' - 'forward' 
'''
The argument norm indicates which direction of the pair of direct/inverse transforms is scaled 
and with what normalization factor. The default normalization ("backward") “backward”, meaning 
no normalization on the forward transforms and scaling by 1/n on the ifft. “forward” instead applies the 1/n factor on the forward tranform. For norm="ortho", both directions are scaled by 1/sqrt(n). 
'''

win_py_case = 0 # 0 scipy - 1 numpy
if win_py_case ==0:  # scipy
    '''
        available scipy Window types: 
        boxcar,triang,blackman,hamming,hann,bartlett,flattop,parzen,bohman
        blackmanharris,nuttall,barthann,cosine,exponential,tukey,taylor,kaiser (needs beta)
        gaussian (needs standard deviation), general_cosine (needs weighting coefficients)
        general_gaussian (needs power, width), general_hamming (needs window coefficient)
        dpss (needs normalized half-bandwidth), chebwin (needs attenuation)

        If the window requires no parameters, then window can be a string.
        If the window requires parameters, then window must be a tuple with the first argument
            the string name of the window, and the next arguments the needed parameters.
        If window is a floating point number, it is interpreted as the beta parameter of the kaiser window.
    '''
    win = 'flattop' # win = 0 no window
elif win_py_case ==1: # numpy
    # window functions for numpy
    win=0 # 0 no window - 1 Hannning - 2 Hamming - 3 Bartlett - 4 blackman - 5 kaiser

R12=2.0
E_true_H2p_R12_asc2_R2 = 0#-0.48731561628865982527 # True ground state energy for H2+ at R=2 and a_sc=2

# plt.style.use('fivethirtyeight')

landa_l = 800  # nm
I_l = 1.0e14  # W/cm2
E0_l = 5.33802569e-9*np.sqrt(I_l)
t_1c = landa_l*0.1378999
w_l = 2*np.pi/t_1c

Pond_R = E0_l/(w_l**2) # Ponderomotive radius (quiver distance)(a.u.) 
# Ip = 0.5  #Ionization potential (a.u.) = 1.6 eV for H atom
Ip = 1/R12-E_true_H2p_R12_asc2_R2 #Ionization potential (a.u.) for H2+ atom at Ground state
Up = E0_l**2/(4*(w_l**2))  #Ponderomotive potential
Keldysh_par = np.sqrt(Ip/(2*Up)) #Keldysh parameter
Ip_Up = Ip/Up
# af, bf, cf, df = -0.03076, 0.00277, 0, 1.31712 # (2nd order fitting is valid for Ip_Up up to 4)
af, bf, cf, df = -0.03592582958843294, 0.006105735198241882, -0.0005683712838491628, 1.3187535133864776  # (3nd order fitting is valid for Ip_Up up to 10)
f_Ip_Up = af*Ip_Up+bf*Ip_Up**2+cf*Ip_Up**3+df  #F(Ip/Up)
Kmax_3Step = Ip+3.17*Up #Max Kinetic energy (a.u.)  - 3 step model
n_cut_off_3Step = Kmax_3Step/w_l #n_cut_off - 3 step model
Kmax_Lewenstein = f_Ip_Up*Ip+3.17*Up #Max Kinetic energy (a.u.) - Lewenstein model
n_cut_off_Lewenstein = Kmax_Lewenstein/w_l #n_cut_off - Lewenstein model

print("Pondermotive radius = ", Pond_R)
print("Ionization potential (a.u.) = ", Ip)
print("Ponderomotive potential = ", Up)
print("Keldysh parameter = ", Keldysh_par)
print("Ip/Up = ", Ip_Up)
print("F(Ip/Up) = ", f_Ip_Up)
print("Max Kinetic energy (a.u.)  - 3 step model = ", Kmax_3Step)
print("n_cut_off  - 3 step model = ", n_cut_off_3Step)
print("Max Kinetic energy (a.u.)  - Lewenstein model = ", Kmax_Lewenstein)
print("n_cut_off - Lewenstein model = ", n_cut_off_Lewenstein)

t=t_au
dt=t_au[1]-t_au[0]
nt=len(t_au)

# applying window function
def windowed(f_t,win):
    M_win=len(f_t)
    if win_py_case ==0:  # scipy
        if win==0:
            f_t=f_t
        else:
            f_t=f_t*window.get_window(win, M_win, fftbins=False)
    elif win_py_case ==1: # numpy
        if win==0:
            f_t=f_t
        elif win==1: #hanning
            f_t=f_t*np.hanning(M_win)
        elif win==2: #hanning
            f_t=f_t*np.hamming(M_win)
        elif win==3: #hanning
            f_t=f_t*np.bartlett(M_win)
        elif win==4: #hanning
            f_t=f_t*np.blackman(M_win)
        elif win==5: #hanning
            betta=0 # 0 Rectangular - 5 Similar to a Hamming - 6 Similar to a Hanning - 8.6 Similar to a Blackman
            f_t=f_t*np.kaiser(M_win,betta)
    return f_t

# zero_padding
n_zero_pad=5*nt
nTot=nt+n_zero_pad

a_dipol_SSFM_zero_pad=np.zeros(nTot, dtype=np.double)
a_dipol_SSFM_zero_pad[:nt]=windowed(a_dipol_SSFM,win=win)

a_dipol_zero_pad=np.zeros(nTot, dtype=np.double)
a_dipol_zero_pad[:nt]=windowed(a_dipol,win=win)

# w_HHG=2*np.pi*np.linspace(0,1/dt,nTot)[:nTot//2] # numpy or np.linspace(0,1/(2*dt),nTot//2)
w_HHG=2*np.pi*fftfreq(nTot, dt)[:nTot//2] # scipy

harm_HHG = w_HHG/w_l
power_spec_HHG_SSFM = np.log10((np.abs(fft(a_dipol_SSFM_zero_pad,norm=norm_FFT)[0:nTot//2]))**2) 
power_spec_HHG = np.log10((np.abs(fft(a_dipol_zero_pad,norm=norm_FFT)[0:nTot//2]))**2) 

# Creating plot
plt.plot(harm_HHG, power_spec_HHG_SSFM, color='C0', linewidth=2, label= label0)  
plt.plot(harm_HHG, power_spec_HHG, color='C1', linewidth=2, label=label1)  

# plot settings
# plt.title("title_temp") 
plt.ylabel("Power Spectrum", fontweight ='bold')
plt.xlabel("Harmonics", fontweight ='bold')

# plt.legend(["Time evolution - no optimization", "Time evolution - with optimization"])

plt.xscale('linear')
plt.yscale('linear')

plt.grid(axis="x")

# plt.xticks(range(math.floor(min(harm_HHG[:nTot // 2])), math.ceil(max(harm_HHG[:nTot // 2]))+1,5))
plt.xticks(range(1, math.ceil(max(harm_HHG[:nTot // 2]))+1,2))

plt.xlim([0, 47])
plt.ylim([-25, -4])

#add vertical line at the perdicted cutoff by the 3 Step Model
plt.axvline(n_cut_off_3Step, color='red', linestyle='--', label='3 Step Model')

#add vertical line at the perdicted cutoff by the Lewenstein Model
plt.axvline(n_cut_off_Lewenstein, color='black', linestyle='--', label='Lewenstein Model')

plt.legend(loc='best', fontsize=7.5, frameon=False)

plt.tight_layout()

# Save plot
plt.savefig('plots/HHG_Power_Spec.pdf', bbox_inches='tight', pad_inches=0.1)

plt.show()
