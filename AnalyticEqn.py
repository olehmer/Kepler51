from scipy.optimize import fsolve
from math import log, pi, exp
import numpy as np
import matplotlib.pyplot as plt


M_Earth = 5.97E24 #mass of Earth in [kg]
R_Earth = 6.37E6 #Earth radius [m]
R_H2 = 4124.0 #gas constant for H2 [J kg-1 K-1]
rho = 5510.0 #density of planets [kg m-3]
GG = 6.672E-11 #Gravitational constant [N m^2 kg^-2]
T = 880.0 #isothermal temp [K]
p_xuv = 0.1 #pressure at XUV level [Pa]
SECONDS_PER_YEAR = 3.154E7 #seconds in a year
e_xuv = 0.2 #XUV absorption efficiency
F_xuv = 55.0 #XUV flux at 0.1 AU

def calc_M(mass, time=1.0E8*SECONDS_PER_YEAR, FRAC=True):
    """
    Calculate the total mass lost from the atmosphere in the given amount of 
    time. If FRAC is true, return the remaining atmospheric fraction instead.
    """

    r = (3.0*mass/(4.0*pi*rho))**(1.0/3.0) #surface radius [m]
    M0 = 0.03*M_Earth #0.03*mass #initial atmosphere mass [kg]
    g = GG*mass/r**2.0 #gravity [m s-1]
    H = R_H2*T/g #scale height [m]

    def M_eqn(M):
        """
        The integrated mass loss equation, just the part that depends on M
        """

        val = (M-M0)*(2.0*H**2.0+2.0*H*r+r**2.0+2.0*H*(H+r)*log(4.0*pi*p_xuv*r**2.0/\
                (g*(M0-M)))+H**2.0*log(4.0*pi*p_xuv*r**2.0/(g*(M0-M)))**2.0)

        return val

    #define the constant terms
    const = -M_eqn(0)
    C = e_xuv*pi*F_xuv/(GG*mass)

    def solver_eqn(M):
        M_val = M_eqn(M)
        return M_val+const-C*r**5.0*time

    guess = M0/2.0
    result = fsolve(solver_eqn,guess)

    if FRAC:
        result = 1.0-result/M0

    return result

def calculate_rs(M0, time=1.0E8*SECONDS_PER_YEAR):
    #val = (3.0*M0*R_H2**2.0*T**2.0/(4.0*e_xuv*F_xuv*pi**2.0*time*GG*rho)*\
    #        log(4.5*p_xuv*R_Earth/(GG*M0*rho))**2.0)**0.25

    ln = log(4.5*p_xuv*R_Earth/(GG*M0*rho))
    rs = (1.5/pi)**0.5*((-2.0*GG*M0*R_H2*rho*T*(1.0+ln)\
            +(-GG*M0*R_H2**2.0*rho*T**2.0*(4.0*GG*M0*rho-3.0*e_xuv*F_xuv*time*\
            (2.0+2*ln+ln**2.0)))**0.5)/(GG*rho*(4.0*GG*M0*rho-3.0*e_xuv*F_xuv*time)))**0.5

    return rs
    


def calc_M_simple(mass, time=1.0E8*SECONDS_PER_YEAR, FRAC=True):
    """
    Calculate the total mass lost from the atmosphere in the given amount of 
    time. If FRAC is true, return the remaining atmospheric fraction instead.

    This version uses the simplified equation
    """

    r = (3.0*mass/(4.0*pi*rho))**(1.0/3.0) #surface radius [m]
    M0 = 0.06*M_Earth #initial atmosphere mass [kg]
    g = GG*mass/r**2.0 #gravity [m s-1]
    H = R_H2*T/g #scale height [m]


    #define the constant term
    C = e_xuv*pi*F_xuv/(GG*mass)

    def solver_eqn(M):
        val = M*(H**2.0*log(4.0*pi*p_xuv*r**2.0/(g*(M0-M)))**2.0)
        return val-C*r**5.0*time

    guess = M0/2.0
    result = fsolve(solver_eqn,guess)

    if FRAC:
        result = 1.0-result/M0

    return result


def M_over_time():
    masses = np.linspace(1,5,8)
    for m in masses:
        mass = m*M_Earth
        times = np.linspace(0,1.0E8*SECONDS_PER_YEAR,100)
        fracs = np.zeros(len(times))
        for i in range(len(times)):
            #print("Time [Myr] = %0.3f"%(times[i]/(1.0E6*SECONDS_PER_YEAR)))
            try:
                fracs[i] = calc_M(mass,time=times[i])
            except:
                print("For %0.1f Earth masses cutoff at: %0.1f Myr"%\
                        (m,times[i]/(1.0E6*SECONDS_PER_YEAR)))
                break
        plt.plot(times/(1.0E6*SECONDS_PER_YEAR),fracs, label="%0.1f M$_{Earth}$"%(m))

    plt.xlabel("Time [Myr]")
    plt.ylabel("Remaining Atmospheric\nFraction")
    plt.grid()
    plt.legend()
    plt.show()


def M_frac_at_100Myr():
    masses = np.linspace(1.5,10,100)
    time_sec = 1.0E8*SECONDS_PER_YEAR

    fracs_norm = np.zeros(len(masses))
    fracs_simple = np.zeros(len(masses))
    for i in range(len(masses)):
        try:
            fracs_norm[i] = calc_M(masses[i]*M_Earth, time=time_sec)
        except:
            continue
        try:
            fracs_simple[i] = calc_M_simple(masses[i]*M_Earth, time=time_sec)
        except:
            continue

    lim = calculate_rs(M_Earth*0.03, time=time_sec)
    mass = rho*4.0/3.0*pi*lim**3.0
    print("cutoff at: r=%0.3f, M=%0.3f"%(lim/R_Earth, mass/M_Earth))

    plt.plot(masses,fracs_norm,"b:", linewidth=4, label="Complete")
    plt.plot(masses,fracs_simple, "r--",label="Simple", linewidth=4)
    plt.legend()
    plt.grid()
    plt.xlabel("Mass [M$_{Earth}$]")
    plt.ylabel("Remaining Atmospheric\nFraction")
    plt.show()






#M_over_time()
#M_frac_at_100Myr()

M0 = 0.03*M_Earth
time = 1.0E8*SECONDS_PER_YEAR
a = 0.3
"""
ln = log(4.5*p_xuv*R_Earth/(GG*M0*rho))
v1 = -2.0*GG*M0*R_H2*T*rho*ln
v2 = -(2.0*GG*M0*rho*R_H2*T)**2.0
v3 = GG*M0*R_H2**2.0*T**2.0*rho*e_xuv*F_xuv*time*ln**2.0
v4 = 4.0*GG**2.0*rho**2.0*M0
v5 = -3.0*e_xuv*F_xuv*time*GG*rho
"""

def get_vs(r_s):
    log_val = log(9.0*p_xuv/(4.0*a*GG*pi*rho**2*r_s**2))
    v1 = 16.0*a*r_s**5.0*GG*pi**2*rho
    v2 = -9.0*e_xuv*F_xuv*pi*r_s**2*time/rho
    v3 = 24.0*a*pi*R_H2*r_s**3*T
    v4 = 18.0*a*R_H2**2*r_s*T**2/(GG*rho)
    v5 = 24.0*a*pi*R_H2*r_s**3*T*log_val
    v6 = 18.0*a*R_H2**2*r_s*T**2/(GG*rho)*log_val
    v7 = 9.0*a*R_H2**2*r_s*T**2/(GG*rho)*log_val**2
    return (v1,v2,v3,v4,v5,v6,v7)

def eqn_rs(r_s):
    v1,v2,v3,v4,v5,v6,v7 = get_vs(r_s)
    return v1+v2+v3+v4+v5+v6+v7

guess = R_Earth*3.0
result = fsolve(eqn_rs,guess)
print("cutoff at r_s=%1.2f"%(result/R_Earth))

v1,v2,v3,v4,v5,v6,v7 = get_vs(R_Earth*1.47)
print("v1 = %2.3e"%(v1))
print("v2 = %2.3e"%(v2))
print("v3 = %2.3e"%(v3))
print("v4 = %2.3e"%(v4))
print("v5 = %2.3e"%(v5))
print("v6 = %2.3e"%(v6))
print("v7 = %2.3e"%(v7))
print(v1+v2+v3+v4+v5+v6+v7)

"""
r_s = np.linspace(1,2.5,100)
vals = []
for r in r_s:
    val = eqn_rs(r*R_Earth)
    vals.append(val)

plt.plot(r_s,vals)
plt.yscale('log')
plt.show()

"""













#print(1.0-calc_M(M_Earth*3)/(M_Earth*3.0*0.03))
