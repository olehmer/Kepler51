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
p_xuv = 5.0 #pressure at XUV level [Pa]
SECONDS_PER_YEAR = 3.154E7 #seconds in a year
e_xuv = 0.1 #XUV absorption efficiency
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


def get_vs_sum_new(r_s,T,rho,F_xuv, a, n, p_xuv, R, time):
    log_val = log(9.0*p_xuv/(4.0*a*GG*pi*rho**2.0*r_s**2.0))
    v1 = 64.0*a*GG**2.0*pi**3*rho**2*r_s**5
    v2 = -36.0*F_xuv*GG*n*pi**2*r_s**2*time
    v3 = 144.0*a*GG*pi**2*R*rho*r_s**3*T
    v4 = 216.0*a*pi*R**2*r_s*T**2
    v5 = 162.0*a*R**3*T**3/(GG*rho*r_s)
    v6 = 144.0*a*GG*pi**2*R*rho*r_s**3*T*log_val
    v7 = 216.0*a*pi*R**2*r_s*T**2*log_val
    v8 = 162.0*a*R**3*T**3*log_val/(GG*rho*r_s)
    v9 = 108.0*a*pi*R**2*r_s*T**2*log_val**2
    v10 = 81.0*a*R**3*T**3*log_val**2.0/(GG*rho*r_s)
    v11 = 27.0*a*R**3*T**3*log_val**3/(GG*rho*r_s)

    return v1+v2+v3+v4+v5+v6+v7+v8+v9+v10+v11

def get_vs(r_s, T, rho, F_xuv, a, e_xuv, p_xuv, R, time):
    log_val = log(9.0*p_xuv/(4.0*a*GG*pi*rho**2*r_s**2))
    v1 = 16.0*a*r_s**5.0*GG*pi**2*rho
    v2 = -9.0*e_xuv*F_xuv*pi*r_s**2*time/rho
    v3 = 24.0*a*pi*R*r_s**3*T
    v4 = 18.0*a*R**2*r_s*T**2/(GG*rho)
    v5 = 24.0*a*pi*R*r_s**3*T*log_val
    v6 = 18.0*a*R**2*r_s*T**2/(GG*rho)*log_val
    v7 = 9.0*a*R**2*r_s*T**2/(GG*rho)*log_val**2
    return (v1,v2,v3,v4,v5,v6,v7)

def print_vs(vs):
    v1,v2,v3,v4,v5,v6,v7 = vs
    print("v1 = %2.3e"%(v1))
    print("v2 = %2.3e"%(v2))
    print("v3 = %2.3e"%(v3))
    print("v4 = %2.3e"%(v4))
    print("v5 = %2.3e"%(v5))
    print("v6 = %2.3e"%(v6))
    print("v7 = %2.3e"%(v7))
    print("Sum = %2.3e"%(v1+v2+v3+v4+v5+v6+v7))

def rs_cutoff(T, rho, F_xuv, a, e_xuv, p_xuv, R, time):
    
    guess = R_Earth*4.0 #guess the radius is at 4 Earth radii

    def eqn_rs(r_s):
        #v1,v2,v3,v4,v5,v6,v7 = get_vs(r_s, T, rho, F_xuv, a, e_xuv, p_xuv, R, time)
        #return v1+v2+v3+v4+v5+v6+v7
        return get_vs_sum_new(r_s, T, rho, F_xuv, a, e_xuv, p_xuv, R, time)

    result = fsolve(eqn_rs, guess)

    return result

TEMP = 0 #isothermal temperature
DENS = 1 #core density
FLUX = 2 #XUV flux
ATMO = 3 #atmospheric mass fraction
EFFI = 4 #XUV absorption efficiency
PRES = 5 #pressure where XUV is absorbed
GASC = 6 #specific gas constant
TIME = 7 #XUV saturation time
param_labels = {0: "Isothermal Temperature [K]",
                1: "Core Density [g cm$^{-3}$]",
                2: "F$_{XUV}$ [W m$^{-2}$]",
                3: "Initial Atmospheric Mass Fraction",
                4: "$\eta$",
                5: "p$_{XUV}$ [Pa]",
                6: "Specific Gas Constant",
                7: "XUV Saturation Time [Myr]"}
def vary_parameter(param_min, param_max, param_type, count=100, save_fig=False,\
        show_fig=True):
    """
    Plot the effect of a single parameter on the cutoff radius.

    Inputs:
    param_min - the minimum value to look at for the given parameter
    param_max - the maximum value to look at for the given parameter
    param_type - which parameter is being looked at (see above list)
    count - the number of points to calculate
    save_fig - whether to save the figure to file, plot not shown if true
    show_fig - whether to show the figure or not

    IMPORTANT:
    When passing in a density pass it in in g/cc, it will be converted to kg/m3
    When passing in XUV sat time, pass it in in Myr
    """
    #the default parameters
    T = 880.0
    rho = 5510.0
    F_xuv = 55.0
    a = 0.03
    e_xuv = 0.1
    p_xuv = 5.0 
    R = R_H2
    time = 1.0E8*SECONDS_PER_YEAR

    param_vals = np.linspace(param_min, param_max, count)
    results = np.zeros(count)

    analytic = np.zeros(count)
    
    def analytic_eqn(T, F, a):
        val = 100.0*T + 100.0*F**2.0 + 1.0/a**2.4 + R_Earth*1.2

        return val

    for i in range(count):
        param = param_vals[i]
        
        if param_type == TEMP:
            results[i] = rs_cutoff(param, rho, F_xuv, a, e_xuv, p_xuv, R, time)
            #analytic[i] = 1.65*param*F_xuv/a + R_Earth*1.05
            analytic[i] = analytic_eqn(param, F_xuv, a)

        if param_type == DENS:
            results[i] = rs_cutoff(T, param*1000.0, F_xuv, a, e_xuv, p_xuv, R, time)

        if param_type == FLUX:
            results[i] = rs_cutoff(T, rho, param, a, e_xuv, p_xuv, R, time)
            analytic[i] = analytic_eqn(T, param, a)

        if param_type == ATMO:
            results[i] = rs_cutoff(T, rho, F_xuv, param, e_xuv, p_xuv, R, time)
            analytic[i] = analytic_eqn(T, F_xuv, param)

        if param_type == EFFI:
            results[i] = rs_cutoff(T, rho, F_xuv, a, param, p_xuv, R, time)

        if param_type == PRES:
            results[i] = rs_cutoff(T, rho, F_xuv, a, e_xuv, param, R, time)

        if param_type == GASC:
            results[i] = rs_cutoff(T, rho, F_xuv, a, e_xuv, p_xuv, param, time)

        if param_type == TIME:
            results[i] = rs_cutoff(T, rho, F_xuv, a, e_xuv, p_xuv, R, \
                    param*(1.0E6)*SECONDS_PER_YEAR)


    plt.plot(param_vals, results/R_Earth)
    plt.xlabel(param_labels[param_type])
    plt.ylabel("R$_{s}$ [R$_{\oplus}$]")
    plt.grid()

    plt.xlim(param_min,param_max)

    if param_type == TEMP or param_type == FLUX or param_type == ATMO:
        #plt.plot(param_vals, analytic/R_Earth, "g--") #ORL TODO
        were = 0 #bs parameter since I'm not ready to delete this yet

    if param_type == PRES:
        plt.xscale('log')
    
    if show_fig:
        plt.show()

    if save_fig:
        title = "vary_param_%d"%(param_type)
        print("Saving figure as: "+title) 
        plt.savefig(title)
    


def all_params_plotted():
    fig, axs = plt.subplots(4,2, figsize=(11,11))
    fig.subplots_adjust(hspace=0.3, top=0.95, bottom=0.05, left=0.1, right=0.95)

    plt.axes(axs[0,0])
    vary_parameter(500.0, 3000.0, TEMP, show_fig=False)

    plt.sca(axs[0,1])
    vary_parameter(5.0,8.0, DENS, show_fig=False)

    plt.axes(axs[1,0])
    vary_parameter(50,200,FLUX, show_fig=False)

    plt.axes(axs[1,1])
    vary_parameter(0.01, 0.1, ATMO, show_fig=False)

    plt.axes(axs[2,0])
    vary_parameter(0.1,10, PRES, show_fig=False)

    plt.axes(axs[2,1])
    vary_parameter(2500,R_H2, GASC, show_fig=False)

    plt.axes(axs[3,0])
    vary_parameter(50,200, TIME, show_fig=False)

    #plt.delaxes(axs[3,1])
    plt.axes(axs[3,1])
    vary_parameter(0.1,0.4, EFFI, show_fig=False)

    plt.show()

def rs_histogram():
    """
    Calculate the r_s cutoff across a range of parameters and plot the result
    in a histogram
    """

    num_steps = 5 
    r_vals = []

    Temps = np.linspace(500,3000,num_steps)
    Densities = np.linspace(5,8,num_steps)
    Fluxes = np.linspace(50,200, num_steps)
    Efficiencies = np.linspace(0.1,0.4, num_steps)
    Atmos_mass_fracs = np.linspace(0.01, 0.1, num_steps)
    Pressures = np.linspace(0.1, 10, num_steps)
    Gas_consts = np.linspace(2500,4157,num_steps)
    Times = np.linspace(50, 200, num_steps)

    min_r = 1000.0 #huge number
    max_r = 0

    for T in Temps:
        for rho in Densities:
            for F in Fluxes:
                for n in Efficiencies:
                    for a in Atmos_mass_fracs:
                        for p in Pressures:
                            for R in Gas_consts:
                                for t in Times:
                                    r = rs_cutoff(T, rho*1000.0, F, a, n, p, R, \
                                            time=1.0E8*SECONDS_PER_YEAR)
                                    r = r/R_Earth

                                    if r < min_r:
                                        min_r = r
                                    if r > max_r:
                                        max_r = r

                                    r_vals.append(r)


    #sort the r values into bins
    bins = np.linspace(min_r, max_r, 100)
    counts = np.zeros(len(bins))

    for r in r_vals:
        for i in range(len(bins)):
            if r < bins[i]:
                counts[i] += 1
                break

    plt.bar(bins,counts, width=0.025)
    plt.xlabel("Radius Cutoff [R$_{Earth}$]")
    plt.ylabel("Count")
    #plt.gca().yaxis.set_visible(False)
    plt.show()
    
    return



#vary_parameter(500.0, 3000.0, TEMP)
#vary_parameter(3.0,8.0, DENS)
#vary_parameter(1,200,FLUX)
#vary_parameter(0.003, 0.3, ATMO)
#vary_parameter(0.001,1000, PRES)
#vary_parameter(200,R_H2, GASC)
#vary_parameter(10,1000, TIME)

all_params_plotted() #ORL use this one in paper
#rs_histogram() #ORL use this one in paper


#r = rs_cutoff(T, rho, F_xuv, 0.03, e_xuv, p_xuv, R_H2, time=1.0E8*SECONDS_PER_YEAR)
#print("r=%0.2f"%(r/R_Earth))













