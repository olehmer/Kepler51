from scipy.optimize import fsolve
from math import log, pi, exp, floor
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from random import uniform, randint

#make the font bigger
matplotlib.rc('font', size=18)

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


def get_vs_comb(rs,T,F_xuv, a, n, p_xuv, R, time):
    """
    what should have been done in the first place, just def everything in 
    pieces...
    """

    rho = 5510.0
    Mp = 4.0/3.0*pi*rs**3.0*rho
    log_val = log(4.0*p_xuv*pi*rs**4/(a*GG*Mp**2))

    val = a/(GG**3*Mp**2*rs**3)*(\
            GG**3*Mp**3 + \
            3.0*GG**2*Mp**2*R*rs*T + \
            6.0*GG*Mp*R**2*rs**2*T**2 +\
            6.0*R**3*rs**3*T**3 +\
            3.0*R*rs*T*(GG**2*Mp**2 + 2.0*GG*Mp*R*rs*T + 2.0*R**2*rs**2*T**2)*log_val +\
            3.0*R**2*rs**2*T**2*(GG*Mp + R*rs*T)*log_val**2 +\
            R**3*rs**3*T**3*log_val**3) - F_xuv*n*pi*time/(GG*Mp)

    return val

    
def get_vs_sum_Zeng(rs,T,F_xuv, a, n, p_xuv, R, time):
    """
    This equation uses the density-radius relationship from Zeng (2015)
    """
    log_val = log(p_xuv/(a*GG*rs**3.4))
    #11 terms total... no typos ftw!
    v1 = 0.3788*a*rs**0.7
    v2 = 16.4176*a*R*T/(GG*rs**2)
    v3 = 245.103*a*R**2*T**2/(GG**2*rs**4.7)
    v4 = 1270.56*a*R**3*T**3/(GG**3*rs**7.4)
    v5 = 3.0*a*R*T*log_val/(GG*rs**2)
    v6 = 86.682*a*R**2*T**2*log_val/(GG**2*rs**4.7)
    v7 = 647.055*a*R**3*T**3*log_val/(GG**3*rs**7.4)
    v8 = 7.91975*a*R**2*T**2*log_val**2/(GG**2*rs**4.7)
    v9 = 114.417*a*R**3*T**3*log_val**2/(GG**3*rs**7.4)
    v10 = 6.96917*a*R**3*T**3*log_val**3/(GG**3*rs**7.4)
    v11 = -8.29355*F_xuv*n*time/(GG*rs**3.7)
    return v1+v2+v3+v4+v5+v6+v7+v8+v9+v10+v11

def get_vs_sum_new(r_s,T,rho,F_xuv, a, n, p_xuv, R, time):
    """
    This version uses the dM/dt from Kevin with Rxuv^3
    """
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
    """
    This version uses the original equation from Luger et al (2015)
    """
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
    
    guess = R_Earth*2.0 #guess the radius is at 4 Earth radii

    def eqn_rs(r_s):
        #return get_vs_sum_new(r_s, T, rho, F_xuv, a, e_xuv, p_xuv, R, time)
        return get_vs_sum_Zeng(r_s, T, F_xuv, a, e_xuv, p_xuv, R, time)


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
param_titles = {0:"A",1:"B",2:"B",3:"C",4:"D",5:"E",6:"F",7:"G"}
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


    plt.title(param_titles[param_type], y=0.85)
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

    #plt.sca(axs[0,1])
    #vary_parameter(4.0,8.0, DENS, show_fig=False)
    plt.delaxes(axs[0,1])

    plt.axes(axs[1,0])
    vary_parameter(50,200,FLUX, show_fig=False)

    plt.axes(axs[1,1])
    vary_parameter(0.01, 0.1, ATMO, show_fig=False)
    
    plt.axes(axs[2,0])
    vary_parameter(0.1,0.4, EFFI, show_fig=False)

    plt.axes(axs[2,1])
    vary_parameter(0.1,10, PRES, show_fig=False)

    plt.axes(axs[3,0])
    vary_parameter(2500,R_H2, GASC, show_fig=False)

    plt.axes(axs[3,1])
    vary_parameter(50,200, TIME, show_fig=False)

    
    plt.show()

def rs_histogram():
    """
    Calculate the r_s cutoff across a range of parameters and plot the result
    in a histogram
    """

    num_steps = 6 
    r_vals = []

    Temps = np.linspace(500,3000,num_steps)
    Fluxes = np.linspace(43,172, num_steps)
    Efficiencies = np.linspace(0.1,0.6, num_steps)
    Atmos_mass_fracs = np.linspace(0.01, 0.1, num_steps)
    Pressures = np.linspace(0.1, 10, num_steps)
    Gas_consts = np.linspace(3615,4157,num_steps)
    Times = np.linspace(80, 120, num_steps)

    min_r = 1000.0 #huge number
    max_r = 0

    for T in Temps:
        for F in Fluxes:
            for n in Efficiencies:
                for a in Atmos_mass_fracs:
                    for p in Pressures:
                        for R in Gas_consts:
                            for t in Times:
                                r = rs_cutoff(T, 1.0, F, a, n, p, R, \
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


def make_hist_arrays(data, min_val, max_val, num_bins):
    bins = np.linspace(min_val, max_val, num_bins)
    counts = np.zeros_like(bins)

    for d in data:
        for i in range(num_bins):
            if d <= bins[i]:
                counts[i] += 1
                break

    return (bins, counts)

def bootstrap(data, CL=0.95, title=""):
    """
    Calculate the mean and confidence level for the given data using the 
    bootstrap method

    Input:
    data - the array of data to work with
    CL - the desired confidence level. Defaults to 99%

    Returns:
    x_mean - the mean of the original data
    x_plus - the upper error for the confidence level
    x_minus - the negative error for the confidence level
    sorted_bs - the array of sorted bootstrap calculated means
    returned as: (x_mean, x_plus, x_minus, bootstrap_means)
    """

    num_pts = len(data)
    num_straps = num_pts*2

    if num_straps < 20000:
        num_straps = 20000

    x_mean = np.mean(data)

    bootstrap_means = np.zeros(num_straps)
    for c in range(num_straps):
        b_mean = np.mean(np.random.choice(data,size=num_pts))
        bootstrap_means[c] = b_mean

    #sort the bootstrap means
    sorted_bs = np.sort(bootstrap_means)

    ind = int(floor((1.0-CL)*num_straps/2.0))
    x_min = sorted_bs[ind-1]
    x_max = sorted_bs[-ind]

    #calculate the +/- values
    x_plus = x_max - x_mean
    x_minus = x_min - x_mean

    if len(title)>0:
        print("%s: %0.2f%% CL gives mean=%0.4f (+%0.4f,%0.4f)"%\
                (title,CL,x_mean,x_plus,x_minus))

    return (x_mean, x_min, x_max, sorted_bs)



def rs_histogram_bootstrap():
    """
    Calculate the r_s cutoff across a range of parameters and plot the result
    in a histogram
    """

    num_pts = 10000
    num_straps = 2*num_pts
    r_vals = np.zeros(num_pts)
    param_vals = []

    min_r = 1000.0 #huge number
    max_r = 0

    for c in range(num_pts):
        T = uniform(500,3000)
        F = uniform(43,172)
        n = uniform(0.1,0.6)
        a = uniform(0.01,0.1)
        p = uniform(0.1,10)
        R = uniform(3615,4157)
        time = uniform(80,120)*(1.0E6*SECONDS_PER_YEAR)

        r = rs_cutoff(T, 1.0, F, a, n, p, R, time)
        r = r/R_Earth

        if r < min_r:
            min_r = r
        if r > max_r:
            max_r = r

        r_vals[c] = r

        params = (T,F,n,a,p,R,time)
        param_vals.append(params)


    r_mean, r_min, r_max, r_bs_means = bootstrap(r_vals,title="r_s")
    r_vals_in_99 = np.where(np.logical_and(r_vals>=r_min, r_vals<=r_max))

    #collect the parameters into arrays, formatted as (T,F,n,a,p,R,time)
    T_vals = []
    F_vals = []
    n_vals = []
    a_vals = []
    p_vals = []
    R_vals = []
    time_vals = []
    for ind in r_vals_in_99[0]:
        T_vals.append(param_vals[ind][0])
        F_vals.append(param_vals[ind][1])
        n_vals.append(param_vals[ind][2])
        a_vals.append(param_vals[ind][3])
        p_vals.append(param_vals[ind][4])
        R_vals.append(param_vals[ind][5])
        time_vals.append(param_vals[ind][6]/(1.0E6*SECONDS_PER_YEAR))


    fig, axs = plt.subplots(2,2, figsize=(11,9))
    fig.subplots_adjust(hspace=0.3)

    plt.axes(axs[0,0])
    r_bins, r_counts = make_hist_arrays(r_vals, min_r, max_r, 100)
    plt.bar(r_bins, r_counts, width=0.025)
    plt.xlabel("$R_{s}$: Cutoff Radius [R$_{Earth}$]")
    plt.title("A", x=0.05, y=0.85)
    plt.xlim(min_r,max_r)

    plt.axes(axs[0,1])
    R_bs_bins, R_bs_counts = make_hist_arrays(r_bs_means, np.min(r_bs_means), \
            np.max(r_bs_means), 100)
    plt.bar(R_bs_bins, R_bs_counts, width=0.00025)
    plt.xlabel(r"$\bar{R}_{s}$: Mean Cutoff Radius [R$_{Earth}$]")
    plt.title("B", x=0.05, y=0.85)
    plt.xlim(np.min(r_bs_means), np.max(r_bs_means))

    plt.axes(axs[1,0])
    T_bins, T_counts = make_hist_arrays(T_vals, 500, 3000, 20)
    plt.bar(T_bins, T_counts, width=125)
    plt.xlabel(r"$T^{*}$: Temperature for $\bar{R}_{s}\pm 2 \sigma$ [K]")
    plt.title("C", x=0.05, y=0.85)
    plt.xlim(500+125,3000)

    plt.axes(axs[1,1])
    T_mean, T_min, T_max, T_bs = bootstrap(T_vals,title="T")
    T_bs_bins, T_bs_counts = make_hist_arrays(T_bs, np.min(T_bs), np.max(T_bs),100)
    plt.bar(T_bs_bins, T_bs_counts, width=5)
    plt.title("D", x=0.05, y=0.85)
    plt.xlabel(r"$\bar{T}^{*}$: Mean Temperature for $\bar{R}_{s}\pm 2 \sigma$ [K]")
    plt.xlim(np.min(T_bs), np.max(T_bs))

    plt.show()

    bootstrap(F_vals,title="F")
    bootstrap(n_vals,title="n")
    bootstrap(a_vals,title="a")
    bootstrap(p_vals,title="p")
    bootstrap(R_vals,title="R")
    bootstrap(time_vals,title="time")


    """ 
    fig, axs = plt.subplots(4,2, figsize=(11,11))
    fig.subplots_adjust(hspace=0.3, top=0.95, bottom=0.05, left=0.1, right=0.95)

    plt.axes(axs[0,0])
    r_bins, r_counts = make_hist_arrays(r_vals, min_r, max_r, 100)
    plt.bar(r_bins, r_counts, width=0.025)
    plt.xlabel("Cutoff  Radius [R$_{Earth}$]")
    plt.xlim(min_r,max_r)

    plt.axes(axs[0,1])
    T_bins, T_counts = make_hist_arrays(T_vals, 500, 3000, 20)
    plt.bar(T_bins, T_counts, width=125)
    plt.xlabel("T")
    plt.xlim(500+125,3000)
    bootstrap(T_vals,title="T")

    plt.axes(axs[1,0])
    F_bins, F_counts = make_hist_arrays(F_vals, 43, 172, 20)
    plt.bar(F_bins, F_counts, width=6.45)
    plt.xlabel("F$_{XUV}$")
    plt.xlim(43+6.45,172)
    bootstrap(F_vals,title="F")

    plt.axes(axs[1,1])
    n_bins, n_counts = make_hist_arrays(n_vals, 0.1, 0.6, 20)
    plt.bar(n_bins, n_counts, width=0.025)
    plt.xlabel("$\eta$")
    plt.xlim(0.1+0.025,0.6)
    bootstrap(n_vals,title="n")

    plt.axes(axs[2,0])
    a_bins, a_counts = make_hist_arrays(a_vals, 0.01, 0.1, 20)
    plt.bar(a_bins, a_counts, width=0.0045)
    plt.xlabel(r"$\alpha$")
    plt.xlim(0.01+0.0045,0.1)
    bootstrap(a_vals,title="a")

    plt.axes(axs[2,1])
    p_bins, p_counts = make_hist_arrays(p_vals, 0.1, 10, 20)
    plt.bar(p_bins, p_counts, width=0.495)
    plt.xlabel("p$_{XUV}$")
    plt.xlim(0.1+0.495,10)
    bootstrap(p_vals,title="p")

    plt.axes(axs[3,0])
    R_bins, R_counts = make_hist_arrays(R_vals, 3615, 4157, 20)
    plt.bar(R_bins, R_counts, width=27.1)
    plt.xlabel("R$_{g}$")
    plt.xlim(3615+27.1,4157)
    bootstrap(R_vals,title="R")

    plt.axes(axs[3,1])
    time_bins, time_counts = make_hist_arrays(time_vals, 80, 120, 20)
    plt.bar(time_bins, time_counts, width=2)
    plt.xlabel(r"$\tau$")
    plt.xlim(80+2,120)
    bootstrap(time_vals,title="time")

    plt.show()
    """
   
    return



def plot_rs_eqn():
    sur_rads = np.linspace(0.5,5,100)
    T = 880.0
    F_xuv = 55.0
    a = 0.03
    n = 0.1
    p_xuv = 5.0
    R = R_H2
    rho = 5510.0
    time = 1.0E8*SECONDS_PER_YEAR

    vals = []
    vals1 = []
    for r_s in sur_rads:
        val = get_vs_sum_Zeng(r_s*R_Earth, T, F_xuv, a, n, p_xuv, R, time)
        val1 = get_vs_sum_new(r_s*R_Earth, T, rho, F_xuv, a, n, p_xuv, R, time)
        vals.append(val)
        vals1.append(val1)

    plt.plot(sur_rads,vals, label="Zeng")
    plt.xlim(0.8,3)
    plt.ylim(-1000,1000)
    #plt.plot(sur_rads,vals1, label="Old")
    plt.legend()
    plt.show()


#vary_parameter(500.0, 3000.0, TEMP)
#vary_parameter(3.0,8.0, DENS)
#vary_parameter(1,200,FLUX)
#vary_parameter(0.003, 0.3, ATMO)
#vary_parameter(0.001,1000, PRES)
#vary_parameter(200,R_H2, GASC)
#vary_parameter(10,1000, TIME)

#plot_rs_eqn()

#all_params_plotted() #ORL use this one in paper
#rs_histogram() #ORL use this one in paper

#rs_histogram_bootstrap() #ORL use this one

#rs_cutoff(T, rho, F_xuv, a, e_xuv, p_xuv, R, time):
r = rs_cutoff(1690.0, 1, 105.5, 0.059, 0.362, 5.05, 3873.0, time=101.2*1.0E6*SECONDS_PER_YEAR)
print("r=%0.2f"%(r/R_Earth))













