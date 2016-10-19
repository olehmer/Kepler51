import matplotlib.pyplot as plt
import numpy as np

from math import pi,exp

GG = 6.674E-11 #gravitational constant
e_xuv = 0.2 #typically between 0.1 and 0.3
AU = 1.49598E11 #AU in m
k51b_orbital_dist = 0.25*AU #orbit of k51b 
k51b_mass = 1.2E25 #[kg]
k51b_rad = 4.48E7  #[m]
k51_mass = 2.1E30  #mass of Kepler 51 in [kg], star
k51_rad = 7.0E8 #the radius of K51 (assumed ~rad_sun) [m]

r_xuv = k51b_rad #typically within 10-15%?

#k51b_rad = k51b_rad*0.3


def calculate_ktide(mp, ms, dist, r_xuv):
    """
    mp = mass of planet in kg
    ms = mass of star in kg
    dist = orbital distance [m]
    r_xuv = radius of XUV absorption [m]
    """

    r_roche = (mp/(3.0*ms))**(1.0/3.0)*dist
    epsilon = r_roche/r_xuv
    k_tide = 1.0 - 3.0/(2.0*epsilon) + 1.0/(2.0*epsilon**3.0)

    return k_tide


def calculate_xuv_lammer2012(dist):
    """
    Return the XUV flux in W/m2 based on Lammer et al 2012 (equation 13)

    T_corona = 0.34L_xuv^0.25 #the paper equation in [erg s-1]
    """
    #the corona is likely ~10,000,000K for young, sun-like stars (under 700 Myrs)
    T_corona = 10000000.0 #coronal temperature, 10,000,000 [K]
    L_xuv = (T_corona/0.34)**4.0
    
    #from the above equation the luminosity is given in [ergs s-1]
    #convert ro [W]
    L_xuv = L_xuv*(1.0E-7) #convert to W

    L_xuv_orb = L_xuv/dist**2.0

    return L_xuv_orb
    


def calculate_xuv_planck(T, star_rad, dist, start=0.01, stop=0.124, nsteps=100):
    """
    Calculate the XUV based on the given temp. Start and stop are the 
    wavelengths to bound the flux by. Given in microns.
    """

    flux = 0.0
    cur = start

    step = (stop-start)/float(nsteps)
    if step < 0:
        print("Error - calculate_xuv() start after stop")

    c1 = 1.1914042E8
    c2 = 1.4387752E4
    for i in range(0,nsteps):
        B = c1/(cur**5.0*(exp(c2/(cur*T))-1.0))
        flux += B*step
        cur = cur+step

    #convert from W/m2-sr to W/m2 and scale by the radius of the star
    flux = pi*flux 
    flux = flux*star_rad**2.0/dist**2.0
    return flux
            

def planet_mass_over_time(core_rho, core_mass, mass, rad, dist, star_mass, time=1000, ts=1):
    """
    Calculate the mass of the planet over time. Time is in Myr, the default is
    1 Byr at 1 Myr timesteps (given by ts)

    Returns an array of mass values calculated at each time step
    """
    
    mass_array = np.zeros(time/ts)
    mass_array.fill(core_mass) #init to core mass
    mass_array[0] = mass 

    xuv = calculate_xuv_lammer2012(dist)

    #the core radius is easily calculated
    rad_core = (3.0*core_mass/(4.0*pi*core_rho))**(1.0/3.0)

    #assume the radius of XUV absorption is equal to the radius of the whole
    #planet (usually within ~10% from Luger et el (2015)
    r_xuv = rad

    #TODO the radius of the planet will change as mass is lost, but that's tricky
    #so I've ignored it for now

    for i in range(1,len(mass_array)):
        k_tide = calculate_ktide(mass, star_mass, dist, r_xuv)
        dMdt = (e_xuv*pi*xuv*rad_core*r_xuv**2.0)/(GG*mass*k_tide)
        dMdt = dMdt * (3.154E13)*ts #convert from seconds to Myr
        #print("i=%3d : dMdt=%3.3e"%(i,dMdt))
        mass = mass - dMdt
        #print("mass=%3.3e, core_mass=%3.3e"%(mass,core_mass))
        if mass < core_mass:
            #this can't be!
            break
        else:
            mass_array[i] = mass

    return mass_array


def plot_lifetime():
    mass_frac = 0.20
    masses = planet_mass_over_time(1000.0, mass_frac*k51b_mass,k51b_mass,k51b_rad,\
            k51b_orbital_dist,k51_mass)
    plt.plot(np.linspace(0,1000,1000), masses/k51b_mass, label="Core Density: 1 g/cc")

    masses = planet_mass_over_time(3000.0, mass_frac*k51b_mass,k51b_mass,k51b_rad,\
            k51b_orbital_dist,k51_mass)
    plt.plot(np.linspace(0,1000,1000), masses/k51b_mass, label="Core Density: 3 g/cc")

    masses = planet_mass_over_time(5000.0, mass_frac*k51b_mass,k51b_mass,k51b_rad,\
            k51b_orbital_dist,k51_mass)
    plt.plot(np.linspace(0,1000,1000), masses/k51b_mass, label="Core Density: 5 g/cc")

    masses = planet_mass_over_time(8000.0, mass_frac*k51b_mass,k51b_mass,k51b_rad,\
            k51b_orbital_dist,k51_mass)
    plt.plot(np.linspace(0,1000,1000), masses/k51b_mass, label="Core Density: 8 g/cc")

    plt.xlabel("Time [Myr]")
    plt.ylabel("Mass Fraction")
    plt.title("XUV Driven Mass Loss for Kepler-51b - Core mass fraction: %0.2f"%(mass_frac))
    plt.legend()
    plt.show()


plot_lifetime()
   
