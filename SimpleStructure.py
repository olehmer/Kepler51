"""
Owen Lehmer 11/17/2016
University of Washington

This file will calculate the atmospheric structure for a young planet with a
hydrogen rich atmosphere.

"""


import matplotlib.pyplot as plt
import numpy as np
from math import pi, exp, log


###########################UNIVERSAL CONSTANTS#################################
kB = 1.380662E-23        #Boltzmann's constant [J K^-1]
GG = 6.672E-11           #Gravitational constant [N m^2 kg^-2]
SIGMA = 5.6704E-8       #Stefan-Boltzmann constant [W m^-2 K^-4]
m_H = 1.66E-27          #Mass of H atom [kg]
e_xuv = 0.2 #typically between 0.1 and 0.3
AU = 1.49598E11 #AU in m
R_H2 = 4124.0 #gas constant for H2 [J kg-1 K]
R_AIR = 287.0 #gas constant for air [J kg-1 K]
M_Earth = 5.972E24 #mass of Earth in [kg]
R_Earth = 6.371E6 #radius of Earth [m]
###############################################################################

k51b_orbital_dist = 0.25*AU #orbit of k51b 
k51b_mass = 1.2E25 #[kg]
k51b_rad = 4.48E7  #[m]
k51_mass = 2.1E30  #mass of Kepler 51 (the star) in [kg]
k51_rad = 7.0E8 #the radius of K51 (assumed ~rad_sun) [m]



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
    #convert to [W]
    L_xuv = L_xuv*(1.0E-7) #convert to W

    L_xuv_orb = L_xuv/dist**2.0

    return L_xuv_orb



def calculate_flux_planck(T, star_rad, dist, start=0.1, stop=10.0, nsteps=1000):
    """
    Calculate the total flux based on the given temp. Start and stop are the 
    wavelengths to bound the flux by. Given in microns.
    """

    flux = 0.0
    cur = start

    step = (stop-start)/float(nsteps)
    if step < 0:
        print("Error - calculate_flux_planck() start after stop")

    c1 = 1.1914042E8 #constant in the planck function for wavelength
    c2 = 1.4387752E4 #constant in the planck function for wavelength
    for i in range(0,nsteps):
        B = c1/(cur**5.0*(exp(c2/(cur*T))-1.0))
        flux += B*step
        cur = cur+step

    #convert from W/m2-sr to W/m2 and scale by the radius of the star
    flux = pi*flux 
    flux = flux*star_rad**2.0/dist**2.0
    return flux
            
def calculate_loss_rate(mass, core_rad, rad_1bar, dist, star_mass):
    """
    This function will calculate the hydrodynamic loss rate from the planet
    based on Luger (2015).

    Inputs:
    mass - the total mass of the planet
    core_rad - the radius of the planet
    rad_1bar - the radius of the 1 bar level in the atmosphere
    dist - the orbital distance of the planet
    star_mass - the mass of the host star

    Returns:
    dMdt - the loss rate in kg/s
    """

    xuv = calculate_xuv_lammer2012(dist)

    #assume the radius of XUV absorption is equal to the radius of the whole
    #planet (usually within ~10% from Luger et al (2015)
    r_xuv = rad_1bar

    k_tide = calculate_ktide(mass, star_mass, dist, core_rad)

    #Equation 5 from Luger et al. (2015)
    dMdt = (e_xuv*pi*xuv*core_rad*r_xuv**2.0)/(GG*mass*k_tide)

    return dMdt

def calculate_rad(p_r, core_mass, core_rho, mass, R_gas, T):
    """
    Calculate the radius of the level at p_r.

    Inputs:
    p_r - pressure that we want to know the radius of [Pa]
    core_mass - the mass of the core [kg]
    core_rho - the density of the core [kg m-3]
    mass - the total mass of the planet [kg]
    R_gas - the specific gas constant for the atmosphere [J kg-1 K]
    T - the isothermal atmospheric temperature [K]

    Returns:
    r - the radius at which the pressure is p_r [m]
    r_s - the radius of the core
    """

    #calculate the radius of the rocky core
    r_s = (3.0*core_mass/(4.0*pi*core_rho))**(1.0/3.0)

    #calculate the surface gravity
    g_s = GG*core_mass/r_s**2.0

    atmos_mass = mass - core_mass

    #calculate the surface pressure
    p_s = atmos_mass*g_s/(4.0*pi*r_s**2.0)


    H = R_gas*T/g_s

    r = r_s**2.0/(H*log(p_r/p_s)+r_s)

    return (r,r_s)




##################################TESTING FUNCTIONS############################
def test_earth():
    r, r_s = calculate_rad(100.0, M_Earth-5.148E18, 5510.0, M_Earth, R_AIR, 280.0)
    r = r-r_s

    print("r = %0.2f km"%(r/1000.0))

    #compare to plain old hydrostatic simple equation for Earth
    H = R_AIR*280.0/9.8
    t_r = -log(100.0/100000.0)*H
    print("test r = %0.2f, H=%0.1f"%(t_r/1000.0,H/1000.0))

def test_kepler51b():
    k51b_mass = 1.2E25 #[kg]
    k51b_rad = 4.48E7  #[m]

    p_top = 100000.0 #1 bar or 100000 [Pa]

    m_core = k51b_mass*0.95
    core_rho = 5510.0 #earth-like density
    T = 1700.0 #isothermal temp [K]

    r, r_s = calculate_rad(p_top, m_core, core_rho, k51b_mass, R_H2, T)

    print("found altitude = %0.1f km (%0.2f of observed)"%((r-r_s)/1000.0, r/k51b_rad))
    print("core is %0.2f of observed"%(r_s/k51b_rad))

test_kepler51b()






















