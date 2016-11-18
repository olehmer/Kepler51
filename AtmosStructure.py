"""
Owen Lehmer 11/17/2016
University of Washington

This file will calculate the atmospheric structure for a young planet with a
hydrogen rich atmosphere.

"""


import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from math import pi, exp


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
            

def atmos_radius(core_rho, core_mass, mass, rad, dist, star_mass, star_rad,\
        star_T, R_gas, p_top):
    """
    This function will calculate the height of the TOA level in the 
    atmosphere. 1 bar is the expected edge for transiting planets (Seager, 2000)

    Inputs:
    core_rho - the density of the planetary core
    core_mass - the mass of the planetary core
    mass - the total mass of the planet
    rad - the radius of the planet
    dist - the orbital distance of the planet
    star_mass - the mass of the host star
    star_rad - the radius of the star
    star_T - the surface temperature of the star
    R_gas - the molar gas constant for the atmosphere
    p_top - the pressure at the limb of the planet (typically ~1 bar)
    p_s - the surface pressure of the atmosphere

    Returns:
    r - the height of the p_top point in the atmosphere [m]
    core_rad - the radius of the core [m]
    """

    xuv = calculate_xuv_lammer2012(dist)
    
    #the core radius is easily calculated
    core_rad = (3.0*core_mass/(4.0*pi*core_rho))**(1.0/3.0)

    #calculate the surface gravity
    g_s = GG*core_mass/core_rad**2.0

    #calculate the surface pressure
    p_s = (mass-core_mass)*g_s/(4.0*pi*core_rad**2.0)

    print("surface pressure is: %0.2f bar"%(p_s/100000.0))

    bol_flux = calculate_flux_planck(star_T, star_rad, dist) #bolometric flux
    T = (0.25*bol_flux/SIGMA)**0.25 #assume the temperature is just in radiative balance
    #assumed here albedo is zero

    #assume the radius of XUV absorption is equal to the radius of the whole
    #planet (usually within ~10% from Luger et al (2015)
    #so r_xuv = rad

    k_tide = calculate_ktide(mass, star_mass, dist, rad)

    #Equation 5 from Luger et al. (2015)
    dMdt = (e_xuv*pi*xuv*rad**3.0)/(GG*mass*k_tide)

    print("dMdt=%3.3e kg/s"%(dMdt))

    def eqn(r):
        """
        Closure equation to be called by fsolve() to find the radius at which 
        p_top occurs.
        """

        return p_s*exp(-1.0/(R_gas*T)*(g_s*core_rad-g_s*core_rad**2.0/r-0.5*(dMdt*T*R_gas/(r**2.0*p_top)))) - p_top

    r = optimize.fsolve(eqn, core_rad)

    return (r, core_rad) 



def earth_test():
    r, core_rad = atmos_radius(5515.0,M_Earth-1.2E18, M_Earth, R_Earth, AU, 1.99E30,\
            6.957E8 ,5800.0, 287.0, 100.0)

    print("got r=%f"%(r-R_Earth))



def kepler51b_rad(core_mass, core_rho):
    """
    Calculate the radius of the 1bar point on Kepler 51b
    """

    #Some measured constants from Masuda (2014)
    k51b_orbital_dist = 0.25*AU #orbit of k51b 
    k51b_mass = 1.2E25 #[kg]
    k51b_rad = 4.48E7  #[m]
    k51_mass = 2.1E30  #mass of Kepler 51 (the star) in [kg]
    k51_rad = 6.957E8 #the radius of K51 (assumed ~rad_sun) [m]
    k51_T = 6018.0 #surface temp of K51

    p_top = 1.0E5 #100,000 Pa or 1 bar

    r, cr = atmos_radius(core_rho, core_mass, k51b_mass, k51b_rad, k51b_orbital_dist,\
            k51_mass,k51_rad,k51_T, R_H2, p_top)

    print("Kepler-51b atmos height is: %0.2f km, cr=%0.2f km"%((r-cr)/1000.0,cr/1000.0))
    print("Fraction of observed radius: %f"%(r/k51b_rad))

kepler51b_rad(M_Earth*0.95, 3515.0)













