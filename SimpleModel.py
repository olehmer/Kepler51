"""
Owen Lehmer 11/17/2016
University of Washington

This file will calculate the atmospheric structure for a young planet with a
hydrogen rich atmosphere.

"""


import numpy as np
from math import pi, exp, log
import matplotlib.pyplot as plt



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
SECONDS_PER_YEAR = 3.154E7 #seconds in a year
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

def calculate_xuv_at_t(L_xuv, t, t_sat=1.0E8):
    """
    Calculate the XUV flux after the specified amount of time.

    Inputs:
    L_xuv - the initial XUV flux
    t - the time since saturation age in years
    t_sat - the saturation age for the XUV in years, default to 100 Myr

    Returns:
    F_xuv - the modified XUV flux

    NOTE: see equation (1) of Luger et al (2015)
    """

    F_xuv = L_xuv

    if t > t_sat:
        F_xuv = L_xuv*(t_sat/t) #beta is of order -1

    return F_xuv


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
            
def calculate_loss_rate(mass, core_rad, rad_1bar, dist, star_mass, time):
    """
    This function will calculate the hydrodynamic loss rate from the planet
    based on Luger (2015).

    Inputs:
    mass - the total mass of the planet
    core_rad - the radius of the planet
    rad_1bar - the radius of the 1 bar level in the atmosphere
    dist - the orbital distance of the planet
    star_mass - the mass of the host star
    time - the time since formation of the planet [in years]

    Returns:
    dMdt - the loss rate in kg/s
    """

    xuv = calculate_xuv_lammer2012(dist)
    F_xuv = calculate_xuv_at_t(xuv, time)


    #assume the radius of XUV absorption is equal to the radius of the whole
    #planet (usually within ~10% from Luger et al (2015)
    r_xuv = rad_1bar

    k_tide = calculate_ktide(mass, star_mass, dist, core_rad)

    #Equation 5 from Luger et al. (2015)
    dMdt = (e_xuv*pi*F_xuv*core_rad*r_xuv**2.0)/(GG*mass*k_tide)

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

    if r < r_s:
        #on really small hot bodies this can happen
        r = r_s

    return (r,r_s)

def plot_radius_over_time(time, radius, r_s):
    """
    Plot the radius of the planet at 5 points across the time array. Plotted
    in terms of core radius.

    Inputs:
    time - array of time values
    radius - array of corresponding radius values
    r_s - the surface radius of the core
    """

    plt.subplot(311,aspect="equal")

    #find the 5 points
    step = len(time)/4 #3 interior points, both ends

    r_0 = (time[0],radius[0]/r_s)
    r_1 = (time[step],radius[step]/r_s)
    r_2 = (time[step*2],radius[step*2]/r_s)
    r_3 = (time[step*3],radius[step*3]/r_s)
    r_4 = (time[-1],radius[-1]/r_s)

    #the height of the largest circle
    h = r_0[1]
    

    x_pts = np.zeros(5) #keep track of the x locations

    #create the circle objects
    spacing = 2.0

    x_pos = r_0[1]+spacing
    x_pts[0] = x_pos
    circ0 = plt.Circle((x_pos,0),radius=r_0[1], alpha=0.2, color="blue")
    core0 = plt.Circle((x_pos,0),radius=1.0, alpha=1.0, color="black")

    x_pos = x_pos+r_0[1]+r_1[1]+ spacing
    x_pts[1] = x_pos
    circ1 = plt.Circle((x_pos,0),radius=r_1[1], alpha=0.2, color="blue")
    core1 = plt.Circle((x_pos,0),radius=1.0, alpha=1.0, color="black")

    x_pos = x_pos + r_1[1]+r_2[1]+spacing
    x_pts[2] = x_pos
    circ2 = plt.Circle((x_pos,0),radius=r_2[1], alpha=0.2, color="blue")
    core2 = plt.Circle((x_pos,0),radius=1.0, alpha=1.0, color="black")

    x_pos = x_pos + r_2[1] + r_3[1] + spacing
    x_pts[3] = x_pos
    circ3 = plt.Circle((x_pos,0),radius=r_3[1], alpha=0.2, color="blue")
    core3 = plt.Circle((x_pos,0),radius=1.0, alpha=1.0, color="black")

    x_pos = x_pos + r_3[1] + r_4[1] + spacing
    x_pts[4] = x_pos
    circ4 = plt.Circle((x_pos,0),radius=r_4[1], alpha=0.2, color="blue")
    core4 = plt.Circle((x_pos,0),radius=1.0, alpha=1.0, color="black")

    end_x = x_pos + r_4[1] + spacing

    #radius over time, plot 5 different times
    plt.gcf().gca().add_artist(circ0)
    plt.gcf().gca().add_artist(core0)
    plt.gcf().gca().add_artist(circ1)
    plt.gcf().gca().add_artist(core1)
    plt.gcf().gca().add_artist(circ2)
    plt.gcf().gca().add_artist(core2)
    plt.gcf().gca().add_artist(circ3)
    plt.gcf().gca().add_artist(core3)
    plt.gcf().gca().add_artist(circ4)
    plt.gcf().gca().add_artist(core4)

    plt.xlim(0,end_x)
    plt.ylim(-(h+1),h+1)

    #update the tick marks
    plt.gcf().gca().set_xticks(x_pts)

    x0 = str("%0.1f"%(r_0[0]/SECONDS_PER_YEAR/1.0E6)) #tick mark in Myr
    x1 = str("%0.1f"%(r_1[0]/SECONDS_PER_YEAR/1.0E6)) 
    x2 = str("%0.1f"%(r_2[0]/SECONDS_PER_YEAR/1.0E6)) 
    x3 = str("%0.1f"%(r_3[0]/SECONDS_PER_YEAR/1.0E6)) 
    x4 = str("%0.1f"%(r_4[0]/SECONDS_PER_YEAR/1.0E6)) 
    x_labels = [x0,x1,x2,x3,x4]
    
    ax = plt.gcf().gca()
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels([str(abs(x)) for x in ax.get_yticks()])

    plt.ylabel("Core Radii")
    plt.title("A", loc="left")


def plot_mass_over_time(time, atmos_mass):
    """
    Plot the mass of the planet over time

    Inputs:
    time - the array of time values
    atmos_mass - the array of atmospheric mass values
    """

    time_myr = time/SECONDS_PER_YEAR/1.0E6 #time in Myr
    plt.subplot(312)
    plt.plot(time_myr, atmos_mass/atmos_mass[0])
    plt.xlim(0,time_myr[-1])
    plt.ylabel("Atmosphere Mass\nFraction", multialignment="center")
    plt.title("B", loc="left")

def plot_density_over_time(time,atmos_mass,core_mass,radius):
    """
    Plot the density of the planet over time.

    Inputs:
    time - the array of time values
    atmos_mass - the array of atmospheric mass values
    core_mass - the mass of the core
    radius - the array of radius values
    """

    density = np.zeros(len(time))
    for i in range(0,len(time)):
        #calculate density in [kg m-3]
        rho = (core_mass + atmos_mass[i])/(4.0/3.0*pi*radius[i]**3.0)
        density[i] = rho


    time_myr = time/SECONDS_PER_YEAR/1.0E6 #time in Myr
    plt.subplot(313)
    plt.plot(time_myr, density*0.001) #convert density to g/cc
    plt.xlim(0,time_myr[-1])
    plt.ylabel("Density [g/cc]")
    plt.title("C", loc="left")
    
    
def planet_over_time(mass, dist, T, core_mass, core_rho, R_gas,\
        star_mass, timestep=1.0E4, duration=3.0E8):
    """
     Calculate the planet over time taking into account the hydrodynamic escape.

    Inputs:
    mass - the total mass of the planet [kg]
    dist - the orbital distance of the planet [m]
    T - the isothermal temperature of the planetary atmosphere [K] 
    core_mass - the mass in the planet's core [kg]
    core_rho - the density of the planet's core [kg m-3]
    R_gas - the specific gas constant of the atmosphere [J kg-1 K]
    star_mass - the mass of the parent star [kg]
    timestep - the timestep to use in the simulation, given in years
    duration - the duration of the simulation in years

    Returns:
    time - the array of time values
    atmos_mass - the atmospheric mass at the corresponding time
    radius - the planetary radius at the time
    r_s - the radius of the core
    """

    ts = timestep*SECONDS_PER_YEAR #10,000 years in seconds as our timestep

    dur = duration*SECONDS_PER_YEAR #the duration of the simulation, 300 Myr

    p_r = 1.0E5 #the TOA pressure, assume we see at the 1 bar level

    num_steps = int(round(dur/ts)) #the number of iterations to perform

    time = np.zeros(num_steps)
    atmos_mass = np.zeros(num_steps)
    radius = np.zeros(num_steps)

    r, r_s = calculate_rad(p_r, core_mass, core_rho, mass, R_gas, T)


    #enter the starting values
    time[0] = 0.0
    atmos_mass[0] = mass - core_mass
    radius[0] = r

    #keep track of the index at which the entire atmosphere is lost
    end_i = num_steps - 1

    cur_mass = mass

    for i in range(1, num_steps):
        r = r_s
        if cur_mass > core_mass:
            r, r_s = calculate_rad(p_r, core_mass, core_rho, cur_mass, R_gas, T)
            dMdt = calculate_loss_rate(cur_mass, r_s, r, dist, star_mass, i*ts/SECONDS_PER_YEAR)

            total_loss = dMdt*ts

            cur_mass = cur_mass - total_loss
            if cur_mass < core_mass or r < r_s:
                #we've lost the whole atmosphere!
                cur_mass = core_mass
                r = r_s
                end_i = i

        radius[i] = r
        time[i] = i*ts
        atmos_mass[i] = cur_mass - core_mass


    #only return the interesting parts
    time = time[0:end_i+1]
    atmos_mass = atmos_mass[0:end_i+1]
    radius = radius[0:end_i+1]

    return (time, atmos_mass, radius, r_s)


def plot_planet_over_time(mass, dist, T, core_mass, core_rho, R_gas,\
        star_mass, timestep=1.0E4, duration=3.0E8):
    """
    Plot the planet over time taking into account the hydrodynamic escape.

    Inputs:
    mass - the total mass of the planet [kg]
    dist - the orbital distance of the planet [m]
    T - the isothermal temperature of the planetary atmosphere [K] 
    core_mass - the mass in the planet's core [kg]
    core_rho - the density of the planet's core [kg m-3]
    R_gas - the specific gas constant of the atmosphere [J kg-1 K]
    star_mass - the mass of the parent star [kg]
    timestep - the timestep to use in the simulation, given in years
    duration - the duration of the simulation in years
    """

    time, atmos_mass, radius, r_s = planet_over_time(mass,dist,T,core_mass,\
            core_rho,R_gas,star_mass,timestep,duration)

    plt.subplots_adjust(hspace=0.3)
    plot_radius_over_time(time, radius, r_s)
    plot_mass_over_time(time, atmos_mass)
    plot_density_over_time(time,atmos_mass,core_mass,radius)
    plt.xlabel("Time [Myr]")
    plt.show()


def plot_kepler51b():
    T = 1500.0
    core_rho = 5800.0
    core_mass = k51b_mass*0.99
    plot_planet_over_time(k51b_mass, k51b_orbital_dist, T, \
            core_mass, core_rho, R_H2, k51_mass, duration=1.0E9)







def plot_escape_parameter_space():
    """
    Plot the parameter space for mass loss. Explore what determines if a planet
    is rocky or gaseous
    """

    core_rho = 5510.0 #core density [kg m-3]
    core_mass_percent = 0.99 #core represents 97% of the mass
    dist = 0.25*AU #orbital distance [m]
    T = 1500.0 #isothermal atmospheric temperature [k], depends on dist
    R_gas = R_H2
    star_mass = k51_mass


    min_mass = 0.5*M_Earth
    max_mass = 10.0*M_Earth

    masses = np.linspace(min_mass,max_mass,20)
    radii = np.zeros(len(masses))
    mass_loss = np.zeros(len(masses))

    for i in range(0,len(masses)):
        core_mass = masses[i]*core_mass_percent
        ts, ms, rs, r_sur = planet_over_time(masses[i],dist,T,core_mass,\
                core_rho,R_gas,star_mass,duration=1.0E9)

        #we only want the last r from this
        radii[i] = rs[-1]

        mass_loss[i] = ms[-1]/(core_mass-core_mass*core_mass_percent)

    #create the Earth density curve
    line_masses = np.linspace(min_mass,max_mass,200)
    line_radii_earth = np.zeros(len(line_masses))
    line_radii_water = np.zeros(len(line_masses))
    rho_earth = 5510.0 #earth density
    rho_water = 1000.0
    for i in range(0,200):
        r_earth = (line_masses[i]/(4.0/3.0*pi*rho_earth))**(1.0/3.0)
        r_water = (line_masses[i]/(4.0/3.0*pi*rho_water))**(1.0/3.0)
        line_radii_earth[i] = r_earth
        line_radii_water[i] = r_water

    #plot the line of Earth density
    plt.plot(line_masses/M_Earth, line_radii_earth/R_Earth, "g--")
    plt.plot(line_masses/M_Earth, line_radii_water/R_Earth, "b-.")

    cm = plt.cm.get_cmap("coolwarm")
    sc = plt.scatter(masses/M_Earth,radii/R_Earth, c=mass_loss, cmap=cm, s=80.0)
    plt.colorbar(sc).set_label("Remaining Atmospheric\nFraction")
    plt.xlim(min_mass/M_Earth,max_mass/M_Earth)
    plt.xlabel("Mass [Earth Masses]")
    plt.ylabel("Radius [Earth Radii]")
    plt.show()


plot_escape_parameter_space()
        








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























