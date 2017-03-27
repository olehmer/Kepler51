"""
Owen Lehmer 11/17/2016
University of Washington

This file will calculate the atmospheric structure for a young planet with a
hydrogen rich atmosphere.

"""


import numpy as np
from math import pi, exp, log, floor
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation 
import sys

from moviepy.video.io.bindings import mplfig_to_npimage #used for animation
import moviepy.editor as mpy

###########################UNIVERSAL CONSTANTS#################################
kB = 1.380662E-23        #Boltzmann's constant [J K^-1]
GG = 6.672E-11           #Gravitational constant [N m^2 kg^-2]
SIGMA = 5.6704E-8       #Stefan-Boltzmann constant [W m^-2 K^-4]
m_H = 1.66E-27          #Mass of H atom [kg]
e_xuv = 0.362 #typically between 0.1 and 0.3
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

#make the font bigger
matplotlib.rc('font', family="serif", serif="Times New Roman", size=18)



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



def total_mass_loss(time, mass, dist):
    """
    Calculate the total mass lost after time seconds

    Inputs:
    time - total time to consider [seconds]
    mass - mass of planet (assumed Earth-like density)
    dist - orbital distance of planet [m]

    Returns:
    loss - total mass loss [kg]
    """

    e_xuv = 0.1
    rho = 5510.0 #density in [kg m-3]
    flux = calculate_xuv_lammer2012(dist)

    r_s = (mass/(4.0/3.0*pi*rho))**(1.0/3.0)

    loss = (2.0*e_xuv*pi*flux*r_s**3.0*time/GG)**0.5

    new_loss = 1.5*3.0*e_xuv*flux*time/(4.0*GG*rho)

    return new_loss #loss


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

    F_xuv = 105.5 #L_xuv ORL hardcoded to 100*modern Sun at 0.1 AU

    if t > t_sat: 
        F_xuv = F_xuv*(t_sat/t) #beta is of order -1

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
    #but we're not actually doing that, it's calculated
    r_xuv = rad_1bar

    k_tide = 1.0 #calculate_ktide(mass, star_mass, dist, core_rad)

    #Equation 5 from Luger et al. (2015)
    #dMdt = (e_xuv*pi*F_xuv*core_rad*r_xuv**2.0)/(GG*mass*k_tide)
    dMdt = (e_xuv*pi*F_xuv*r_xuv**3.0)/(GG*mass*k_tide)

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

    #calculate the radius of the rocky core based on Zeng (2015)
    r_s = 1.3*core_mass**0.27 #(3.0*core_mass/(4.0*pi*core_rho))**(1.0/3.0)

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

    plt.subplot(211,aspect="equal")

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
    spacing = 6.0

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
    plt.locator_params(axis='y', nbins=6)
    ax.set_yticklabels([str(abs(x)) for x in ax.get_yticks()])

    plt.ylabel("$R_{XUV}$\n[Core Radii]")
    plt.title("A", y=0.7, x=0.975)


def plot_mass_over_time(time, atmos_mass):
    """
    Plot the mass of the planet over time

    Inputs:
    time - the array of time values
    atmos_mass - the array of atmospheric mass values
    """

    time_myr = time/SECONDS_PER_YEAR/1.0E6 #time in Myr
    plt.subplot(212)
    plt.plot(time_myr, atmos_mass/atmos_mass[0])
    plt.xlim(0,time_myr[-1])
    #plt.xticks(np.arange(0,time_myr[-1]+1,1))
    plt.minorticks_on()
    plt.ylabel("Atmosphere Mass\nFraction", multialignment="center")
    plt.title("B", y=0.85, x=0.975)

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
    plt.ylabel("Density [g cm"+r'$^{-1}$'+"]")
    plt.title("C", loc="left")
    
    
def planet_over_time(mass, dist, T, core_mass, core_rho, R_gas,\
        star_mass, timestep=1.0E6, duration=1.0E8, p_r=5.0):
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
    p_r = the pressure to calculate the radius to (defaults to base of 
          thermosphere, which is at 5 Pa [See Owen & Jackson (2012)])

    Returns:
    time - the array of time values
    atmos_mass - the atmospheric mass at the corresponding time
    radius - the planetary radius at the time
    r_s - the radius of the core
    """

    ts = timestep*SECONDS_PER_YEAR #10,000 years in seconds as our timestep

    dur = duration*SECONDS_PER_YEAR #the duration of the simulation, 100 Myr


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
            if cur_mass <= core_mass or r <= r_s:
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
        star_mass, timestep=1.0E6, duration=1.0E8):
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
            core_rho,R_gas,star_mass,timestep,duration, p_r=5.0)

    plt.subplots_adjust(hspace=0)
    plot_radius_over_time(time, radius, r_s)
    plot_mass_over_time(time, atmos_mass)
    #plot_density_over_time(time,atmos_mass,core_mass,radius) #IMPORTANT if 
    #uncommented change subplot numbers back to 311 and 312 in previous funcs
    plt.xlabel("Time [Myr]")
    plt.show()


def plot_kepler51b():
    """
    Plot Kepler-51b
    """
    T = 2000.0
    core_rho = 5510.0
    core_mass = k51b_mass*0.97
    print("Mass = %0.4f Earth Masses"%(k51b_mass/M_Earth))
    plot_planet_over_time(k51b_mass, k51b_orbital_dist, T, \
            core_mass, core_rho, R_H2, k51_mass, timestep=1.0E4, duration=1.0E9)

def plot_planet_raius():
    matplotlib.rc('font',size=14)
    T = 880.0
    core_rho = 5510.0
    mass = 2.0*M_Earth
    core_mass = mass*0.97
    orb_dist = 0.1*AU
    plot_planet_over_time(mass, orb_dist, T, \
            core_mass, core_rho, R_H2, k51_mass, timestep=1.0E4, duration=1.0E8)






def plot_escape_parameter_space(DUR=1.0E8, TS=1.0E6, T=1000.0, NO_PLOT=False):
    """
    Plot the parameter space for mass loss. Explore what determines if a planet
    is rocky or gaseous

    Inputs:
    DUR - the duration of the model [yrs]
    TS - the time step to use [yrs]
    T - the isothermal atmospheric temperature [K]
    NO_PLOT - don't plot the result, just return the data

    Returns:
    masses - the masses at the end of the model run
    radii - the radii at the end of the model
    atmos_mass - the remaining atmospheric mass
    """

    core_rho = 5510.0 #core density [kg m-3]
    core_mass_percent = 0.941 #core represents 94.1% of the mass
    dist = 0.1*AU #orbital distance [m]
    R_gas = 3873.0 #R_H2
    star_mass = k51_mass

    min_mass = 0.5*M_Earth
    max_mass = 13.0*M_Earth

    masses = np.linspace(min_mass,max_mass,20)
    radii = np.zeros(len(masses))
    atmos_mass = np.zeros(len(masses))

    for i in range(0,len(masses)):
        #just set the core mass to the percent of the total mass specified
        core_mass = masses[i]*core_mass_percent

        #calculate the planet over time!
        ts, ms, rs, r_sur = planet_over_time(masses[i],dist,T,core_mass,\
                core_rho,R_gas,star_mass,duration=DUR, timestep=TS)

        #we only want the last values for the plot from this
        radii[i] = rs[-1]
        atmos_mass[i] = ms[-1]/(core_mass-core_mass*core_mass_percent)

        if atmos_mass[i] > 1.0:
            #for very short timescales numerical issues sometimes crop up 
            atmos_mass[i] = 1.0



    if not NO_PLOT:
        #create the Earth density curve and water density curve
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
        plt.plot(line_masses/M_Earth, line_radii_earth/R_Earth, "k--", zorder=1)
        #plt.plot(line_masses/M_Earth, line_radii_water/R_Earth, "b-.")

        #plot the data
        cm = plt.cm.get_cmap("gray") #bwr previously
        sc = plt.scatter(masses/M_Earth,radii/R_Earth, c=atmos_mass, cmap=cm, s=80.0, zorder=2)
        plt.colorbar(sc).set_label("Remaining Atmospheric\nFraction")
        plt.xlim(min_mass/M_Earth,max_mass/M_Earth)
        plt.xlabel("Mass [Earth Masses]")
        plt.ylabel("Radius [Earth Radii]")
        #plt.title("Atmospheric Temperature of %0.0f [K]"%(T))
        plt.grid()
        plt.show()

    return masses, radii, atmos_mass

def animate_loss(SAVE_TO_FILE=False):
    """
    Animate the plot_escape_parameter_space() function over time

    Inputs:
    SAVE_TO_FILE - if true the animation will be written to the file
                   protoatmosphere_loss.gif
    """

    save_duration = 10 #the amount of time the saved gif should last [s]

    temp = 1690.0 # temperature [K]
    start = 1.0E7 #start at 10 Myr
    end = 1.0E9 #end at 1 Gyr
    count = 100
    dur_steps = np.linspace(start, end, count) #generate count frames

    timestep = 1.0E5 #the time step to use

    fig = plt.figure(facecolor="white")
    ax = fig.add_subplot(111)
    cm = plt.cm.get_cmap("bwr")

    frames = []
    for i in range(0, count):
        #print the status just cause
        sys.stderr.write("\r%2.0f%% done"%(100.0*float(i)/float(count)))
        sys.stderr.flush()

        cur_dur = dur_steps[i]
        curVals = plot_escape_parameter_space(DUR=cur_dur, \
                TS=timestep, T=temp, NO_PLOT=True)
        frames.append(curVals)

    sys.stderr.write("\r100%% done\n")
    sys.stderr.flush()

    #init the colorbar and plot
    masses, radii, atmos_mass = frames[0]
    sc = plt.scatter(masses/M_Earth,radii/R_Earth, c=atmos_mass, cmap=cm, s=80.0)
    clb = plt.colorbar(sc)
    clb.set_label("Remaining Atmospheric\nFraction")
    plt.xlim(np.min(masses)/M_Earth,np.max(masses)/M_Earth)
    plt.ylim(0.5,4)
    plt.xlabel("Mass [Earth Masses]")
    plt.ylabel("Radius [Earth Radii]")
    #plt.title("Atmospheric Temperature set to %0.0f [K]"%(temp))


    #set up the time label
    ax.text(7,3.75,"Time: %4.0d [Myr]"%(0))

    #calculate the Earth density line
    line_masses = np.linspace(0.5*M_Earth,13*M_Earth, 200)
    line_radii_earth = np.zeros(len(line_masses))
    rho_earth = 5510.0 #earth density
    for i in range(0,200):
        r_earth = (line_masses[i]/(4.0/3.0*pi*rho_earth))**(1.0/3.0)
        line_radii_earth[i] = r_earth

    
    def animate(i):
        if SAVE_TO_FILE:
            #i is passed in as the amount of time in the duration
            i = int(floor(i/float(save_duration)*float(count)))
        masses, radii, atmos_mass = frames[i]
        plt.cla() #clear the frame
        sc = plt.scatter(masses/M_Earth,radii/R_Earth, c=atmos_mass, cmap=cm, s=80.0)

        plt.plot(line_masses/M_Earth, line_radii_earth/R_Earth, "k--")

        plt.xlim(np.min(masses)/M_Earth,np.max(masses)/M_Earth)
        plt.ylim(0.5,4)
        plt.xlabel("Mass [Earth Masses]")
        plt.ylabel("Radius [Earth Radii]")
        #plt.title("Atmospheric Temperature set to %0.0f [K]"%(temp))

        #update the time text
        ax.text(7,3.75,"Time: %5.0d [Myr]"%(dur_steps[i]/1.0E6))

        if SAVE_TO_FILE:
            return mplfig_to_npimage(fig)


    if SAVE_TO_FILE:
        ani = mpy.VideoClip(animate, duration=save_duration) 
        ani.write_gif("protoatmosphere_loss.gif", fps=20, opt="nq")
    else:
        ani = animation.FuncAnimation(fig,animate, frames=count, repeat_delay=1000,\
            blit=False)
        plt.show()


def plot_Rxuv_at_time(time=1.0E8):
    """
    Plot the radius of the 1 bar level at time years
    """

    time = time*SECONDS_PER_YEAR

    rho = 1000.0 #density of core [kg m-3]
    dist = 0.25*AU #orbital distance [m]
    T = 2000.0
    p = 1.0E5

    xuv = calculate_xuv_lammer2012(dist)
    F_xuv = calculate_xuv_at_t(xuv, time)

    loss_const = 0.0 #3.0*e_xuv*F_xuv/(4.0*GG*rho)

    print("loss = %0.2f kg/s, after 100 Myr loss = %0.5f Earth masses"%(loss_const,(loss_const*time)/M_Earth))
    
    masses = np.linspace(0.5*M_Earth,10.0*M_Earth,500)
    rxuv = np.zeros(len(masses))



    for i in range(len(masses)):
        m = masses[i]
        ma = 0.03*m #initial atmospheric mass
        R_s = (3.0*m/(4.0*pi*rho))**(1.0/3.0)

        g_s = GG*m/R_s**2.0
        H = R_H2*T/g_s

        p_s = ma*g_s/(4.0*pi*R_s**2.0)

        rx = R_s
        if True:
            #rx = R_s**2.0/(H*log(p*4.0*pi*R_s**2.0/(g_s*(ma-loss_const*time)))+R_s)

            rx = R_s**2.0/(H*log(p/p_s)+R_s)
            #print("%3d: p_s=%2.3e Pa, H=%2.3e m, R_s=%2.3e m, log(p/p_s)=%0.2f, rx=%2.3e"%(i,p_s,H,R_s,log(p/p_s),rx))
            
        rxuv[i] = rx

    plt.plot(masses/M_Earth, rxuv/R_Earth)
    plt.xlabel("Mass [M$_{\oplus}$]")
    plt.ylabel(r'$R_{XUV}$'+" [R$_{\oplus}$]")
    plt.ylim(-100,100)
    plt.xlim(0.5,10)
    plt.show()



def plot_radius_mass_raltionship():
    """
    Plot the radius for a given mass on a single curve.
    """
    core_rho = 5510.0 #core density, [kg m-3]
    R_gas = R_H2
    p_r = 1.0E5 #1 bar in Pa

    #atmos_mass = M_Earth*0.05 #static %5 of an Earth mass in the atmosphere
    core_mass_percent = 0.97

    masses = np.linspace(1.75*M_Earth,10.0*M_Earth,100)

    radii = np.zeros(len(masses))
    
    temps = np.linspace(500,2000,4)

    for T in temps:
        for i in range(len(masses)):
            core_mass = masses[i]*core_mass_percent
            r, r_s = calculate_rad(p_r, core_mass, core_rho, masses[i], R_gas, T)
            radii[i] = r

        label_str = "T=%0.0f"%(T)
        plt.plot(masses/M_Earth, radii/R_Earth, label=label_str)
    plt.xlim(1.75,10.0)
    plt.ylim(1.5,4)
    plt.xlabel("Mass [Earth Masses]")
    plt.ylabel("Radius [Earth Radii]")
    plt.title("Atmospheric mass: %0.0fwt.%%"%((1.0-core_mass_percent)*100.0))
    plt.legend()
    plt.grid()
    plt.show()


def plot_radius_over_individual_mass(T=1000.0):
    """
    Plot the radius of each body for a given mass
    """

    core_rho = 5510.0 #core density, [kg m-3]
    R_gas = R_H2
    core_mass_percent = 0.97
    p_r = 1.0E5 #1 bar in Pa

    masses = np.linspace(1.5*M_Earth,7.0*M_Earth,10)

    for j in range(len(masses)):
        core_mass = masses[j]*core_mass_percent
        atmos_mass = masses[j] - core_mass
        atmos_masses = np.linspace(0,atmos_mass,100)
        radii = np.zeros(len(atmos_masses))

        for i in range(len(atmos_masses)):
            r, r_s = calculate_rad(p_r, core_mass, core_rho, atmos_masses[i]+core_mass, R_gas, T)
            radii[i] = r


        if j < 5:
            #show only some labels
            label_str = "%0.1f Earth Masses"%(masses[j]/M_Earth)
            plt.plot(atmos_masses/M_Earth,radii/R_Earth, label=label_str)
        else:
            plt.plot(atmos_masses/M_Earth,radii/R_Earth)



    plt.legend(loc="top left")
    plt.xlabel("Atmospheric Mass [Earth Masses]")
    plt.ylabel("Radius of 1 bar level [Earth Radii]")
    plt.title("Temperature Set to %0.0f [K]"%(T))
    plt.xlim(0,0.05)
    plt.ylim(0,10)
    plt.grid()
    plt.show()

def plot_escape_parameter_space_side_by_side():
    
    m3000, r3000, am3000 = plot_escape_parameter_space(DUR=1.0E8, T=1760, NO_PLOT=True)
    m2000, r2000, am2000 = plot_escape_parameter_space(DUR=1.0E8, T=880, NO_PLOT=True)

    min_mass = np.min(m2000)
    max_mass = np.max(m2000)

    #create the Earth density curve and water density curve
    line_masses = np.linspace(min_mass,max_mass,200)
    line_radii_earth = np.zeros(len(line_masses))
    rho_earth = 5510.0 #earth density
    for i in range(0,200):
        r_earth = (line_masses[i]/(4.0/3.0*pi*rho_earth))**(1.0/3.0)
        line_radii_earth[i] = r_earth


    fig, (ax1,ax2) = plt.subplots(1,2,sharey=True)
    fig.subplots_adjust(wspace=0.05)
    fig.set_size_inches(12,6, forward=True)

    cm = plt.cm.get_cmap("bwr")

    ax1.plot(line_masses/M_Earth, line_radii_earth/R_Earth, "k--", zorder=1)
    sc1 = ax1.scatter(m2000/M_Earth,r2000/R_Earth, c=am2000, cmap=cm, s=80.0, zorder=2)
    ax1.set_xlabel("Mass [Earth Masses]")
    ax1.set_title("A", y=0.9, x=0.075)
    ax1.set_ylabel("Radius [Earth Radii]")
    ax1.set_xlim(min_mass/M_Earth,max_mass/M_Earth)
    ax1.xaxis.grid()
    ax1.yaxis.grid()

    ax2.plot(line_masses/M_Earth, line_radii_earth/R_Earth, "k--", zorder=1)
    sc2 = ax2.scatter(m3000/M_Earth,r3000/R_Earth, c=am3000, cmap=cm, s=80.0, zorder=2)
    ax2.set_xlabel("Mass [Earth Masses]")
    ax2.set_title("B", y=0.9, x=0.075)
    ax2.set_xlim(min_mass/M_Earth,max_mass/M_Earth)
    plt.grid()


    plt.colorbar(sc1, ax=[ax1,ax2]).set_label("Remaining Atmospheric\nFraction")
    plt.show()

    title = "880K_1760K_compared.png"
    print("SAVING FIGURE AS: %s"%(title)) 
    fig.savefig(title, dpi=100)



def plot_rxuv_denominator():
    """
    Plot the denominator of the r_xuv function (from the hydrostatic equation).
    """

    masses = np.linspace(0.5*M_Earth,10.0*M_Earth,200)
    denom = []

    rho = 5510.0 #density of rocky core [kg m-3]
    T = 880.0 #isothermal temp [K]
    dur = 1.0E8 #duration in Years
    tstep = 1.0E5 #timestep in Years
    dist = 0.1*AU 
    times = []
    num_lines = 5 

    for i in range(len(masses)):
        #just set the core mass to the percent of the total mass specified
        core_mass = masses[i]*0.97

        #calculate the planet over time!
        times, ams, r, r_s = planet_over_time(masses[i],dist,T,core_mass,\
                rho,R_H2,k51_mass, timestep=tstep, duration=dur) #get the results for 10,000,000

        g_s = GG*masses[i]/r_s**2.0



        num_times = int(round(dur/tstep)) #the number of items that should be produced
        cur_time = 0.0
        for k in range(len(times),num_times):
            #loop over the time to pad the arrays we use
            times = np.append(times,cur_time)
            ams = np.append(ams,0.0)
            r = np.append(r,r_s)
            cur_time += tstep


        time_vals = [] 
        spacing = len(ams)/(num_lines-1) 
        for j in range(num_lines):
            ind = j*spacing
            if ind >= len(ams)-1:
                ind = len(ams)-1
            atmos_mass = ams[ind]

            #calculate the surface pressure
            p_s = atmos_mass*g_s/(4.0*pi*r_s**2.0)

            H = R_H2*T/g_s

            if p_s==0:
                #this is a super cludgy way to correct for timstep anomalies
                time_vals.append((r_s/R_Earth)**3.0)
                #print("%3d, adding mod for mass: %0.2f"%(i,masses[i]/M_Earth))
            else:
                #print("%3d, ind was: %d"%(i,ind))
                #time_vals.append((r[ind]/R_Earth)**2.0)
                time_vals.append( ((r_s**2.0/(H*log(0.1/p_s)+r_s))/R_Earth)**3.0 )

        denom.append(time_vals)


    for i in range(num_lines):
        y_array = []
        for j in range(len(denom)):
            y_array.append(denom[j][i])
        spacing = len(times)/(num_lines-1)
        ind = i*spacing
        line_label = ""
        if ind >= len(times)-1:
            ind = len(times)-1
            line_label = "100 Myr"
        else:
             line_label = "%3.0f Myr"%(times[ind]/SECONDS_PER_YEAR/1E6)
        plt.plot(masses/M_Earth,y_array, label=line_label)
    plt.xlabel("Mass [Earth Masses]")
    plt.ylabel("$R^{3}_{XUV}$ [Earth Radii]")
    plt.ylim(0,100)
    plt.xlim(1.75,10)
    plt.legend(loc="bottom right")
    plt.grid()
    plt.show()










def plot_total_loss_over_time():
    masses = np.linspace(0.5*M_Earth,10.0*M_Earth,100)
    losses = []
    
    for m in masses:
        loss = total_mass_loss(SECONDS_PER_YEAR*1.0E8,m, k51b_orbital_dist)
        losses.append(loss/m)

    plt.plot(masses/M_Earth,losses)
    plt.show()


        
#plot_kepler51b()
#plot_planet_raius() #ORL use this one for paper



#plot_escape_parameter_space(DUR=1.0E8, T=1000.0)
#plot_rxuv_denominator() #ORL use this one for paper
#plot_escape_parameter_space_side_by_side() #ORL use this one for paper
animate_loss(SAVE_TO_FILE=True)

#plot_radius_mass_raltionship()
#plot_Rxuv_at_time()
#plot_total_loss_over_time()


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

    m_core = k51b_mass*0.90
    core_rho = 5510.0 #earth-like density
    T = 1500.0 #isothermal temp [K]

    r, r_s = calculate_rad(p_top, m_core, core_rho, k51b_mass, R_H2, T)

    print("found altitude = %0.1f km (%0.2f of observed)"%((r-r_s)/1000.0, r/k51b_rad))
    print("core is %0.2f of observed"%(r_s/k51b_rad))























