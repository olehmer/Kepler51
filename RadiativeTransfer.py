#Owen Lehmer 11/1/16
#University of Washington, Seattle, WA
#Department of Earth and Space Sciences


import matplotlib.pyplot as plt
import numpy as np
import sys
from math import e, pi, exp, isnan, log
from scipy import optimize




################Constants########################
M_Earth = 5.972E24 #mass of Earth in [kg]
R_Earth = 6.371E6 #radius of Earth [m]
GG = 6.67408E-11 #gravitational constant [m3 kg-1 s-2]
R_H2 = 4124.0 #gas constant for H2 [J kg-1 K]
SIGMA = 5.67E-8 #Stefan Boltzman constant
#################End Constants###################


class Layer:
    """
    This class is the basic structure for our model. The atmosphere is 
    comprised of these Layer class objects. They record all relevant info
    for the given layer (i.e. flux up, flux down, pressure, density,
    T, etc.).

    IMPORTANT: things set to -1 by default may use that -1 as a check to see
               if the variable is set. 
    """
    def __init__(self, T=-1.0, p_top=-1.0, p_bot=-1.0, rho=-1.0, m=-1.0, \
            m_below=-1.0, F_uv=0, F_up=0, F_down=0,\
            F_sol=0, r=-1.0, abs_uv=0, abs_sol=0, abs_long=0):
        self.T = float(T)          #The temp for the layer (assumed isothermal) [K]
        self.p_top = float(p_top)  #The pressure at the top of the layer [Pa]
        self.p_bot = float(p_bot)  #The pressure at the bottom of the layer [Pa]
        self.rho = float(rho)      #The average density of the layer [kg m-3]
        self.m = float(m)          #The mass of the layer [kg]
        self.m_below = float(m_below) #The total mass below this layer [kg]
        self.F_uv = float(F_uv)    #The downward UV flux from the star [W m-2]
        self.F_sol = float(F_sol)  #The downward visible flux from the star [W m-2]
        self.F_up = float(F_up)    #The upward longwave flux at the layer [W m-2]
        self.F_down = float(F_down) #The downward longwave flux [W m-2]
        self.abs_uv = float(abs_uv) #The fraction of absorbed UV in the layer
        self.abs_sol = float(abs_sol) #The fraction of absorbed solar radiation in the layer
        self.abs_long = float(abs_long) #Fraction of absorbed longwave
        self.r = float(r)          #The radial radius of the layer from the surface [m]
                                   #Note about self.r - this is the radius at the
                                   #bottom of the layer




def UpdateLayerDistance(layers, ind, R_gas, core_rad):
    """
    This function will calculate the radial distance of the current layer. This
    is done by assuming the layer is approximately isothermal and gravity is
    constant throughout the layer. The first time this function is called g is
    assumed constant throughout the atmosphere. Once r has been calculated 
    gravity can be calculated for each layer (based on previous r).

    IMPORTANT: this function starts from the ground layer first, then works
    up!

    Input:
    layers - the array of layers in the atmosphere
    ind - the index of the current layer in the layers array
    R_gas - the gas constant for the atmosphere
    core_rad - the radius of the planetary core

    Updates layers directly
    """

    #make a temporary index (t_ind) to reverse the direction
    t_ind = len(layers)-ind-1 #reverse the direction, we want to start at the ground

    g = 0.0 #define gravity, g [m s-2]

    if layers[t_ind].r == -1: #not set yet
        g = GG*layers[len(layers)-1].m_below/core_rad**2.0 #gravity!
    else:
        g = GG*layers[t_ind].m_below/layers[t_ind].r**2.0
       
    #scale height is given by RT/g
    #scale height, the altitude for which pressure decreases by e (~1/3)
    H = R_gas*layers[t_ind].T/g

    delta_r = -H*log(layers[t_ind].p_top/layers[t_ind].p_bot) #this is the height of the layer
    print("%d: H=%0.2f km, T=%0.2f K, delta_r=%0.2f km, g=%0.2f m/s2, r=%0.2f km, p=%0.1f bar"%(t_ind,H/1000.0,layers[t_ind].T, delta_r/1000.0, g, (layers[t_ind].r-core_rad)/1000.0, layers[t_ind].p_bot/100000.0))

    if ind == 0:
        #this is the bottom layer
        layers[t_ind].r = core_rad
        
    if t_ind != 0:
        #this isn't the top layer, set the radius of the above layer
        layers[t_ind-1].r = layers[t_ind].r + delta_r



def LayersInit(N, total_flux, core_mass, core_rad, atmos_mass, R_gas,\
        p_toa=1.0E5):
    """
    This function will generate an array of N layers for the atmosphere assuming
    the initial atmosphere is in hydrostatic equilibrium and isothermal.

    Inputs:
    N - the number of atmospheric layers to use
    total_flux - the total incident flux [W m-2]. This will be used to set the isothermal
                 atmospheric temperature 
    core_mass - the mass of the planet core [kg]
    core_rad - the radius of the planet core [m]
    atmos_mass - the mass of the atmosphere [kg]
    R_gas - the specific gas constant for the atmosphere [J kg-1 K]
    p_toa - optional parameter to set the pressure at the top of the model. If
            not set the top of the model will be set to 1E5 [Pa] (1 bar)

    Returns:
    layers - the array of atmospheric layers
    """

    g = GG*core_mass/core_rad**2.0 #calculate gravity
    iso_T = (total_flux/SIGMA)**(0.25) #estimate the isothermal temperature

    #calculate the surface pressure
    p_s = atmos_mass*g/(4.0*pi*core_rad**2.0)

    #calculate the TOA pressure
    if p_toa > p_s: #don't reset p_toa if it's passed in and valid
        p_toa = p_s*1.0E-4

    #Create the pressure profile for the atmosphere, log scale and linear are below
    p_profile = np.logspace(np.log10(p_s),np.log10(p_toa), N+1) #log scale
    #p_profile = np.linspace(p_s, p_toa, N) #linear scale

    #create the layers array that we'll be working with
    layers = []

    #keep track of the total mass so far
    cur_mass = core_mass

    #planetary surface area, assumed ~constant in the hydrostatic atmosphere
    area = 4.0*pi*core_rad**2.0

    for i in range(0,N):
        p_bot = p_profile[i]
        p_top = p_profile[i+1]

        mass = (p_bot-p_top)*area/g #approximate mass of the layer
        
        T = iso_T

        rho = ((p_bot-p_top)/2.0)/(R_gas*T) #estimate the density with the average pressure

        layer = Layer(p_bot=p_bot, p_top=p_top, m=mass, m_below=cur_mass, T=T, rho=rho)
        layers.insert(0,layer) #prepend the layer in our array

        cur_mass = cur_mass + mass #increment the total mass

    for i in range(0,N):
        #Update the distance to each layer
        UpdateLayerDistance(layers, i, R_gas, core_rad)

    return layers



def BalanceRadiativeTransfer(layers, F_uv, F_sol, F_long, kappa_uv, \
        kappa_sol, kappa_long, uv_p_ref, sol_p_ref, long_p_ref, R_gas, iter_lim):
    """
    This is a wrapper function to call UpdateLayerRadTrans() and balance the 
    radiative transfer in the atmosphere. Each time the radial outflow is changed
    we need to calculate the full radiative balance to keep the model from 
    going nuts. This function will balance the radiation the layers.

    Inputs:
    layers - the atmospheric layers layer
    ground_ T- The temperature of the ground (assumed a blackbody absorber) [K]
    F_uv - the TOA flux [W m-2]
    F_sol - the TOA ~visible flux [W m-2]
    F_long - the TOA longwave flux, typically 0
    kappa_uv - the mass absorption coefficient for the UV
    kappa_sol - the mass absorption coefficient for the visible
    kappa_long - the mass absorption coefficient for the longwave radiation
    uv_p_ref - the reference pressure that kappa_uv was measured at
    sol_p_ref - the reference pressure that kappa_sol was measured at
    long_p_ref - the reference pressure that kappa_long was measured at
    R_gas - the specific gas constant for the atmosphere
    iter_lim - the limit on the number of iterations to perform

    Updates layers directly
    """

    count = 0

    while count < iter_lim:
        for i in range(0,len(layers)):
            UpdateLayerRadTrans(layers, i, F_uv, F_sol, F_long, kappa_uv, \
                    kappa_sol, kappa_long, uv_p_ref, sol_p_ref, long_p_ref, R_gas)

        count += 1



def UpdateLayerRadTrans(layers, ind, F_uv, F_sol, F_long, kappa_uv, \
        kappa_sol, kappa_long, uv_p_ref, sol_p_ref, long_p_ref, R_gas):
    """
    This function will calculate the radiative profile of the atmosphere. This 
    is a purely radiative model with a single, pressure scaled, optical depth
    for each flux channel.

    IMPORTANT: this model assumes that the atmosphere will only emit in the
               longwave. Any photons absorbed in the UV or sol band are assumed
               to go to kinetic energy and photons in the longwave.

    Inputs:
    layers - the atmospheric layers layer
    ind - the atmospheric layer to work with
    ground_ T- The temperature of the ground (assumed a blackbody absorber) [K]
    F_uv - the TOA flux [W m-2]
    F_sol - the TOA ~visible flux [W m-2]
    F_long - the TOA longwave flux, typically 0
    kappa_uv - the mass absorption coefficient for the UV
    kappa_sol - the mass absorption coefficient for the visible
    kappa_long - the mass absorption coefficient for the longwave radiation
    uv_p_ref - the reference pressure that kappa_uv was measured at
    sol_p_ref - the reference pressure that kappa_sol was measured at
    long_p_ref - the reference pressure that kappa_long was measured at
    R_gas - the specific gas constant for the atmosphere

    Updates layers directly
    """

    g = GG*layers[ind].m_below/layers[ind].r**2.0 #gravity at the layer

    #calculate the optical depth for each channel TODO need equation number here
    #optical depth is approximated by tau=kappa_ref*pressure^2/(2*g*pressure_ref)
    p_avg = (layers[ind].p_bot-layers[ind].p_top)/2.0
    tau_uv = kappa_uv*(p_avg)**2.0/(2.0*g*uv_p_ref)
    tau_sol = kappa_sol*p_avg**2.0/(2.0*g*sol_p_ref)
    tau_long = kappa_long*p_avg**2.0/(2.0*g*long_p_ref)

    #calculate the absorption in each band
    abs_uv = 1.0 - exp(-tau_uv)
    abs_sol = 1.0 - exp(-tau_sol)
    abs_long = 1.0 - exp(-tau_long)

    #update the absorption in each band for the layer
    layers[ind].abs_uv = abs_uv
    layers[ind].abs_sol = abs_sol
    layers[ind].abs_long = abs_long

    if ind == 0:
        #this is the top layer, set the fluxes from the TOA
        #This could be done outside this function, but was kept here for readability
        layers[ind].F_uv = F_uv
        layers[ind].F_sol = F_sol
        layers[ind].F_down = F_long

    #calculate the total flux absorbed in the layer
    F_in = abs_uv*layers[ind].F_uv + abs_sol*layers[ind].F_sol + \
            abs_long*layers[ind].F_up + abs_long*layers[ind].F_down

    #now calculate the new energy balance to get the temperature
    #TODO this needs an equation
    #F_in = GM/r*rho*u + simga*T^4
    M = layers[ind].m_below
    r = layers[ind].r
    rho = layers[ind].rho
    T = (F_in/(abs_long*SIGMA))**(0.25)

    #with the new temperature update the layer density and T
    if T > 0.0: 
        #on the first pass through the atmosphere (if no shortwave absorption)
        #the layers won't absorb until the flux reaches the surface
        new_rho = p_avg/(R_gas*T)
        layers[ind].T = T
        layers[ind].rho = new_rho

    else:
        #the calculated T isn't good yet
        T = layers[ind].T

    #calculate the longwave emission from the layer
    F_emit = abs_long*SIGMA*T**4.0

    #now calculate how much radiation is passed to the layer below
    if ind < len(layers)-1:
        #this isn't the bottom layer
        layers[ind+1].F_uv = layers[ind].F_uv*(1.0-abs_uv)
        layers[ind+1].F_sol = layers[ind].F_sol*(1.0-abs_sol)
        layers[ind+1].F_down = layers[ind].F_down*(1.0-abs_long) + 0.5*F_emit
    else:
        #this is the bottom layer, update the ground upward flux 
        F_total = layers[ind].F_uv*(1.0-abs_uv) + layers[ind].F_sol*(1.0-abs_sol) +\
                layers[ind].F_down*(1.0-abs_long) + 0.5*F_emit

        #ground_T = (F_total/SIGMA)**(0.25)
        #from the ground temp the upward flux is thus F_total
        #update the upward flux of the bottom layer
        layers[ind].F_up = F_total

    #now calculate the flux passed in the up direction
    if ind > 0:
        #not the top layer
        #note, only longwave flux can be passed upward in this model
        layers[ind-1].F_up = layers[ind].F_up*(1.0-abs_long) + 0.5*F_emit




        
def BalanceAtmosphere(core_mass, core_rad, atmos_mass, R_gas, F_uv, F_sol, F_long,\
        kappa_uv, kappa_sol, kappa_long, uv_p_ref, sol_p_ref, long_p_ref,\
        N=300, iter_lim=600, p_toa_in=1.0E5):
    """
    This is the top level function to balance the atmospheric model. Given the 
    above parameters this will compute the temperature profile, pressure 
    profile, height profile, and the loss rate from the atmosphere. This model uses a 3-stream
    approach to handle UV, visible, and IR flux.

    IMPORTANT: all the passed in flux values (F_uv, F_sol, F_long) should have
               the albedo already taken into account. This model will not 
               calculate scattering or any other form of reflection.

    Inputs:
    core_mass - the mass of the planetary core [kg]
    core_rad - the radius of the planetary core [m]
    atmos_mass - the mass of the atmosphere [kg]
    R_gas - the specific gas constant for the atmosphere [J kg-1 K]
    F_uv - the downward UV flux at the top of the atmosphere [W m-2]
    F_sol - the non-UV non-longwave downward flux at the top of atmos [W m-2]
    F_long - the longwave downward flux at TOA, typically 0 [w m-2]. 
    kappa_uv - the mass absorption coefficient for the UV band [m2 kg-1]
    kappa_sol - the mass absorption coefficient to use in the solar band [m2 kg-1]
    kappa_long - the mass absorption coefficient to use in the longwave band [m2 kg-1]
    uv_p_ref - the reference pressure for kappa_uv
    sol_p_ref - the reference pressure for kappa_sol
    long_p_ref - the reference pressure for kappa_long
    N - the number of atmospheric layers to use in the simulation, defaults to 100
    iter_lim - the number of iterations after which the model will stop
    p_toa_in - the pressure at the top of the atmosphere to be used in the model [Pa].
                defaults to one bar

    NOTE: TOA = top of atmosphere, and TOA is at and index of 0 in our layers array

    Output:
    p_profile - the pressure profile of the atmosphere
    r_profile - the radial distance of each layer
    T_profile - the temperature profile of the atmosphere
    mass_flux - the mass flux loss rate from the atmosphere [kg s-1]
    """

    #initialize our array to hold the layers
    total_flux = F_uv + F_sol + F_long
    layers = LayersInit(N, total_flux, core_mass, core_rad, atmos_mass, R_gas,\
            p_toa=p_toa_in)

    #calculate the radiative transfer in the layers
    BalanceRadiativeTransfer(layers,F_uv,F_sol,F_long,kappa_uv,kappa_sol,\
            kappa_long,uv_p_ref,sol_p_ref,long_p_ref,R_gas,iter_lim)



    #for i in range(0,N):
    #    UpdateLayerDistance(layers, i, R_gas, core_rad)

    #get the pressure and distance profiles
    p_profile = np.zeros(N)
    r_profile = np.zeros(N)
    T_profile = np.zeros(N) 
    for i in range(0,N):
        p_profile[i] = layers[i].p_bot
        r_profile[i] = layers[i].r
        T_profile[i] = layers[i].T


    return (p_profile, r_profile, T_profile)




    



def kinda_earth_test():
    results = BalanceAtmosphere(M_Earth, R_Earth, 5.148E18, 285.0, 10.0, 240.0, 0.0,\
        0.10, 0.0, 0.01, 1.0E4, 1.0, 1.0E4,\
        N=400, iter_lim=800, p_toa_in=100.0)

    p_profile, r_profile, T_profile = results

    fig, ax1 = plt.subplots()
    ax1.plot(T_profile, (r_profile-R_Earth)/1000.0, color='white')
    ax1.set_ylabel("Altitude [km]")
    ax1.set_xlabel("Temperature [K]")

    ax2 = ax1.twinx()
    ax2.plot(T_profile, p_profile)
    ax2.invert_yaxis()
    ax2.set_ylabel("Pressure [Pa]")
    ax2.set_yscale('log')

    plt.title("Temperature Profile")
    plt.show()















    
