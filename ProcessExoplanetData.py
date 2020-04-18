import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from math import pi


##########CONSTANTS#################################
G = 6.67E-11 #m3/kg/s2
M_jup = 1.898E27 #mass of Jupiter in kg
R_jup = 69911000.0 #radius of Jupiter in m
rho_jup = 1.33 #density of Jupiter [g/cm3]
AU = 149597870700.0 #AU in m
M_earth = 5.972E24      #Mass of the Earth [kg]
R_earth = 6371393.0 #earth radius [m]
M_venus = 4.867E24 #mass of Venus [kg]
M_mars = 6.39E23 #mass of mars [kg]
M_saturn = 5.683E26 #mass of saturn [kg]
M_neptune = 1.024E26 #mass of neptune [kg]
M_uranus = 8.681E25 #mass of Uranus [kg]
################################################################

#increase pyplot font size
matplotlib.rcParams.update({'font.size':17})

class Planet:
    def __init__(self):
        self.name = ""
        self.mass = 0.0
        self.gravity = 0.0
        self.period = 0.0
        self.eccentricity = 0.0
        self.distance = 0.0
        self.radius = 0.0
        self.density = 0.0
        self.density_e_m = 0.0
        self.density_e_p = 0.0
        self.temp = 0.0
        self.stellar_temp = 0.0
        self.stellar_mass = 0.0
        self.stellar_rad = 0.0
        self.flux = 0.0



def read_exoplanet_data():
    data = np.genfromtxt("allplanets-ascii_Jan_2017.txt", skip_header=1)
    names = np.genfromtxt("allplanets-ascii_Jan_2017.txt",skip_header=1, usecols=0, dtype=str)
    #data = np.genfromtxt("exoplanet_data.txt", skip_header=1)
    #names = np.genfromtxt("exoplanet_data.txt",skip_header=1, usecols=0, dtype=str) #data[:,0] #array of system names

    stellar_temp = data[:,1:4] #[stellar temperature [K], plus error, minus error]
    stellar_mass = data[:,7:10] #mass of the star in M_sun units
    stellar_radius = data[:,10:13] #radius of the star in R_sun units
    orbital_period = data[:,19] #orbital period, [days]
    eccentricity = data[:,20:23] #[eccentricity, plus error, minus error]
    orbital_dist = data[:,23:26] #[orbit distance [AU], plus error, minus error]
    planet_masses = data[:,26:29] #[planetary mass [MJup], plus error, minus error]
    planet_radii = data[:,29:32] #[planetary radius [RJup], plus error, minus error]
    planet_gravity = data[:,32:35] #planetary gravity [m/s2]
    planet_density = data[:,35:38] #[planetary density [RhoJup], plus error, minus error]
    planetary_temp = data[:,38:41] #[planetary temp [K], plus error, minus error]
    
    return (names, stellar_temp, stellar_mass, stellar_radius, orbital_period,\
            eccentricity, orbital_dist, planet_masses, planet_radii, \
            planet_density, planetary_temp, planet_gravity)



def create_planet_array():
    names, stellar_temp, stellar_mass, stellar_radius, orbital_period,\
           eccentricity, orbital_dist, planet_masses, planet_radii, \
           planet_density, planetary_temp, planet_gravity = read_exoplanet_data()

    planets = []
    for i in range(0,len(names)):
        p = Planet()
        p.name = names[i]
        p.stellar_temp = stellar_temp[i,0]
        p.stellar_mass = stellar_mass[i,0]
        p.stellar_rad = stellar_radius[i,0]
        p.gravity = planet_gravity[i,0]
        p.period = orbital_period[i]
        p.eccentricity = eccentricity[i]
        if planet_masses[i,0] == 0:
            p.mass = planet_masses[i,1]
        else:
            p.mass = planet_masses[i,0]

        p.distance = orbital_dist[i,0]
        if planet_radii[i,0] == 0:
            p.radius = planet_radii[i,1]
        else:
            p.radius = planet_radii[i,0]

        p.density = planet_density[i,0]
        p.density_e_p = planet_density[i,1] #positive density error
        p.density_e_m = planet_density[i,2] #negative density error
        p.temp = planetary_temp[i,0]
        if p.stellar_rad > 0 and p.stellar_temp > 0 and p.distance > 0:
            p.flux = (p.stellar_rad*695700099)**2.0*(5.67E-8)*(p.stellar_temp)**4.0/(p.distance*AU)**2.0

        planets.append(p)
    return planets

def plot_sunlike_planets():
    """
    Plot the orbital distance vs mass for planets orbiting Sun-like stars 
    i.e. stars between 0.8 and 1.2 times the mass of the Sun
    """

    planets = create_planet_array()
    min_mass = 0.5*M_earth
    max_mass = 10.0*M_earth

    mass = []
    orb_dist = []
    rho = []
    rad = []
    for i in range(len(planets)):
        p = planets[i]
        m = p.mass*M_jup
        r = p.radius*R_jup
        if p.stellar_mass > 0.9 and p.stellar_mass < 1.1 and p.distance>0 and\
                m < max_mass and m > min_mass:
            mass.append(m/M_earth)
            orb_dist.append(p.distance)
            rho.append(p.density)
            rad.append(r/R_earth)

            print("%s has mass: %0.2f [Earth Masses], radius: %0.2f [Earth Radii]\n\t\tdensity: %0.2f [g/cc]"%(p.name,m/M_earth,r/R_earth,p.density))

    """
    #create the Earth density curve and water density curve
    line_masses = np.linspace(min_mass,max_mass,200)
    line_radii_earth = np.zeros(len(line_masses))
    rho_earth = 5510.0 #earth density
    for i in range(0,200):
        r_earth = (line_masses[i]/(4.0/3.0*pi*rho_earth))**(1.0/3.0)
        line_radii_earth[i] = r_earth

    #plot the line of Earth density
    plt.plot(line_masses/M_earth, line_radii_earth/R_earth, "k--", zorder=1)
    """

    #cm = plt.cm.get_cmap("bwr")
    #sc = plt.scatter(orb_dist, mass, c=rad, s=80, alpha=0.5, zorder=2,\
    #        cmap=cm)#, norm=matplotlib.colors.LogNorm())
    #plt.colorbar(sc).ax.set_title("Radius [Earth Radii]")

    #make bins for bar plot
    num_bins = 10
    bins = np.logspace(-2,0,num_bins)
    #bins = np.linspace(0.01,1,num_bins)
    counts = np.zeros(num_bins)
    for d in orb_dist:
        for i in range(num_bins):
            if d <= bins[i]:
                counts[i] += 1
                break

    #for 10 bins nothing in the first one, throw it out
    counts = counts[1:]
    bins = bins[1:]
    x = range(len(bins))
    labels = np.round(bins*100.0)/100.0
    plt.bar(x,counts,0.9, align='center')
    #plt.gca().tick_params(axis=u'x', which=u'both',length=0)
    plt.xticks(x,labels)#, rotation='vertical')
    plt.ylim(0,11)
    plt.xlabel("Orbital Distance [AU]")
    plt.ylabel("Planet Count")
    plt.show()

    """
    sc = plt.scatter(orb_dist, mass, s=80, alpha=0.5)
    
    plt.xlabel("Orbital Distance [AU]")
    plt.ylabel("Mass [Earth Masses]")
    #plt.xlim(min_mass/M_earth, max_mass/M_earth)
    plt.xscale("log")
    plt.xlim(0.01,1)
    plt.ylim(0,10)
    plt.grid()
    plt.show()
    """


def plot_exoplanet_mass_radius():
    """
    Plot the mass radius relationship of all known exoplanets. Color the planets
    according to the recieved stellar flux.
    """
    planets = create_planet_array()

    mass = []
    radius = []
    flux = []

    prob_count = 0
    good_count = 0
    for i in range(0,len(planets)):
        if planets[i].mass > 0 and planets[i].radius > 0 and \
                planets[i].flux > 0 and planets[i].mass*M_jup/M_earth < 20\
                and planets[i].radius*R_jup/R_earth < 10\
                and planets[i].flux/1366.0 < 500:
            mass.append(planets[i].mass*M_jup/M_earth)
            radius.append(planets[i].radius*R_jup/R_earth)
            flux.append(planets[i].flux/1366.0)

    m_gcc_1 = []
    m_gcc_e = []
    rads = np.linspace(0,20,100)
    for i in range(0,len(rads)):
        #plot the density contours
        v = 4.0/3.0*pi*rads[i]**3.0
        m_gcc_e.append(v/5.51)
        m_gcc_1.append(v*1000.0/5510.0)


    #plt.plot(m_gcc_1, rads, label="Water Density (1.0 g/cc)")
    plt.plot(m_gcc_e, rads, label="Earth Density (5.5 g/cc)")
    cm = plt.cm.get_cmap("coolwarm")
    sc = plt.scatter(mass,radius, c=flux, cmap=cm)
    plt.colorbar(sc).ax.set_title("Incident Flux [Earth Fluxes]")
    plt.grid()
    plt.ylabel("Radius [Earth Radii]")
    plt.xlabel("Mass [Earth Masses]")
    plt.yscale("log")
    plt.xscale("log")
    plt.ylim(0,10)
    plt.xlim(0.5,20)
    plt.legend()
    plt.show()









def list_low_density_planets():
    """
    Plot the mass radius relationship of all known exoplanets. Color the planets
    according to the recieved stellar flux.
    """
    planets = create_planet_array()

    count = 1 
    for i in range(0,len(planets)):
        if planets[i].mass > 0 and planets[i].radius > 0 \
                and planets[i].mass*M_jup/M_earth < 20 \
                and planets[i].radius*R_jup/R_earth < 10:
            mass = (planets[i].mass*M_jup/M_earth)
            radius = (planets[i].radius*R_jup/R_earth)
            name = planets[i].name
            rho = (mass*M_earth*1000.0)/(4.0/3.0*pi*(radius*R_earth*100.0)**3.0)
            
            if rho < 0.5 and mass < 10.0:
                #print("%3d (%3d) - Planet: %12s, density: %2.3f g/cc, Mass: %2.2f Earth Masses, Radius: %2.2f Earth Radii"%(count,i,name,rho,mass,radius))
                count += 1

    




#list_low_density_planets()
plot_exoplanet_mass_radius()
#plot_sunlike_planets()


