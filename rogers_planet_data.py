import numpy as np
import matplotlib.pyplot as plt
from math import pi

SOLAR_MASS = 1.989E30 #solar mass [kg]
SOLAR_RAD = 6.957E8 #radius of sun [m]
GG = 6.67E-11 #gravitational constant [m3 kg-1 s-2]
SIGMA = 5.67E-8
SECONDS_PER_DAY = 86400.0

#Planet data used by Rogers (2015), taken from marcy et al (2014)
class Star:
    def __init__(self, name, temp, radius, mass, planets):
        self.name = name 
        self.radius = radius #in solar radii
        self.mass = mass #in solar masses
        self.temp = temp #in Kelvin
        self.planets = planets #array or orbital periods in days

def init_stars():
    stars = [\
            Star("Kepler-100",5825.0,1.49,1.08, [12.8159,6.88705,35.3331]),\
            Star("Kepler-93",5669.0,0.92,0.91, [4.72674,1460.0]),\
            Star("Kepler-102",4903.0,0.74,0.8, [16.1457,10.3117,27.4536,7.07,5.287]),\
            Star("Kepler-94",4781.0,0.76,0.81,[2.508,820.3]),\
            Star("Kepler-103",5845.0,1.44,1.09,[15.97,179.61]),\
            Star("Kepler-106",5858.0,1.04,1.0, [13.57,43.84,6.16,23.98]),\
            Star("Kepler-95",5699.0,1.41,1.08, [11.52]),\
            Star("Kepler-109",5952.0,1.32,1.04, [6.48,21.22]),\
            Star("Kepler-48",5194.0,0.89,0.88, [4.78,9.67,42.90,982.0]),\
            Star("Kepler-113",4725.0,0.69,0.75,[8.92,4.75]),\
            Star("Kepler-25",6270.0,1.31,1.19, [12.72,6.24,123.0]),\
            Star("Kepler-37",5417.0,0.77,0.8, [39.79,21.30,13.37]),\
            Star("Kepler-68",5793.0,1.24,1.08, [5.40,9.61,625.0]),\
            Star("Kepler-96",5690.0,1.02,1.0, [16.24]),\
            Star("Kepler-131",5685.0,1.03,1.02, [16.09,25.52]),\
            Star("Kepler-97",5779.0,0.98,0.94, [2.59,789.0]),\
            Star("Kepler-98",5539.0,1.11,0.99, [1.54]),\
            Star("Kepler-99",4782.0,0.73,0.79, [4.60]),\
            Star("Kepler-406",5538.0,1.07,1.07, [2.43, 4.62]),\
            Star("Kepler-407",5476.0,1.01,1.0, [0.67, 3000.0]),\
            Star("NO_NAME",6104.0,1.23,1.08, [2.47]),\
            Star("Kepler-409",5460.0,0.89,0.92, [68.96]),\
            Star("Kepler-10",5708.0,1.065,0.91,[0.84]),\
            Star("Kepler-19",5541.0,0.85,0.936,[9.3]),\
            Star("Kepler-20",5466.0,0.944,0.912,[77.61]),\
            Star("Kepler-21",6131.0,1.86,1.34,[2.76]),\
            Star("Kepler-22",5518.0,0.979,0.97,[290.0])]
        

    return stars


def make_hist_arrays(data, min_val, max_val, num_bins):
    bins = np.linspace(min_val, max_val, num_bins)
    counts = np.zeros_like(bins)

    for d in data:
        for i in range(num_bins):
            if d <= bins[i]:
                counts[i] += 1
                break

    return (bins, counts)


def plot_orbital_dists_as_sunlike():
    orbits = []
    fluxes = []

    for star in init_stars():
        lum = SIGMA*star.temp**4.0*4.0/3.0*pi*(star.radius*SOLAR_RAD)**3.0


        for planet in star.planets:
            a = (GG*(star.mass*SOLAR_MASS)*(planet*SECONDS_PER_DAY)**2.0/(4.0*pi**2.0))**(1.0/3.0)
            flux = lum/(4.0/3.0*pi*a**3.0)
            solar_equiv = (1366.0/flux)**0.5
            orbits.append(solar_equiv)
            fluxes.append(flux/1366.0)

            if flux/1366.0 > 100.0:
                print("\n%s"%(star.name))
                #print("\t%0.2f AU"%(solar_equiv))
                print("\t%0.2f E Flux (period: %0.2f [days], equiv dist: %0.2f [AU])"%(flux/1366.0,planet, solar_equiv))

    #print("Orbital median: %0.2f"%(np.median(orbits)))

    #bins, counts = make_hist_arrays(orbits, np.min(orbits), 0.5,10)
    #plt.bar(bins,counts, width=0.05)
    #plt.ylim(0,np.max(counts)+2)
    #plt.show()

    bmin = 0.0
    bmax = 50.0 #np.max(fluxes)
    nbins = 30
    bins,counts = make_hist_arrays(fluxes,bmin,bmax,nbins)
    w = (bmax - bmin)/nbins
    plt.bar(bins,counts,width=w)
    plt.ylim(0,np.max(counts)+1)
    plt.show()

    return orbits


plot_orbital_dists_as_sunlike()
