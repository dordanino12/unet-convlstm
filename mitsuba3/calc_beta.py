import numpy as np


def process_cloud_vars(QN, NC, T, p):
    # QN is in [g/kg]
    # NC is in [cm^-3]
    # T is in [K]
    # p is in [millibar]

    Z, X, Y = QN.shape

    kB = 1.380649e-23 # [J/K]
    m = 4.81e-26 # [kg] (mass of air molecule)
    rho_water = 1000  # [kg/m^3]
    NC = NC * 1e6  # convert from [cm^-3] to [m^-3]

    P = np.multiply(p*100, np.ones([X, Y, Z])).transpose((2, 0, 1))  # multiply by 100 to convert millibar to Pa
    rho_air = (P*m) / (kB*T) # [kg/m^3]
    LWC = QN * rho_air # [g/m^3]

    rho_water_g_m3 = rho_water * 1000  # 1,000,000 [g/m^3]
    droplet_vol = (LWC / (NC + 1e-15)) / rho_water_g_m3  # [m^3]
    Reff_meters = ((3 * droplet_vol) / (4 * np.pi)) ** (1 / 3)  # [m]
    beta_ext = (3 * LWC) / (2 * rho_water_g_m3 * Reff_meters + 1e-15)  # [m^-1]

    Reff = Reff_meters * 1e6  # Convert to [microns]

    return LWC, Reff, beta_ext