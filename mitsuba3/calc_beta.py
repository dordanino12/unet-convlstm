import numpy as np


def process_cloud_vars(QN, NC, T, p):
    Z, X, Y = QN.shape

    kB = 1.380649e-23
    m = 4.81e-26
    rho_water = 1000  # [kg/m^3]

    P = np.multiply(p*100, np.ones([X, Y, Z])).transpose((2, 0, 1))  # multiply by 100 to convert millibar to Pa
    rho_air = (P*m) / (kB*T)
    LWC = QN * rho_air
    Reff = 100 * ((3 * LWC) / (4 * np.pi * NC + 1e-6)) ** (1/3)
    beta_ext = 2 * LWC / (3 * 1e3*rho_water * 1e-6*Reff)

    return LWC, Reff, beta_ext