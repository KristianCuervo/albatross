# %%
import casadi as cas
import numpy as np
import matplotlib.pyplot as plt
from src import Albatross

def run(psi0: float):
    opti = cas.Opti()

    # Granularity
    N = 128

    # Spatial coordinate
    h = opti.variable(N)    # altitude above sea-level 

    # Inertial (ground-frame) velocities
    u = opti.variable(N)    # x-direction 
    v = opti.variable(N)    # y-direction
    w = opti.variable(N)    # z-direction

    # Control variables
    mu = opti.variable(N)   # Bank angle
    cl = opti.variable(N)   # Coefficient of lift

    # Minimisation variable
    V_ref = opti.variable(1)    # Reference wind speed - determines the wind shear strength in simulation.

    # Time step
    dt = opti.variable(1)

    # Initialise container class for bird attributes.
    bird = Albatross()

    h_ref = 10
    p = 0.143
    V_w = V_ref * (h/h_ref)**p


    # %%
    h_shear = np.linspace(0, 25, 100)
    V_w_shear = (h_shear / h_ref)**p

    fig, ax = plt.subplots(figsize=(4, 6))
    ax.plot(V_w_shear, h_shear, lw=2, color='red')
    n_arrows = 12
    h_arrows = np.linspace(1, 24, n_arrows)
    V_arrows = (h_arrows / h_ref)**p
    for ha, va in zip(h_arrows, V_arrows):
        ax.annotate('', xy=(va, ha), xytext=(0, ha),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    ax.set_xlabel(r'Wind speed $V_w / V_{ref}$')
    ax.set_ylabel('Altitude [m]')
    ax.set_title('Wind Shear Profile')
    ax.set_xlim(0, None)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    V_a = cas.sqrt((u + V_w)**2 + v**2 + w**2) 
    gamma = cas.arcsin(-w/V_a)      # flight path angle
    xi = cas.arctan2(v, u + V_w)    # heading angle

    cd = bird.cd_0 + bird.k * cl**2

    rho = 1.225 # Sea-level pressure

    L = lambda V_a : 0.5 * rho * V_a**2 * bird.S * cl
    D = lambda V_a : 0.5 * rho * V_a**2 * bird.S * cd

    a_u1 = cas.cos(gamma)*cas.cos(xi)
    a_u2 = cas.cos(mu)*cas.sin(gamma)*cas.cos(xi) + cas.sin(mu)*cas.sin(xi)
    a_v1 = cas.cos(gamma)*cas.sin(xi)
    a_v2 = cas.cos(mu)*cas.sin(gamma)*cas.sin(xi) - cas.sin(mu)*cas.cos(xi)
    a_w1 = -cas.sin(gamma)
    a_w2 = cas.cos(mu)*cas.cos(gamma)

    m = bird.m
    g = 9.80665

    dudt = -a_u1*(D(V_a)/m) - a_u2*(L(V_a)/m)
    dvdt = -a_v1*(D(V_a)/m) - a_v2*(L(V_a)/m)
    dwdt = -a_w1*(D(V_a)/m) - a_w2*(L(V_a)/m) + g

    dhdt = -w


    # Objective: minimise reference wind speed for sustained soaring
    opti.minimize(V_ref)

    # Periodic finite differences (wrap last → first)
    du = cas.diff(cas.vertcat(u[-1], u))
    dv = cas.diff(cas.vertcat(v[-1], v))
    dw = cas.diff(cas.vertcat(w[-1], w))
    dh = cas.diff(cas.vertcat(h[-1], h))

    # Shifted derivatives for trapezoidal collocation
    dudt_prev = cas.vertcat(dudt[-1], dudt[:-1])
    dvdt_prev = cas.vertcat(dvdt[-1], dvdt[:-1])
    dwdt_prev = cas.vertcat(dwdt[-1], dwdt[:-1])
    dhdt_prev = cas.vertcat(dhdt[-1], dhdt[:-1])

    # Trapezoidal dynamics constraints (2nd-order accurate)
    opti.subject_to(du == 0.5 * (dudt + dudt_prev) * dt)
    opti.subject_to(dv == 0.5 * (dvdt + dvdt_prev) * dt)
    opti.subject_to(dw == 0.5 * (dwdt + dwdt_prev) * dt)
    opti.subject_to(dh == 0.5 * (dhdt + dhdt_prev) * dt)


    # Bounds
    ## Altitude
    opti.subject_to(h >= 0.5)       # min altitude [m]
    opti.subject_to(h[0] == 0.5)

    # Positive flight
    u_avg = cas.sum1(u)
    v_avg = cas.sum1(v)
    psi = cas.arctan2(u_avg, v_avg)

    opti.subject_to(psi >= psi0)

    ## Control bounds
    ### Lift coefficient
    opti.subject_to(cl >= 0.1)
    opti.subject_to(cl <= 1.5)

    ### Bank angle
    opti.subject_to(-np.pi/2 < mu)
    opti.subject_to(mu < np.pi/2)

    ## Airspeed
    opti.subject_to(V_a >= 5.0)       # stall speed [m/s]

    ## Time step
    opti.subject_to(dt >= 0.02)
    opti.subject_to(dt <= 1.0)

    ## Total cycle period — prevent degenerate zero-period solution
    T_cycle = dt * N
    opti.subject_to(T_cycle >= 5.0)   # minimum 5 s cycle
    opti.subject_to(T_cycle <= 15.0)  # maximum 15 s cycle

    ## Reference shear wind speed
    opti.subject_to(V_ref >= 1.0)
    opti.subject_to(V_ref <= 30.0)



    T = 7.5
    tv = np.linspace(0, T, N)
    l = 2 * np.pi * tv / T          # phase angle [0, 2π]

    h0 = 1 + 9 * (1 - np.cos(l))   

    u0 = -1*(-5 - 5 * np.cos(l))        

    v0 = +(- 5 - 5 * np.sin(l)     )    

    w0 = -9 * np.sin(l) * (2 * np.pi / T)

    opti.set_initial(h, h0)
    opti.set_initial(u, u0)
    opti.set_initial(v, v0)
    opti.set_initial(w, w0)


    opti.set_initial(mu, +0.7 * np.sin(l))         
    opti.set_initial(cl, 0.8 + 0.3 * np.cos(l))      

    opti.set_initial(V_ref, 7.0)    
    opti.set_initial(dt, T / N)


    opts = {
        'ipopt.max_iter': 10000,
        'ipopt.mu_strategy': 'adaptive',
        'ipopt.tol': 1e-6,
        'ipopt.print_level': 5,
    }
    opti.solver('ipopt', opts)
    try:
        sol = opti.solve()
    except RuntimeError:
        sol = opti.debug
    print(sol.value(V_ref))
    return sol.value(V_ref)

ps = np.linspace(0, np.pi, 10)
lst = []
for p in ps:
    lst.append(run(p))

print(lst)