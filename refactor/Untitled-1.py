# %%
import casadi as cas
import numpy as np
import matplotlib.pyplot as plt
from albatross import Albatross

# %% [markdown]
# This block initialises the optimisation problem using the CasADI python API. 
# 
# N represents the granularity of the simulation. This is used then to initialies a 4-dimensional state space including the altitude of the bird trajectory (h) and the 3-dimensional velocities (u, v, w). Furthermore, the control variables are created which include the bank angle (mu) and the coefficient of lift (cl). 
# 
# The time step must also be included as an optimisation parameter as the optimisation software will adapt a suitable dt for the simulation.
# 
# The optimisation objective to be minimised is chosen to the be the wind shear reference strength (V_ref). The efficiency of a gliding flight strategy can be seen as the minimum amount of wind strength required to sustain the same altitude; thus a lower V_ref would mean a more efficient flight path.
# 
# Finally the Albatross class is initialised, which is a container for the parameters of the albatross model used in this simulation. This data, which includes e.g mass, wing span, drag coefficients, is taken from Sachs (2005).

# %%
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


# %% [markdown]
# The wind-speed is simulated using an exponential model, wherein the values for the reference altitude (h_ref) and exponential parameter (p) are taken from Sachs (2005). The wind-speed is then simulated in the negative x-direction (hence the negative sign).

# %%
# Windspeed (wind blows in -x direction)
h_ref = 10
p = 0.143
V_w = V_ref * (h/h_ref)**p

# %% [markdown]
# The wind-shear gradient is graphed here:

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


# %% [markdown]
# The aerodynamic forces are not based on the inertial speeds, which are relative to the ground, but rather the airspeed. The airspeed then includes the windspeed as an extra drag force. 
# 
# The airspeed is also used to calculate the final two flight angles; the flight path angle(gamma) and the heading angle (xi).
# 
# The flight path angle is the angle of the airpseed in reference to the horizon and determines if you are climbing (+) or descending (-).
# 
# The heading angle describe the horizontal movement in reference to the coordinate system; if you are yawing right (+) or left(-). 

# %%
# Airspeed
V_a = cas.sqrt((u + V_w)**2 + v**2 + w**2) 

# Resulting flight angles
gamma = cas.arcsin(-w/V_a)      # flight path angle
xi = cas.arctan2(v, u + V_w)    # heading angle

# %% [markdown]
# The drag coefficient is calculated using the drag-polar; 
# 
# $$
# C_d = C_{d0} + \kappa C_L^2
# $$
# 
# From there, the ambient sea-level pressure of $\rho=1.225$ can be used to calculate the lift and drag forces:
# $$
# L = \frac{1}{2} C_L \rho V^2 S 
# $$
# $$
# D = \frac{1}{2} C_D \rho V^2 S
# $$

# %%
# Drag characteristics
cd = bird.cd_0 + bird.k * cl**2


# Aerodynamic forces
rho = 1.225 # Sea-level pressure

L = lambda V_a : 0.5 * rho * V_a**2 * bird.S * cl
D = lambda V_a : 0.5 * rho * V_a**2 * bird.S * cd

# %% [markdown]
# Flight angle coefficients are used such as to simplify the equations of motions

# %%
# Flight angle coefficients (Sachs 2005 convention: positive mu = bank right)
a_u1 = cas.cos(gamma)*cas.cos(xi)
a_u2 = cas.cos(mu)*cas.sin(gamma)*cas.cos(xi) + cas.sin(mu)*cas.sin(xi)
a_v1 = cas.cos(gamma)*cas.sin(xi)
a_v2 = cas.cos(mu)*cas.sin(gamma)*cas.sin(xi) - cas.sin(mu)*cas.cos(xi)
a_w1 = -cas.sin(gamma)
a_w2 = cas.cos(mu)*cas.cos(gamma)

# %% [markdown]
# Here the equations of motions are defined.
# 
# Furthermore, as the reference frame has the z-coordinate aligned with gravity, then the altitude is oriented in the opposite direction to the z-direction.
# 

# %%
# Equations of motion (inertial frame — wind enters via V_a, gamma, xi)
m = bird.m
g = 9.80665

dudt = -a_u1*(D(V_a)/m) - a_u2*(L(V_a)/m)
dvdt = -a_v1*(D(V_a)/m) - a_v2*(L(V_a)/m)
dwdt = -a_w1*(D(V_a)/m) - a_w2*(L(V_a)/m) + g

dhdt = -w

# %% [markdown]
# Here the periodic boundary conditions are defined. It is required for the simulation over one period length to then return to the same initial velocities and altitude. Here it should be noted, that the x-y coordinates are free variables, and it is expected for the flight path to traverse in a non-zero direction.

# %%
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

# %% [markdown]
# Here, boundary conditions are imposed on the different variables. 
# 
# On ther altitude, it is required that the albatross is always above 0.5m, such that it is not crashing into the sea. It is also subjected to start at 0.5m.
# 
# The maximum lift coefficient is given to be 1.5, which can be calculated from the data given by Sachs (2005) on the characteristics of the albatross. Furthermore, it is expected that the albatross can significantly reduce its lift by for example retracting its wings, and thus a lower bound is given. This lower bound is kept above 0.0 to prevent negative lift.
# 
# The bank angle is imposed such that the albatross does not bank more than -90 to +90 degrees. If it were to exceed this bank angle, then it would be oriented with its head downwards.
# 
# A minimum airspeed is given which is to represent a stall speed.
# 
# A minimum cycle period is used to prevent trivial zero-period solutions. 
# 
# Finally, bounds on the wind reference shear strength is used to prevent inconsistent solutions.

# %%
# Bounds
## Altitude
opti.subject_to(h >= 0.5)       # min altitude [m]
opti.subject_to(h[0] == 0.5)

# Positive flight
u_avg = cas.sum1(u)
v_avg = cas.sum1(v)
psi = cas.arctan2(u_avg, v_avg)

opti.subject_to(psi >= 0.0)

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


# %% [markdown]
# An initial trajectory needs to be used such that the optimisation model has something to start iterating from. Apriori information is then given to resemble the shape of the expected graph seen from Sachs (2005); although I found that with many types of initial conditions it still converges to a very similar result.
# 

# %%
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

# %% [markdown]
# Now the actual dynamic optimisation is taken care of by CasADI.

# %%
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

# %%
## Results
print(f"\n=== Solution ===")
print(f"Minimum wind V_ref = {sol.value(V_ref):.2f} m/s")
print(f"Cycle period       = {sol.value(dt) * N:.2f} s")
print(f"Altitude range     = {sol.value(h).min():.1f} - {sol.value(h).max():.1f} m")
print(f"Airspeed range     = {sol.value(V_a).min():.1f} - {sol.value(V_a).max():.1f} m/s")

h_sol_res = sol.value(h)
V_w_sol = sol.value(V_w)
i_low, i_high = h_sol_res.argmin(), h_sol_res.argmax()
print(f"Wind speed at lowest  point (h={h_sol_res[i_low]:.1f} m) = {V_w_sol[i_low]:.2f} m/s")
print(f"Wind speed at highest point (h={h_sol_res[i_high]:.1f} m) = {V_w_sol[i_high]:.2f} m/s")

# %% [markdown]
# The following section graphs both the initial guessed trajectory, and the final optimised trajectory. I've adjusted the initial trajectory to be similar to the final one, but it was quite robust to converge to this same final optimised path regardless of initial conditions. Although; there is an arbitrary preference in the y-direction in which the bird can sway towards, thus there are two mirror orbits which represent the same optimality.

# %%
## Recover x, y by trapezoidal integration of ground velocities
dt_val = sol.value(dt)
u_sol = sol.value(u)
v_sol = sol.value(v)
h_sol = sol.value(h)
speed = sol.value(V_a)

x_sol = np.zeros(N)
y_sol = np.zeros(N)
for i in range(1, N):
    x_sol[i] = x_sol[i-1] + 0.5 * (u_sol[i] + u_sol[i-1]) * dt_val
    y_sol[i] = y_sol[i-1] + 0.5 * (v_sol[i] + v_sol[i-1]) * dt_val

## Interactive 3D flight path plot with Plotly
import plotly.graph_objects as go

fig = go.Figure()
dt_guess = T / N
x0 = np.zeros(N)
y0 = np.zeros(N)
for i in range(1, N):
    x0[i] = x0[i-1] + 0.5 * (u0[i] + u0[i-1]) * dt_guess
    y0[i] = y0[i-1] + 0.5 * (v0[i] + v0[i-1]) * dt_guess
fig.add_trace(go.Scatter3d(
    x=x0, y=y0, z=h0,
    mode='lines',
    line=dict(color='red', width=3, dash='dot'),
    name='Initial guess',
))

# Main trajectory coloured by airspeed
fig.add_trace(go.Scatter3d(
    x=x_sol, y=y_sol, z=h_sol,
    mode='lines',
    line=dict(color=speed, colorscale='RdBu_r', width=5,
              colorbar=dict(title='Airspeed [m/s]', len=0.5, thickness=15,
                            x=1.0, xpad=5, y=0.5)),
    name='Trajectory',
    hovertemplate=(
        'x: %{x:.1f} m<br>y: %{y:.1f} m<br>'
        'h: %{z:.1f} m<br>Airspeed: %{marker.color:.1f} m/s'
    ),
))

# Start marker
fig.add_trace(go.Scatter3d(
    x=[x_sol[0]], y=[y_sol[0]], z=[h_sol[0]],
    mode='markers',
    marker=dict(size=6, color='green'),
    name='Start',
))

# Compute equal-scale axis ranges centred on the data (needed for arrow sizing)
_all_x = np.concatenate([x_sol, x0])
_all_y = np.concatenate([y_sol, y0])
_all_z = np.concatenate([h_sol, h0])
_mid = np.array([(_all_x.max()+_all_x.min())/2,
                 (_all_y.max()+_all_y.min())/2,
                 (_all_z.max()+_all_z.min())/2])
_half = max(_all_x.max()-_all_x.min(), _all_y.max()-_all_y.min(), _all_z.max()-_all_z.min()) / 2 * 1.15  # 15 % padding

arrow_z = h_sol.max() * 0.8
V_w_mean = sol.value(V_w).mean()
wind_vec = np.array([-V_w_mean, 0.0, 0.0])            
wind_dir = wind_vec / np.linalg.norm(wind_vec)         
arrow_len = 0.35 * _half
arrow_base = np.array([x_sol.mean() + 0.3 * _half, y_sol.mean(), arrow_z])
arrow_tip = arrow_base + arrow_len * wind_dir

fig.add_trace(go.Scatter3d(
    x=[arrow_base[0], arrow_tip[0]],
    y=[arrow_base[1], arrow_tip[1]],
    z=[arrow_base[2], arrow_tip[2]],
    mode='lines', line=dict(color='dodgerblue', width=8),
    name='Wind dir', showlegend=True,
))
fig.add_trace(go.Cone(
    x=[arrow_tip[0]], y=[arrow_tip[1]], z=[arrow_tip[2]],
    u=[wind_dir[0]], v=[wind_dir[1]], w=[wind_dir[2]],
    sizemode='absolute', sizeref=arrow_len * 0.4,
    colorscale=[[0, 'dodgerblue'], [1, 'dodgerblue']],
    showscale=False, name='Wind dir',
))

fig.update_layout(
    title=f'Dynamic Soaring Orbit  (V_ref = {sol.value(V_ref):.2f} m/s)',
    scene=dict(
        xaxis_title='x [m]',
        yaxis_title='y [m]',
        zaxis_title='h [m]',
        xaxis=dict(range=[_mid[0]-_half, _mid[0]+_half]),
        yaxis=dict(range=[_mid[1]-_half, _mid[1]+_half]),
        zaxis=dict(range=[max(0, _mid[2]-_half), _mid[2]+_half]),
        aspectmode='cube',
    ),
    width=800, height=700,
    margin=dict(l=0, r=80, t=40, b=0),
    legend=dict(x=0.0, y=1.0, bgcolor='rgba(255,255,255,0.6)'),
)

fig.show()

# %%
dx_total = x_sol.max() - x_sol.min()
dy_total = y_sol.max() - y_sol.min()
dh_total = h_sol.max() - h_sol.min()
print(f"Total dx = {dx_total}") 
print(f"Total dy = {dy_total}") 
print(f"Total dh = {dh_total}") 

# %%
# Individual summary plots
t_cycle = sol.value(dt) * N
t_norm = np.linspace(0, 1, N)  # t / t_cycle

h_sol = sol.value(h)
cl_sol = sol.value(cl)
mu_sol = sol.value(mu)
speed_air = sol.value(V_a)
wind_speed = sol.value(V_w)
speed_inertial = np.sqrt(sol.value(u)**2 + sol.value(v)**2 + sol.value(w)**2)

# 1) Altitude
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(t_norm, h_sol)
ax.set_ylabel('Altitude [m]')
ax.set_title('Altitude')
ax.set_xlabel(r'$t\,/\,t_{\mathrm{cycle}}$')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('altitude.jpg', dpi=150, bbox_inches='tight')
plt.show()

# 2) Speeds
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(t_norm, speed_inertial, label='Inertial speed')
ax.plot(t_norm, speed_air, '--', label='Airspeed')
ax.set_ylabel('Speed [m/s]')
ax.set_title('Speeds')
ax.set_xlabel(r'$t\,/\,t_{\mathrm{cycle}}$')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('speeds.jpg', dpi=150, bbox_inches='tight')
plt.show()

# 3) Lift coefficient
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(t_norm, cl_sol)
ax.set_ylabel(r'$C_L$')
ax.set_title('Lift Coefficient')
ax.set_xlabel(r'$t\,/\,t_{\mathrm{cycle}}$')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lift_coefficient.jpg', dpi=150, bbox_inches='tight')
plt.show()

# 4) Bank angle
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(t_norm, np.degrees(mu_sol))
ax.set_ylabel('Bank angle [deg]')
ax.set_title('Bank Angle')
ax.set_xlabel(r'$t\,/\,t_{\mathrm{cycle}}$')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('bank_angle.jpg', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# All in all the results are very close to those found in Sachs (2005), except for that the bank angle is shown in the opposite direction. However, there are two 

# %%
import json

# Extract convergence history from the solver

# Get the stats from the optimizer
stats = opti.stats()

# CasAdi stores iterations as a dict of lists, not a list of dicts
iterations = stats['iterations']
objective_values = np.array(iterations['obj'])
constraint_violations = np.array(iterations['inf_pr'])

# Feasibility threshold (same as solver tolerance)
tol = opts.get('ipopt.tol', 1e-6)
feasible = constraint_violations <= tol
iters = np.arange(len(objective_values))

# Colour by log constraint violation (continuous gradient)
log_cv = np.log10(np.clip(constraint_violations, 1e-12, None))

# Plot convergence history
fig, ax = plt.subplots(figsize=(12, 5))

# Objective function over iterations — colour by constraint violation level
ax.plot(iters, objective_values, '-', color='grey', lw=1, alpha=0.4)
sc = ax.scatter(iters, objective_values, c=log_cv, cmap='RdYlGn_r',
                edgecolors='k', linewidths=0.3, s=35, zorder=3)

# Highlight the final (optimal) solution with a large star
final_idx = len(objective_values) - 1
ax.scatter(final_idx, objective_values[final_idx], marker='*', s=400,
           c='gold', edgecolors='black', linewidths=1.2, zorder=5,
           label=f'Optimal: $V_{{ref}}$ = {objective_values[final_idx]:.2f} m/s (iter {final_idx})')
ax.axhline(objective_values[final_idx], ls='--', color='gold', alpha=0.5, lw=1)

cbar = plt.colorbar(sc, ax=ax, pad=0.02)
cbar.set_label(r'$\log_{10}$(constraint violation)')
ax.set_ylabel('Objective Function: $V_{ref}$ [m/s]')
ax.set_xlabel('Iteration')
ax.set_title('Optimization Convergence: Objective (coloured by feasibility)')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print summary
n_iters = len(objective_values)
print(f"Total iterations: {n_iters}")
print(f"Final objective: {objective_values[-1]:.6f}")
print(f"Final constraint violation: {constraint_violations[-1]:.2e}")


