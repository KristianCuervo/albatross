"""
Coordinate Frame Diagrams

Interactive 3-D diagrams illustrating the coordinate conventions used in the
dynamic soaring model (after Sachs 2005). Run this script to open the figures
in a browser via Plotly.

Diagram 1 — Velocity vectors: VK = V + Vw in the ground frame (xg, yg, zg).
Diagram 2 — Flight angles: heading xi, flight path gamma, bank angle mu.
"""

import subprocess
import tempfile
import numpy as np
import plotly.graph_objects as go


def _show(fig):
    """Write figure to a temp HTML file and open in the Windows browser (WSL2)."""
    with tempfile.NamedTemporaryFile(suffix='.html', delete=False, mode='w') as f:
        fig.write_html(f.name)
        path = f.name
    win_path = subprocess.check_output(['wslpath', '-w', path]).decode().strip()
    subprocess.Popen(['explorer.exe', win_path])


# ── Shared helpers ────────────────────────────────────────────────────────────

def _vec(fig, start, end, color='black', dash='solid', width=3, sizeref=0.09):
    s, e = np.asarray(start, float), np.asarray(end, float)
    d    = (e - s) / np.linalg.norm(e - s)
    fig.add_trace(go.Scatter3d(
        x=[s[0], e[0]], y=[s[1], e[1]], z=[s[2], e[2]],
        mode='lines', line=dict(color=color, width=width, dash=dash),
        showlegend=False, hoverinfo='skip',
    ))
    fig.add_trace(go.Cone(
        x=[e[0]], y=[e[1]], z=[e[2]],
        u=[d[0]], v=[d[1]], w=[d[2]],
        sizemode='absolute', sizeref=sizeref,
        colorscale=[[0, color], [1, color]],
        showscale=False, hoverinfo='skip', anchor='tip',
    ))


def _lbl(fig, pos, text, color='#222', size=15):
    fig.add_trace(go.Scatter3d(
        x=[pos[0]], y=[pos[1]], z=[pos[2]], mode='text',
        text=[text], textfont=dict(size=size, color=color),
        showlegend=False, hoverinfo='skip',
    ))


def _box(fig, start, end):
    s, e = np.asarray(start, float), np.asarray(end, float)
    fig.add_trace(go.Scatter3d(
        x=[s[0], e[0]], y=[s[1], e[1]], z=[s[2], e[2]],
        mode='lines', line=dict(color='#888', width=2),
        showlegend=False, hoverinfo='skip',
    ))


def _arc_cone(fig, pts, color):
    tang = pts[-1] - pts[-3]
    fig.add_trace(go.Cone(
        x=[pts[-1,0]], y=[pts[-1,1]], z=[pts[-1,2]],
        u=[tang[0]], v=[tang[1]], w=[tang[2]],
        sizemode='absolute', sizeref=0.07,
        colorscale=[[0, color], [1, color]],
        showscale=False, hoverinfo='skip', anchor='tip',
    ))


_SCENE = dict(
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    zaxis=dict(visible=False),
    bgcolor='white',
    aspectmode='cube',
    camera=dict(eye=dict(x=1.5, y=-1.0, z=0.9), up=dict(x=0, y=0, z=1)),
)

# ── Diagram 1: Velocity vectors (VK = V + Vw) ────────────────────────────────

fig1 = go.Figure()

AL = 1.4
for tip, label, offset in [
    ([ AL, 0,   0  ], 'x<sub>g</sub>', [ AL+0.13,  0,       0     ]),
    ([  0, AL,  0  ], 'y<sub>g</sub>', [-0.05,     AL+0.13, 0.07  ]),
    ([  0,  0, -AL ], 'z<sub>g</sub>', [ 0.08,     0,      -AL-0.18]),
]:
    _vec(fig1, [0,0,0], tip, color='#222', width=4, sizeref=0.10)
    _lbl(fig1, offset, label)

V_a = np.array([ 0.65,  0.70,  0.30])
Vw  = np.array([ 0.00, -0.55,  0.00])
VK  = V_a + Vw

_vec(fig1, [0,0,0], VK,  color='black', width=2, sizeref=0.08)
_vec(fig1, [0,0,0], V_a, color='black', width=2, sizeref=0.08)
_vec(fig1, V_a,     VK,  color='black', width=2, sizeref=0.07)

_lbl(fig1, VK  + [ 0.08, -0.06,  0.06], 'V<sub>K</sub>')
_lbl(fig1, V_a + [ 0.08,  0.06,  0.06], 'V')
_lbl(fig1, (V_a + VK)/2 + [0.10, 0.0, 0.06], 'V<sub>w</sub>')

Va_f = np.array([V_a[0], V_a[1], 0.0])
VK_f = np.array([VK[0],  VK[1],  0.0])
_box(fig1, np.zeros(3), Va_f)
_box(fig1, np.zeros(3), VK_f)
_box(fig1, Va_f, VK_f)
_box(fig1, Va_f, V_a)
_box(fig1, VK_f, VK)

z_up     = np.array([0., 0., 1.])
V_u      = V_a / np.linalg.norm(V_a)
wing_dir = np.cross(V_u, z_up); wing_dir /= np.linalg.norm(wing_dir)
for d, arm in [(V_u, 0.20), (wing_dir, 0.42)]:
    fig1.add_trace(go.Scatter3d(
        x=[-arm*d[0], arm*d[0]], y=[-arm*d[1], arm*d[1]], z=[-arm*d[2], arm*d[2]],
        mode='lines', line=dict(color='black', width=6),
        showlegend=False, hoverinfo='skip',
    ))

fig1.update_layout(
    scene=_SCENE, paper_bgcolor='white',
    margin=dict(l=0, r=0, t=0, b=0), width=660, height=560,
)

# ── Diagram 2: Flight angles (xi, gamma, mu) ──────────────────────────────────

N       = 120
gamma_v = np.radians(20)
xi_v    = np.radians(35)
VL      = 1.5

vhat = np.array([
    np.cos(gamma_v) * np.sin(xi_v),
    np.cos(gamma_v) * np.cos(xi_v),
    np.sin(gamma_v),
])
V3  = VL * vhat
V_h = np.array([V3[0], V3[1], 0.])

fig2 = go.Figure()

AL2 = 1.2
ax_col = '#444'
for tip, label, offset in [
    ([ AL2, 0,    0   ], 'x<sub>g</sub>', [ AL2+0.12,  0,       0      ]),
    ([   0, AL2,  0   ], 'y<sub>g</sub>', [-0.05,      AL2+0.12, 0.07  ]),
    ([   0,   0, -AL2 ], 'z<sub>g</sub>', [ 0.08,      0,       -AL2-0.16]),
]:
    _vec(fig2, [0,0,0], tip, color=ax_col, width=3, sizeref=0.08)
    _lbl(fig2, offset, label, color=ax_col)

fig2.add_trace(go.Scatter3d(
    x=[0, V_h[0]], y=[0, V_h[1]], z=[0, V_h[2]],
    mode='lines', line=dict(color='#ccc', width=2),
    showlegend=False, hoverinfo='skip',
))

_vec(fig2, [0,0,0], V3, color='#222', width=4, sizeref=0.10)
_lbl(fig2, V3 + [0.08, 0, 0.08], 'V', color='#222', size=15)

wing = np.cross(vhat, z_up); wing /= np.linalg.norm(wing)
for d, arm in [(vhat, 0.12), (wing, 0.25)]:
    fig2.add_trace(go.Scatter3d(
        x=[-arm*d[0], arm*d[0]], y=[-arm*d[1], arm*d[1]], z=[-arm*d[2], arm*d[2]],
        mode='lines', line=dict(color='black', width=5),
        showlegend=False, hoverinfo='skip',
    ))

# xi arc — heading angle
xi_a   = np.linspace(0, xi_v, N)
xi_pts = 0.55 * np.column_stack([np.sin(xi_a), np.cos(xi_a), np.zeros(N)])
fig2.add_trace(go.Scatter3d(
    x=xi_pts[:,0], y=xi_pts[:,1], z=xi_pts[:,2],
    mode='lines', line=dict(color='black', width=3),
    showlegend=False, hoverinfo='skip',
))
_arc_cone(fig2, xi_pts, 'black')
_lbl(fig2, xi_pts[N//2] + [-0.06, 0.12, -0.14], 'ξ', color='black', size=16)

# gamma arc — flight path angle
Vh_n    = V_h / np.linalg.norm(V_h)
gam_a   = np.linspace(0, gamma_v, N)
gam_pts = 0.70 * (np.outer(np.cos(gam_a), Vh_n) + np.outer(np.sin(gam_a), z_up))
fig2.add_trace(go.Scatter3d(
    x=gam_pts[:,0], y=gam_pts[:,1], z=gam_pts[:,2],
    mode='lines', line=dict(color='black', width=3),
    showlegend=False, hoverinfo='skip',
))
_arc_cone(fig2, gam_pts, 'black')
_lbl(fig2, gam_pts[N//2] + [0.10, 0, -0.12], 'γ', color='black', size=16)

# mu arc — bank angle (full rotation around V axis)
e1 = np.cross(-z_up, vhat); e1 /= np.linalg.norm(e1)
e2 = np.cross(vhat, e1);    e2 /= np.linalg.norm(e2)
c_mu   = 0.72 * V3
r_mu   = 0.32
mu_a   = np.linspace(0, 2*np.pi - 0.18, N)
mu_pts = c_mu + r_mu * (np.outer(np.cos(mu_a), e2) + np.outer(np.sin(mu_a), e1))
fig2.add_trace(go.Scatter3d(
    x=mu_pts[:,0], y=mu_pts[:,1], z=mu_pts[:,2],
    mode='lines', line=dict(color='black', width=3),
    showlegend=False, hoverinfo='skip',
))
_arc_cone(fig2, mu_pts, 'black')
_lbl(fig2, mu_pts[3*N//4] + [0.08, 0, -0.06], 'μ', color='black', size=16)

fig2.update_layout(
    scene=_SCENE, paper_bgcolor='white',
    margin=dict(l=0, r=0, t=0, b=0), width=660, height=560,
)

_show(fig1)
_show(fig2)
