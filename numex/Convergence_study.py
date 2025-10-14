# --- convergence_driver.py
from ngsolve import *
from netgen.occ import *
import numpy as np
from numpy.fft import rfft, rfftfreq
import matplotlib.pyplot as plt
import time 
# ----- problem constants (your values)
R = 5
r = 0.25
lam, mu = 2, 1
order = 2
tend = 1.0
f0 = 2.0                  # Hz  (u ~ sin(2πx - 2*2π t))
c_true = 2.0              # exact phase speed for this plane wave
lambda0 = 1.0
dx_probe = 0.25           # ≤ λ/2 to avoid aliasing
probe1 = (0.00, 0.0)
probe2 = (probe1[0] + dx_probe, probe1[1])

def build_mesh(maxh):
    wp = WorkPlane()
    rect = wp.RectangleC(R, R).Face()
    rect.edges[0].name = "bottom"
    rect.edges[1].name = "right"
    rect.edges[2].name = "top"
    rect.edges[3].name = "left"
    hole = wp.Circle(0, 0, r).Face()
    hole.edges[0].name = "inner"
    hole.edges[0].maxh = maxh

    # periodic identifications
    rect.edges[0].Identify(rect.edges[2], "top", IdentificationType.PERIODIC)
    rect.edges[3].Identify(rect.edges[1], "right", IdentificationType.PERIODIC)

    geom = rect  # no hole if you want: geom = rect - hole
    print('grid setup')
    return Mesh(OCCGeometry(geom, dim=2).GenerateMesh(maxh=maxh))

def c_from_xcorr_unwrapped(u1, u2, dt, dx, f0, c_guess=None):
    # lag modulo T
    u1 = u1 - np.mean(u1); u2 = u2 - np.mean(u2)
    u1 /= (np.linalg.norm(u1) + 1e-15); u2 /= (np.linalg.norm(u2) + 1e-15)
    c = np.correlate(u2, u1, mode='full')
    k = np.argmax(c)
    lag_samples = k - (len(u1) - 1)
    dtau = lag_samples * dt

    # unwrap with integer multiples of period
    T = 1.0 / f0
    if c_guess is None:
        n = int(np.round(-dtau / T))
    else:
        n = int(np.round((dx / (c_guess * T)) - dtau / T))
    dtau_unw = dtau + n * T
    return dx / dtau_unw, dtau, dtau_unw, n

def run_once_dg(maxh,
             order=3,
             tend=1.0,
             dt_scale=5,
             lam=2.0, mu=1.0,
             init_amp=0.1,
             probe1=probe1,
             probe2=probe2,
             dx_probe=None,   # set to |x2-x1| if not along x, pass explicit distance
             f0=f0,          # dominant frequency for c_from_xcorr_unwrapped
             c_true=2.0,      # expected phase/group speed used for rel error
             draw=False):
    """
    Mixed DG (sigma,u) first-order elastodynamics with error metrics and probes.

    Returns a dict with keys identical to your H1-lumped run_once:
        {maxh, dt, dof_u, dof_s, dof, h_mean, c_est, disp_rel_err,
         relL2_final, relL2_mean, gfu}
    """
    # --- mesh & dt ---
    mesh = build_mesh(maxh)
    dt   = maxh / (dt_scale * order)

    # --- spaces: DG L2 ---
    fes  = Periodic(L2(mesh, order=order, dgjumps=True))
    X    = fes**3 * fes**2          # sigma(3) ⊕ u(2)

    # trial / test
    sigma_vec, u = X.TrialFunction()
    tau_vec,   v = X.TestFunction()

    # pack/unpack 2x2 tensors for constitutive ops
    sigma      = CF(( sigma_vec[0], sigma_vec[2],
                      sigma_vec[2], sigma_vec[1] ), dims=(2,2))
    tau        = CF(( tau_vec[0],   tau_vec[2],
                      tau_vec[2],   tau_vec[1] ), dims=(2,2))

    sigma_other = CF(( sigma_vec.Other()[0], sigma_vec.Other()[2],
                       sigma_vec.Other()[2], sigma_vec.Other()[1] ), dims=(2,2))
    tau_other   = CF(( tau_vec.Other()[0],   tau_vec.Other()[2],
                       tau_vec.Other()[2],   tau_vec.Other()[1] ), dims=(2,2))

    n = specialcf.normal(2)

    def tr(S): return S[0,0] + S[1,1]
    def Cinv(S):
        return 1/(2*mu) * (S - lam/(2*mu + 2*lam) * tr(S) * Id(2))

    # --- bilinear forms ---
    aa  = InnerProduct(sigma, Grad(v)) * dx
    aa += InnerProduct(tau,   Grad(u)) * dx

    # symmetric interior numerical fluxes
    aa += -0.5 * (v - v.Other()) * (sigma*n) * dx(element_boundary=True)
    aa += -0.5 * (u - u.Other()) * (tau*n)   * dx(element_boundary=True)

    mm  = -InnerProduct(Cinv(sigma), tau) * dx   # stress block (compliance)
    mm += u * v * dx                             # displacement block

    # system operators
    mstar = BilinearForm(X); mstar += mm; mstar += dt * aa; mstar.Assemble()
    mstarinv = mstar.mat.Inverse(X.FreeDofs(), inverse="sparsecholesky")
    a = BilinearForm(X); a += aa; a.Assemble()

    # --- initial & analytic fields (plane wave along +x) ---
    tpar   = Parameter(0.0)
    u0     = init_amp * CF((-2, 0)) * sin(2*pi*x)
    u_ana  = init_amp * CF((-2, 0)) * sin(2*pi*x - 2*2*pi*tpar)     # speed 2
    s0_vec = init_amp * CF(( 4, 2, 0)) * sin(2*pi*x)
    s_ana  = init_amp * CF(( 4, 2, 0)) * sin(2*pi*x - 2*2*pi*tpar)

    gfu = GridFunction(X)
    gfu.components[0].Set(s0_vec)   # sigma triple
    gfu.components[1].Set(u0)       # u vector

    if draw:
        Draw(gfu.components[0], mesh, "sigma")
        Draw(gfu.components[1], mesh, "u")
        visoptions.scalfunction = "sigma:1"
        visoptions.vecfunction  = "None"

    # --- probes (σ_xx component) ---
    if dx_probe is None:
        dx_probe = ((probe2[0]-probe1[0])**2 + (probe2[1]-probe1[1])**2)**0.5
    station1 = mesh(*probe1)
    station2 = mesh(*probe2)

    s1_num, s2_num = [], []
    s1_ana, s2_ana = [], []
    err_L2_times   = []

    # helpers to access numeric fields
    # sigma_xx from first component of the sigma triple:
    sigma_xx_num = gfu.components[0][0]
    # displacement vector:
    ux_num = gfu.components[1][0]
    uy_num = gfu.components[1][1]
    u_num  = CF((ux_num, uy_num))

    # --- time loop ---
    t = 0.0
    steps = int(np.ceil(tend/dt))
    w = gfu.vec.CreateVector()

    with TaskManager():
        for n in range(steps):
            # RHS and update
            w.data = dt * a.mat * gfu.vec
            gfu.vec.data -= mstarinv * w

            t = (n+1)*dt
            tpar.Set(t)

            # probe σ_xx (index 0 of analytic s_ana)
            s1_num.append(float(sigma_xx_num(station1)))
            s2_num.append(float(sigma_xx_num(station2)))
            s1_ana.append(float(s_ana(station1)[0]))
            s2_ana.append(float(s_ana(station2)[0]))

            # dissipation metric (L2 error of displacement vs analytic)
            if n % 100 == 0:
                eL2 = sqrt(Integrate(InnerProduct(u_num - u_ana, u_num - u_ana), mesh, order=8))
                err_L2_times.append(eL2)
                print(f"t={t:.3f}, step {n+1}/{steps}, L2(u-uan)={float(eL2):.3e}")
            if draw:
                Redraw()

    # --- dispersion from unwrapped xcorr of σ_xx traces ---
    s1_num = np.array(s1_num); s2_num = np.array(s2_num)
    c_est, dtau, dtau_unw, nT = c_from_xcorr_unwrapped(s1_num, s2_num, dt, dx_probe, f0, c_guess=c_true)
    disp_rel_err = abs(c_est - c_true) / c_true

    # --- aggregate dissipation ---
    relL2_final = sqrt(Integrate(InnerProduct(u_num - u_ana, u_num - u_ana), mesh, order=8))
    relL2_mean  = float(np.mean(err_L2_times)) if err_L2_times else relL2_final

    # --- mesh & dof metrics ---
    h = specialcf.mesh_size
    h_mean = float(Integrate(h**2, mesh))**0.5
    #dof_s  = X.Space(0).ndof
    #dof_u  = X.Space(1).ndof
    dof_total = X.ndof

    # --- return identical dictionary signature ---
    return dict(
        maxh=maxh,
        dt=dt,
        #dof_u=dof_u,
        #dof_s=dof_s,
        dof=dof_total,
        h_mean=h_mean,
        c_est=c_est,
        disp_rel_err=disp_rel_err,
        relL2_final=relL2_final,
        relL2_mean=relL2_mean,
        gfu=gfu,
    )


def run_once(maxh):
    mesh = build_mesh(maxh)
    # time step tied to mesh size (keep CFL comparable)
    dt = maxh / (5*order)

    # spaces
    fesi = H1LumpingFESpace(mesh, order=order)
    fesp = Periodic(fesi)
    fes = fesp**2

    Si = Periodic(L2(mesh, order=order+1))
    S = Si**3

    # vars
    u, v = fes.TnT()
    sigma_vec, tau_vec = S.TnT()
    sigma = CF(( sigma_vec[0], sigma_vec[2], sigma_vec[2], sigma_vec[1] ), dims=(2,2))
    tau   = CF(( tau_vec[0],   tau_vec[2],   tau_vec[2],   tau_vec[1] ), dims=(2,2))

    def eps(u): return 0.5*(grad(u)+grad(u).trans)
    def tr(s):  return s[0,0]+s[1,1]
    def C(s):   return 2*mu*s + lam*tr(s)*Id(2)
    def Cinv(s): return 1/(2*mu) * (s - lam/(2*mu+2*lam) * tr(s) * Id(2))

    # forms
    mform = u*v*dx(intrules=fesi.GetIntegrationRules())
    m  = BilinearForm(mform, diagonal=True).Assemble()
    a  = BilinearForm(trialspace=S, testspace=fes)
    a += -InnerProduct(sigma, grad(v))*dx
    a.Assemble()
    minv = m.mat.Inverse(fes.FreeDofs())

    # import scipy.sparse as sp
    # A = sp.csr_matrix(a.mat.CSR())
    # plt.rcParams['figure.figsize'] = (4,4)
    # plt.spy(A)
    # plt.show()

    mS = BilinearForm(S, diagonal=True)
    mS += InnerProduct(sigma, tau)*dx   # mass-like for stress variable
    mS.Assemble()
    mSinv = mS.mat.Inverse(S.FreeDofs(), inverse="sparsecholesky")

    aS = BilinearForm(trialspace=fes, testspace=S)
    aS += InnerProduct(C(eps(u)), tau)*dx()
    aS.Assemble()

    print('bilinear forms setup')

    # initial + analytic
    tpar = Parameter(0.0)
    u0    = 0.1 * CF((-2,0)) * sin(2*pi*x)
    u_ana = 0.1 * CF((-2,0)) * sin(2*pi*x - 2*2*pi*tpar)
    s0vec = 0.1 * CF((4,2,0)) * sin(2*pi*x)
    s_ana = 0.1 * CF((4,2,0)) * sin(2*pi*x - 2*2*pi*tpar)

    gfu = GridFunction(fes); gfu.Set(u0)
    gfs = GridFunction(S);   gfs.Set(s0vec)

    # probes
    station1 = mesh(*probe1)
    station2 = mesh(*probe2)
    s1_num, s2_num = [], []
    # (optional) analytic probe on sigma_xx component:
    s1_ana, s2_ana = [], []

    err_L2_times = []
    steps = int(np.ceil(tend/dt))
    with TaskManager():
        for n in range(steps):
            # leapfrog-like update you had
            gfu.vec.data += dt * (minv @ a.mat)  * gfs.vec
            gfs.vec.data += dt * (mSinv @ aS.mat)* gfu.vec

            tnow = n*dt
            tpar.Set(tnow)

            # probe sigma_xx (component 0 of sigma_vec) and store
            s1_num.append(float(gfs[0](station1)))
            s2_num.append(float(gfs[0](station2)))
            s1_ana.append(float(s_ana(station1)[0]))
            s2_ana.append(float(s_ana(station2)[0]))

            # dissipation metric (relative L2 of displacement vs analytic)
            if n % 100 == 0:
                print(f"t={tnow:.3f}, step {n}/{steps}")
                eL2 = sqrt(Integrate(InnerProduct(gfu - u_ana, gfu - u_ana), mesh, order=8))
                #L2u = sqrt(Integrate(InnerProduct(u_ana, u_ana), mesh, order=8))
                err_L2_times.append(eL2)# / (L2u + 1e-15))
                print(err_L2_times[-1])

    # dispersion from unwrapped xcorr
    s1_num = np.array(s1_num); s2_num = np.array(s2_num)
    c_est, dtau, dtau_unw, nT = c_from_xcorr_unwrapped(s1_num, s2_num, dt, dx_probe, f0, c_guess=c_true)
    disp_rel_err = abs(c_est - c_true) / c_true

    # aggregate dissipation: final-time relative L2 and time-average
    relL2_final = sqrt(Integrate(InnerProduct(gfu - u_ana, gfu - u_ana), mesh, order=8))# / \
                  #(sqrt(Integrate(InnerProduct(u_ana, u_ana), mesh, order=8)) + 1e-15)
    print(relL2_final)
    relL2_mean = float(np.mean(err_L2_times)) if err_L2_times else relL2_final

    # mesh metrics
    h = specialcf.mesh_size
    h_mean = float(Integrate(h**2, mesh))**0.5
    dof_u = fes.ndof; dof_s = S.ndof; dof_total = dof_u + dof_s

    return dict(maxh=maxh, dt=dt, dof_u=dof_u, dof_s=dof_s, dof=dof_total,
                h_mean=h_mean, c_est=c_est, disp_rel_err=disp_rel_err,
                relL2_final=relL2_final, relL2_mean=relL2_mean, gfu = gfu)

def run_convergence(maxh0, nlevels):
    results = []
    for lev in range(nlevels):
        maxh = maxh0 - lev * 0.02
        print(f"\n--- level {lev}, maxh={maxh} ---")
        timestart = time.time()
        res = run_once(maxh)
        timeend = time.time()
        print(f"done in {timeend - timestart:.1f} sec")
        print(f"DoF={res['dof']}, c≈{res['c_est']:.4f}, disp_err={res['disp_rel_err']:.3e}, "
              f"relL2_final={res['relL2_final']:.3e}")
        results.append(res)
    return results


if __name__ == "__main__":
    # choose a starting mesh size (close to your original factor*maxh)
    results = run_convergence(maxh0=0.1, nlevels=5)

    # plots: errors vs DoF (or vs h_mean)
    dof   = np.array([r["dof"] for r in results], dtype=float)
    hbar  = np.array([r["h_mean"] for r in results], dtype=float)
    edisp = np.array([r["disp_rel_err"] for r in results], dtype=float)
    eL2   = np.array([r["relL2_final"] for r in results], dtype=float)
    gfu  = results[-1]["gfu"]  # finest mesh solution

    # slope helper
    def slope(x, y):
        # linear fit in log-log
        coeff = np.polyfit(np.log(x), np.log(y), 1)
        return coeff[0]

    print("\nEOC (error vs h_mean):")
    print("  disp:", slope(hbar, edisp))
    print("  L2  :", slope(hbar, eL2))

    # suppose your current solution is gfu (displacement) on 'mesh'
    # pick a test-mass point just outside the domain to avoid singularity:
    mesh = build_mesh(maxh=0.1)
    x0s = [(0.0, 0.0, 0.0)] if mesh.dim==3 else [(0.0, 0.0)]  # add z if 3D

    # if your model is 2D, assume thickness, e.g. 1 m:
    # acc = newtonian_noise_accel(mesh, gfu, x0s, rho=1.0, G=1.0,
    #                             order=8, eps=0.05*dof[-1]**(-1/3),
    #                             thickness=1.0)
    # print("δa(x0) =", acc)
    # station1 = mesh(*probe1)
    # print(8 * np.pi / 3 * float(gfu[0](station1)),8 * np.pi / 3 * float(gfu[1](station1)))

    print(dof, edisp, eL2)
    L2_n2_m3 = [0.09439739857335111, 0.07470435300646619, 0.0554050845523585, 0.03646866738771914, 0.01800453238584756]
    disp_n2_m3 = [1.364e-01, 1.161e-01, 9.649e-02, 7.759e-02, 7.759e-02]

    dof_dg_3 = [285100, 453900, 807900]
    L2_dg_3 = [2.891e-01, 2.433e-01, 1.914e-01]
    disp_dg_3 = [7.143e-02, 1.029e-01, 8.696e-02]
    plt.figure()
    plt.plot(dof[::-1], L2_n2_m3, label="L2 for pol. order 3", c = 'r')
    plt.plot(dof[::-1], edisp, label="dispersion for pol order 2", c = 'g')
    plt.plot(dof[::-1], eL2, label=" L2 for pol. order 2", c = 'b')
    plt.plot(dof[::-1], disp_n2_m3, label="dispersion for pol order 3", c = 'y')
    plt.plot(dof_dg_3[::-1], L2_dg_3, label="L2 DG order 3", c = 'm')
    plt.plot(dof_dg_3[::-1], disp_dg_3, label="dispersion DG order 3", c = 'c')
    plt.scatter(dof[::-1], L2_n2_m3, c = 'r')
    plt.scatter(dof[::-1], edisp, c = 'g')
    #plt.scatter(dof[::-1], eL2, c = 'b')
    plt.scatter(dof[::-1], disp_n2_m3, c = 'y')
    plt.gca().invert_xaxis()  # finer mesh -> right; invert if you prefer
    plt.xlabel("total DoF")
    plt.ylabel("error")
    plt.xscale("log")  
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.show()
