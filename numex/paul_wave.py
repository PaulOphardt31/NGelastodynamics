from ngsolve import *
from netgen.occ import *
# from ngsolve.webgui import Draw          # optional: Browser-Viewer
import matplotlib
matplotlib.use("Agg")                      # avoid macOS GUI clash
import matplotlib.pyplot as plt
import numpy as np

R = 2.5
r = 0.25
omega = 2*pi 

c = np.sqrt((2 + 2.0*1) / 1)

# --- Geometry ---
wp = WorkPlane()
wp.MoveTo(-2*R, -R/2)            

outer_face = wp.Rectangle(4*R, R).Face()
for e in outer_face.edges:
    e.name = "outer"                      

wp.MoveTo(-R, -R/2)            

inner_face = wp.Rectangle(2*R, R).Face()
for e in inner_face.edges:
    e.name = "interface"                 

cavity = wp.Circle(0, 0, r).Face()
for e in cavity.edges:
    e.name = "cavity"

h_cavity_edge = 0.02         
h_inner_face  = 0.06        
h_outer_face  = 0.15         

for e in cavity.edges:
    e.maxh = h_cavity_edge
inner_face.maxh = h_inner_face
outer_face.maxh = h_outer_face

outer_shell = outer_face - inner_face     
inner_solid = inner_face - cavity         
geo = Glue([outer_shell, inner_solid])

mesh = Mesh(OCCGeometry(geo, dim=2).GenerateMesh(maxh=h_outer_face))
mesh.Curve(3)
Draw(mesh)


print("Materials:", mesh.GetMaterials())
print("Boundaries:", mesh.GetBoundaries())

station1 = mesh(1, 0)
station2 = mesh(0.5,0.5)
print('station object', station1)

# --- FE spaces ---
order = 4
fes1 = L2(mesh, order=order)  #TODO complex = True raises an error for SolveM, could be nice to have for PML
fes = fes1 * fes1 * fes1      #TODO subdomain formalism could be extended, use different sol on outer domain and use coupling

p, ux, uy = fes.TrialFunction()
q, vx, vy = fes.TestFunction()         

n = specialcf.normal(2)
v = CoefficientFunction((vx, vy))
u = CoefficientFunction((ux, uy))

# --- DG ---
a1 = BilinearForm(fes)
a1 += c * grad(p) * v * dx
a1 += c * -0.5 * (p - p.Other()) * (v*n) * dx(element_boundary=True)


a2 = BilinearForm(fes)
a2 += -c * grad(q) * u * dx
a2 +=  c * 0.5 * (q - q.Other()) * (u*n) * dx(element_boundary=True)

# --- Absorbing boundary ---
#a1 += ( -0.5*p*(v*n) + 0.5*(u*n)*(v*n) ) * ds(definedon=mesh.Boundaries("outer")) #TODO fix zsh: bus error  netgen NGelastodynamics/numex/alternative_wave.py
#a2 += (  0.5*p*q      - 0.5*(u*n)*q    ) * ds(definedon=mesh.Boundaries("outer"))

# --- Rigid cavity wall ---
#alpha = 1.0
#a1 += alpha * (u*n)*(v*n) * ds(definedon=mesh.Boundaries("cavity")) #TODO same error as last, other BND condition needed (grad(p)*n = 0)


# --- Initial conditions ---
tpar = Parameter(0.0)
u0     = sin(omega * x)
u0ana  = sin(omega * (x - tpar)) 

U = GridFunction(fes)
U.components[0].Set(u0)   # p
U.components[1].Set(u0)   # u_x
U.components[2].Set(0)    # u_y

res = U.vec.CreateVector()
w   = U.vec.CreateVector()

Draw(U.components[1], mesh, "ux")
Draw(U.components[2], mesh, "uy")
Draw(U.components[0], mesh, "p")
SetVisualization(min=-0.1, max=0.1, deformation=True)

# --- Time stepping ---
tau  = 1e-4
tend = 1.0
t    = 0.0
nd   = fes1.ndof

t_list = []
p_list_station1 = []
p_list_station2 = []

with TaskManager():
    while t < tend:
        a1.Apply(U.vec, w)
        fes1.SolveM(rho=CoefficientFunction(1), vec=w.Range(nd,   2*nd))
        fes1.SolveM(rho=CoefficientFunction(1), vec=w.Range(2*nd, 3*nd))
        U.vec.data -= tau * w

        a2.Apply(U.vec, w)
        fes1.SolveM(rho=CoefficientFunction(1), vec=w.Range(0, nd))
        U.vec.data -= tau * w

        t_list.append(t)
        p_list_station1.append(U.components[0](station1))
        p_list_station2.append(U.components[0](station2))

        t += tau
        tpar.Set(t)     
        Redraw()
        sys.stdout.write(f"\r t = {t:.3f}")
        sys.stdout.flush()

plt.figure()
plt.plot(t_list, p_list_station1, lw=1.2, label="Station (1,0)")
plt.plot(t_list, p_list_station2, lw=1.2, label="Station (0.5, 0.5)")
plt.ylabel("Pressure ")
plt.xlabel("Time [s]")
plt.legend()
plt.tight_layout()
plt.savefig("Benchmarksolutions/pressure_stations.png", dpi=150)
print("Saved: pressure_stations.png")
