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
S = HDivDiv(mesh, order=order-1, dirichlet="bottom|right|top", plus = True)
V = HCurl(mesh, order=order, dirichlet="left", type1 = True)

v, dv = V.TnT()
sigma, dsigma = S.TnT()  

n = specialcf.normal(2)

def tang(u): return u-(u*n)*n

def eps(u):
    return 0.5 * (grad(u) + grad(u).trans)

aV = BilinearForm(trialspace = S, testspace = V)
aV += -div(sigma) * dv * dx
aV += sigma*n * tang(dv) * dx(element_boundary = True)

bV = BilinearForm(V)
bV += v * dv * ds(skeleton = True, definedon=mesh.Boundaries("outer"))
bV.Assemble()

# C = Id(2)

aS = BilinearForm(trialspace = V, testspace = S)
# grad since C = Id and dsigma is symmetric
aS += - InnerProduct(grad(v), dsigma)  * dx
aS += (dsigma*n)*n * v*n * dx(element_boundary = True)

mV = BilinearForm(V, symmetric=True)
mV += InnerProduct(v, dv) * dx
mV.Assemble()

MinvV = mV.mat.Inverse(V.FreeDofs(), inverse="umfpack")

mS = BilinearForm(S, symmetric=True)
mS += InnerProduct(sigma, dsigma) * dx
mS.Assemble()

MinvS = mS.mat.Inverse(S.FreeDofs(), inverse="umfpack")

# --- Initial conditions ---
tpar = Parameter(0.0)
u0     = sin(omega * x)
u0ana  = sin(omega * (x - tpar)) 

gfv = GridFunction(V)
gfs = GridFunction(S)

gfv.Set(CF((u0,0)))
gfs.Set((CF((4*u0,0)),CF((0, 2*u0))))

resV = gfv.vec.CreateVector()
wV = gfv.vec.CreateVector()

# resS = gfs.vec.CreateVector()
wS = gfs.vec.CreateVector()

Draw (gfv, mesh, "gfv")
Draw (gfs[0], mesh, "gfs")
#visoptions.scalfunction="gfv:2"
#visoptions.vecfunction="None"

# --- Time stepping ---
tau  = 1e-4
tend = 1.0
t    = 0.0

t_list = []
sxx_list_station1 = []
sxx_list_station2 = []

with TaskManager():
    while t < tend:
        aS.Apply (gfv.vec, wS) 
        # print(Norm(wS))
        gfs.vec.data -= tau * MinvS * wS

        aV.Apply (gfs.vec, wV)    
        resV.data = bV.mat * gfv.vec   
        # print(Norm(resV))
        resV.data += wV        
        gfv.vec.data -= tau * MinvV * resV


        t_list.append(t)
        #sxx_list_station1.append(gfs.components[0](station1))
        #sxx_list_station2.append(gfs.components[0](station2))

        t += tau
        Redraw()

        sys.stdout.write(f"\r t = {t:.3f}")
        sys.stdout.flush()
        # input()

plt.figure()
plt.plot(t_list, sxx_list_station1, lw=1.2, label="sxx Station (1,0)")
plt.plot(t_list, sxx_list_station2, lw=1.2, label="sxx Station (0.5, 0.5)")
plt.ylabel("Pressure ")
plt.xlabel("Time [s]")
plt.legend()
plt.tight_layout()
plt.savefig("Benchmarksolutions/pressure_stations.png", dpi=150)
print("Saved: pressure_stations.png")
