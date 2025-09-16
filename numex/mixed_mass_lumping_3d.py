from ngsolve import *
# from ngsolve.webgui import Draw
from time import sleep, time
from netgen.occ import *

from ngsolve.internal import visoptions, viewoptions

R = 2.5
r = 0.25

lam = 2
mu = 1

box = Box(Pnt(-R,-R,-R), Pnt(R,R,R))
box.faces.Max(X).name = "front"
box.faces.Min(X).name = "back"
box.faces.Max(Y).name = "right"
box.faces.Min(Y).name = "left"
box.faces.Max(Z).name = "top"
box.faces.Min(Z).name = "bottom"

sp = Sphere(Pnt(0,0,0), r)
sp.faces[0].name = "inner"


maxh = 0.05
factor = 5

sp.faces[0].maxh = maxh


box.faces[5].Identify(box.faces[4], "top", IdentificationType.PERIODIC)
box.faces[1].Identify(box.faces[0], "front", IdentificationType.PERIODIC)
box.faces[3].Identify(box.faces[2], "right", IdentificationType.PERIODIC)

geom = box -sp
# circ.edges[3].Identify(circ.edges[1], "right", IdentificationType.PERIODIC)



# wp = WorkPlane()
# circ = wp.RectangleC(R,R).Face() 
# circ.edges[0].name = "bottom"
# circ.edges[1].name = "right"
# circ.edges[2].name = "top"
# circ.edges[3].name = "left"

# circ_inner = wp.Circle(0,0,r).Face()
# circ_inner.edges[0].name = "inner"



mesh = Mesh(OCCGeometry(geom, dim=3).GenerateMesh(maxh=factor * maxh))
# mesh.Curve(3)
Draw(mesh)
# print(mesh.GetBoundaries())
# input()

order=2

dt =  0.2 * maxh / (5*order)
tend = 1

# u0 = exp(-100**2*( (x-0.5)**2 + (y-0.5)**2))
# v0 = 0

# damp = (x**2 + y**2 - r**2)
# damp = IfPos(x + 1, 0, 1)
damp = exp(-2*(x + 2)**2)

u0 = 0.1 * CF((-2,0,0)) * sin(2*pi*x)
# u0 = exp(-10**2*( (x-0.5)**2 + (y-0.5)**2)) * CF((1,1))



s0_vec = 0.1 * CF((4,2,0,0,0,0)) * sin(2*pi*x) * damp
# s0 = 0.1 * sin(2*pi*x) * CF((4,0,0,2), dims = (2,2)) * damp

# ds0 = CF((s0[0,0].Diff(x) + s0[0,1].Diff(y), s0[1,0].Diff(x) + s0[1,1].Diff(y)))



fesi = H1LumpingFESpace(mesh, order=order)
fesp = Periodic(fesi)

fes = fesp**3

Si = Periodic(L2(mesh, order=order+1))
S = Si**6

u,v = fes.TnT()
sigma_vec, tau_vec = S.TnT()

sigma = CF(( sigma_vec[0], sigma_vec[3], sigma_vec[4], 
            sigma_vec[3],sigma_vec[1], sigma_vec[5], 
            sigma_vec[4],sigma_vec[5], sigma_vec[2] ), dims = (3,3))

tau = CF(( tau_vec[0], tau_vec[3], tau_vec[4], 
            tau_vec[3],tau_vec[1],tau_vec[5], 
            tau_vec[4],tau_vec[5],tau_vec[2] ), dims = (3,3))


def eps(u):
    return 0.5 * (grad(u) + grad(u).trans)

def tr(sigma):
    return sigma[0,0] + sigma[1,1] + sigma[2,2]

def C(sigma):
    return 2 * mu * sigma + lam * tr(sigma) * Id(3)

def Cinv(sigma):
    return 1/(2 * mu) * (sigma - lam/(2*mu + 3*lam) * tr(sigma) * Id(3))



# points = [(0,0), (1,0), (0,1), (0.5,0), (0.5,0.5), (0,0.5), (1/3, 1/3)]
# weights = [1/40, 1/40, 1/40, 1/15, 1/15, 1/15, 9/40]

n = specialcf.normal(2)

# ir = IntegrationRule(points, weights)

mform = u*v*dx(intrules=fesi.GetIntegrationRules())
# aform = InnerProduct(grad(u),grad(v))*dx
aform = InnerProduct(C(eps(u)), grad(v))*dx


m = BilinearForm(mform, diagonal=True)
a = BilinearForm(trialspace=S, testspace=fes)
a += -InnerProduct(sigma, grad(v)) * dx
# a.Assemble()

 

mS = BilinearForm(S, diagonal=True) 
# mS += InnerProduct(Cinv(sigma), tau) * dx
mS += InnerProduct(sigma, tau) * dx
# mS.Assemble()

aS = BilinearForm(trialspace=fes, testspace=S)
# aS += InnerProduct(eps(u), tau) * dx()
aS += InnerProduct(C(eps(u)), tau) * dx()
# aS.Assemble()




gfu = GridFunction(fes)


gfs = GridFunction(S)
with TaskManager():
    gfu.Set(u0)
    gfs.Set(s0_vec)

Draw(gfu, mesh, name="u")
Draw(gfs, mesh, name="sigma", draw_surf = False)



# visoptions.scalfunction="u:1"
visoptions.scalfunction="sigma:1"


visoptions.vecfunction="None"

viewoptions.clipping.nx = 0
viewoptions.clipping.ny = 0
viewoptions.clipping.nz = 1

viewoptions.clipping.enable = 1
visoptions.clipsolution = 'scal'

w= gfu.vec.CreateVector()

# input()
with TaskManager():
    a.Assemble()
    m.Assemble()
    aS.Assemble()
    mS.Assemble()

    minv = m.mat.Inverse(fes.FreeDofs()) 
    mSinv = mS.mat.Inverse(S.FreeDofs()) #, inverse="sparsecholesky")

with TaskManager(): 
    for n in range(int(tend/dt)):
        gfu.vec.data += dt * minv@a.mat * gfs.vec 
        gfs.vec.data += dt * mSinv@aS.mat * gfu.vec

        # 
        if n % 100 == 0:
            print("t =", n*dt)
            Redraw()
        # input()




Redraw()

