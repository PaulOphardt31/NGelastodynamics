from ngsolve import *
# from ngsolve.webgui import Draw
from time import sleep, time
from netgen.occ import *
from ngsolve.internal import visoptions, viewoptions

R = 5
r = 0.25

lam = 2
mu = 1


wp = WorkPlane()
circ = wp.RectangleC(R,R).Face() 
circ.edges[0].name = "bottom"
circ.edges[1].name = "right"
circ.edges[2].name = "top"
circ.edges[3].name = "left"

circ_inner = wp.Circle(0,0,r).Face()
circ_inner.edges[0].name = "inner"


maxh = 0.01
factor = 5

circ_inner.edges[0].maxh = maxh

circ.edges[0].Identify(circ.edges[2], "top", IdentificationType.PERIODIC)

circ.edges[3].Identify(circ.edges[1], "right", IdentificationType.PERIODIC)
    

geom = circ - circ_inner



mesh = Mesh(OCCGeometry(geom, dim=2).GenerateMesh(maxh=factor * maxh))
mesh.Curve(3)
Draw(mesh)


order=2

dt = maxh / (5*order)
tend = 1
# u0 = exp(-100**2*( (x-0.5)**2 + (y-0.5)**2))
# v0 = 0

u0 = 0.1 * CF((-2,0)) * sin(2*pi*x)
# u0 = exp(-10**2*( (x-0.5)**2 + (y-0.5)**2)) * CF((1,1))
s0_vec = 0.1 * CF((4,2,0)) * sin(2*pi*x)
s0 = 0.1 * sin(2*pi*x) * CF((4,0,0,2), dims = (2,2))

ds0 = CF((s0[0,0].Diff(x) + s0[0,1].Diff(y), s0[1,0].Diff(x) + s0[1,1].Diff(y)))



fesi = H1LumpingFESpace(mesh, order=order)
fesp = Periodic(fesi)

fes = fesp**2

Si = Periodic(L2(mesh, order=order))
S = Si**3

u,v = fes.TnT()
sigma_vec, tau_vec = S.TnT()

sigma = CF(( sigma_vec[0], sigma_vec[2], sigma_vec[2], sigma_vec[1] ), dims = (2,2))
tau = CF(( tau_vec[0], tau_vec[2], tau_vec[2], tau_vec[1] ), dims = (2,2))


def eps(u):
    return 0.5 * (grad(u) + grad(u).trans)

def tr(sigma):
    return sigma[0,0] + sigma[1,1]

def C(sigma):
    return 2 * mu * sigma + lam * tr(sigma) * Id(2)

def Cinv(sigma):
    return 1/(2 * mu) * (sigma - lam/(2*mu + 2*lam) * tr(sigma) * Id(2))



points = [(0,0), (1,0), (0,1), (0.5,0), (0.5,0.5), (0,0.5), (1/3, 1/3)]
weights = [1/40, 1/40, 1/40, 1/15, 1/15, 1/15, 9/40]

n = specialcf.normal(2)

ir = IntegrationRule(points, weights)

mform = u*v*dx(intrules=fesi.GetIntegrationRules())
# aform = InnerProduct(grad(u),grad(v))*dx
aform = InnerProduct(C(eps(u)), grad(v))*dx


m = BilinearForm(mform, diagonal=True).Assemble()
a = BilinearForm(trialspace=S, testspace=fes)
a += -InnerProduct(sigma, grad(v)) * dx
a.Assemble()

minv = m.mat.Inverse(fes.FreeDofs())  

mS = BilinearForm(S) 
mS += InnerProduct(Cinv(sigma), tau) * dx
mS.Assemble()
mSinv = mS.mat.Inverse(S.FreeDofs(), inverse="sparsecholesky")

aS = BilinearForm(trialspace=fes, testspace=S)
aS += InnerProduct(eps(u), tau) * dx()
aS.Assemble()


gfu = GridFunction(fes)
gfu.Set(u0)

gfs = GridFunction(S)
gfs.Set(s0_vec)

Draw(gfu, mesh, name="u")
Draw(gfs, mesh, name="sigma")



visoptions.scalfunction="u:1"
visoptions.scalfunction="sigma:1"


visoptions.vecfunction="None"


w= gfu.vec.CreateVector()

with TaskManager(): 
    for n in range(int(tend/dt)):
        gfu.vec.data += dt * minv@a.mat * gfs.vec 
        gfs.vec.data += dt * mSinv@aS.mat * gfu.vec

        
        if n % 100 == 0:
            print("t =", n*dt)
            Redraw()
        # input()




Redraw()

