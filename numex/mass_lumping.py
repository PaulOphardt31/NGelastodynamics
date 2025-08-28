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
# circ_inner.edges[0].maxh = 0.1

circ.edges[0].Identify(circ.edges[2], "top", IdentificationType.PERIODIC)

circ.edges[3].Identify(circ.edges[1], "right", IdentificationType.PERIODIC)
    

geom = circ - circ_inner


maxh = 0.05 

mesh = Mesh(OCCGeometry(geom, dim=2).GenerateMesh(maxh=maxh))
mesh.Curve(3)
Draw(mesh)


order=1

tau = 1 *  maxh / (5*order)
tend = 1
# u0 = exp(-100**2*( (x-0.5)**2 + (y-0.5)**2))
# v0 = 0

u0 = 0.1 * CF((-2,0)) * sin(2*pi*x)
# u0 = exp(-10**2*( (x-0.5)**2 + (y-0.5)**2)) * CF((1,1))
s0_vec = 0.1 * CF((4,0,0,2)) * sin(2*pi*x)
s0 = 0.1 * sin(2*pi*x) * CF((4,0,0,2), dims = (2,2))

ds0 = CF((s0[0,0].Diff(x) + s0[0,1].Diff(y), s0[1,0].Diff(x) + s0[1,1].Diff(y)))



# fesi = H1LumpingFESpace(mesh, order=order)
fesi = H1(mesh, order=order)
# fesi = NodalFESpace(mesh, order=order)

fesp = Periodic(fesi)

fes = fesp**2

u,v = fes.TnT()

def eps(u):
    return 0.5 * (grad(u) + grad(u).trans)

def tr(sigma):
    return sigma[0,0] + sigma[1,1]

def C(sigma):
    return 2 * mu * sigma + lam * tr(sigma) * Id(2)


points = [(0,0), (1,0), (0,1)]
weights = [1/6, 1/6, 1/6]

# points = [(0,0), (1,0), (0,1), (0.5,0), (0.5,0.5), (0,0.5), (1/3, 1/3)]
# weights = [1/40, 1/40, 1/40, 1/15, 1/15, 1/15, 9/40]



ir = IntegrationRule(points, weights)

mform = u*v*dx(intrules = {TRIG: ir}) #intrules=fesi.GetIntegrationRules())
# aform = InnerProduct(grad(u),grad(v))*dx
aform = InnerProduct(C(eps(u)).Compile(), grad(v))*dx

m = BilinearForm(mform, diagonal=True).Assemble()
# m = BilinearForm(mform).Assemble()
a = BilinearForm(aform).Assemble()
minv = m.mat.Inverse(fes.FreeDofs())  

gfu = GridFunction(fes)
gfu.Set(u0)

Draw(gfu, mesh, name="u")

def GetCeps(u):
    return C(0.5 * (grad(gfu) + grad(gfu).trans))




visoptions.scalfunction="u:1"


# sleep (3)
unew = gfu.vec.CreateVector()
uold = gfu.vec.CreateVector()
uold.data = gfu.vec


Si = L2(mesh, order=order-1)
S = Si**4
gfs_old = GridFunction(S)
gfs = GridFunction(S)
gfs_old.Set(s0_vec)


ms = BilinearForm(S, diagonal=True)
ms += S.TrialFunction() * S.TestFunction() * dx
ms.Assemble()
minvs = ms.mat.Inverse(S.FreeDofs())

fs = LinearForm(S)
fs += InnerProduct(GetCeps(gfu).Compile(), S.TestFunction()) * dx

# Draw(gfs, mesh, name="sigma")
# visoptions.scalfunction="sigma:1"
visoptions.vecfunction="None"


gfsd = GridFunction(fes)
gfsd.Set(ds0)

gfu.vec.data = gfu.vec + tau * gfsd.vec 

# input()
with TaskManager(): 
    for n in range(int(tend/tau)):
        unew.data = 2*gfu.vec - uold 
        unew.data -= tau**2 * minv@a.mat * gfu.vec
        uold.data = gfu.vec
        gfu.vec.data = unew.data

        # fs.Assemble()
        # gfs.vec.data -= tau * minvs * fs.vec
        # gfs_old.vec.data = gfs.vec
        # gfs.Set(gfs_old + tau * GetCeps(gfu))

        # if n % 100 == 0:
        #     print("t =", n*tau)
        Redraw()
        # input()

Redraw()

