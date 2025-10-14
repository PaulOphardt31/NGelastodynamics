from ngsolve import *
# from ngsolve.webgui import Draw
from time import sleep, time
from netgen.occ import *
from ngsolve.internal import visoptions, viewoptions

R = 4
r = 0.25

lam = 2
mu = 1

# SetNumThreads(12)

box = Box(Pnt(-R,-R,-R), Pnt(R,R,R))
box.faces.Max(X).name = "front"
box.faces.Min(X).name = "back"
box.faces.Max(Y).name = "right"
box.faces.Min(Y).name = "left"
box.faces.Max(Z).name = "top"
box.faces.Min(Z).name = "bottom"

box.faces[5].Identify(box.faces[4], "top", IdentificationType.PERIODIC)
box.faces[1].Identify(box.faces[0], "front", IdentificationType.PERIODIC)
box.faces[3].Identify(box.faces[2], "right", IdentificationType.PERIODIC)


local = Sphere(Pnt(0,0,0), 3 * r)

sp = Sphere(Pnt(0,0,0), r)
sp.faces[0].name = "inner"

maxh = 0.05 #05
factor = 20

sp.faces[0].maxh = maxh


small = local - sp
large = box - local - sp

small.name = "small"
large.name = "large"

geom = Glue ([large,small])

mesh = Mesh(OCCGeometry(geom, dim=3).GenerateMesh(maxh=factor*maxh, grading=0.5))
mesh.Curve(3)

Draw(mesh)
# print(mesh.GetBoundaries())
# print(mesh.GetMaterials())
# input()

order=2

tau = 0.1 *  maxh / (5*order)
tend = 10
# u0 = exp(-100**2*( (x-0.5)**2 + (y-0.5)**2))
# v0 = 0

# u0 = 0.1 * CF((-2,0,0)) * sin(2*pi*x)
# s0 = 0.1 * sin(2*pi*x) * CF((4,0,0,0,2,0,0,0,0), dims = (3,3))

omega = 0.5

u0 = 0.1 * CF((-2,0,0)) * sin(omega * pi*x)
s0 = 0.1 * sin(omega * pi*x) * CF((4,0,0,0,2,0,0,0,0), dims = (3,3))


ds0 = CF((s0[0,0].Diff(x) + s0[0,1].Diff(y) + s0[0,2].Diff(z),
          s0[1,0].Diff(x) + s0[1,1].Diff(y) + s0[1,2].Diff(z),
          s0[2,0].Diff(x) + s0[2,1].Diff(y) + s0[2,2].Diff(z)), dims=(3,1))



# fesi = H1LumpingFESpace(mesh, order=1)
fesi = H1LumpingFESpace(mesh, order=order)
# fesi = H1(mesh, order=1)
# fesi = Periodic(H1(mesh, order=order))
# fesi = NodalFESpace(mesh, order=order)

# gfu = GridFunction(fesi)

# Draw(gfu)
# for i in range(fesi.ndof):
#     print("dof", i)
#     gfu.vec[:] = 0
#     gfu.vec[i] = 1
#     Redraw()
#     input()

fesp = Periodic(fesi)

fes = fesp**3

localdofs = fes.GetDofs(mesh.Materials("small"))
print ("local dofs: ", localdofs.NumSet(),"/",len(localdofs))
Ps = Projector(localdofs, True)   # projection to small
Pl = Projector(localdofs, False)  # projection to large



u,v = fes.TnT()

def eps(u):
    return 0.5 * (grad(u) + grad(u).trans)

def tr(sigma):
    return sigma[0,0] + sigma[1,1] + sigma[2,2]

def C(sigma):
    return 2 * mu * sigma + lam * tr(sigma) * Id(3)


# points = [(0,0), (1,0), (0,1)]
# weights = [1/6, 1/6, 1/6]

points = [(0,0), (1,0), (0,1), (0.5,0), (0.5,0.5), (0,0.5), (1/3, 1/3)]
weights = [1/40, 1/40, 1/40, 1/15, 1/15, 1/15, 9/40]

n = specialcf.normal(mesh.dim)

# lumping = IntegrationRule( [(0,0),(1,0),(0,1)], [1/6, 1/6, 1/6])
# mform = u*v*dx(intrules = { TRIG: lumping })


mform = u*v*dx(intrules=fesi.GetIntegrationRules())
# aform = InnerProduct(grad(u),grad(v))*dx
aform = InnerProduct(C(eps(u)), grad(v))*dx



m = BilinearForm(mform, diagonal=True).Assemble()
# m = BilinearForm(mform).Assemble()
a = BilinearForm(aform).Assemble()
minv = m.mat.Inverse(fes.FreeDofs())  

mmat = m.mat
amat = a.mat
minva = minv@amat

APs = minv@(amat@Ps.CreateSparseMatrix()).DeleteZeroElements(1e-12)
APl = minva@Pl



gfu = GridFunction(fes)
gfu.Set(u0)

Draw(gfu, mesh, name="u")

def GetCeps(u):
    return C(0.5 * (grad(gfu) + grad(gfu).trans))



visoptions.scalfunction="u:1"

visoptions.vecfunction="None"

viewoptions.clipping.nx = 0
viewoptions.clipping.ny = 0
viewoptions.clipping.nz = -1

viewoptions.clipping.enable = 1
visoptions.clipsolution = 'scal'



# sleep (3)
unew = gfu.vec.CreateVector()
uold = gfu.vec.CreateVector()
uold.data = gfu.vec



gfsd = GridFunction(fes)
gfsd.Set(ds0)

gfu.vec.data = uold + tau * gfsd.vec 

unew = gfu.vec.CreateVector()
# uold = gfu.vec.CreateVector()
z = gfu.vec.CreateVector()
znew = gfu.vec.CreateVector()
zold = gfu.vec.CreateVector()
w = gfu.vec.CreateVector()
# uold.data = gfu.vec

substeps = 20

# input()

with TaskManager(): # pajetrace=10**8):
    for n in range(int(tend/tau)):
        w.data = APl * gfu.vec
        zold.data = gfu.vec
        z.data = zold - (tau/substeps)**2/2*(w+APs*zold)
        for m in range(1, substeps):
            znew.data = 2*z-zold
            znew.data -= (tau/substeps)**2*(w+APs*z)
            zold,z,znew = z,znew,zold
        unew.data = 2*z-uold
        uold.data = gfu.vec
        gfu.vec.data = unew.data
        if n % 1 == 0:
            Redraw()
            # input()
