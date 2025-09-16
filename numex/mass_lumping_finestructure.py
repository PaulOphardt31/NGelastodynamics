from netgen.occ import *
from ngsolve import *
import netgen.geom2d as geom2d

rect = MoveTo(-1,-1).Rectangle(2.5,2).Face()
hole = MoveTo(0.5,0.01).Rectangle(0.001,0.8).Face() + \
    MoveTo(0.5,-0.81).Rectangle(0.001,0.8).Face()
local = Circle((0.5,0.8),0.1).Face() + \
    Circle((0.5,0),0.1).Face() + \
    Circle((0.5,-0.8),0.1).Face()
large = rect-local-hole
small = local-hole
large.faces.name="large"
small.faces.name="small"

shape = Glue ([large,small])
geo = OCCGeometry(shape, dim=2)
mesh = Mesh(geo.GenerateMesh(maxh=0.02, grading=0.5))
Draw (mesh);

tau = 0.01
tend = 1
u0 = exp(-10**2*( x**2 + y**2))
v0 = 0
substeps = 20

fes = H1(mesh, order=1)
u,v = fes.TnT()

localdofs = fes.GetDofs(mesh.Materials("small"))
print ("local dofs: ", localdofs.NumSet(),"/",len(localdofs))
Ps = Projector(localdofs, True)   # projection to small
Pl = Projector(localdofs, False)  # projection to large

lumping = IntegrationRule( [(0,0),(1,0),(0,1)], [1/6, 1/6, 1/6])
mform = u*v*dx(intrules = { TRIG: lumping })
aform = grad(u)*grad(v)*dx

mmat = BilinearForm(mform).Assemble().mat
amat = BilinearForm(aform).Assemble().mat
minva = mmat.CreateSmoother()@amat
# APs = minva@Ps    # operator composition
APs = mmat.CreateSmoother()@(amat@Ps.CreateSparseMatrix()).DeleteZeroElements(1e-12)
APl = minva@Pl

gfu = GridFunction(fes)
gfu.Set(u0)

scene = Draw(gfu, order=1, deformation=True, settings = { "subdivision" : 1, "Objects" : { "Wireframe" : False}})
#sleep (3)
unew = gfu.vec.CreateVector()
uold = gfu.vec.CreateVector()
z = gfu.vec.CreateVector()
znew = gfu.vec.CreateVector()
zold = gfu.vec.CreateVector()
w = gfu.vec.CreateVector()
uold.data = gfu.vec

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