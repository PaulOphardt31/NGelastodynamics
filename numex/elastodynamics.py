from ngsolve import *
from netgen.occ import *
from ngsolve.internal import visoptions, viewoptions

R = 2
r = 0.1

Mx = 0.5 #(R - r)/2
My = 0
damp = 100

wp = WorkPlane()
circ = wp.Circle(0,0,R).Face() 
circ.edges[0].name = "outer"

circ_inner = wp.Circle(0,0,r).Face()
circ_inner.edges[0].name = "inner"

geom = circ - circ_inner

mesh = Mesh(OCCGeometry(geom, dim=2).GenerateMesh(maxh=0.1))
mesh.Curve(3)
Draw(mesh)

# print(mesh.GetBoundaries())

order=4

order = 3
S = HDivDiv(mesh, order=order-1, dirichlet="bottom|right|top", plus = True)
V = HCurl(mesh, order=order, dirichlet="left", type1 = True)

# X = V*Q


# fes1 = L2(mesh, order=order)
# fes = fes1*fes1*fes1

v, dv = V.TnT()
sigma, dsigma = S.TnT()


# p,ux,uy = fes.TrialFunction()
# q,vx,vy = fes.TestFunction()

u0 = exp (-damp * ( (x-Mx)*(x-Mx) + (y-My)*(y-My) ))

n = specialcf.normal(2)

def tang(u): return u-(u*n)*n

def eps(u):
    return 0.5 * (grad(u) + grad(u).trans)

# a1 = BilinearForm(fes)
# a1 += grad(p) * v * dx
# a1 += -0.5 * (p - p.Other()) * (v*n) * dx(element_boundary = True)

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


gfv = GridFunction(V)
gfs = GridFunction(S)

gfv.Set(CF((u0,u0)))

resV = gfv.vec.CreateVector()
wV = gfv.vec.CreateVector()

# resS = gfs.vec.CreateVector()
wS = gfs.vec.CreateVector()

# Draw (u.components[1], mesh, "ux")
# Draw (u.components[2], mesh, "uy")
Draw (gfv, mesh, "gfv")
Draw (gfs, mesh, "gfs")
visoptions.scalfunction="gfv:2"
visoptions.vecfunction="None"


SetVisualization(min=-0.1, max=0.1, deformation=True)


mV = BilinearForm(V, symmetric=True)
mV += InnerProduct(v, dv) * dx
mV.Assemble()

MinvV = mV.mat.Inverse(V.FreeDofs(), inverse="umfpack")

mS = BilinearForm(S, symmetric=True)
mS += InnerProduct(sigma, dsigma) * dx
mS.Assemble()

MinvS = mS.mat.Inverse(S.FreeDofs(), inverse="umfpack")

tau = 1e-3
tend = 10

t = 0
# nd = fes1.ndof

input ("<press enter>")

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

        t += tau
        Redraw()
        # input()








