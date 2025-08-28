from ngsolve import *
from netgen.occ import *
from ngsolve.internal import visoptions, viewoptions

R = 5
r = 0.25

Mx = 2 #(R - r)/2
My = 0
damp = 2

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
circ_inner.edges[0].maxh = 0.01

circ.edges[0].Identify(circ.edges[2], "top", IdentificationType.PERIODIC)

circ.edges[3].Identify(circ.edges[1], "right", IdentificationType.PERIODIC)
    

geom = circ - circ_inner

mesh = Mesh(OCCGeometry(geom, dim=2).GenerateMesh(maxh=0.1))
mesh.Curve(3)
Draw(mesh)

order = 2
# S = Periodic(HDivDiv(mesh, order=order-1, dirichlet="inner", plus = True))
# V = Periodic(HCurl(mesh, order=order)) #, type1 = True))

S = Periodic(HDivDiv(mesh, order=order, dirichlet="inner")) #, plus = True))
V = Periodic(HCurl(mesh, order=order)) #, type1 = True))



v, dv = V.TnT()
sigma, dsigma = S.TnT()

peak = 1 #IfPos(x+R/2-1,0,1)# exp(-damp * ( (x-Mx)*(x-Mx) + (y-My)*(y-My) ))


v0 = peak * 0.1 * CF((-2,0)) * sin(2*pi*x)
s0 = peak * 0.1 * CF((4,0,0,2), dims = (2,2)) * sin(2*pi*x)


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

bS = BilinearForm(S)
bS += InnerProduct(sigma*n, dsigma*n) * ds(skeleton = True, definedon=mesh.Boundaries("inner"))
bS.Assemble()

# C = Id(2)
def tr(sigma):
    return sigma[0,0] + sigma[1,1]

def Cinv(sigma):
    return 1/(2 * mu) * (sigma - lam/(2*mu + 2*lam) * tr(sigma) * Id(2))
    

aS = BilinearForm(trialspace = V, testspace = S)
# grad since C = Id and dsigma is symmetric
aS += -  InnerProduct(grad(v), dsigma)  * dx
aS += (dsigma*n)*n * v*n * dx(element_boundary = True)



gfv = GridFunction(V)
gfs = GridFunction(S)

gfv.Set(v0)
gfs.Set(s0)

for i in range(S.ndof):
    if S.FreeDofs()[i] == 0:
        gfs.vec[i] = 0

resV = gfv.vec.CreateVector()
wV = gfv.vec.CreateVector()

resS = gfs.vec.CreateVector()
wS = gfs.vec.CreateVector()

# Draw (u.components[1], mesh, "ux")
# Draw (u.components[2], mesh, "uy")
Draw (gfv, mesh, "gfv")
Draw (gfs, mesh, "gfs")
visoptions.scalfunction="gfs:1"
# visoptions.vecfunction="None"

# nn = CF((x,y)) * 1/(sqrt(x*x + y*y))
# Draw(gfs*nn, mesh, "bnd")

# SetVisualization(min=-0.2, max=0.2, deformation=True)


mV = BilinearForm(V, symmetric=True)
mV += InnerProduct(v, dv) * dx
mV.Assemble()

MinvV = mV.mat.Inverse(V.FreeDofs(), inverse="sparsecholesky")

mS = BilinearForm(S, symmetric=True)
mS += InnerProduct(Cinv(sigma), dsigma) * dx
mS.Assemble()

MinvS = mS.mat.Inverse(S.FreeDofs(), inverse="sparsecholesky")

tau = 0.001
tend = 1

t = 0
# nd = fes1.ndof

input ("<press enter>")

P = Projector(S.FreeDofs(), True)


with TaskManager():
    while t < tend:
        print("t=", t)
        aS.Apply (gfv.vec, wS) 
        # resS.data = bS.mat * gfs.vec  # project residual
        # resS.data += wS
        # print(Norm(wS))
        gfs.vec.data -= tau * MinvS * wS
        # gfs.vec.data -= tau * MinvS * resS

        aV.Apply (gfs.vec, wV)    
        # resV.data = bV.mat * gfv.vec   
        # print(Norm(resV))
        # resV.data += wV        
        gfv.vec.data -= tau * MinvV * wV

        t += tau
        Redraw()
        # input()








