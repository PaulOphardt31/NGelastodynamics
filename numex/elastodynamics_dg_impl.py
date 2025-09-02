from ngsolve import *
from netgen.occ import *
from ngsolve.internal import visoptions, viewoptions

R = 5
r = 0.25

lam = 2
mu = 1

ss = 100
Mx = -2.5/2
My = 0
peak = 1 #2 * exp (-ss * ( (x-Mx)*(x-Mx) + (y-My)*(y-My) ))


# v0 = peak * 0.1 * CF((2,0)) * sin(2*pi*x)
# s0 = peak * 0.1 * CF((-2,4,4,0), dims = (2,2)) * sin(2*pi*x)

v0 = peak * 0.1 * CF((-2,0)) * sin(2*pi*x)
s0_vec = peak * 0.1 * CF((4,2,0)) * sin(2*pi*x)

# s0_vec = CF((0,0,0,0))

# s0_vec = CF((v0[0].Diff(x), v0[0].Diff(y), v0[1].Diff(x), v0[1].Diff(y)), dims = (2,2))
# s0_vec = 0.1 * CF((4,2,0,0)) * sin(2*pi*x)
# s0_vec = 0.1 * CF((1,0,0,0)) # * sin(2*pi*x)


wp = WorkPlane()
circ = wp.RectangleC(R,R).Face() 
circ.edges[0].name = "bottom"
circ.edges[1].name = "right"
circ.edges[2].name = "top"
circ.edges[3].name = "left"

circ_inner = wp.Circle(0,0,r).Face()
circ_inner.edges[0].name = "inner"
circ_inner.edges[0].maxh = 0.02

circ.edges[0].Identify(circ.edges[2], "top", IdentificationType.PERIODIC)

circ.edges[3].Identify(circ.edges[1], "right", IdentificationType.PERIODIC)
    

geom = circ - circ_inner

mesh = Mesh(OCCGeometry(geom, dim=2).GenerateMesh(maxh=0.2))
mesh.Curve(3)
Draw(mesh)

# Draw(v0[0], mesh, "v0")
# input()


order = 3

fes = Periodic(L2(mesh, order=order, dgjumps=True))
# fes2 = Periodic(L2(mesh, order=order-1))

X = fes**3 * fes**2


sigma_vec, u = X.TrialFunction()
tau_vec, v = X.TestFunction()


sigma = CF(( sigma_vec[0], sigma_vec[2], sigma_vec[2], sigma_vec[1] ), dims = (2,2))
sigma_other = CF(( sigma_vec.Other()[0], sigma_vec.Other()[2], sigma_vec.Other()[2], sigma_vec.Other()[1] ), dims = (2,2))
tau = CF(( tau_vec[0], tau_vec[2], tau_vec[2], tau_vec[1] ), dims = (2,2))
tau_other = CF(( tau_vec.Other()[0], tau_vec.Other()[2], tau_vec.Other()[2], tau_vec.Other()[1] ), dims = (2,2))


n = specialcf.normal(2)

def eps(u):
    return 0.5 * (grad(u) + grad(u).trans)

def tr(sigma):
    return sigma[0,0] + sigma[1,1]

def Cinv(sigma):
    # return sigma
    return 1/(2 * mu) * (sigma - lam/(2*mu + 2*lam) * tr(sigma) * Id(2))

def C(sigma):
    return 2 * mu * sigma + lam * tr(sigma) * Id(2)

dt = 1e-3

gfu = GridFunction(X)
# gfu_old = GridFunction(X)

# sigma_old = sigma = CF(( sigma_vec[0], sigma_vec[2], sigma_vec[3], sigma_vec[1] ), dims = (2,2))


aa = InnerProduct(sigma, Grad(v)) * dx 
# aa += -0.5 * (v - v.Other()) * (sigma*n) * dx(element_boundary = True)
aa += -0.5 * (v - v.Other(bnd = CF((0,0)))) * (sigma*n) * dx(element_boundary = True)
aa += 0.5 * (v) * (sigma*n) * ds(skeleton = True, definedon=mesh.Boundaries("inner"))

# aa += - InnerProduct(v - v.Other(), 0.5 * (sigma + sigma_other) * n) * dx(skeleton=True)

aa += InnerProduct(tau, Grad(u)) * dx()
# aa += -0.5 * (u - u.Other()) * (tau*n) * dx(element_boundary = True)
aa += -0.5 * (u - u.Other(bnd = CF((0,0)))) * (tau*n) * dx(element_boundary = True)
aa += 0.5 * (u) * (tau*n) * ds(skeleton = True, definedon=mesh.Boundaries("inner"))

mm = -InnerProduct(Cinv(sigma),tau) * dx()
mm += u * v * dx()


mstar = BilinearForm(X)
mstar += dt * aa 
mstar += mm

mstar.Assemble()
mstarinv = mstar.mat.Inverse(X.FreeDofs(), inverse="sparsecholesky")

a = BilinearForm(X)
a += aa 
a.Assemble()

w = gfu.vec.CreateVector()

gfu.components[0].Set(s0_vec)
gfu.components[1].Set(v0)


Draw (gfu.components[0], mesh, "sigma")
Draw (gfu.components[1], mesh, "u")

# Draw (gfu.components[2], mesh, "sxy")
# Draw (gfu.components[3], mesh, "ux")
# Draw (gfu.components[4], mesh, "uy")
# Draw (CF((gfu.components[3],gfu.components[4])), mesh, "u")

visoptions.scalfunction="sigma:1"
# visoptions.scalfunction="u:1"
visoptions.vecfunction="None"


# SetVisualization(min=-0.1, max=0.1)

# Cinv = Id(2)


# m = BilinearForm(X) #, diag = True)
# m += InnerProduct(u, v) * dx
# m += InnerProduct(Cinv(sigma), tau) * dx
# # m += InnerProduct(sigma, tau) * dx
# m.Assemble()


# Minv = m.mat.Inverse(X.FreeDofs(), inverse="sparsecholesky")


tend = 1

t = 0

with TaskManager():
    while t < tend:
        print("t=", t)
        w.data = dt * a.mat * gfu.vec
        gfu.vec.data -= mstarinv * w

        # a.Apply (gfu.vec, w) 
        # print(Norm(wS))
        # gfu.vec.data -= tau * Minv * w

        
        t += dt
        Redraw()
        # input()








