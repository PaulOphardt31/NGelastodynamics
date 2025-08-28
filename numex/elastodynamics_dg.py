from ngsolve import *
from netgen.occ import *
from ngsolve.internal import visoptions, viewoptions

R = 5
r = 0.25

lam = 2
mu = 1

# v0 = peak * 0.1 * CF((2,0)) * sin(2*pi*x)
# s0 = peak * 0.1 * CF((-2,4,4,0), dims = (2,2)) * sin(2*pi*x)

v0 = 0.1 * CF((-2,0)) * sin(2*pi*x)
# s0 = 0.1 * CF((4,0,0,2), dims = (2,2)) * sin(2*pi*x)
s0_vec = 0.1 * CF((4,2,0)) * sin(2*pi*x)

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
# circ_inner.edges[0].maxh = 0.02

circ.edges[0].Identify(circ.edges[2], "top", IdentificationType.PERIODIC)

circ.edges[3].Identify(circ.edges[1], "right", IdentificationType.PERIODIC)
    

geom = circ - circ_inner

mesh = Mesh(OCCGeometry(geom, dim=2).GenerateMesh(maxh=0.05))
mesh.Curve(3)
Draw(mesh)

order = 2

fes = Periodic(L2(mesh, order=order))
# fes2 = Periodic(L2(mesh, order=order-1))

X = fes**3 * fes**2

# sigma_xx, sigma_yy, sigma_xy, ux, uy = X.TrialFunction()
# tau_xx, tau_yy, tau_xy, vx, vy = X.TestFunction()

sigma_vec, u = X.TrialFunction()
tau_vec, v = X.TestFunction()


sigma = CF(( sigma_vec[0], sigma_vec[2], sigma_vec[2], sigma_vec[1] ), dims = (2,2))
sigma_other = CF(( sigma_vec.Other()[0], sigma_vec.Other()[2], sigma_vec.Other()[2], sigma_vec.Other()[1] ), dims = (2,2))
tau = CF(( tau_vec[0], tau_vec[2], tau_vec[2], tau_vec[1] ), dims = (2,2))
tau_other = CF(( tau_vec.Other()[0], tau_vec.Other()[2], tau_vec.Other()[2], tau_vec.Other()[1] ), dims = (2,2))


# sigma = CF(( sigma_vec[0], sigma_vec[2], sigma_vec[3], sigma_vec[1] ), dims = (2,2))
# sigma_other = CF(( sigma_vec.Other()[0], sigma_vec.Other()[2], sigma_vec.Other()[3], sigma_vec.Other()[1] ), dims = (2,2))
# tau = CF(( tau_vec[0], tau_vec[2], tau_vec[3], tau_vec[1] ), dims = (2,2))
# tau_other = CF(( tau_vec.Other()[0], tau_vec.Other()[2], tau_vec.Other()[3], tau_vec.Other()[1] ), dims = (2,2))



# tau = CF(( (tau_xx, tau_xy), (tau_xy, tau_yy) ), dims = (2,2))

# u = CF((ux, uy))
# v = CF((vx, vy))

# grad_u = CF((grad(ux), grad(uy)), dims = (2,2))
# grad_v = CF((grad(vx), grad(vy)), dims = (2,2))

# u_other = CF((ux.Other(), uy.Other()))
# v_other = CF((vx.Other(), vy.Other()))



# u0 = exp (-damp * ( (x-Mx)*(x-Mx) + (y-My)*(y-My) ))

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


a = BilinearForm(X)
a += InnerProduct(sigma, grad(v)) * dx 
a += -0.5 * (v - v.Other()) * (sigma*n) * dx(element_boundary = True)

# a += - InnerProduct(v - v.Other(), 0.5 * (sigma + sigma_other) * n) * dx(skeleton=True)

a += - InnerProduct(tau, grad(u)) * dx()
a += 0.5 * (u - u.Other()) * (tau*n) * dx(element_boundary = True)

gfu = GridFunction(X)

w = gfu.vec.CreateVector()

gfu.components[0].Set(s0_vec)
gfu.components[1].Set(v0)


Draw (gfu.components[0], mesh, "sigma")
Draw (gfu.components[1], mesh, "u")

# Draw (gfu.components[2], mesh, "sxy")
# Draw (gfu.components[3], mesh, "ux")
# Draw (gfu.components[4], mesh, "uy")
# Draw (CF((gfu.components[3],gfu.components[4])), mesh, "u")

# visoptions.scalfunction="sigma:1"
visoptions.scalfunction="u:1"
visoptions.vecfunction="None"


# SetVisualization(min=-0.1, max=0.1, deformation=True)

# Cinv = Id(2)


m = BilinearForm(X) #, diag = True)
m += InnerProduct(u, v) * dx
m += InnerProduct(Cinv(sigma), tau) * dx
# m += InnerProduct(sigma, tau) * dx
m.Assemble()


Minv = m.mat.Inverse(X.FreeDofs(), inverse="sparsecholesky")

tau = 1e-4
tend = 1

t = 0

with TaskManager():
    while t < tend:
        print("t=", t)
        a.Apply (gfu.vec, w) 
        # print(Norm(wS))
        gfu.vec.data -= tau * Minv * w

        
        t += tau
        Redraw()
        # input()








