from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as s

eps = 0.3
g = 0.0
w0 = 0.8
dt = 1e-5
gamma = 1.0

file = File('./test.pvd')

eps = 0.3
g = 0.0
w0 = 0.8
dt = 0.01
gamma = 1.0

img = plt.imread("./img/newton.jpeg")/256
(Nx, Ny) = img.shape
print(img)
L = 100
x = np.linspace(0, L, Nx)
y = np.linspace(0, L, Ny)
f = s.interp2d(x, y, img)

mesh = RectangleMesh(Point((0, 0)), Point(L, L), Nx-1, Ny-1)

class Field(UserExpression):
    def eval(self, value, x):
        value[0] = f(x[0], x[1])

    def value_shape(self):
        return ()

X = FunctionSpace(mesh, 'CG', 1)
rho = Function(X)
field = Field()
rho.interpolate(field)

Vh = FiniteElement('CG', mesh.ufl_cell(), 2)
ME = FunctionSpace(mesh, Vh*Vh)
X = VectorFunctionSpace(mesh, 'CG', 1)
Y = FunctionSpace(mesh, 'CG', 1)
R = FunctionSpace(refine(mesh), 'CG', 1)

class GaussianRandomField(UserExpression):
    def eval(self, val, x):
        val[0] = np.sqrt(0.001)*np.random.randn()
        val[1] = np.sqrt(0.001)*np.random.randn()
    def value_shape(self):
        return (2,)

class VectorField(UserExpression):
    def eval(self, val, x):
        val[0] = 0
        val[1] = 1
    def value_shape(self):
        return (2,)

class Problem(NonlinearProblem):
    def __init__(self, a, L):
        NonlinearProblem.__init__(self)
        self.L = L
        self.a = a
    def F(self, b, x):
        assemble(self.L, tensor=b)
    def J(self, A, x):
        assemble(self.a, tensor=A)

class CustomSolver(NewtonSolver):
    def __init__(self):
        NewtonSolver.__init__(self, MPI.comm_world, PETScKrylovSolver(), PETScFactory.instance())
    def solver_setup(self, A, P, problem, iteration):
        self.linear_solver().set_operator(A)
        PETScOptions.set("ksp_type", "gmres")
        PETScOptions.set("ksp_monitor")
        PETScOptions.set("pc_type", "ilu")
        self.linear_solver().set_from_options()

def G1(w, v):
    return -eps/2 - g/3*(w+v) + 1/4*(w**2+w*v+v**2)

def G2(w):
    return -eps/2*w - g/3*w**2 + 1/4*w**3

def A(w, v, k):
    return (dot(grad(w), grad(v)) - k**2*w*v)*dx

def B(w, v, theta, gamma):
    D = outer(theta, theta)
    return 2*gamma**2*dot(grad(w), dot(D, grad(v)))*dx

theta = Function(X)
vectorField = VectorField()
theta.interpolate(vectorField)
k = sqrt((np.pi*rho/w0)**2 - rho**2*gamma**2)

Uh = Function(ME)
Uh_0 = Function(ME)
U = TrialFunction(ME)
phi, psi = TestFunctions(ME)

initial = GaussianRandomField()
Uh.interpolate(initial)
Uh_0.interpolate(initial)

uh, qh = split(Uh)
uh_0, qh_0 = split(Uh_0)

qh_mid = 0.5*qh + 0.5*qh_0
dPhi = G1(uh, uh_0)*uh + G2(uh_0)

L0 = (uh-uh_0)*phi*dx + dt*A(qh_mid, phi, k) - dt*B(uh, phi, theta, gamma) + dt*dPhi*phi*dx
L1 = qh*psi*dx - A(uh, psi, k)

L = L0 + L1
a = derivative(L, Uh, U)

SH_problem = Problem(a, L)
solver = CustomSolver()

t = 0
T = 5

file = File('./result/newton.pvd')
while (t < T):
    print('time: {}'.format(t))
    t += dt
    Uh_0.vector()[:] = Uh.vector()
    solver.solve(SH_problem, Uh.vector())
    sol_c = project(Uh.split()[0], Y)
    sol_r = Function(R)
    LagrangeInterpolator.interpolate(sol_r, sol_c)
    sol_r.rename('field', 'label')
    file << (sol_r, t)