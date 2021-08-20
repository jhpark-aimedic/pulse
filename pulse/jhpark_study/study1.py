from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
# Set pressure function:

T = 10.0 # tension
A = 1.0 # pressure amplitude
R = 0.3 # radius of domain
theta = 0.2
x0 = 0.6*R*cos(theta)
y0 = 0.6*R*sin(theta)
sigma = 0.025
# sigma = 50 # large value for verification
n = 40 # approx no of elements in radial direction
# mesh = UnitCircle(n)
elementDegree = 1 # It means linear element
gDim = 3 # It means 2D mesh
mesh = UnitDiscMesh.create(MPI.comm_world, n, elementDegree, gDim) 
V = FunctionSpace(mesh, "Lagrange", 1)

# Define boundary condition w=0
def boundary(x, on_boundary):
    return on_boundary
bc = DirichletBC(V, Constant(0.0), boundary)

# Define variational problem
w = TrialFunction(V)
v = TestFunction(V)
a = inner(nabla_grad(w), nabla_grad(v))*dx
f = Expression("4*exp(-0.5*(pow((R*x[0] - x0)/sigma, 2)) "\
                " - 0.5*(pow((R*x[1] - y0)/sigma, 2)))",
                R=R, x0=x0, y0=y0, sigma=sigma, degree=1)
L = f*v*dx
# Compute solution
w = Function(V)
problem = LinearVariationalProblem(a, L, w, bc)
solver = LinearVariationalSolver(problem)
solver.parameters["linear_solver"] = "cg"
solver.parameters["preconditioner"] = "ilu"
solver.solve()
# Plot scaled solution, mesh and pressure
plt.figure(1)
plot(mesh, title="Mesh over scaled domain")
plt.figure(2)
plot(w, title="Scaled deflection")
f = interpolate(f, V)
plt.figure(3)
plot(f, title="Scaled pressure")
# Find maximum real deflection
# max_w = w.vector().array().max()
max_w = w.vector().get_local().max()
max_D = A*max_w/(8*pi*sigma*T)
print ("Maximum real deflection is", max_D)

# Verification for "flat" pressure (large sigma)
if sigma >= 50:
    w_exact = Expression("1 - x[0]*x[0] - x[1]*x[1]", degree=1)
    w_e = interpolate(w_exact, V)
    # dev = np.abs(w_e.vector().array() - w.vector().array()).max()
    dev = np.abs(w_e.vector().get_local() - w.vector().get_local()).max()
    # print("sigma=%g: max deviation=%e" % dev)
    print("sigma:", sigma, " max deviation=", dev)
# Should be at the end
plt.show()

