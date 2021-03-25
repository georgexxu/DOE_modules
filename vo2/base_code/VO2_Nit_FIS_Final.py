'''
Note: VO2 phase transition simulation 
Adaptive time stepping based on error tolerance and absolute error;
Fully implicit scheme; 
Nitsche's trick for special BCs;
P1 element for variables
Result: this version leads to correct phase transition
Newton solver, linear solver: direct parallel solver mumps

Last modified by Xiaofeng Xu 2021/03/24
1. Fix try ... except bug and refactor the code. Potential issue: if tolerance too big, may need fix later
2. Refactor records: - change solve4dt, move parameters setting before time evolution
                  - delete computation of relative error
                  - Comment out n = FacetNormal(mesh)

'''


from __future__ import print_function
from fenics import *
import math
import numpy as np
# from numpy.random import rand
import time

list_linear_algebra_backends()
print(parameters["linear_algebra_backend"])
list_linear_solver_methods()

start_time = time.time()

rank = MPI.rank(MPI.comm_world)

#===============================================
# Step 1.   Define parameters:
#           Geometry and simulation time, time step 
#           Adaptive time paramteres 
#           Newton solver parameters 
#           filenames
#           Other paramters, e.g. Nitsche's trick 
#===============================================
Lx = 60.0
Ly = 20.0
nx = 96   # 96
ny = 32   # 32
thickness = 100.0

tf = 5000         # Final time in nanoseconds  5000.0
dt = 1E-6       # Initial time step
save_frequency = 5
alpha = 1e-6 #epsilon = alpha * h

#adaptive time stepping parameters
tol_tstep = 0.1  # Goal relative error for time-stepping
r_t_max = 5.0
r_t_min = 0.2
s_t = 0.9

#newton solver parameters
rel_par = 0.9
max_iter = 30
kr_solver_iter = 2000
newton_rel_tol = 1E-2

#filenames
fig_file = 'mumps_Nit_tol1'+str(tol_tstep)+'/'
time_steps_file1 = 'mumps_Nit_eps1_record'+str(tol_tstep)+'.txt'
time_steps_file2 = 'mumps_Nit_eps1_time_step'+str(tol_tstep)+'.txt'
#not tuned parameters
num_steps = 1000     # Number of time steps
dt_out = tf / num_steps  # Time interval for output

tol = 1E-14   # T


#===============================================
# Step 2.   Define physical parameters
#           Or PDE paramters
#           Initial conditions
#===============================================
UCVOL = 0.059
TC = 1.0
T1 = 0.81361
AN1 = 34.867
AN2 = (-10.561 + 2.0547)/2
AN3 = (5.6028 + 71.008)/4
T2 = 0.79882
AU1 = 66.828
AU2 = (23.181 - 62.358)/2
AU3 = (6.7797 + 33.898)/4
GNU1 = 5.0847
GNU2 = (3.3898 - 25.424)/2 + 1.2712
GNU3 = (0.84746 + 3*11.299)/2
LN = 154.05
KAPPAN = 34.333
LU = 924.01
KAPPAU = 34.333
CHI = 19.954
NC = 15.727
NV = 15.727
ECHARGE = 34.333
KB = 1.0
MEA = 25000.0
MEC = 50000.0
MHA = 20833.0
MHC = 41667.0
K0 = 0.068432
EPSILON = 113.84
CPV = 216.9
THETA = 434580.0
HTRAN = 48.528
# ETA_IN = 0
# MU_IN = 0
# ETA_IN = 1.0
# MU_IN = -1.0
# Lian changed
ETA_IN = 0.791296*math.sqrt(2)    # 0.791296 for 300 K
MU_IN = -0.914352*math.sqrt(2)    # -0.914352 for 300 K
CHP_IN = 0.0   # Intrinsic chemical potential

#----------------------------------------------------------------
# Define initial and boundary values
#----------------------------------------------------------------
Ts = 300 / 338.0
#Lian change
#Ts = 300 / 338.0             # K / [Chosen temperature unit]
delV = 2.0                   # V
Rresistor = 9.0E4 / 2.1429E11  # Omega / [Chosen resistance unit]

eta_i = ETA_IN
mu_i = MU_IN
gamma_ei = (-CHI*mu_i**2/2 + CHP_IN)/(KB*Ts)
gamma_hi = (-CHI*mu_i**2/2 - CHP_IN)/(KB*Ts)

#-----------------------------------------------------------
# Define expressions used in variational forms
#-----------------------------------------------------------
thickness = Constant(thickness)
delt = Constant(dt)
UCVOL = Constant(UCVOL)
TC = Constant(TC)
T1 = Constant(T1)
AN1 = Constant(AN1)
AN2 = Constant(AN2)
AN3 = Constant(AN3)
T2 = Constant(T2)
AU1 = Constant(AU1)
AU2 = Constant(AU2)
AU3 = Constant(AU3)
GNU1 = Constant(GNU1)
GNU2 = Constant(GNU2)
GNU3 = Constant(GNU3)
LN = Constant(LN)
KAPPAN = Constant(KAPPAN)
LU = Constant(LU)
KAPPAU = Constant(KAPPAU)
CHI = Constant(CHI)
NC = Constant(NC)
NV = Constant(NV)
ECHARGE = Constant(ECHARGE)
KB = Constant(KB)
MEA = Constant(MEA)
MEC = Constant(MEC)
MHA = Constant(MHA)
MHC = Constant(MHC)
K0 = Constant(K0)
EPSILON = Constant(EPSILON)
CPV = Constant(CPV)
THETA = Constant(THETA)
HTRAN = Constant(HTRAN)
ETA_IN = Constant(ETA_IN)
MU_IN = Constant(MU_IN)
CHP_IN = Constant(CHP_IN)

# Yin: Seems using 'Constant()' is important for achieving convergence #Xu: I am not sure about this claim
phi_i = Expression('(delV - delV*x[1]/Ly)', degree=1, delV=delV, Ly=Ly)
Ts = Constant(Ts)
delV = Constant(delV)
Rresistor = Constant(Rresistor)
eta_i = Constant(eta_i)
mu_i = Constant(mu_i)
gamma_ei = Constant(gamma_ei)
gamma_hi = Constant(gamma_hi)

# Set a small region of impurity where the transition temperature is lower than pristine case
Tcimp = Expression('TCIMP0*(tanh(2*(pow(x[0] - Lx/2, 2) + pow(x[1], 2) - pow(RIMPDIS, 2))/pow(DWWIDTH, 2)) - 1)/2', \
                  degree=1, Lx=Lx, TCIMP0=20.0/338.0, RIMPDIS=3.0, DWWIDTH=10.0)

#===============================================
# Step 3.   Generate mesh
#           Define finite elements and function spaces
#           Define physical variables, test functions
#           Initilization if needed 
#           Define boundary conditions, Dirichelet on Function space 
#           Define Weak forms
#===============================================
mesh = RectangleMesh(Point(0, 0), Point(Lx, Ly), nx, ny)

#--------------------------------------------------------------
# 3.1 Define function space and functions for multiple variables
#--------------------------------------------------------------
P1 = FiniteElement('Lagrange', triangle, 1)
R0 = FiniteElement('R', triangle, 0)
V1 = FunctionSpace(mesh, P1)
VR = FunctionSpace(mesh, R0)

n_f = 9  # Number of functions that need to be solved
# First 6 are for unknown variables and last one is for an auxiliary variable
element = MixedElement([P1,P1,P1,P1,P1,P1,R0])
V = FunctionSpace(mesh, element)
# Create function space for the 5 previous-solution functions that need to be updated
element2 = MixedElement([P1]*5)
V_n = FunctionSpace(mesh, element2)
# Define test functions
v_1, v_2, v_3, v_4, v_5, v_6, v_10 = TestFunctions(V)

# Define trial functions
du = TrialFunction(V)   # Newton iteration step

# Define functions
u = Function(V)     # The most recently computed solution
u_n = Function(V_n)  # The previous solution

# Split system functions to access components
eta, mu, gamma_e, gamma_h, phi, T, integral_phi = split(u)
eta_n, mu_n, gamma_en, gamma_hn, T_n = split(u_n)

#------------------------------------------------------
# 3.2 Assign initial values to the functions u_n
#------------------------------------------------------
eta_0 = project(eta_i, V1)
mu_0 = project(mu_i, V1)
gamma_e0 = project(gamma_ei, V1)
gamma_h0 = project(gamma_hi, V1)
phi_0 = project(phi_i, V1)
T_0 = project(Ts, V1)
lam_10 = project(Constant(0), V1)
lam_20 = project(Constant(0), V1)
lam_30 = project(Constant(0), V1)
integral_phi_0 = project(Constant(0), VR)

# lam_1n = project(Expression('x[1]>=tol && Ly-x[1]>=tol ? 0.0 : 0.1', \
#                             degree=1, tol=tol, Ly=Ly), V1)
# lam_2n = project(Expression('x[1]>=tol && Ly-x[1]>=tol ? 0.0 : 0.1', \
#                             degree=1, tol=tol, Ly=Ly), V1)
# lam_3n = project(Expression('x[1]>=tol ? 0.0 : 0.1', \
#                             degree=1, tol=tol), V1)

fa0 = FunctionAssigner(V_n, [V1]*5)
fa0.assign(u_n, [eta_0, mu_0, gamma_e0, gamma_h0, T_0])

#---------------------------------------------------------
# Assign initial guess to the functions u
#---------------------------------------------------------
fa = FunctionAssigner(V, [V1,V1,V1,V1,V1,V1,VR])
fa.assign(u, [eta_0, mu_0, gamma_e0, gamma_h0, phi_0, T_0, integral_phi_0])
# u.vector().set_local(rand(u.vector().size()))  # Set the initial guess to random fields
# u.vector().apply("")

#-----------------------------------------------------------------
# 3.3 Mark boundaries and define normal boundary conditions
#-----------------------------------------------------------------
boundary_markers = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
boundary_markers.set_all(9)
# n = FacetNormal(mesh)

class BoundaryX0(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0, tol)
bx0 = BoundaryX0()
bx0.mark(boundary_markers, 0)   # 0 marks y = 0 boundary

class BoundaryX1(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], Ly, tol)
bx1 = BoundaryX1()
bx1.mark(boundary_markers, 1)   # 1 marks y = Ly boundary

ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)  # Measure ds

bc_phi_1 = DirichletBC(V.sub(4), Constant(0), bx1)  # Dirichlet boundary condition for phi

def domain_lam_1_2(x):
    return x[1] > tol and x[1] < Ly - tol

def domain_lam_3(x):
    return x[1] > tol

bcs = [bc_phi_1] # Nitsche used for gamma_e,gamma_h

#-------------------------------------------------------------------
# 3.4 Define python functions used in the weak form  
#-------------------------------------------------------------------
def dfb_deta(T, eta, mu):
    return AN1*(T - T1 - Tcimp)/TC*eta + AN2*eta**3 + AN3*eta**5 + GNU1*mu \
    - GNU2*eta*mu**2 + 1.5*GNU3*eta**2*mu

def dfb_dmu(T, eta, mu):
    return AU1*(T - T2 - Tcimp)/TC*mu + AU2*mu**3 + AU3*mu**5 + GNU1*eta \
    - GNU2*eta**2*mu + 0.5*GNU3*eta**3

def Fermi(gamma):
    return 1 / (exp(-gamma) + 3*sqrt(DOLFIN_PI)/4*(4 + gamma**2)**(-0.75))

def dFermi(gamma):   # Derivative of Fermi(gamma)
    return Fermi(gamma)**2 * (exp(-gamma) + 9*sqrt(DOLFIN_PI)/8 \
    *(4 + gamma**2)**(-1.75)*gamma)

def Fermi_b(gamma):
    return conditional(lt(gamma, -100), 0, conditional(gt(gamma, 100), \
                4/(3*sqrt(DOLFIN_PI))*(4 + gamma**2)**0.75, Fermi(gamma)))

def dFermi_b(gamma):
    return conditional(lt(gamma, -100), 0, conditional(gt(gamma, 100), \
                2/sqrt(DOLFIN_PI)*gamma*(4 + gamma**2)**(-0.25), dFermi(gamma)))

# n = FacetNormal(mesh)

#----------------------------------------------------------------
# Define intermediate variables for the variational problem
#----------------------------------------------------------------
nd_e = NC*Fermi_b(gamma_e)    # Electron density
nd_h = NV*Fermi_b(gamma_h)    # Hole density

nd_en = NC*Fermi_b(gamma_en)    # Electron density
nd_hn = NV*Fermi_b(gamma_hn)    # Hole density

j_ex = -nd_e*MEA/ECHARGE*(KB*T*gamma_e.dx(0) + gamma_e*KB*T.dx(0) \
                          + CHI*mu*mu.dx(0) - ECHARGE*phi.dx(0))
j_ey = -nd_e*MEC/ECHARGE*(KB*T*gamma_e.dx(1) + gamma_e*KB*T.dx(1) \
                          + CHI*mu*mu.dx(1) - ECHARGE*phi.dx(1))
j_e = as_vector([j_ex, j_ey])   # Electron flux

j_exn = -nd_en*MEA/ECHARGE*(KB*T_n*gamma_en.dx(0) + gamma_en*KB*T_n.dx(0) \
                          + CHI*mu_n*mu_n.dx(0) - ECHARGE*phi.dx(0))
j_eyn = -nd_en*MEC/ECHARGE*(KB*T_n*gamma_en.dx(1) + gamma_en*KB*T_n.dx(1) \
                          + CHI*mu_n*mu_n.dx(1) - ECHARGE*phi.dx(1))
j_en = as_vector([j_exn, j_eyn])   # Electron flux



j_hx = -nd_h*MHA/ECHARGE*(KB*T*gamma_h.dx(0) + gamma_h*KB*T.dx(0) \
                          + CHI*mu*mu.dx(0) + ECHARGE*phi.dx(0))
j_hy = -nd_h*MHC/ECHARGE*(KB*T*gamma_h.dx(1) + gamma_h*KB*T.dx(1) \
                          + CHI*mu*mu.dx(1) + ECHARGE*phi.dx(1))
j_h = as_vector([j_hx, j_hy])   # Hole flux

j_hxn = -nd_hn*MHA/ECHARGE*(KB*T_n*gamma_hn.dx(0) + gamma_hn*KB*T_n.dx(0) \
                          + CHI*mu_n*mu_n.dx(0) + ECHARGE*phi.dx(0))
j_hyn = -nd_hn*MHC/ECHARGE*(KB*T_n*gamma_hn.dx(1) + gamma_hn*KB*T_n.dx(1) \
                          + CHI*mu_n*mu_n.dx(1) + ECHARGE*phi.dx(1))
j_hn = as_vector([j_hxn, j_hyn])   # Hole flux

nd_in = NC*Fermi_b((-CHI*mu**2/2 + CHP_IN)/(KB*T))   # Intrinsic carrier density
nd_eeq = NC*Fermi_b((-CHI*mu**2/2 + ECHARGE*phi + CHP_IN)/(KB*T))  # Equilibrium electron density
nd_heq = NV*Fermi_b((-CHI*mu**2/2 - ECHARGE*phi - CHP_IN)/(KB*T))  # Equilibrium hole density

nd_inn = NC*Fermi_b((-CHI*mu_n**2/2 + CHP_IN)/(KB*T_n))   # Intrinsic carrier density
nd_eeqn = NC*Fermi_b((-CHI*mu_n**2/2 + ECHARGE*phi + CHP_IN)/(KB*T_n))  # Equilibrium electron density
nd_heqn = NV*Fermi_b((-CHI*mu_n**2/2 - ECHARGE*phi - CHP_IN)/(KB*T_n))  # Equilibrium hole density

Jy = ECHARGE*(j_hy - j_ey)    # y component of the total current
Jy_n = ECHARGE*(j_hyn - j_eyn)    # y component of the total current


Jy_testn = -v_5.dx(1) * NV*dFermi_b(gamma_hn)*MHC*(  ECHARGE) \
           +v_5.dx(1) * NC*dFermi_b(gamma_en)*MEC*(- ECHARGE)

Jy_testn2 = -v_5.dx(1) * assemble(NV*dFermi_b(gamma_hn)*MHC*(  ECHARGE)  * ds(0))\
           +v_5.dx(1) * assemble(NC*dFermi_b(gamma_en)*MEC*(- ECHARGE) * ds(0))

#-------------------------------------------------------------------------
# 3.5 Define variational problem
#-------------------------------------------------------------------------
# Lian times a factor 2
Feta = ((eta - eta_n)/delt + 2.0 * LN*dfb_deta(T, eta, mu))*v_1*dx \
       + LN*KAPPAN*dot(grad(eta), grad(v_1))*dx


# Lian times a factor 2
Fmu = ((mu - mu_n)/delt + 2.0 * LU*(dfb_dmu(T, eta, mu) \
       + CHI*mu*(nd_e + nd_h - 2*nd_in)))*v_2*dx + LU*KAPPAU*dot(grad(mu), grad(v_2))*dx


Fe = (NC*dFermi_b(gamma_e)*(gamma_e - gamma_en)/delt \
      - K0*mu**2*(nd_eeq*nd_heq - nd_e*nd_h))*v_3*dx - dot(j_e, grad(v_3))*dx

Fh = (NV*dFermi_b(gamma_h)*(gamma_h - gamma_hn)/delt \
      - K0*mu**2*(nd_eeq*nd_heq - nd_e*nd_h))*v_4*dx - dot(j_h, grad(v_4))*dx

Fphi = EPSILON*dot(grad(phi), grad(v_5))*dx - ECHARGE*(nd_h - nd_e)*v_5*dx

FT = (CPV*(T - T_n)/delt \
      - ECHARGE*((j_hx - j_ex)**2/(nd_e*MEA + nd_h*MHA) \
                 + (j_hy - j_ey)**2/(nd_e*MEC + nd_h*MHC)) \
      - AN1*eta*(eta - eta_n)/delt - AU1*mu*(mu - mu_n)/delt \
      + HTRAN*(T - Ts)/thickness)*v_6*dx \
     + THETA*dot(grad(T), grad(v_6))*dx
# -------------------------------------------------------
# Nitsche's trick: Define nonstandard boundary conditions
# -------------------------------------------------------

epsilon = alpha*(60.0/nx)
Fbc_e_Nitsche = -1.0/epsilon*(gamma_e*KB*T +CHI*mu**2/2 - CHP_IN )*v_3*ds(0) \
                -1.0/epsilon*(gamma_e*KB*T +CHI*mu**2/2 - CHP_IN )*v_3*ds(1)
Fbc_h_Nitsche = -1.0/epsilon*(gamma_h*KB*T +CHI*mu**2/2 - CHP_IN )*v_4*ds(0) \
                -1.0/epsilon*(gamma_h*KB*T +CHI*mu**2/2 - CHP_IN )*v_4*ds(1)
Fbc_phi_Nitsche = -1/epsilon*(phi + Rresistor*thickness*integral_phi- delV)*v_5*ds(0) \
                  + (integral_phi - Lx * Jy)*v_10*ds(0)

F = Feta + Fmu + Fe + Fh + Fphi + FT + Fbc_e_Nitsche + Fbc_h_Nitsche + Fbc_phi_Nitsche

# Gateaux derivative in direction of du (variational form for solving for Jacobian of F)
Jac = derivative(F, u, du)

#----------------------------------------------------------
# Solve the problem and save the solution
#----------------------------------------------------------
# Create VTK files for visualization output
vtkfile_eta = File(fig_file+'eta.pvd')
vtkfile_mu = File(fig_file+'mu.pvd')
vtkfile_nd_e = File(fig_file+'n.pvd')
vtkfile_nd_h = File(fig_file+'p.pvd')
vtkfile_phi = File(fig_file+'phi.pvd')
vtkfile_T = File(fig_file+'T.pvd')
vtkfile_gamma_e = File(fig_file+'gamma_e.pvd')
vtkfile_gamma_h = File(fig_file+'gamma_h.pvd')


# _eta, _mu, _gamma_e, _gamma_h, _phi, _T, _lam_1, _lam_2, _lam_3 = u.split()
# vtkfile_phi << (_phi, 0)


# Encapsulate the process of updating the previous solution "u_n"
def update_u_n(u):
    assign(u_n.sub(0), u.sub(0))
    assign(u_n.sub(1), u.sub(1))
    assign(u_n.sub(2), u.sub(2))
    assign(u_n.sub(3), u.sub(3))
    assign(u_n.sub(4), u.sub(5)) # update T, phi: no need

# Encapsulate the solving process for a given time step "dt"
def solve4dt(dt): #used to update variables in F, e.g delt 
    delt.assign(Constant(dt))
    # problem = NonlinearVariationalProblem(F, u, bcs, Jac)
    # solver = NonlinearVariationalSolver(problem)
    num, converged = solver.solve()
    # Update previous solution
    update_u_n(u)
    return converged

# Encapsulate the exportation of solutions to files
def save_sol(u):
    _u = u.split()   # Not a deep copy, which is fast and memory saving

    # Save solution to file (VTK)
    _u[0].rename("eta", "eta")
    _u[1].rename("mu", "mu")
    vtkfile_eta << (_u[0], t)
    vtkfile_mu << (_u[1], t)

    _u[2].rename("gamma_e", "gamma_e")
    vtkfile_gamma_e << (_u[2], t)

    _nd_e = project(NC*Fermi(_u[2])*UCVOL, V1)
    _nd_e.rename("n","n")
    vtkfile_nd_e << (_nd_e, t)

    _u[3].rename("gamma_h", "gamma_h")
    vtkfile_gamma_h << (_u[3], t)

    _nd_h = project(NV*Fermi(_u[3])*UCVOL, V1)
    _nd_h.rename("p","p")
    vtkfile_nd_h << (_nd_h, t)

    _u[4].rename("phi","phi")
    vtkfile_phi << (_u[4], t)
    _T = project(_u[5]*338.0, V1)
    _T.rename("T","T")
    vtkfile_T << (_T, t)


def rel_err_L2(u1, u2): 
    # Xu: errornorm is already in parallel mode 
    error1 = errornorm(u1.sub(0),u2.sub(0),'L2')
    error2 = errornorm(u1.sub(1),u2.sub(1),'L2')
    error3 = errornorm(u1.sub(2),u2.sub(2),'L2')
    error4 = errornorm(u1.sub(3),u2.sub(3),'L2')
    error5 = errornorm(u1.sub(4),u2.sub(4),'L2')
    error6 = errornorm(u1.sub(5),u2.sub(5),'L2')

    #total_error = error1 + error2 +error3 +error4+error5+error6 
    #return MPI.sum(MPI.comm_world, total_error)
    return error1, error2,error3, error4, error5, error6 
    
def compute_value_at_vertex(u,mesh,nx,ny): #Note: compute_vertex_value() is a fenics function
    eta_vertex = np.reshape(u.sub(0).compute_vertex_values(mesh),(ny+1,nx+1))
    mu_vertex = np.reshape(u.sub(1).compute_vertex_values(mesh),(ny+1,nx+1))
    gamma_e_vertex =np.reshape(u.sub(2).compute_vertex_values(mesh),(ny+1,nx+1))
    gamma_h_vertex =np.reshape(u.sub(3).compute_vertex_values(mesh),(ny+1,nx+1))
    phi_vertex = np.reshape(u.sub(4).compute_vertex_values(mesh),(ny+1,nx+1))
    T_vertex = np.reshape(u.sub(5).compute_vertex_values(mesh),(ny+1,nx+1))

    return [eta_vertex,mu_vertex,gamma_e_vertex,gamma_h_vertex,phi_vertex,T_vertex]

def save_arrays_to_txt(dict,filename): # input must be a dictionary 
    with open(filename, 'w') as outfile:
      for key,value in dict.items():
          outfile.write(key+'\n')       
          np.savetxt(outfile, value, fmt='%-8.4f') #a 8-character wide field with 4 digits following the decimal point

def get_diagonal_entries(u_list): #list of function arrays
    diagonals_values = []
    x,y = u_list[0].shape 
    for array_ in u_list:
        diagonal_single = []
        for i in range(x): # avoid index out of range 
              diagonal_single.append(round(array_[i][3*i],4))
        diagonals_values.append(diagonal_single)
    return diagonals_values

#===============================================
# Step 4.   Time evolution of PDEs
#           Prepare variables for adaptive time stepping 
#           Define nonlinear problem and solver
#           Time loop with adaptive time stepping 
#===============================================

# Define reservoir functions for adaptive time-stepping
u_coarse_t = Function(V)
u_n_start = Function(V_n)
u_non_converge_n = Function(V) #added 
#u_start = Function(V) #never used
# Create progress bar
progress = Progress('Time-stepping')

# Time-stepping
t_out = dt_out
t = 0.0
n_step = 0
# save_sol(u) Xu:deleted

myfile1 = open(time_steps_file1,'w+')
myfile2 = open(time_steps_file2,'w+')
myfile1.close()
myfile2.close()
time_steps = []

#Define and set up solver
problem = NonlinearVariationalProblem(F, u, bcs, Jac)
solver = NonlinearVariationalSolver(problem)
prm = solver.parameters
prm['newton_solver']['relaxation_parameter'] = rel_par
prm['newton_solver']['maximum_iterations'] = max_iter
prm['newton_solver']['absolute_tolerance'] = 1E-6
prm['newton_solver']['relative_tolerance'] = newton_rel_tol 

prm['newton_solver']['linear_solver'] = 'mumps'#'bicgstab'  # 'mumps', 'gmres', 'bicgstab'
prm['newton_solver']['krylov_solver']['monitor_convergence'] = True
#prm['newton_solver']['preconditioner'] = 'hypre_amg' #'hypre_euclid'   # 'hypre_euclid'
prm["newton_solver"]["krylov_solver"]["maximum_iterations"] = kr_solver_iter
prm["newton_solver"]["krylov_solver"]['error_on_nonconvergence'] = False
prm['newton_solver']['error_on_nonconvergence'] = False

while t < tf + 1E-6*dt_out:
    #set_log_level(LogLevel.INFO)
    #print('User message ===>', assemble((integral_phi /60.0 - Rresistor * thickness * Jy)*ds(0)))

    #file for recording adaptive time steps
    myfile1 = open(time_steps_file1,'a')
    myfile2 = open(time_steps_file2,'a')

    if rank == 0: 
        # Test and rerun the time-stepping to refine "dt"
        print('User message ===> Start adapative time-stepping at t = '+str(t)+'; dt = '+str(dt))

        myfile1.write('User message ===> Start adapative time-stepping at t = '+str(t)+'; dt = '+str(dt)+'\n')
        time_steps.append((t,dt))   
        #t_n = t #keep previous time 
        #t += dt 

    rerror = 1000 # set big to initiate
    u_n_start.assign(u_n) # Keep a copy of the previous solution "u_n"
    u_non_converge_n.assign(u) ## Keep a copy of the previous solution 
                                  # in case newton iteration non convergent

    while rerror > tol_tstep:
        '''
        2021/03/24 Xu:
        (1) this try ... except ... is added because error in the mumps linear solver
        would terminate the program even if we set error_on_nonconvergence to false for both Newton sovler and krylov solver. Possibly due to mumps being in an external package
        (2) The original implementation does not have this try .. catch ...function , then a relatively larger
        adaptive error tolerance sometimes lead to a large time step size that makes the mumps solver fail, hence terminating the program. In this case, we are forced to pick a small error tolerance yet still not sure about if the mumps solver would fail. This is unwanted.
       '''
        try:
            converged = solve4dt(dt) #u_n updated
            if converged!= True: # if not converged, then we need to take a smaller dt and get out the current adaptive time stepping.
                u_n.assign(u_n_start) #correct previous time step value
                u.assign(u_non_converge_n) #correct inital value for newton iteration
         #       t = t_n
                dt = dt/2.0
                if rank == 0:
                    print('User messages ====> Newton solver did not converge!!Start adapative time-stepping at t = '+str(t) +'; dt = '+str(dt),flush=True)
                    myfile1.write('User messages ====> Newton solver did not converge!!Start adapative time-stepping at t = '+str(t) +'; dt = '+str(dt)+'\n')
                    time_steps.append((t,dt))
                break
        except Exception as e:
            u_n.assign(u_n_start)
            u.assign(u_non_converge_n)
     #       t = t_n
            dt = dt/2.0
            if rank == 0:
                print('User messages ====> Error not catched by Newton solver occured at t = '+str(t) +'; dt = '+str(dt),flush=True)
                myfile1.write('User messages ====> Did not converge!! Error not catched by Newton solver occured. Start adapative time-stepping at t = '+str(t) +'; dt = '+str(dt)+'\n')
                myfile1.write("Error occured in corase time step \n")
                time_steps.append((t,dt))
            break
        u_coarse_t.assign(u)
        u_n.assign(u_n_start)
        
        try:
            '''
            2021/03/24 Xu: Here we didn't check if newton solver converged or not, we assume that this would have happened in the coarse step, so whenever we are here, newton solver should converge.
            This might be a potential issue, then we just need to add the same code as in the coarse adaptive step.
           '''
            converged = solve4dt(dt/2.0) #u_n updated
            converged = solve4dt(dt/2.0) #u_n updated
        except Exception as e:
            u_n.assign(u_n_start)
            u.assign(u_non_converge_n) #added today
     #       t = t_n
            dt = dt/2.0
            if rank == 0:
                print('User messages ====> Newton solver did not converge!!Start adapative time-stepping at t = '+str(t) +'; dt = '+str(dt))
                myfile1.write('User messages ====> Newton solver did not converge!!Start adapative time-stepping at t = '+str(t) +'; dt = '+str(dt)+'\n')
                myfile1.write("Error occured in fine time step \n")
                time_steps.append((t,dt))
            break

        error1, error2,error3, error4, error5, error6 = rel_err_L2(u_coarse_t, u)

        # rerror = error1/norm1 + error2/norm2 +error3/norm3 +error4/norm4+error5/norm5+error6/norm6 
        rerror = error1 + error2 +error3 +error4 +error5 +error6
        if rank == 0: 
            myfile1.write('adapative time-stepping at t = '+str(t)+'; with dt = '+str(dt)+'\t rerror = '+str(rerror)+'\n')
            myfile1.write('eta:        '+str(round(error1,6))+", \t \n")
            myfile1.write('mu:         '+str(round(error2,6))+", \t \n")
            myfile1.write('gamma_e:    '+str(round(error3,6))+", \t \n")
            myfile1.write('gamma_h:    '+str(round(error4,6))+", \t \n")
            myfile1.write('phi:        '+str(round(error5,6))+", \t \n")
            myfile1.write('T:          '+str(round(error6,6))+", \t \n")
            print("L2 error between Corser and finer: "+str(round(rerror,6))+ 'tol = ' + str(tol_tstep))
            myfile1.write("L2 error between Corser and finer: "+str(round(rerror,6))+ 'tol = ' + str(tol_tstep)+'\n')
            
        #Key formula for our method based on Richardson's extrapolation
        error_ratio = min(max(s_t*math.sqrt(tol_tstep/max(rerror, 1E-10)), r_t_min),r_t_max) 

        if rerror <= tol_tstep: # good to exit and go to next time step 
            n_step += 1
            if rank == 0: 
                myfile2.write(str(n_step)+',  '+str(dt)+' , '+str(t)+"\n")
            t += dt  #potential issue: if RHS or BC depends on t, then make sure you assign correct t beforehand
            if error_ratio > 1:  # 
                dt *= error_ratio
            if n_step%save_frequency==1:
                save_sol(u)
            break
        else: # redo this time step with new dt
            if rank == 0: 
                print()
                print("User message ===> Error too big, start redo this step! ", end = '||')
                myfile1.write("User message ===> Error too big, start redo this step! || ")
            u_n.assign(u_n_start)  # for 2nd time entry to loop, change u_n
            u.assign(u_non_converge_n) #correct inital value for newton iteration 
            dt *= error_ratio
            if rank == 0:
                print('Start adapative time-stepping at t = '+str(t)+'; dt = '+str(dt))
                myfile1.write('Start adapative time-stepping at t = '+str(t)+'; dt = '+str(dt))
                time_steps.append((t,dt))
    myfile1.close()
    myfile2.close()
    #save_sol(u)

MPI.barrier(MPI.comm_world)
elapsed_time = time.time() - start_time
if rank == 0:
    print('User message ===> Computation time: ', elapsed_time, ' s', flush=True)

# Hold plot
# interactive()
