from fenics import *
import time
import numpy as np
import math
from scipy import stats
import matplotlib.pyplot as plt
import os.path

def array_2d_to_list(array_2d):
    a = list(array_2d) # a list passed in is fine
    for i in range(len(a)):
        a[i] = list(a[i])[::-1] # #reversed since FEniCS array from Right to Left
    return a 

def write_2d_list_to_file(filename, list_2d):
    myfile = open(filename,'w')
    for i in range(len(list_2d)):
        for j in range(len(list_2d[i])):
            if j == len(list_2d[i]) - 1: 
                myfile.write(str(list_2d[i][j]))# must be a string
            else: 
                myfile.write(str(list_2d[i][j])+' , ')
        myfile.write('\n')
    myfile.close()

def read_2d_list_from_file(filename):
    list_2d = []
    myfile = open(filename,'r')
    lines = myfile.readlines()
    for line in lines:
        temp = line.split(',')  
        temp = [float(item) for item in temp]
        list_2d.append(temp)
    return list_2d    

def ac1d_fis1st_solver(epsilon,grid_num,t_step,T,output_freq,
                solver_choice,precond,
                sol_slice = 1,domain = [-1,1]):
    figure = False 
    x_0 = domain[0]
    x_N = domain[1]
    dt = t_step
    mesh = IntervalMesh(grid_num,x_0,x_N)
    #mesh = UnitSquareMesh(grid_point,grid_point)
    folder_name = "ac_1st_fis_eps"+str(epsilon)+"grid"+str(grid_num)+"_dt"+str(t_step)+"/"
    os.mkdir(folder_name) #Issue 1: fix if file already exist
    filename1 = os.path.join(folder_name, "solution.txt" ) 
    #folder_name = ""
    # filename1 =folder_name + "solution.txt" 
    P1 = FiniteElement('Lagrange',mesh.ufl_cell(),1)
    V1 = FunctionSpace(mesh,P1)
    eta = Function(V1)
    v = TestFunction(V1)
    eta_n = Function(V1)
    eta_0 = Expression('cos(pi*x[0])',degree = 2)
    eta_n.assign(eta_0)            
    eta.assign(eta_0)
    #weak form
    #F_imp = (eta-eta_n)/Constant(dt)*v*dx-eta*v*dx + eta*eta*eta*v*dx + dot(grad(eta),grad(v))*dx
    F_imp = (eta-eta_n)/Constant(dt)*v*dx + dot(grad(eta),grad(v))*dx - 1/(Constant(epsilon)*Constant(epsilon))*(- eta * ( eta-1 ) * ( eta+1 ) * v * dx)
    deta = TrialFunction(V1)
    Jac = derivative(F_imp,eta,deta)
    #define the solver 
    problem = NonlinearVariationalProblem(F_imp, eta, None, Jac) # Jacobian is passed in here
    solver = NonlinearVariationalSolver(problem)

    prm = solver.parameters
    prm['newton_solver']['relaxation_parameter'] = 0.9
    prm['newton_solver']['maximum_iterations'] = 200
    prm['newton_solver']['absolute_tolerance'] = 1E-6
    prm['newton_solver']['relative_tolerance'] = 1E-6
    #prm['newton_solver']['linear_solver'] = 'pcg'  # 'mumps', 'gmres', 'bicgstab'
    #prm['newton_solver']['linear_solver'] = 'superlu_dist'  # 'mumps', 'gmres', 'bicgstab'
    #prm['newton_solver']['preconditioner'] = 'hypre_amg'   # 'hypre_euclid'
    prm['newton_solver']['krylov_solver']['nonzero_initial_guess'] = True
    #solver.set_operator(A)  
    prm['newton_solver']['krylov_solver']['monitor_convergence'] = True
    prm['newton_solver']['krylov_solver']['absolute_tolerance'] = 1E-20
    prm['newton_solver']['krylov_solver']['relative_tolerance'] = 1E-20
    #prm['newton_solver']['linear_solver'] = 'mumps'  # 'mumps', 'gmres', 'bicgstab'
    prm['newton_solver']['linear_solver'] = solver_choice  # 'mumps', 'gmres', 'bicgstab'
    #prm['newton_solver']['preconditioner'] = 'amg'   # 'hypre_euclid'
    if solver_choice != "lu": 
        prm['newton_solver']['preconditioner'] = precond   # 'hypre_euclid'

    start_time = time.time()
    t = 0 
    counter = 0
    solutions = []
    solutions.append(eta.vector().get_local()[::sol_slice])
    ts = [] 
    ts.append(t)
    while t < T: 
        t += dt
        counter += 1
        print("===========================> current time: ",t)

        solver.solve()
        eta_n.assign(eta)

        if counter % output_freq  == 0:
            if figure:
                plt.figure()
                plot(eta)
                plt.ylim([-1.1,1.1])
                plt.show()
            solutions.append(eta.vector().get_local()[::sol_slice])
            ts.append(t)
    #solutions is a list contain 1d arrays, save them to human readable txt file
    solutions = array_2d_to_list(solutions)
    write_2d_list_to_file(filename1,solutions)
    print("solutions are stored at the following time: ", ts)

ac1d_fis1st_solver(0.04,513,t_step = 0.0005,T = 0.01,output_freq = 1,solver_choice = 'gmres',precond = 'hypre_amg',sol_slice = 1,domain = [-1,1])

#if we want a numerical solution on a very small mesh with small dt
#as reference solution, we change grid_num, t_step. Moreover
# we change output_freq, sol_slice
