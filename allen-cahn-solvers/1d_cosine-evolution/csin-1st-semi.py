from fenics import *
import time
import numpy as np
import math
from scipy import stats
import matplotlib.pyplot as plt

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

def f_(phi): 
    return phi**3 - phi
def F_(phi):
    return 1.0/4*(phi**2-1)**2

def func_U(phi): 
    return sqrt(F_(phi))

def H(phi):
    return f_(phi)/func_U(phi) 

for epsilon in [0.01]:#,0.04,0.08,0.16]:
    print("========================>Current epsilon: ",epsilon)
    for grid_num in [512]:
        for dt in [0.00003]: #,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
        # for dt in [0.9]:
            x_0 = -1 
            x_N = 1
            mesh = IntervalMesh(grid_num,x_0,x_N)
            #mesh = UnitSquareMesh(grid_point,grid_point)
            # dt = 1
            T =0.0006 #0.000016
            print_frequency = 1
            fine_coarse = "coarse" #"reference" # "reference" 
            sol_space = 1 #when use fine mesh, set to other integer
            figure = False
            write_to_file = True
            filename = "epsilon"+str(epsilon)+"_1st_semi_"+fine_coarse+".txt"
            P1 = FiniteElement('Lagrange',mesh.ufl_cell(),1)
            V1 = FunctionSpace(mesh,P1)

            eta = TrialFunction(V1)
            v = TestFunction(V1)
            eta_n = Function(V1)

            #initial condition
            #eta_0 = Expression('sqrt((x[0]*x[0]+ x[1]*x[1])) < 100 + DOLFIN_EPS ? 1 : -1',degree = 2)
            eta_0 = Expression('cos(pi*x[0])',degree = 2)
            eta_n.assign(eta_0)
            
            #initilize solution, this is very important
            #eta.assign(eta_0)

            #weak form
            #F_imp = (eta-eta_n)/Constant(dt)*v*dx-eta*v*dx + eta*eta*eta*v*dx + dot(grad(eta),grad(v))*dx
            #F_imp = (eta-eta_n)/Constant(dt)*v*dx + dot(grad(eta),grad(v))*dx - 1/(Constant(epsilon)*Constant(epsilon))*(- eta * ( eta-1 ) * ( eta+1 ) * v * dx)

            a = (eta)/dt*v*dx + dot(grad(eta),grad(v))*dx 

            L = (eta_n)/dt*v*dx - 1/(Constant(epsilon)*Constant(epsilon))*(eta_n * ( eta_n-1 ) * ( eta_n+1 ))* v * dx

            eta = Function(V1)

            parameters["linear_algebra_backend"] = "PETSc"
        
            # krylov_method="cg"
            # solver = KrylovSolver(krylov_method)
            krylov_method="gmres"
            precond="ilu"
            #precond="none"    
            solver = KrylovSolver(krylov_method, precond)
            #solver = KrylovSolver(krylov_method)

            solver.parameters["relative_tolerance"] = 1.0e-10
            solver.parameters["absolute_tolerance"] = 1.0e-10
            solver.parameters["monitor_convergence"] = True
            solver.parameters["maximum_iterations"] = 1000

            A, b = assemble_system(a, L)
            solver.set_operator(A)

            t = 0 
            counter = 0
            solutions = []
            solutions.append(eta_n.vector().get_local()[::sol_space])
            Rs = [] 
            ts = []
            a1 = time.time()

            if figure:
                plt.figure()
                plot(eta_n)
                plt.ylim([-1.1,1.1])
                plt.show()

            start_time = time.time()
            while t < T: 
                t += dt
                print("===========================> current time: ",t)

                # A, b = assemble_system(a, L)
                b = assemble(L)
                solver.solve(eta.vector(), b)

                # phi_n_1.assign(phi_n)
                eta_n.assign(eta)

                if counter % print_frequency  == 0:
                    if figure:
                        plt.figure()
                        plot(eta)
                        plt.ylim([-1.1,1.1])
                        plt.show()
                        #plt.savefig(str(counter)+"_fine.png")
                    solutions.append(eta.vector().get_local()[::sol_space])
                counter += 1
            #solutions is a list contain 1d arrays, save them to human readable txt file
            if write_to_file: 
                solutions = array_2d_to_list(solutions)
                write_2d_list_to_file(filename,solutions)

