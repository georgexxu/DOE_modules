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


def f_(phi): 
    return phi**3 - phi
def F_(phi):
    return 1.0/4*(phi**2-1)**2

def func_U(phi): 
    return sqrt(F_(phi))

def H(phi):
    return f_(phi)/func_U(phi) 

for epsilon in [0.02]:#,0.04,0.08,0.16]:
    print("========================>Current epsilon: ",epsilon)
    for grid_num in [512]:
        for dt in [0.0001]: #,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
        # for dt in [0.9]:
            x_0 = -1 
            x_N = 1
            mesh = IntervalMesh(grid_num,x_0,x_N)
            #mesh = UnitSquareMesh(grid_point,grid_point)
            # dt = 1
            T =0.002 #0.000016
            print_frequency = 1
            fine_coarse = "coarse" #"reference" # "reference" 
            sol_space = 1 #when use fine mesh, set to other integer
            figure = False
            write_to_file = True
            filename = "epsilon"+str(epsilon)+"_bdf2_ieq_"+fine_coarse+".txt"

            P1 = FiniteElement('Lagrange',mesh.ufl_cell(),1)
            element = MixedElement([P1,P1])
            V = FunctionSpace(mesh,element)
            V1 = FunctionSpace(mesh, P1)

            # varaibles for the new scheme 
            u = TrialFunction(V)
            # u = Function(V)
            # du = TrialFunction(V)
            phi, U = split(u)
            v1,v2 = TestFunction(V)
            phi_n = Function(V1)
            phi_n_1 = Function(V1)

            #variables for 1st step FIS
            eta = Function(V1)
            eta_n = Function(V1)
            w = TestFunction(V1)

            #Initilize solutions
            phi_0 = Expression('cos(pi*x[0])',degree = 2)
            phi_n.assign(phi_0) # assign is a member function of phi_n object 
            phi_n_1.assign(phi_0)

            eta_n.assign(phi_0)
            eta.assign(phi_0)
            if figure: #initial condition figure 
                plt.figure()
                plot(eta)
                plt.ylim([-1.1,1.1])
                plt.show()
            solutions = [] #store initial condition
            solutions.append(phi_n.vector().get_local()[::sol_space])


            #2. linear solver 
            #First step use 1st order Fully implicit scheme to get u_n, u_n_1 
            #check if this is correct 
            t = 0
            # vtkfile = File(fig_folder_name+"/phi.pvd")
            # eta_n.rename("eta","eta") # phi: consistent with later variable name 
            # vtkfile << (eta_n,t)

            F_imp = (eta-eta_n)/Constant(dt)*w*dx + dot(grad(eta),grad(w))*dx - 1/(Constant(epsilon)*Constant(epsilon))*(- eta * ( eta-1 ) * ( eta+1 ) * w * dx)
            deta = TrialFunction(V1)
            Jac = derivative(F_imp,eta,deta)

            t += dt 
            problem1 = NonlinearVariationalProblem(F_imp, eta, None, Jac) # Jacobian is passed in here
            solver1 = NonlinearVariationalSolver(problem1)
            solver1.solve()
            eta_n.assign(eta) # now eta_n is the solution at first dt
            # eta_n.rename("eta","eta")
            # vtkfile << (eta_n,t)
            if figure: # first dt figure
                plt.figure()
                plot(eta)
                plt.ylim([-1.1,1.1])
                plt.show()

           #Initilization with the computed variable values 
            phi_n.assign(eta_n)

            a = 3*phi/(2*dt)*v1*dx + dot(grad(phi),grad(v1))*dx \
                + 1/(epsilon*epsilon)*H(2*phi_n - phi_n_1)*U*v1*dx \
                + 3*U*v2*dx - 1.0/2*H(2*phi_n - phi_n_1)*(3*phi)*v2*dx
            L = -(- 4*phi_n +phi_n_1)/(2*dt)*v1*dx - (- 4*func_U(phi_n) + func_U(phi_n_1))*v2*dx \
                + 1.0/2*H(2*phi_n - phi_n_1)*(- 4*phi_n +phi_n_1)*v2*dx

            parameters["linear_algebra_backend"] = "PETSc"
            # krylov_method="lu"
            # precond="ilu"
            # precond="none"    
            solver = PETScKrylovSolver()
            #solver = KrylovSolver(krylov_method)
            # 1. mumps option 
            # PETScOptions.set("ksp_type", "preonly") 
            # PETScOptions.set("pc_type", 'lu')
            # PETScOptions.set("pc_factor_mat_solver_type", 'mumps')
            PETScOptions.set("ksp_type", "gmres") 
            PETScOptions.set('pc_type', 'ilu')
            # PETScOptions.set('pc_hypre_type', 'boomeramg')
            PETScOptions.set('ksp_rtol', 1e-10)
            PETScOptions.set('ksp_atol', 1e-10)
            PETScOptions.set('ksp_max_it', 1000)
            PETScOptions.set('ksp_monitor_true_residual')
            # PETScOptions.set('ksp_error_if_not_converged',False)
            # PETScOptions.set("ksp_view")
            # PETScOptions.set('ksp_view_mat_explicit')            
            # PETScOptions.set('ksp_view_eigenvalues')

            solver.set_from_options()

            # solver.parameters["relative_tolerance"] = 1.0e-10
            # solver.parameters["absolute_tolerance"] = 1.0e-10
            # solver.parameters["monitor_convergence"] = True
            # solver.parameters["maximum_iterations"] = 1000

            A, b = assemble_system(a, L)
            
            counter = 1
            solutions.append(phi_n.vector().get_local()[::sol_space]) #store first dt solution
            Rs = [] 
            ts = []
            a1 = time.time()

            if figure:
                plt.figure()
                plot(phi_n)
                plt.ylim([-1.1,1.1])
                plt.show()

            start_time = time.time()
            u = Function(V)
            while t < T: 
                t += dt
                print("===========================> current time: ",t)
                #2. linear solver 

                A, b = assemble_system(a, L)
                solver.set_operator(A)
                solver.solve(u.vector(), b)

                phi_n_1.assign(phi_n)
                assign(phi_n,u.sub(0))

                counter += 1
                phi_V1 = project(u.sub(0),V1)
                if counter % print_frequency  == 0:
                    if figure:
                        plt.figure()
                        plot(phi_V1)
                        plt.ylim([-1.1,1.1])
                        plt.show()
                        #plt.savefig(str(counter)+"_fine.png")
                    solutions.append(phi_V1.vector().get_local()[::sol_space])
                counter += 1
            #solutions is a list contain 1d arrays, save them to human readable txt file
            if write_to_file: 
                solutions = array_2d_to_list(solutions)
                write_2d_list_to_file(filename,solutions)

            #compute slope and calculate relative error
            # slope, intercept, r_value, p_value, std_err = stats.linregress(ts,Rs)
            # rel_error = abs(exact_slope-slope)/abs(exact_slope)
            # with open('implicit_cg_2D_slope_1_g1.txt', 'a') as f_slope:
            #     f_slope.write(str(grid_point)+'\t')
            #     f_slope.write(str(dt) + '\t' + str(slope)+'\t'+ str(rel_error)+'\t'+str(end_time)+'\n')

            # # with open('implicit_amg_2D_CPU.txt', 'a') as fCPU:
            #     fCPU.write("time for solving the PDE: %s\n" % a2)
            # print("time for solving the PDE: ",a2)
            # print(Rs)
            # print(ts)

            # elapsed_time = time.time() - start_time
            # print('User message ===> Computation time: ', elapsed_time, ' s', flush=True)

            # with open('implicit_1st_2D_Area_'+str(grid_point)+'.txt', 'w') as fR:
            #     for item in Rs:
            #         fR.write("%s\n" % item)

            # with open('implicit_1st_2D_t_'+str(grid_point)+'.txt', 'w') as ft:
            #     for item in ts:
            #         ft.write("%s\n" % item)

        # h = [math.sqrt(max(0.36-2*t,0)) for t in ts]
        # Rs = [R + h[0] - Rs[0] for R  in Rs]

        # plt.plot(ts, h,'-', label='Area: reference')
        # plt.plot(ts, Rs,'*', label='Area: implicit')

        # plt.legend()
        # #plt.savefig('implicit_2D.png')
        # plt.show()
