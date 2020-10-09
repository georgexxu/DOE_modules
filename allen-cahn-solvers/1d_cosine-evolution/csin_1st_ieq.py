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

for epsilon in [0.02]:#,0.04,0.08,0.16]:
    print("========================>Current epsilon: ",epsilon)
    for grid_num in [512]:
        for dt in [0.001]: #,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
        # for dt in [0.9]:
            x_0 = -1 
            x_N = 1
            mesh = IntervalMesh(grid_num,x_0,x_N)
            #mesh = UnitSquareMesh(grid_point,grid_point)
            # dt = 1
            T =0.02 #0.000016
            print_frequency = 1
            fine_coarse = "coarse" #"reference" # "reference" 
            sol_space = 1 #when use fine mesh, set to other integer
            figure = True
            write_to_file = True
            filename = "epsilon"+str(epsilon)+"_bdf1_ieq_"+fine_coarse+".txt"

            # P1 = FiniteElement('Lagrange',mesh.ufl_cell(),1)
            # V1 = FunctionSpace(mesh,P1)

            # eta = Function(V1)
            # v = TestFunction(V1)
            # eta_n = Function(V1)
            P1 = FiniteElement('Lagrange',mesh.ufl_cell(),1)
            V = FunctionSpace(mesh, P1)

            phi = TrialFunction(V)
            # u = Function(V)
            # du = TrialFunction(V)
            v1 = TestFunction(V)
            phi_n = Function(V)
            # phi_n_1 = Function(V)

            #Initilize solutions
            phi_0 = Expression('cos(pi*x[0])',degree = 2)
            phi_n.assign(phi_0) # assign is a member function of phi_n object 

            #initial condition
            #eta_0 = Expression('sqrt((x[0]*x[0]+ x[1]*x[1])) < 100 + DOLFIN_EPS ? 1 : -1',degree = 2)
            #eta_0 = Expression('tanh((sqrt(x[0]*x[0]+ x[1]*x[1])-0.6)/(sqrt(2)*eps))',degree = 2,eps = epsilon )
            #eta_0 = Expression('-sin(pi*x[0])',degree = 2)
            # eta_0 = Expression('cos(pi*x[0])',degree = 2)
            #eta_0 = Expression('x[0]+0.5',degree = 2)
            #eta_0 = project(eta_0,V1)
            # eta_n.assign(eta_0)
            
            # #initilize solution
            # eta.assign(eta_0)


            #weak form
            #F_imp = (eta-eta_n)/Constant(dt)*v*dx-eta*v*dx + eta*eta*eta*v*dx + dot(grad(eta),grad(v))*dx
            a = phi/dt*v1*dx + dot(grad(phi),grad(v1))*dx \
                + 1/(epsilon*epsilon)*H(phi_n)*H(phi_n)*1.0/2*phi*v1*dx
            L = phi_n/dt*v1*dx - 1/(epsilon*epsilon)*H(phi_n)*H(phi_n)*(-1.0/2)*phi_n*v1*dx \
                - 1/(epsilon*epsilon)*H(phi_n)*func_U(phi_n)*v1*dx

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

            A, b = assemble_system(a, L)

            t = 0 
            counter = 0
            solutions = []#store initial condition figure
            solutions.append(phi_n.vector().get_local()[::sol_space])
            Rs = [] 
            ts = []
            a1 = time.time()

            if figure:
                plt.figure()
                plot(phi_n)
                plt.ylim([-1.1,1.1])
                plt.show()

            start_time = time.time()
            phi = Function(V)
            while t < T: 
                t += dt
                print("===========================> current time: ",t)

                #1. newton solver 
                #                 #solve(F_imp == 0,eta,None)
                # problem = NonlinearVariationalProblem(F, u, None, Jac) # Jacobian is passed in here
                # solver = NonlinearVariationalSolver(problem)

                # prm = solver.parameters
                # prm['newton_solver']['relaxation_parameter'] = 0.9
                # prm['newton_solver']['maximum_iterations'] = 100
                # prm['newton_solver']['absolute_tolerance'] = 1E-5
                # prm['newton_solver']['relative_tolerance'] = 1E-7
                # #prm['newton_solver']['linear_solver'] = 'pcg'  # 'mumps', 'gmres', 'bicgstab'
                # #prm['newton_solver']['linear_solver'] = 'superlu_dist'  # 'mumps', 'gmres', 'bicgstab'
                # #prm['newton_solver']['preconditioner'] = 'hypre_amg'   # 'hypre_euclid'
                # prm['newton_solver']['krylov_solver']['nonzero_initial_guess'] = True
                # #solver.set_operator(A)  
                # prm['newton_solver']['krylov_solver']['monitor_convergence'] = True
                # prm['newton_solver']['krylov_solver']['absolute_tolerance'] = 1E-10
                # prm['newton_solver']['krylov_solver']['relative_tolerance'] = 1E-6
                # # prm['newton_solver']['linear_solver'] = 'gmres'  # 'mumps', 'gmres', 'bicgstab'
                # # prm['newton_solver']['preconditioner'] = 'ilu'   # 'hypre_euclid'
                
                # # prm['newton_solver']['linear_solver'] = 'bicgstab'  # 'mumps', 'gmres', 'bicgstab'
                # # prm['newton_solver']['preconditioner'] = 'ilu'   # 'hypre_euclid'

                # prm['newton_solver']['linear_solver'] = 'mumps'  # 'mumps', 'gmres', 'bicgstab'
                # #prm['newton_solver']['linear_solver'] = 'gmres'  # 'mumps', 'gmres', 'bicgstab'
                # # prm['newton_solver']['preconditioner'] = 'amg'   # 'hypre_euclid'

                # solver.solve()

                # phi_n_1.assign(phi_n)
                # assign(phi_n,u.sub(0))

                #2. linear solver 

                A, b = assemble_system(a, L)
                solver.set_operator(A)
                solver.solve(phi.vector(), b)

                # phi_n_1.assign(phi_n)
                phi_n.assign(phi)

                if counter % print_frequency  == 0:
                    if figure:
                        plt.figure()
                        plot(phi)
                        plt.ylim([-1.1,1.1])
                        plt.show()
                        #plt.savefig(str(counter)+"_fine.png")
                    solutions.append(phi.vector().get_local()[::sol_space])
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
