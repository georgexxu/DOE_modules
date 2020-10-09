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

# 2nd Order Crank-Nicolson implicit scheme 
for epsilon in [0.04]:#,0.04,0.08,0.16]:
    print("========================>Current epsilon: ",epsilon)
    for grid_num in [512]:
        for dt in [0.0005]: #,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
        # for dt in [0.9]:
            x_0 = -1 
            x_N = 1
            mesh = IntervalMesh(grid_num,x_0,x_N)
            #mesh = UnitSquareMesh(grid_point,grid_point)
            # dt = 1
            T = 0.01
            print_frequency = 2
            fine_coarse = "coarse" #"reference" # "reference" 
            sol_space = 1 #when use fine mesh, set to other integer
            figure = True 

            P1 = FiniteElement('Lagrange',mesh.ufl_cell(),1)
            V1 = FunctionSpace(mesh,P1)

            eta = Function(V1)
            v = TestFunction(V1)
            eta_n = Function(V1)

            #initial condition
            #eta_0 = Expression('sqrt((x[0]*x[0]+ x[1]*x[1])) < 100 + DOLFIN_EPS ? 1 : -1',degree = 2)
            #eta_0 = Expression('tanh((sqrt(x[0]*x[0]+ x[1]*x[1])-0.6)/(sqrt(2)*eps))',degree = 2,eps = epsilon )
            eta_0 = Expression('cos(pi*x[0])',degree = 2)
            #eta_0 = project(eta_0,V1)
            eta_n.assign(eta_0)
            
            #initilize solution
            eta.assign(eta_0)


            #weak form
            #F_imp = (eta-eta_n)/Constant(dt)*v*dx-eta*v*dx + eta*eta*eta*v*dx + dot(grad(eta),grad(v))*dx
            F_imp = (eta-eta_n)/Constant(dt)*v*dx + Constant(0.5)*dot(grad(eta),grad(v))*dx \
                    + Constant(0.5)*dot(grad(eta_n),grad(v))*dx \
                    + 1/(Constant(2)*Constant(epsilon)*Constant(epsilon))*( eta * ( eta-1 ) * ( eta+1 ) * v * dx) \
                    + 1/(Constant(2)*Constant(epsilon)*Constant(epsilon))*( eta_n * ( eta_n-1 ) * ( eta_n+1 ) * v * dx)

            deta = TrialFunction(V1)
            Jac = derivative(F_imp,eta,deta)

            # def computeRadius(eta,V): # robust in non-parallel mode

            #     '''
            #     robust in non-parallel mode
            #     time consuming, when interpolate on 120*120*120, takes 93s, yet solving only takes 20s
            #     the most time consuming step is interpolate
            #     '''
            #     #eta = interpolate(eta,V)
            #     #eta = project(eta,V)
            #     values = eta.vector().get_local()
            #     values = np.where(values > - DOLFIN_EPS,1,values)
            #     values = np.where(values != 1, 0, values)
            #     volume = 100*100*100*np.sum(values)/float(len(values))
            #     R = np.power((3/(4.0*np.pi))*volume,1.0/3)
            #     return R

            t = 0 
            counter = 0
            solutions = []
            solutions.append(eta.vector().get_local()[::sol_space])
            Rs = [] 
            ts = []
            a1 = time.time()

            if figure: 
                plt.figure()
                plot(eta)
                plt.show()

            start_time = time.time()
            while t < T: 
                t += dt
                print("===========================> current time: ",t)

                #solve(F_imp == 0,eta,None)
                problem = NonlinearVariationalProblem(F_imp, eta, None, Jac) # Jacobian is passed in here
                solver = NonlinearVariationalSolver(problem)

                prm = solver.parameters
                prm['newton_solver']['relaxation_parameter'] = 0.9
                prm['newton_solver']['maximum_iterations'] = 100
                prm['newton_solver']['absolute_tolerance'] = 1E-5
                prm['newton_solver']['relative_tolerance'] = 1E-7
                #prm['newton_solver']['linear_solver'] = 'pcg'  # 'mumps', 'gmres', 'bicgstab'
                #prm['newton_solver']['linear_solver'] = 'superlu_dist'  # 'mumps', 'gmres', 'bicgstab'
                #prm['newton_solver']['preconditioner'] = 'hypre_amg'   # 'hypre_euclid'
                prm['newton_solver']['krylov_solver']['nonzero_initial_guess'] = True
                #solver.set_operator(A)  
                prm['newton_solver']['krylov_solver']['monitor_convergence'] = True
                prm['newton_solver']['krylov_solver']['absolute_tolerance'] = 1E-10
                prm['newton_solver']['krylov_solver']['relative_tolerance'] = 1E-10
                # prm['newton_solver']['linear_solver'] = 'gmres'  # 'mumps', 'gmres', 'bicgstab'
                # prm['newton_solver']['preconditioner'] = 'ilu'   # 'hypre_euclid'
                
                # prm['newton_solver']['linear_solver'] = 'bicgstab'  # 'mumps', 'gmres', 'bicgstab'
                # prm['newton_solver']['preconditioner'] = 'ilu'   # 'hypre_euclid'

                #prm['newton_solver']['linear_solver'] = 'mumps'  # 'mumps', 'gmres', 'bicgstab'
                prm['newton_solver']['linear_solver'] = 'cg'  # 'mumps', 'gmres', 'bicgstab'
                #prm['newton_solver']['preconditioner'] = 'amg'   # 'hypre_euclid'

                solver.solve()
                eta_n.assign(eta)

                counter += 1
                if counter % print_frequency  == 0:
                    if figure: 
                        plt.figure()
                        plot(eta)
                        plt.show()
                    solutions.append(eta.vector().get_local()[::sol_space])
            
            #solutions is a list contain 1d arrays, save them to human readable txt file
            solutions = array_2d_to_list(solutions)
            filename = "2nd_eps"+str(epsilon)+"_"+fine_coarse+".txt"
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
