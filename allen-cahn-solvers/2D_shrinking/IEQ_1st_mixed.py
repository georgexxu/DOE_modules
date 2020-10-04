from fenics import *
import time
import numpy as np
import math
from scipy import stats
# import matplotlib.pyplot as plt

def computeArea(eta): # robust in non-parallel mode
    '''
    robust in non-parallel mode
    time consuming, when interpolate on 120*120*120, takes 93s, yet solving only takes 20s
    the most time consuming step is interpolate
    '''
    #eta = interpolate(eta,V)
    #eta = project(eta,V)
    values = eta.vector().get_local()
    #values = np.where(values > - DOLFIN_EPS,1,values)
    values = np.where(values < 0,2,values)
    values = np.where(values != 2, 0, values)
    values = np.where(values == 2, 1, values)
    Area = 2*2* np.sum(values)/float(len(values))
    radius = math.sqrt(Area/math.pi)
    #return Area
    return radius
def f_(phi): 
    return phi**3 - phi
def F_(phi):
    return 1.0/4*(phi**2-1)**2

def func_U(phi): 
    return sqrt(F_(phi))

def H(phi):
    return f_(phi)/func_U(phi)


for epsilon in [0.06]:#,0.04,0.08,0.16]:
    print("========================>Current epsilon: ",epsilon)
    for grid_point in [512]:
        for dt in [0.002]: #,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
            mesh = RectangleMesh(Point(-1,-1),Point(1,1),grid_point,grid_point)
            #mesh = UnitSquareMesh(grid_point,grid_point)
            T = 0.2

            #======== Output parameters======
            write_file_name1 = "mixed_1st_order_2D_radius_"+str(epsilon)+'.txt'
            write_file_name2 = "mixed_1st_order_stab_2D_ts_"+str(epsilon)+'.txt'
            fig_folder_name = "solution_1st_order"+str(epsilon)
            write_frequency = 1
            write_to_file = True 
            save_figure = True
            save_fig_frequency = 10
            #================================

            P1 = FiniteElement('Lagrange',mesh.ufl_cell(),1)
            element = MixedElement([P1,P1])
            V = FunctionSpace(mesh,element)
            V1 = FunctionSpace(mesh, P1)

            u = TrialFunction(V)
            # u = Function(V)
            # du = TrialFunction(V)
            phi, U = split(u)

            v1,v2 = TestFunction(V)

            phi_n = Function(V1)
            # phi_n_1 = Function(V1)

            #Initilize solutions
            phi_0 = Expression('tanh((sqrt(x[0]*x[0]+ x[1]*x[1])-0.6)/(sqrt(2)*eps))',degree = 2,eps = epsilon )
            phi_n.assign(phi_0) # assign is a member function of phi_n object 
            # phi_n_1.assign(phi_0)
            # no need to initilize U_n, U_n_1, they are functions in phi_n, phi_n_1 
            #for Newton solver, also important to initialize phi
            # phi_0 = project(phi_0,V1)
            # assign(u.sub(0),phi_0)
            # assign(u.sub(1),U_0)

            #1. newton solver. This problem is linear so no need
            #weak form
            # F_phi = (3*phi - 4*phi_n +phi_n_1)/(2*dt)*v1*dx + dot(grad(phi),grad(v1))*dx \
            #         + 1/(epsilon*epsilon)*H(2*phi_n - phi_n_1)*U*v1*dx
            # F_U = (3*U - 4*func_U(phi_n) + func_U(phi_n_1))*v2*dx - 1.0/2*H(2*phi_n - phi_n_1)*(3*phi - 4*phi_n +phi_n_1)*v2*dx
            # F = F_phi + F_U

            # a = 3*phi/(2*dt)*v1*dx + dot(grad(phi),grad(v1))*dx \
            #     + 1/(epsilon*epsilon)*H(2*phi_n - phi_n_1)*U*v1*dx \
            #     + 3*U*v2*dx - 1.0/2*H(2*phi_n - phi_n_1)*(3*phi)*v2*dx
            # L = -(- 4*phi_n +phi_n_1)/(2*dt)*v1*dx - (- 4*func_U(phi_n) + func_U(phi_n_1))*v2*dx \
            #     + 1.0/2*H(2*phi_n - phi_n_1)*(- 4*phi_n +phi_n_1)*v2*dx
            # Jac  =derivative(F,u,du)

            #2. linear solver 

            # S = 1
            # a_stab = S/(epsilon*epsilon)*phi*v1*dx
            # a_stab = 0
            a = phi/dt*v1*dx + dot(grad(phi),grad(v1))*dx \
                + 1/(epsilon*epsilon)*H(phi_n)*U*v1*dx \
                + U*v2*dx - 1.0/2*H(phi_n)*phi*v2*dx
            # L_stab = S/(epsilon*epsilon)*(2*phi_n - phi_n_1)*v1*dx 
            # L_stab = 0
            L = phi_n/dt*v1*dx + func_U(phi_n)*v2*dx \
                - 1.0/2*H(phi_n)*phi_n*v2*dx 

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

            t = 0 
            counter = 0

            if write_to_file: 
                with open(write_file_name1, 'w') as fR:
                    fR.write("This file is used to record the radius: \n")

                with open(write_file_name2, 'w') as ft:
                    ft.write("This file is used to record the time: \n")            

            Rs = [] # storing radii  
            ts = [] # storing time
            a1 = time.time()
            R = computeArea(phi_n) # input for comupteArea is a function
            print("initial radius is: ",R)
            if write_to_file: 
                R = computeArea(phi_n)
                print("initial radius is: ",R)
                with open(write_file_name1, 'a') as fR:
                    fR.write("%s\n" % R)
                with open(write_file_name2, 'a') as ft:
                    ft.write("%s\n" % t)

            Rs.append(R)
            ts.append(t)
            a2 = time.time()-a1

            if save_figure: 
                vtkfile = File(fig_folder_name+"/phi.pvd")
                phi_n.rename("phi","phi")
                vtkfile << (phi_n,t)

            start_time = time.time()

            u = Function(V)
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
                solver.solve(u.vector(), b)

                # phi_n_1.assign(phi_n)
                assign(phi_n,u.sub(0))


                counter += 1
                phi_V1 = project(u.sub(0),V1)
                if write_to_file: 
                    if counter % write_frequency  == 0:
                        #vtkfile << (eta,t)
                        #project phi on to V1 function space between computing area
                        R = computeArea(phi_V1)
                        with open(write_file_name1, 'a') as fR:
                            fR.write("%s\n" % R)
                        with open(write_file_name2, 'a') as ft:
                            ft.write("%s\n" % t)

                        Rs.append(R) #better to write to a file
                        ts.append(t)
                if save_figure: 
                    if counter % save_fig_frequency == 0:
                        phi_V1.rename("phi", "phi")
                        vtkfile << (phi_V1,t)

            end_time = round(time.time()-start_time,2)
            # a2 = time.time()-a1
            print(ts)
            print(Rs)

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
