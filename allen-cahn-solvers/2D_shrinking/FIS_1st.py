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

for epsilon in [0.001]:#,0.04,0.08,0.16]:
    print("========================>Current epsilon: ",epsilon)
    for grid_point in [2048]:
        for dt in [0.00001]: #,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
            with open('implicit_1st_2D_Area_'+str(epsilon)+'.txt', 'w') as fR:
                fR.write("This file is used to record the radius, dt = "+str(dt)+": \n")
            with open('implicit_1st_2D_t_'+str(epsilon)+'.txt', 'w') as ft:
                ft.write("This file is used to record the time: \n")
        # for dt in [0.9]:

            mesh = RectangleMesh(Point(-1,-1),Point(1,1),grid_point,grid_point)
            #mesh = UnitSquareMesh(grid_point,grid_point)
            # dt = 1
            T = 0.2
            print_frequency = int(0.004/dt)

            P1 = FiniteElement('Lagrange',mesh.ufl_cell(),1)
            V1 = FunctionSpace(mesh,P1)

            eta = Function(V1)
            v = TestFunction(V1)
            eta_n = Function(V1)

            #initial condition
            #eta_0 = Expression('sqrt((x[0]*x[0]+ x[1]*x[1])) < 100 + DOLFIN_EPS ? 1 : -1',degree = 2)
            eta_0 = Expression('tanh((sqrt(x[0]*x[0]+ x[1]*x[1])-0.6)/(sqrt(2)*eps))',degree = 2,eps = epsilon )
            #eta_0 = project(eta_0,V1)
            eta_n.assign(eta_0)
            
            #initilize solution
            eta.assign(eta_0)

            #weak form
            #F_imp = (eta-eta_n)/Constant(dt)*v*dx-eta*v*dx + eta*eta*eta*v*dx + dot(grad(eta),grad(v))*dx
            F_imp = (eta-eta_n)/Constant(dt)*v*dx + dot(grad(eta),grad(v))*dx - 1/(Constant(epsilon)*Constant(epsilon))*(- eta * ( eta-1 ) * ( eta+1 ) * v * dx)

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

            Rs = [] 
            ts = []
            a1 = time.time()
            R = computeArea(eta_n)
            print("initial radius is: ",R)
            with open('implicit_1st_2D_Area_'+str(epsilon)+'.txt', 'a') as fR:
                fR.write("%s\n" % R)
            with open('implicit_1st_2D_t_'+str(epsilon)+'.txt', 'a') as ft:
                ft.write("%s\n" % t)
            Rs.append(R)
            ts.append(t)
            a2 = time.time()-a1

            # with open('implicit_amg_2D_CPU.txt', 'a') as fCPU:
            #     fCPU.write("Mesh:%s\n" % grid_point)
            #     fCPU.write("time for computing radius:%s\n" % a2)

            # print("time for computing radius: ",a2)

            # vtkfile = File("solution/eta.pvd")
            # eta_n.rename(str(epsilon),str(epsilon))
            # vtkfile << (eta_n,t)

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
                prm['newton_solver']['krylov_solver']['relative_tolerance'] = 1E-6
                # prm['newton_solver']['linear_solver'] = 'gmres'  # 'mumps', 'gmres', 'bicgstab'
                # prm['newton_solver']['preconditioner'] = 'ilu'   # 'hypre_euclid'
                
                # prm['newton_solver']['linear_solver'] = 'bicgstab'  # 'mumps', 'gmres', 'bicgstab'
                # prm['newton_solver']['preconditioner'] = 'ilu'   # 'hypre_euclid'

                #prm['newton_solver']['linear_solver'] = 'mumps'  # 'mumps', 'gmres', 'bicgstab'
                prm['newton_solver']['linear_solver'] = 'cg'  # 'mumps', 'gmres', 'bicgstab'
                # prm['newton_solver']['preconditioner'] = 'amg'   # 'hypre_euclid'

                solver.solve()
                eta_n.assign(eta)

                counter += 1
                if counter % print_frequency  == 0:
                    # vtkfile << (eta,t)
                    R = computeArea(eta)
                    with open('implicit_1st_2D_Area_'+str(epsilon)+'.txt', 'a') as fR:
                        fR.write("%s\n" % R)
                    with open('implicit_1st_2D_t_'+str(epsilon)+'.txt', 'a') as ft:
                        ft.write("%s\n" % t)

                    Rs.append(R) #better to write to a file
                    ts.append(t)
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
