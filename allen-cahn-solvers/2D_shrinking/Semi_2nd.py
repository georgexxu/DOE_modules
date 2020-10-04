from fenics import *
import time
import numpy as np
import math
from scipy import stats
#import matplotlib.pyplot as plt

def computeArea(eta,V): # robust in non-parallel mode

    '''
    robust in non-parallel mode
    time consuming, when interpolate on 120*120*120, takes 93s, yet solving only takes 20s
    the most time consuming step is interpolate
    '''
    #eta = interpolate(eta,V)
    #eta = project(eta,V)
    values = eta.vector().get_local()
    values = np.where(values > - DOLFIN_EPS,1,values)
    values = np.where(values != 1, 0, values)
    Area = 256.0 * 256.0 * np.sum(values)/float(len(values))
    return Area

# with open('semi_amg_2D_CPU.txt', 'w') as fCPU:
#     fCPU.write("This file is used to record the CPU time\n")
with open('semi_2nd_2D_slope_time_1.txt', 'a') as f_slope:
    f_slope.write("This file is used to record the slope and relative error\n")
exact_slope = -2*math.pi
for grid_point in [256]:
# for grid_point in [2048]:
    for dt in [0.025,0.05,0.1,0.2,0.4,0.8]:
    #for dt in [2.0]:

        mesh = RectangleMesh(Point(-128,-128),Point(128,128),grid_point,grid_point)
        mesh2 = RectangleMesh(Point(-128,-128),Point(128,128),1000,1000)

        # dt = 1
        T = 201
        print_ferquence = int(50/dt)

        P1 = FiniteElement('Lagrange',mesh.ufl_cell(),1)
        V1 = FunctionSpace(mesh,P1)

        P2 = FiniteElement('Lagrange',mesh2.ufl_cell(),1)
        V2 = FunctionSpace(mesh2,P2)


        eta = TrialFunction(V1)
        v = TestFunction(V1)
        eta_n = Function(V1)
        eta_n_1 = Function(V1)

        #initial condition
        eta_0 = Expression('sqrt((x[0]*x[0]+ x[1]*x[1])) < 100 + DOLFIN_EPS ? 1 : -1',degree = 2)
        #eta_0 = project(eta_0,V1)
        eta_n.assign(eta_0)
        eta_n_1.assign(eta_0)

        #weak form
        #F_imp = (eta-eta_n)/Constant(dt)*v*dx-eta*v*dx + eta*eta*eta*v*dx + dot(grad(eta),grad(v))*dx

        # a = (eta)/dt*v*dx + dot(grad(eta),grad(v))*dx 

        # L = (eta_n)/dt*v*dx + (- eta_n * ( eta_n-1 ) * ( eta_n+1 ) * v * dx)

        a = (3*eta)/(2*dt)*v*dx + dot(grad(eta),grad(v))*dx 

        L = (4*eta_n-eta_n_1)/(2*dt)*v*dx + 2*(- eta_n * ( eta_n-1 ) * ( eta_n+1 ) * v * dx) \
                + eta_n_1 * ( eta_n_1-1 ) * ( eta_n_1+1 ) * v * dx


        p = dot(grad(eta),grad(v))*dx 

        eta = Function(V1)

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
        R = computeArea(eta_n,V2)
        Rs.append(R)
        ts.append(t)
        a2 = time.time()-a1

        # with open('semi_amg_2D_CPU.txt', 'a') as fCPU:
        #     fCPU.write("Mesh:%s\n" % grid_point)
        #     fCPU.write("time for computing radius:%s\n" % a2)

        # print("time for computing radius: ",a2)

        # vtkfile = File("semi_amg_2D/eta.pvd")
        # vtkfile << (eta_n,t)


        a1 = time.time()

        parameters["linear_algebra_backend"] = "PETSc"
        
        # krylov_method="cg"
        # solver = KrylovSolver(krylov_method)
        krylov_method="cg"
        # precond="amg"
        #precond="none"    
        # solver = KrylovSolver(krylov_method, precond)
        solver = KrylovSolver(krylov_method)

        solver.parameters["relative_tolerance"] = 1.0e-10
        solver.parameters["absolute_tolerance"] = 1.0e-10
        solver.parameters["monitor_convergence"] = True
        solver.parameters["maximum_iterations"] = 1000

        A, b = assemble_system(a, L)

        #solver.set_operator(A)


        # P, b_temp = assemble_system(p, L)

        # solver.set_operator(A,P)

        # a2 = time.time()-a1

        # with open('semi_amg_2D_CPU.txt', 'a') as fCPU:
        #     fCPU.write("time for setup:%s\n" % a2)

        # print("time for setup: ",a2)


        start_time = time.time()
        while t < T: 
            t += dt
            print("===========================> current time: ",t)

            #solver.set_operator(A)  

            b = assemble(L)
            solver.solve(A, eta.vector(), b)

            eta_n_1.assign(eta_n)
            eta_n.assign(eta)

            counter += 1
            if counter % print_ferquence == 0:
                #vtkfile << (eta,t)
                R = computeArea(eta,V2)
                Rs.append(R) #better to write to a file
                ts.append(t)
        end_time = round(time.time()-start_time,2)
        print(ts)
        print(Rs)
        slope, intercept, r_value, p_value, std_err = stats.linregress(ts,Rs)
        rel_error = abs(exact_slope-slope)/abs(exact_slope)
        with open('semi_2nd_2D_slope_time_1.txt', 'a') as f_slope:
            f_slope.write(str(grid_point)+'\t')
            f_slope.write(str(dt) + '\t' + str(slope)+'\t'+ str(rel_error)+'\t'+str(end_time)+'\n')


        # a2 = time.time()-a1

        # with open('semi_amg_2D_CPU.txt', 'a') as fCPU:
        #     fCPU.write("time for solving the PDE:%s\n" % a2)
        # print("time for solve the PDE: ",a2)

        # print(Rs)
        # print(ts)


        # elapsed_time = time.time() - start_time
        # print('User message ===> Computation time: ', elapsed_time, ' s', flush=True)

        # with open('semi_1st_2D_Area_dt_1_'+str(grid_point)+'.txt', 'w') as fR:
        #     for item in Rs:
        #         fR.write("%s\n" % item)

        # with open('semi_1st_2D_t_dt_1_'+str(grid_point)+'.txt', 'w') as ft:
        #     for item in ts:
        #         ft.write("%s\n" % item)



    # h = [math.pi * 100 * 100 - 2.0 * math.pi * t for t in ts]
    # Rs = [R + h[0] - Rs[0] for R  in Rs]
    # print("numerical solution: ")
    # print(Rs)


    # plt.plot(ts, h,'-', label='Area: reference')
    # plt.plot(ts, Rs,'*', label='Area: semi-implicit')

    # plt.legend()
    # plt.savefig('semi_amg_2D.png')
    # plt.show()

