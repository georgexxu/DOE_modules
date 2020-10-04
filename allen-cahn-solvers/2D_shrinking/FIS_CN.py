from fenics import *
import time
import numpy as np
import math
#import matplotlib.pyplot as plt
# with open('implicit_amg_2D_CPU.txt', 'w') as fCPU:
#     fCPU.write("This file is used to record the CPU time\n")
for grid_point in [256]:
    start_time = time.time()

    mesh = RectangleMesh(Point(-128,-128),Point(128,128),grid_point,grid_point)
    mesh2 = RectangleMesh(Point(-128,-128),Point(128,128),1000,1000)


    dt = 0.5
    T = 2000
    print_ferquence = 200

    P1 = FiniteElement('Lagrange',mesh.ufl_cell(),1)
    V1 = FunctionSpace(mesh,P1)

    P2 = FiniteElement('Lagrange',mesh2.ufl_cell(),1)
    V2 = FunctionSpace(mesh2,P2)


    eta = Function(V1)
    v = TestFunction(V1)
    eta_n = Function(V1)
    eta_n_1 = Function(V1)

    #initial condition
    eta_0 = Expression('sqrt((x[0]*x[0]+ x[1]*x[1])) < 100 + DOLFIN_EPS ? 1 : -1',degree = 2)
    #eta_0 = project(eta_0,V1)
    eta_n.assign(eta_0)

    #weak form
    #F_imp = (eta-eta_n)/Constant(dt)*v*dx-eta*v*dx + eta*eta*eta*v*dx + dot(grad(eta),grad(v))*dx
    F_imp = (eta-eta_n)/Constant(dt)*v*dx \
            + 0.5*dot(grad(eta),grad(v))*dx - 0.5*(- eta * ( eta-1 ) * ( eta+1 ) * v * dx) \
            + 0.5*dot(grad(eta_n),grad(v))*dx - 0.5*(- eta_n * ( eta_n-1 ) * ( eta_n+1 ) * v * dx)

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

    t = 0 
    counter = 0

    Rs = [] 
    ts = []
    a1 = time.time()
    R = computeArea(eta_n,V2)
    Rs.append(R)
    ts.append(t)
    a2 = time.time()-a1

    # with open('implicit_amg_2D_CPU.txt', 'a') as fCPU:
    #     fCPU.write("Mesh:%s\n" % grid_point)
    #     fCPU.write("time for computing radius:%s\n" % a2)

    print("time for computing radius: ",a2)

    # vtkfile = File("implicit_amg_2D/eta.pvd")
    # vtkfile << (eta_n,t)

    
    while t < T: 
        a1 = time.time()
        t += dt
        print("===========================> current time: ",t)

        #solve(F_imp == 0,eta,None)
        problem = NonlinearVariationalProblem(F_imp, eta, None, Jac) # Jacobian is passed in here
        solver = NonlinearVariationalSolver(problem)

        # added by Lian 
        prm = solver.parameters
        prm['newton_solver']['relaxation_parameter'] = 0.9
        prm['newton_solver']['maximum_iterations'] = 100
        prm['newton_solver']['absolute_tolerance'] = 1E-7
        prm['newton_solver']['relative_tolerance'] = 1E-9
        #prm['newton_solver']['linear_solver'] = 'pcg'  # 'mumps', 'gmres', 'bicgstab'
        #prm['newton_solver']['linear_solver'] = 'superlu_dist'  # 'mumps', 'gmres', 'bicgstab'
        #prm['newton_solver']['preconditioner'] = 'hypre_amg'   # 'hypre_euclid'
        #prm['newton_solver']['krylov_solver']['nonzero_initial_guess'] = True
        #solver.set_operator(A)  
        #prm['newton_solver']['krylov_solver']['monitor_convergence'] = True
        prm['newton_solver']['krylov_solver']['absolute_tolerance'] = 1E-10
        prm['newton_solver']['krylov_solver']['relative_tolerance'] = 1E-6
        # prm['newton_solver']['linear_solver'] = 'gmres'  # 'mumps', 'gmres', 'bicgstab'
        # prm['newton_solver']['preconditioner'] = 'ilu'   # 'hypre_euclid'
        
        # prm['newton_solver']['linear_solver'] = 'bicgstab'  # 'mumps', 'gmres', 'bicgstab'
        # prm['newton_solver']['preconditioner'] = 'ilu'   # 'hypre_euclid'

        #prm['newton_solver']['linear_solver'] = 'mumps'  # 'mumps', 'gmres', 'bicgstab'
        
        prm['newton_solver']['linear_solver'] = 'cg'  # 'mumps', 'gmres', 'bicgstab'
        #prm['newton_solver']['preconditioner'] = 'amg'   # 'hypre_euclid'

        # solver.parameters['newton_solver']['linear_solver'].set_operator(u)
        #PETScOptions.set("mg_levels_ksp_type", "chebyshev")
        #PETScOptions.set("mg_levels_pc_type", "jacobi")
        # PETScOptions.set("mg_levels_esteig_ksp_type", "cg")
        #PETScOptions.set("mg_levels_ksp_chebyshev_esteig_steps", 50)

        
        # prm['newton_solver']['error_on_nonconvergence'] = False

        solver.solve()
        eta_n.assign(eta)

        counter += 1
        if counter % print_ferquence == 0:
            #vtkfile << (eta,t)
            R = computeArea(eta,V2)
            Rs.append(R) #better to write to a file
            ts.append(t)
    
        a2 = time.time()-a1
        print("time used for 1 time step: ",a2)

    # with open('implicit_amg_2D_CPU.txt', 'a') as fCPU:
    #     fCPU.write("time for solving the PDE: %s\n" % a2)
    # print("time for solving the PDE: ",a2)
    # print("numerical solutions: ")
    # print(Rs)
    # print(ts)
    # elapsed_time = time.time() - start_time
    # print('User message ===> Computation time: ', elapsed_time, ' s', flush=True)

    with open('imp_CN_2D_Area_'+str(grid_point)+'.txt', 'w') as fR:
        for item in Rs:
            fR.write("%s\n" % item)

    with open('imp_CN_2D_t_'+str(grid_point)+'.txt', 'w') as ft:
        for item in ts:
            ft.write("%s\n" % item)


h = [math.pi * 100 * 100 - 2.0 * math.pi * t for t in ts]
Rs = [R + h[0] - Rs[0] for R  in Rs]
print("numerical solution: ")
print(Rs)

# plt.plot(ts, h,'-', label='Area: reference')
# plt.plot(ts, Rs,'*', label='Area: implicit')

# plt.legend()
# plt.savefig('implicit_2D.png')
# plt.show()
