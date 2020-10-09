from fenics import *
import time
import numpy as np
import math
from scipy import stats
import os.path
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
    return Area, radius

def ac2d_fis1st_solver(epsilon,grid_num,t_step,output_freq,vtk_save,solver_choice,precond,T = 0.2):

    dt =t_step 
    counter = 0  
    output_count = 0
    mesh = RectangleMesh(Point(-1,-1),Point(1,1),grid_num-1,grid_num-1)
    folder_name = "eps_"+str(epsilon)+"grid_"+str(grid_num)+"dt_"+str(dt)
    os.mkdir(folder_name)
    text_fname = os.path.join(folder_name, 'eps'+str(epsilon)+'_fis_1st_2D'+'.txt' ) 
    with open(text_fname, 'w') as f:
        f.write('\t'.join(["count","time","radius","area"])+'\n')

    P1 = FiniteElement('Lagrange',mesh.ufl_cell(),1)
    V1 = FunctionSpace(mesh,P1)
    eta = Function(V1)
    v = TestFunction(V1)
    eta_n = Function(V1)
    #initial condition
    eta_0 = Expression('tanh((sqrt(x[0]*x[0]+ x[1]*x[1])-0.6)/(sqrt(2)*eps))',degree = 2,eps = epsilon )
    eta_n.assign(eta_0)
    
    #initilize solution
    eta.assign(eta_0)

    #weak form
    F_imp = (eta-eta_n)/Constant(dt)*v*dx + dot(grad(eta),grad(v))*dx - 1/(Constant(epsilon)*Constant(epsilon))*(- eta * ( eta-1 ) * ( eta+1 ) * v * dx)
    deta = TrialFunction(V1)
    Jac = derivative(F_imp,eta,deta)

    t = 0 
    counter = 0
    Rs = [] 
    ts = []
    Area, R = computeArea(eta_n) #compute initial area, radius
    with open(text_fname, 'a') as f:
        f.write('\t'.join([str(output_count),str(t),str(R),str(Area)])+'\n')
    Rs.append(R)
    ts.append(t)

    if vtk_save: 
        vtkfile = File(folder_name+"/eta.pvd")
        eta_n.rename("eta","eta")
        vtkfile << (eta_n,t)
    while t < T: 
        t += dt
        counter += 1
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
        prm['newton_solver']['krylov_solver']['relative_tolerance'] = 1E-8
        # prm['newton_solver']['linear_solver'] = 'gmres'  # 'mumps', 'gmres', 'bicgstab'
        # prm['newton_solver']['preconditioner'] = 'ilu'   # 'hypre_euclid'
        
        # prm['newton_solver']['linear_solver'] = 'bicgstab'  # 'mumps', 'gmres', 'bicgstab'
        # prm['newton_solver']['preconditioner'] = 'ilu'   # 'hypre_euclid'

        #prm['newton_solver']['linear_solver'] = 'mumps'  # 'mumps', 'gmres', 'bicgstab'

        prm['newton_solver']['linear_solver'] = solver_choice  # 'mumps', 'gmres', 'bicgstab'
        if solver_choice != 'lu': 
            prm['newton_solver']['preconditioner'] = precond   # 'hypre_euclid'
        solver.solve()
        eta_n.assign(eta)
        if counter % output_freq == 0:
            output_count += 1 
            if vtk_save: 
                eta.rename("eta","eta")
                vtkfile << (eta,t)
            Area, R = computeArea(eta)
            with open(text_fname, 'a') as f:
                f.write('\t'.join([str(output_count),str(t),str(R),str(Area)])+'\n')
            Rs.append(R) #better to write to a file
            ts.append(t)

    print(ts)
    print(Rs)

ac2d_fis1st_solver(0.04,257,0.001,output_freq = 1,vtk_save = True,solver_choice = 'lu',precond='none',T = 0.2)