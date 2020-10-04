from fenics import *
import time 
# import matplotlib.pyplot as plt 
import numpy as np
# from ufl import *
#define mesh and function spaces


# a = [0.0, -0.0001, -0.00017, -0.00024000000000000003, -0.00031, -0.00038, -0.00045000000000000004, -0.0005200000000000001, -0.00059, -0.0006600000000000001, -0.0007300000000000001, -0.0008, -0.0008700000000000001, -0.0009500000000000001, -0.00102, -0.00109, -0.00116, -0.0012300000000000002, -0.0013000000000000002, -0.0013700000000000001, -0.00144]
# b = [0, 0.49999999999999994, 1.0000000000000002, 1.5000000000000007, 2.000000000000001, 2.499999999999999, 2.9999999999999973, 3.4999999999999956, 3.999999999999994, 4.499999999999992, 4.99999999999999, 5.4999999999999885, 5.999999999999987, 6.499999999999985, 6.999999999999983, 7.499999999999981, 7.99999999999998, 8.499999999999986, 8.999999999999993, 9.5, 10.000000000000007]
# plt.plot(b,a,'o-')
# plt.show()


#helper functions for weak form
def f(eta): 
	return 4*eta**3 - 4*eta	
'''
Here we need to be careful that passed argument
is a UFL object, more specifically a TrialFunction, 
therefore, we need to be careful with the operations 
that are compatible with UFL objects 
'''
def tanh_derivative(eta): 
	return pow(2*exp(eta)/(exp(2*eta)+1),2)
    # return (2*exp(eta)/(exp(2*eta)+1))**2 # these two choices are same

def tanh(eta): #need to check this
	return (exp(-eta)-exp(eta))/(exp(eta)+exp(-eta))

def get_location(eta):
    temp = abs(eta.vector().get_local()) # from right to left order
    index = np.argmin(temp)
    # print(index,len(temp)//2)
    # print(eta.vector().get_local())
    # position = 
    return (len(temp)//2 - index)*(2.0/(len(temp)-1))
    # values = eta.vector().get_local()
    # result = np.where((values> -1e-10)&(values < 1e-10))
    # print(result)
    # if len(result) != 1: 
    #     print("error detecting eta = 0 ")
    # return result[0]

def get_location_interpolation(eta,length):
    temp = eta.vector().get_local() # from right to left order
    index = 0
    grid_size = (length/(len(temp)-1))
    for i in range(len(temp)):
        if temp[i] <= DOLFIN_EPS: #first smaller than zero
        # if temp[i] <= 0:
            index = i
            break 
    coarse_dist =  (len(temp)//2 - index)*grid_size
    x1 = abs(temp[index])
    x2 = abs(temp[index-1]) #bigger or equal to 0   
    #interpolate_dist = coarse_dist + (x1/x2)/(1+x1/x2)*grid_size
    interpolate_dist = coarse_dist + (x1)/(x2+x1)*grid_size
    return interpolate_dist


# eta = Constant(1)
# print(eta)
# print(tanh_derivative(eta))
# print(f(eta))
# x_0 = -25.6
# x_N = 25.6
x_0 = -1
x_N = 1
length = x_N - x_0
grid_num = 512#use even number
mesh = IntervalMesh(grid_num,x_0,x_N)

V = FunctionSpace(mesh,"P", 1)
eta = Function(V)
eta_n = Function(V)
v = TestFunction(V)

T = 10000
dt = 0.1
figure = False

print_frequency = 10
for epsilon in [0.01]:
    for h in [1e-6]:

        with open('implicit_location_'+str(epsilon)+'_h'+str(h)+'.txt', 'w') as fR:
            fR.write("This file is used to record distance, dt = "+str(dt)+": \n")
        with open('implicit_t_'+str(epsilon)+'.txt', 'w') as ft:
            ft.write("This file is used to record distance: \n")

        # eta_0 = Expression('tanh(x[0]/sqrt(eps))',degree = 2,eps = epsilon)
        eta_0 = Expression('tanh(sqrt(2)*x[0]/sqrt(eps))',degree = 2,eps = epsilon)
        eta_n.assign(eta_0)
        eta.assign(eta_0)
        #create variational form 
        # F_imp = (eta - eta_n)/Constant(dt)*v*dx + f(eta)*v*dx \
        # 		+ inner(grad(eta),grad(v))*dx - tanh_derivative(eta)*h*v*dx
        # from ufl import *
        # F_imp = (eta - eta_n)/Constant(dt)*v*dx + f(eta)*v*dx \
        # 		+ Constant(epsilon)*inner(grad(eta),grad(v))*dx \
        #         - 1/Constant(sqrt(epsilon))*tanh_derivative(eta/Constant(sqrt(epsilon)))*Constant(h)*v*dx
        F_imp = (eta - eta_n)/Constant(dt)*v*dx + f(eta)*v*dx \
                + Constant(epsilon)*inner(grad(eta),grad(v))*dx \
                -(Constant(3.0/2)*eta*eta - Constant(3.0/2))*Constant(h)/Constant(sqrt(epsilon))*v*dx
        # F_imp = (eta - eta_n)/Constant(dt)*v*dx + f(eta)*v*dx \
        # + Constant(epsilon)*inner(grad(eta),grad(v))*dx 

        deta = TrialFunction(V)
        Jac = derivative(F_imp,eta,deta)


        # set up solver 
        t = 0
        counter = 0
        # vtkfile = File("solution/eta.pvd")
        # vtkfile << (eta,t)
        locations = []
        ts = []
        loc = get_location_interpolation(eta,length)
        locations.append(loc)
        ts.append(t)
        # with open('implicit_location_'+str(epsilon)+'_h'+str(h)+'.txt', 'a') as fR:
        #     fR.write("%s\n" % loc)
        # with open('implicit_t_'+str(epsilon)+'.txt', 'a') as ft:
        #     ft.write("%s\n" % t)

        # eta_n.rename(str(epsilon),str(epsilon))
        # vtkfile << (eta_n,t)
        # if figure: 
        #     plt.figure()
        #     plot(eta)
        #     plt.show()

        start_time = time.time()
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
            prm['newton_solver']['absolute_tolerance'] = 1E-13
            prm['newton_solver']['relative_tolerance'] = 1E-13
            #prm['newton_solver']['linear_solver'] = 'pcg'  # 'mumps', 'gmres', 'bicgstab'
            #prm['newton_solver']['linear_solver'] = 'superlu_dist'  # 'mumps', 'gmres', 'bicgstab'
            #prm['newton_solver']['preconditioner'] = 'hypre_amg'   # 'hypre_euclid'
            prm['newton_solver']['krylov_solver']['nonzero_initial_guess'] = True
            #solver.set_operator(A)  
            prm['newton_solver']['krylov_solver']['monitor_convergence'] = True
            prm['newton_solver']['krylov_solver']['absolute_tolerance'] = 1E-18
            prm['newton_solver']['krylov_solver']['relative_tolerance'] = 1E-18
            # prm['newton_solver']['linear_solver'] = 'gmres'  # 'mumps', 'gmres', 'bicgstab'
            # prm['newton_solver']['preconditioner'] = 'ilu'   # 'hypre_euclid'
            
            # prm['newton_solver']['linear_solver'] = 'bicgstab'  # 'mumps', 'gmres', 'bicgstab'
            # prm['newton_solver']['preconditioner'] = 'ilu'   # 'hypre_euclid'

            #prm['newton_solver']['linear_solver'] = 'mumps'  # 'mumps', 'gmres', 'bicgstab'
            
            prm['newton_solver']['linear_solver'] = 'cg'  # 'mumps', 'gmres', 'bicgstab'
            # prm['newton_solver']['preconditioner'] = 'amg'   # 'hypre_euclid'

            solver.solve()
            eta_n.assign(eta)
            # vtkfile << (eta,t)
            if counter % print_frequency == 0:
                loc = get_location_interpolation(eta,length)
                locations.append(loc)
                ts.append(t)
                with open('implicit_location_'+str(epsilon)+'_h'+str(h)+'.txt', 'a') as fR:
                    fR.write("%s\n" % loc)
                with open('implicit_t_'+str(epsilon)+'.txt', 'a') as ft:
                    ft.write("%s\n" % t)
                # if figure: 
                #     plt.figure()
                #     plot(eta)
                #     plt.show()

    #     print(locations)
    # reference_sol = [3.0/(2*sqrt(2))*h*i for i in ts]
    # # reference_sol = [-h/sqrt(2)*i for i in ts]
    # print(ts)
    # plt.figure()
    # plt.plot(ts,locations,'o-',label = "numerical")
    # plt.plot(ts,reference_sol,'-',label = "reference")
    # plt.legend()
    # plt.title("epsilon = "+str(epsilon)+" h = "+str(h))
    # plt.show()

    # solve the problem (time dependent loop)