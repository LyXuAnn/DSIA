import numpy as np
import pandas as pd
import tensorflow as tf
import random as rndm
import time
import netCDF4 as nc
import os
import math


# import related projection model ---------------------------------------------------------------
matr_pca = np.array(pd.DataFrame(pd.read_csv('../Data/redc/U_ROMS_RPCA100.txt', header=None)), dtype=np.float32)[:, :30]
cpu_config = tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8,
                            device_count={'CPU': 8})
# -------------------------------------------------------------------------------


# parameter ---------------------------------------------------------------------
# CNOP parameter
initial_file = "860_479-sym.nc"
lastnum = 100  # double variation extraction time
constraint = 400e9  # 600e9
u_base = np.zeros((4, 110, 55), dtype=np.float32)
v_base = np.zeros((4, 109, 56), dtype=np.float32)
zeta_base = np.zeros((110, 56), dtype=np.float32)
u_base_evol = np.zeros((4, 110, 55), dtype=np.float32)
v_base_evol = np.zeros((4, 109, 56), dtype=np.float32)
zeta_base_evol = np.zeros((110, 56), dtype=np.float32)
# dimension parameter
m_origin = 54776  # origin space dimension
m_reduced = 30  # feature space dimension
# algorithm parameter
n = 20  # number of nest
initial_ratio = -0.3  # -0.3
steppara = 0.8  # 0.2, 0.8
#  N_max = 30  # max number of iteration
weight = 0.9
dec_weight = 0.01
lim_weight = 0.6
# algorithm swarm parameter
nest = np.zeros((n, m_reduced), dtype=np.float32)  # PSO swarm array
vel = np.zeros((n, m_reduced), dtype=np.float32)  # PSO velocity array
localnest = np.zeros((n, m_reduced), dtype=np.float32)  # PSO local best swarm array
best = np.zeros(m_reduced, dtype=np.float32)  # PSO best swarm array
# algorithm value parameter
localfit = np.full(n, -1e18)  # PSO local best value
fbest = -1e18  # PSO global best value
# --------------------------------------------------------------------------------


# file operation -----------------------------------------------------------------
# read initial state of double gyre
def read_uvbase():
    global u_base, v_base, zeta_base

    ncfile = nc.Dataset('/home/yikui/workspace/ROMS/Test/double_gyre/CNOP/data/base_504.nc')

    u_base = ncfile.variables['u'][0]
    v_base = ncfile.variables['v'][0]
    zeta_base = ncfile.variables['zeta'][0]

    ncfile.close()

    return


# read origin terminated state of double gyre
def read_uvbase_evol():
    global u_base_evol, v_base_evol, zeta_base_evol
    global lastnum

    ncfile = nc.Dataset('/home/yikui/workspace/ROMS/Test/double_gyre/CNOP/data/base_504_evol.nc')

    u_base_evol = ncfile.variables['u'][lastnum]
    v_base_evol = ncfile.variables['v'][lastnum]
    zeta_base_evol = ncfile.variables['zeta'][lastnum]

    ncfile.close()

    return


# edit new initial state file of double gyre
def edit_netcdf(pert_u, pert_v, pert_zeta):
    global u_base, v_base, zeta_base

    netcdf_u = u_base + pert_u
    netcdf_v = v_base + pert_v
    netcdf_zeta = zeta_base + pert_zeta

    ncfile = nc.Dataset('/home/yikui/workspace/ROMS/Test/double_gyre/CNOP/data/cnopInput_504.nc', mode='a')

    ncfile.variables['u'][0] = netcdf_u
    ncfile.variables['v'][0] = netcdf_v
    ncfile.variables['zeta'][0] = netcdf_zeta

    ncfile.close()

    return


# write final cnop
def write_cnop():
    global sess
    global best
    global m_origin

    # decoder the low dimension solution
    mat1 = tf.placeholder(tf.float32, [None, m_reduced])
    mat2 = tf.placeholder(tf.float32, [m_reduced, m_origin])
    ret_mat = tf.matmul(mat1, mat2)
    best_origin = []
    with tf.Session(config=cpu_config) as sess:
        sess.run(tf.global_variables_initializer())
        generator_out = sess.run(ret_mat, feed_dict={mat1: best.reshape(-1, m_reduced), mat2: matr_pca.T})
        best_origin = generator_out.reshape(m_origin)
    # write the solution into the nc data
    cnop_u = best_origin[0:24200].reshape(4, 110, 55)
    cnop_v = best_origin[24200:48616].reshape(4, 109, 56)
    cnop_zeta = best_origin[48616:m_origin].reshape(110, 56)

    ncfile = nc.Dataset('/home/yikui/workspace/ROMS/Test/double_gyre/CNOP/data/cnop.nc', mode='a')

    ncfile.variables['u'][0] = cnop_u
    ncfile.variables['v'][0] = cnop_v
    ncfile.variables['zeta'][0] = cnop_zeta

    ncfile.close()

    return


# write final cnop state
def cnop_evol():
    global sess
    global best
    global m_origin

    # decoder the low dimension solution
    mat1 = tf.placeholder(tf.float32, [None, m_reduced])
    mat2 = tf.placeholder(tf.float32, [m_reduced, m_origin])
    ret_mat = tf.matmul(mat1, mat2)
    best_origin = []
    with tf.Session(config=cpu_config) as sess:
        sess.run(tf.global_variables_initializer())
        generator_out = sess.run(ret_mat, feed_dict={mat1: best.reshape(-1, m_reduced), mat2: matr_pca.T})
        best_origin = generator_out.reshape(m_origin)
    # write the solution into the nc data
    cnop_u = best_origin[0:24200].reshape(4, 110, 55)
    cnop_v = best_origin[24200:48616].reshape(4, 109, 56)
    cnop_zeta = best_origin[48616:m_origin].reshape(110, 56)

    # run the best solution to record
    edit_netcdf(cnop_u, cnop_v, cnop_zeta)
    os.system('./double_gyre.sh &> log')

    return
# --------------------------------------------------------------------------------


# assistant computation ----------------------------------------------------------
# calculate constraint by detailed data
def cal_constraint(u, v, zeta):
    sum = 0.0

    # potential norm
    for x in range(0, 55):
        for y in range(0, 109):
            sum = sum + 0.5 * 9.8 * (zeta[y][x]**2) * 18500 * 18500

    # kinetic energy
    for x in range(0, 55):
        for y in range(0, 109):
            for z in range(0, 4):
                sum = sum + 0.5 * 125 * (u[z][y][x]**2 + v[z][y][x]**2) * 18500 * 18500

    return sum


# project the solution exceed the limitation to the constraint space
def project(solution, i):
    global m_origin, constraint

    # calculate constraint norm value
    sum = cal_constraint_sol(solution)

    # project the unproper solution
    if sum > constraint:
        print('No.', i, 'projected!!!!!!!!!!')
        print('--origin sum = %e' % sum)
        para = math.sqrt(constraint) / math.sqrt(sum)
        solution = solution * para
        sum = cal_constraint_sol(solution)
        print('nest', i, '--sum after project = %e' % sum)
    else:
        print('nest', i, '--sum without project = %e' % sum)

    return solution


# calculate constraint by the solution
def cal_constraint_sol(solution):
    global sess
    global m_origin

    # decoder the low dimension solution
    mat1 = tf.placeholder(tf.float32, [None, m_reduced])
    mat2 = tf.placeholder(tf.float32, [m_reduced, m_origin])
    ret_mat = tf.matmul(mat1, mat2)
    solution_origin = []
    with tf.Session(config=cpu_config) as sess:
        sess.run(tf.global_variables_initializer())
        generator_out = sess.run(ret_mat, feed_dict={mat1: solution.reshape(-1, m_reduced), mat2: matr_pca.T})
        solution_origin = generator_out.reshape(m_origin)
    # write the solution into the nc data structure
    solution_u = solution_origin[0:24200].reshape(4, 110, 55)
    solution_v = solution_origin[24200:48616].reshape(4, 109, 56)
    solution_zeta = solution_origin[48616:m_origin].reshape(110, 56)

    # calculate constraint norm value
    sum = cal_constraint(solution_u, solution_v, solution_zeta)

    return sum
# --------------------------------------------------------------------------------


# adaption computation -----------------------------------------------------------
# calculate adaption value
def getAdapValue():
    global u_base_evol, v_base_evol, zeta_base_evol
    global lastnum
    adapValue = 0.0

    # obtain the new evol
    ncfile = nc.Dataset('/home/yikui/workspace/ROMS/Test/double_gyre/Forward/gyre3d_his_01.nc')

    nu_base_evol = ncfile.variables['u'][lastnum]
    nv_base_evol = ncfile.variables['v'][lastnum]
    nzeta_base_evol = ncfile.variables['zeta'][lastnum]

    ncfile.close()

    # obtain the development of the pert
    pert_u_end = nu_base_evol - u_base_evol
    pert_v_end = nv_base_evol - v_base_evol
    pert_zeta_end = nzeta_base_evol - zeta_base_evol

    # obtain the energy norm
    for x in range(0, 35):
        for y in range(39, 70):
            adapValue = adapValue + 0.5 * 9.8 * (pert_zeta_end[y][x]**2) * 18500 * 18500
            for z in range(0, 4):
                adapValue = adapValue + 0.5 * 125 * (pert_u_end[z][y][x]**2 + pert_v_end[z][y][x]**2) * 18500 * 18500

    return adapValue


# the process to calculate adaption value
def adapFunction(solution):
    global sess
    global m_origin

    # decoder the low dimension solution
    mat1 = tf.placeholder(tf.float32, [None, m_reduced])
    mat2 = tf.placeholder(tf.float32, [m_reduced, m_origin])
    ret_mat = tf.matmul(mat1, mat2)
    solution_origin = []
    with tf.Session(config=cpu_config) as sess:
        sess.run(tf.global_variables_initializer())
        generator_out = sess.run(ret_mat, feed_dict={mat1: solution.reshape(-1, m_reduced), mat2: matr_pca.T})
        solution_origin = generator_out.reshape(m_origin)
    # write the solution into the nc data structure
    solution_u = solution_origin[0:24200].reshape(4, 110, 55)
    solution_v = solution_origin[24200:48616].reshape(4, 109, 56)
    solution_zeta = solution_origin[48616:m_origin].reshape(110, 56)

    # run the solution in ROMS to compute the adapaValue
    edit_netcdf(solution_u, solution_v, solution_zeta)
    os.system('./double_gyre.sh &> log')
    adapValue = getAdapValue()

    return adapValue
# --------------------------------------------------------------------------------


# algorithm function -------------------------------------------------------------
# self defined matrix multiply interface to avoid the thread lock of numpy
def vec_mul_matrix(vec, mat):
    ret_vec = np.zeros((mat.shape[1]), dtype=np.float32)
    for i in range(mat.shape[1]):
        ret_vec[i] = np.linalg.norm(vec, mat[:][i])
        # for j in range(mat.shape[0]):
        #     ret_vec[i] += vec[j] * mat[j][i]
    return ret_vec


# initial PSO related solution with double gyre data
def initial_solutions():
    global sess
    global initial_file
    global initial_ratio, steppara
    global m_reduced
    global nest, vel

    # read the initial data by the ncfile
    ncfile = nc.Dataset('/home/yikui/workspace/ROMS/Test/double_gyre/CNOP/data/' + initial_file)

    u_diff = ncfile.variables['u'][0]
    v_diff = ncfile.variables['v'][0]
    zeta_diff = ncfile.variables['zeta'][0]

    ncfile.close()

    # calculate constraint of initial data
    sum = cal_constraint(u_diff, v_diff, zeta_diff)
    print('--constraint of initial file: %e' % sum)

    # flatten the data
    diff_origin = np.concatenate((u_diff.reshape(24200), v_diff.reshape(24416), zeta_diff.reshape(6160)))
    origin = diff_origin * initial_ratio

    # encoder the high dimension data
    # encoder_solution = np.dot(origin, matr_pca)
    # encoder_solution = vec_mul_matrix(origin, matr_pca)
    mat1 = tf.placeholder(tf.float32, [None, m_origin])
    mat2 = tf.placeholder(tf.float32, [m_origin, m_reduced])
    ret_mat = tf.matmul(mat1, mat2)
    encoder_solution = []
    with tf.Session(config=cpu_config) as sess:
        sess.run(tf.global_variables_initializer())
        generator_out = sess.run(ret_mat, feed_dict={mat1: origin.reshape(-1, m_origin), mat2: matr_pca})
        encoder_solution = generator_out.reshape(m_reduced)
    nest[0, :] = encoder_solution
    vel[0, :] = 0

    # initialize the swarm
    for i in range(1, n):
        for j in range(0, m_reduced):
            rnd = rndm.random() - 0.5
            vel[i, j] = rnd * steppara
            nest[i, j] = encoder_solution[j] + vel[i, j]

    # project the initial swarm
    for i in range(n):
        nest[i, :] = project(nest[i, :], i)

    return


# update swarm
def getNewNest(weight):
    global n, nest, vel, localnest, best
    c1 = 2.0
    c2 = 2.0

    for i in range(n):
        rnd1 = rndm.random()
        rnd2 = rndm.random()
        vel[i, :] = weight * vel[i, :] + c1 * rnd1 * (best - nest[i, :]) + c2 * rnd2 * (localnest[i, :] - nest[i, :])
        nest[i, :] = nest[i, :] + vel[i, :]
        nest[i, :] = project(nest[i, :], i)

    return


# summary for the solution
def getBestNest():
    global n, nest, localnest, best, localfit, fbest

    for i in range(n):
        fnew = 0.0
        print('call adapFunction No.', i)
        fnew = adapFunction(nest[i, :])
        if fnew > localfit[i]:
            localfit[i] = fnew
            localnest[i, :] = nest[i, :].copy()
            if localfit[i] > fbest:
                fbest = localfit[i]
                best = localnest[i, :].copy()
                print('***bestnest & fbest changed to', i, ', %e' % fbest)

    sum = cal_constraint_sol(best)
    print('constraint of current best: %e' % sum)

    return
# --------------------------------------------------------------------------------


# main process -------------------------------------------------------------------
start_time = time.time()
print('start')

read_uvbase()
print('read_uvbase finished')

read_uvbase_evol()
print('read_uvbase_evol finished')

initial_solutions()
getBestNest()

while weight >= lim_weight:
    print('------------Iteration :', weight, '------------')
    getNewNest(weight)
    getBestNest()
    weight -= dec_weight

print('***AdapValue: %e' % fbest)

write_cnop()
print('------CNOP written in ../data/cnop.nc------')

end_time = time.time()
print('CPU Time = ', end_time-start_time)

cnop_evol()
print('------CNOP evol written in ../../Forward/his.nc------')

sess.close()
# --------------------------------------------------------------------------------
