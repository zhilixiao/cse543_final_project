import scipy
import numpy as np

patch_w = 3
patch_h = 3
k = 9
L = 3

#doge1--Doge's Palace Light--I07
A1 = np.float32([[-0.005896293,  0.003201143,  0.000909221],
                 [0.003201143,  0.007083686,  -0.002286545],
                 [0.000909221,  -0.002286545,  0.034119957]])
b1 = np.float32([-0.227557077,  -0.047123300,  0.205254578]).T
c1 = 0.673823976

A2= np.float32([[-0.006897805,  0.002766256,  0.003926461],
                [0.002766256,  0.008716128,  -0.001902301],
                [0.003926461,  -0.001902301,  0.032048057]])    
b2 = np.float32([-0.234017921,  -0.046864423,  0.214483105]).T
c2 = 0.650575084

A3 = np.float32([[-0.007922342,  0.002600586,  0.007483184],
                 [0.002600586,  0.010639797,  -0.001596472],
                 [0.007483184,  -0.001596472,  0.029887898]])
                 
b3 = np.float32([-0.246965886,  -0.047780764,  0.229818354])
c3 = 0.642377030


#put light into lists
L_A = [np.float32([[-0.005896293,  0.003201143,  0.000909221],
                 [0.003201143,  0.007083686,  -0.002286545],
                 [0.000909221,  -0.002286545,  0.034119957]]),
                 
     np.float32([[-0.006897805,  0.002766256,  0.003926461],
                [0.002766256,  0.008716128,  -0.001902301],
                [0.003926461,  -0.001902301,  0.032048057]]),

     np.float32([[-0.007922342,  0.002600586,  0.007483184],
                 [0.002600586,  0.010639797,  -0.001596472],
                 [0.007483184,  -0.001596472,  0.029887898]])]
                 
L_b = [np.float32([-0.227557077,  -0.047123300,  0.205254578]),
       np.float32([-0.234017921,  -0.046864423,  0.214483105]),
       np.float32([-0.246965886,  -0.047780764,  0.229818354])
       ]

L_c = [0.673823976, 0.650575084, 0.642377030]

#npatch.dim = 3x3x3, R.dim 3x3x3x3
def shade(npatch, R):
    #npatch flatten to 9x1x3
    npatch_flatten = npatch.reshape(k,3)
    R_flatten = R.reshape(k,3,3)
    
    #get R@N
    npatch_flatten = np.matmul(npatch_flatten[:,None],R_flatten.swapaxes(1,2))[:,0]
    
    S = np.zeros((L * k,1), dtype = float)
    
    #for i = 0,1,2, 
    for i in range(3)
        S_tmp = (np.sum(npatch_flatten @ L_A[i] * npatch_flatten, axis = 1) + L_b[i] @ npatch_flatten.T + L_c[i])
        S[i:L * k:3, :] = S_tmp
    
    
    return S



def f_cost(S, I):
    I_X = I.reshape(k,1,3)
    f_cost = S - I
    return f_cost


#return vector with the curl for each n1 to nk in seperate rows
def integrability_cost(npatch, R):
    uv = npatch[:,:, 0:2]
    
    npatch_flatten = npatch.reshape(k,3)
    R_flatten = R.reshape(k,3,3)
    rs_y = R_flatten[:, 0, 0:2].reshape(k,2);
    rs_x = R_flatten[:, 1, 0:2].reshape(k,2);
    
    #can be sub to patch_w and patch_h
    #TODO: need to add boundary
    uv_y = uv[1:3,:,:] - uv[0:2,:,:]
    uv_y_flatten = uv_y.reshape(k,2)
    
    uv_x = uv[:,1:3,:] - uv[:,0:2,:]
    uv_x_flatten = uv_x.reshape(k,2)
    
    c_y = np.sum(uv_y_flatten * rs_y, axis = 1)
    c_x = np.sum(uv_x_flatten * rs_x, axis = 1)
    
    curl = np.zeros((k,1), dtype = float)
    
    curl = c_y - c_x
    
    return curl

    
def smoothness_cost():

    

    return gaussian
