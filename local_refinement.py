import scipy
import math #for atan
import numpy as np
from scipy.ndimage import gaussian_filter
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

#patch_nrm is in uvr form
#npatch.dim = 3x3x3, R.dim 3x3x3x3, return S.dim=27x1
def shade(patch_nrm, R):
    #npatch flatten to 9x3
    npatch_flatten = patch_nrm.reshape(k,3)
    R_flatten = R.reshape(k,3,3)
    
    #get R@N, the true norm
    npatch_flatten = np.matmul(npatch_flatten[:,None],R_flatten.swapaxes(1,2))[:,0]
    
    S = np.zeros((L * k,1), dtype = float)
    
    #for L = 0,1,2, 
    for i in range(3)
        S_tmp = (np.sum(npatch_flatten @ L_A[i] * npatch_flatten, axis = 1) + L_b[i] @ npatch_flatten.T + L_c[i])
        S[i:L * k:3, :] = S_tmp
    
    #S is suppoesd to be 27x1
    return S

#img_nrm.dim = LxWx3, R.dim LxWx3x3, return S.dim=(3*LW)x1
def shade_img(img_nrm, R):
    H, W = img_nrm.shape
    #npatch flatten to 9x3
    img_nrm_flatten = img_nrm.reshape(H * W,3)
    R_flatten = R.reshape(H * W,3,3)
    
    #get R@N, the true norm
    img_nrm_flatten = np.matmul(img_nrm_flatten[:,None],R_flatten.swapaxes(1,2))[:,0]
    
    S = np.zeros((L * H * W,1), dtype = float)
    
    #for L = 0,1,2, 
    for i in range(3)
        S_tmp = (np.sum(img_nrm_flatten @ L_A[i] * img_nrm_flatten, axis = 1) + L_b[i] @ img_nrm_flatten.T + L_c[i])
        S[i:L * H * W:3, :] = S_tmp
    
    #S is suppoesd to be (3*LW)x1
    return S



#wrong dimension now, ingore it
def f_cost(S, I):
    I_X = I.reshape(k,1,3)
    f_cost = S - I
    return f_cost


#return vector with the curl for each n1 to nk in seperate rows
#npatch is the uv of nrms of the patch, R is the rotation matrix of the patch
#c1 should be kx1, with each row the value of curl of the norm
def integrability_cost(patch_nrm, R):
    uv = patch_nrm[:,:, 0:2]
    
    npatch_flatten = patch_nrm.reshape(k,3)
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

#need to modified to calculate the C1 cost of the entire image
def C1_img(patch_nrm, R):
    uv = patch_nrm[:,:, 0:2]
    
    npatch_flatten = patch_nrm.reshape(k,3)
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






#generate a gaussian kernel with std sigma and win_size
# def g2_kernel(win_size, sigma):
    # t = np.arange(win_size)
    # x, y = np.meshgrid(t, t)
    # o = (win_size - 1) / 2
    # r = np.sqrt((x - o)**2 + (y - o)**2)
    # scale = 1 / (sigma**2 * 2 * np.pi)
    # return scale * np.exp(-0.5 * (r / sigma)**2)



#have not integrated results from previous scale
#this can be done to the whole image, img's dim = LxW
#return G2 for the whole image of dim = LXWX3X3,
def compute_G2(img):
    L, W = img.shape 
    #vertical Iy
    I_y = np.gradient(img, axis = 0)
    sqr_I_y = I_y**2
    
    #horizontal Ix
    I_x = np.gradient(img, axis = 1)
    sqr_I_x = I_x**2
    
    #sigma = 1
    G_x = gaussian_filter(I_x, 1)
    
    G_y = gaussian_filter(I_y, 1)
    
    Gxy = gaussian_filter(I_x*I_y, 1)
    
    
    C = (G_x - G_y)/(G_x + G_y)
    
    S = (2 * Gxy)(G_x + G_y)
    
    #local orientation in radians
    theta = 0.5 * np.arctan2(S, C)
    
    ka = np.cos(theta)**2
    kb = -2 * np.cos(theta) * np.sin(theta)
    kc = np.sin(theta)**2
    
    x = np.arange(-1, 2, 1)
    y = np.arange(-1, 2, 1)
    xx, yy = np.meshgrid(x, y)
    G_2a = 0.9213 * (2 * x ** 2 - 1) * np.exp(-(xx**2 + yy**2))
    G_2b = 1.843 * xx * yy * np.exp(-(xx**2 + yy**2))
    G_2c = 0.9213 * (2 * y ** 2 - 1) * np.exp(-(xx**2 + yy**2))
    
    #select the theta at the middle of the patch, G2's dib  m = LxWx3x3
    G2 = (np.einsum('i,jk->ijk', ka.reshape(W*L,), G_2a) + np.einsum('i,jk->ijk', kb.reshape(W*L,) ,G_2b) \
            + np.einsum('i,jk->ijk', kc.reshape(W*L,), G_2c)).reshape(L,W,3,3)
     
    return G2


#compute only for a patch, the patch_G2 is the 3x3 G2 of the pixel in the middle of the patch
#C2 should be 2x1
def smoothness_cost(patch_G2, patch_nrm):
    u = patch_nrm[:,:, 0] 
    v = patch_nrm[:,:, 1]
    
    C2 = np.zeros(2,1)
    
    #might need sum twice for both axes
    C2[0,1] = np.sum(patch_G2 * u)
    C2[0,2] = np.sum(patch_G2 * v)
    return C2

#in progess
def Jacob(gcost):
    #shade entire image
    S = shade_img(img, img_R)
    #calcuate f cost of the entire image
    img_fcost = (img - S) ** 2   
    




