import numpy as np
from skimage.io import imread, imsave
from numpy import linalg as LA
from scipy.spatial.transform import Rotation


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
                 
b3 = np.float32([-0.246965886,  -0.047780764,  0.229818354]).T
c3 = 0.642377030




#https://stackoverflow.com/questions/43507491/imprecision-with-rotation-matrix-to-align-a-vector-to-an-axis for calc_R

e = 0.0001
n_0 = np.array([0,0,1], dtype = float)
max_iter = 100

#return r such that n_0 = R x n
def calc_R(n):
    uvw = np.cross(n, n_0)
    rcos = np.dot(n, n_0)
    rsin = LA.norm(uvw)
    if not np.isclose(rsin, 0):
        uvw /= rsin
    u, v, w = uvw
    
    tmp = np.array([[ 0, -w,  v],
                    [ w,  0, -u],
                    [-v,  u,  0]])
    r =  rcos * np.eye(3) + rsin * tmp + (1.0 - rcos) * uvw[:,None] * uvw[None,:]
    
    #n.shape
    #n_0.shape
    
    #r = Rotation.align_vectors(n.reshape(1,3), n_0.reshape(1,3))
    return r

def shade_x(n):
    s = np.zeros((3,1), dtype = float)
    s[0,:] = n @ A1 @ n.T + b1.T @ n.T + c1
    s[1,:] = n @ A2 @ n.T + b2.T @ n.T + c2
    s[2,:] = n @ A3 @ n.T + b3.T @ n.T + c3
    return s

#E = ||f||^2  = ||s(n) - I_x||^2
#return f 3x1 vector
def calc_f(n, I):
    #apply shading function 
    s = shade_x(n).copy()
    f = np.zeros((3,1), dtype = float)
    
    f = s - I
    return f


def jacob_n(n):
    Jn = np.zeros((3,3), dtype=float)
    Jn[0, :] = n @ A1 + b1.T 
    Jn[1, :] = n @ A2 + b2.T
    Jn[2, :] = n @ A3 + b3.T
    return Jn

#first compute a 3x3 J(n), 
#then compute and return 3x2 Juv 
def jacob_uv(n, R):
    Jn = jacob_n(n)
    u, v, r = n
    nuv = np.array([[1, 0],
                    [0, 1],
                    [-u/r, -v/r]])
    Juv = Jn @ R @ nuv
    return Juv

#Given an observed color (vector) img, minimize the error 
#with respect to  normal vector

#n should be 3x1 vectors, A 3x3 matrix, b 3x1 vector
#h is 3x1, n_{i+1} = n_i + h
#update function: Juv.T @ Juv @ h = -Juv.T @ f(n_i)
#return the corresponding normal that minimize the Error
def G_N(I_x):
    #R_0 map n_0 to [0,0,1]
    R_0 = np.eye(3)
    
    #init guess R @ n_uv = n maps to n_0
    #n = n_0.copy()
    n_uv = n_0.copy()
    #R = R_0.copy()
    
    #other init guess
    n1 = np.sqrt(0.3)
    n2 = np.sqrt(0.3)
    n = np.array([n1,n2, np.sqrt(1 - (n1**2 + n2**2))])
    R = LA.inv(calc_R(n))
    
    n = R @ n_uv
    R = LA.inv(calc_R(n))
    
    err = 0
    iter = 0
    
    f = calc_f(n, I_x)

    err = LA.norm(f)
    
    while(err > e and iter < max_iter):
        iter = iter + 1 
        Juv = jacob_uv(n,R)
        #print("iter:", iter)
        #update
        h = -1.0 * LA.pinv(Juv.T @ Juv) @ Juv.T @ f
        n_uv[0:2] = n_uv[0:2] + h.T
        u, v, r = n_uv
        
        #reset
        if (u**2 + v**2) >= 0.5:
            #new guess
            n = R @ n_uv
            n = n/LA.norm(n)
            R = LA.inv(calc_R(n))
            n_uv = n_0.copy()

            n = R @ n_uv
            R = LA.inv(calc_R(n))       #get rotation matrix
        else: 
            r = np.sqrt(1 - (u**2 + v**2))
            n_uv[2] = r
            n = (R @ n_uv.T).T  
            

        f = calc_f(n, I_x)
        err = LA.norm(f)
        #print("E:", err)    
    return n

#iterate through each pixel         
def natural_SFS(img, mask):
    h, w, c = img.shape
    print(img.shape)
    nrm = np.zeros(img.shape)
    for i in range(h):
        print(i)
        for j in range(w):
            if (mask[i, j] == 1):
                nrm[i, j, :] = G_N(img[i, j, :].reshape(3,1))
    return nrm
   

#CSE543 try





########################## Support code below

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')


# Utility functions to clip intensities b/w 0 and 1
# Otherwise imsave complains
def clip(im):
    return np.maximum(0.,np.minimum(1.,im))


############# Main Program





#simple test of G_N of a single point
n_test = np.array([np.sqrt(0.5), -np.sqrt(0.5), 0 ], dtype = float)
I_test = shade_x(n_test)
print("n_test", n_test)
print("I_test", I_test.T)

n_r = G_N(I_test).T
print("result", n_r)
print("shade result", shade_x(n_r).T)



#R @ n_0 = n_r
r = Rotation.align_vectors(n_r.reshape(1,3), n_0.reshape(1,3))
R = r[0].as_matrix()
print(R)

print(R @ n_0)
#test scipy.optimize


#7
# Load image data
# imgs = []
# for i in range(1,11):
    # imgs = imgs + [np.float32(imread(fn('test/inputs/im_b04_%02d.png' % i)))/255.]

# mask = np.float32(imread(fn('test/blob04_mask.png')) > 0)
# #
# nrm = natural_SFS(imgs[7],mask)
# #
# nimg = nrm/2.0+0.5
# nimg = clip(nimg * mask[:,:,np.newaxis])
# imsave(fn('test/outputs/my_nrm7.png'),nimg)



