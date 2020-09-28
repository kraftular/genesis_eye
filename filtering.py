import numpy as np
from util import shape_assert

from filterpy.kalman import KalmanFilter


class LowPassIIRBase(object):
    """
    represents a generic 1-corner butterworth filter. use the great website
    below to generate constants for subclasses:

    https://www-users.cs.york.ac.uk/~fisher/mkfilter/trad.html

    in practice a low pass filter does not cut it for smoothing out our skeletal
    joint positions, it just spreads out the noise over a larger spectrum. see
    kalman filter code for next attempt.
    """
    NZEROS=2
    NPOLES=2
    def __init__(self,shape):
        self.shape = shape
        self.xv = np.zeros((self.NZEROS+1,)+tuple(shape),np.float32)
        self.yv = np.zeros((self.NPOLES+1,)+tuple(shape),np.float32)

    def filter(self,sample):
        shape_assert(sample,self.shape)
        self.xv[0:2,...] = self.xv[1:3,...]
        self.xv[2,...] = sample / self.GAIN
        self.yv[0:2,...] = self.yv[1:3,...]
        self.yv[2,...] = (self.xv[0,...] + self.xv[2,...]) + 2 * self.xv[1,...] \
                         + (self.MAGIC_1 * self.yv[0,...]) + (self.MAGIC_2 *
                                                              self.yv[1,...])
        return self.yv[2,...].copy()

    def __call__(self,sample):
        return self.filter(sample)
        
class LowPass8Hz(LowPassIIRBase):
    """
    30 hz sample frequency, 8 Hz corner frequency lowpass
    """
    GAIN=3.084091054e+00
    MAGIC_1 = -0.1742373434
    MAGIC_2 = -0.1227412250

class LowPass5Hz(LowPassIIRBase):
    GAIN=6.449489743e+00
    MAGIC_1= -0.2404082058
    MAGIC_2= 0.6202041029

#########################################################################
# wonderful: https://www-users.cs.york.ac.uk/~fisher/mkfilter/trad.html #
#                                                                       #
# static float xv[NZEROS+1], yv[NPOLES+1];                              #
#                                                                       #
# static void filterloop()                                              #
#   { for (;;)                                                          #
#       { xv[0] = xv[1]; xv[1] = xv[2];                                 #
#         xv[2] = next_input_value / GAIN;                              #
#         yv[0] = yv[1]; yv[1] = yv[2];                                 #
#         yv[2] =   (xv[0] + xv[2]) + 2 * xv[1]                         #
#                      + ( MAGIC_1 * yv[0]) + ( MAGIC_2 * yv[1]);       #
#         next_output_value = yv[2];                                    #
#       }                                                               #
#   }                                                                   #
#########################################################################


class Kalman2DPoint(object):
    """
    simplest type of Kalman filtering: treat each skeleton joint as an independent
    point, and use constant-acceleration kalman filtering on it.

    use a heuristic method for converting neural-net joint position confidence to 
    position variance. 

    solve for estimated 2D joint location based on N-way replicated Kalman filter.
    """

    def __init__(self,process_variance=0.001):
        self.kf = KalmanFilter(dim_x = 6, #x,y,x_dot,y_dot,x_accel,y_accel
                               dim_z = 2) #x,y
        #engineering: define dt to be 1 frame, simplifying all coefficients
        A = 0.5 #coeff to get position update from const acceleration
        self.kf.F = np.float32([
            [1,0,1,0,A,0], #y is updated by prev y + yv * dt + 0.5 ya * dt^2
            [0,1,0,1,0,A], #x is updated by prev x +   xv * dt + 0.5 xa dt^2
            [0,0,1,0,1,0], #yv is prev yv + ya * dt
            [0,0,0,1,0,1], #xv is prev xv + xa * dt
            [0,0,0,0,1,0], #ya is prev ya
            [0,0,0,0,0,1]]) #xa is prev xa

        self.kf.H = np.float32([
            [1,0,0,0,0,0],
            [0,1,0,0,0,0]]) #only x and y are observable

        #initialize x to nonsense value:
        self.kf.x = np.zeros((6,),dtype=np.float32)

        #initialize P to very high uncertainty (variance)
        self.kf.P *=10000

        #set the process noise variance. this is the noise introduced
        #by flaws in our model of constant-accel point motion, not by
        #flaws in measurement. The measurement noise is characterized
        #by R, later on.

        Q_a = np.float32([
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,1,0],
            [0,0,0,0,0,1]]) * process_variance
        #the process variance is modeled as a random acceleration applied
        #to points at each time step.

        self.kf.Q = np.matmul(np.matmul(self.kf.F,Q_a),self.kf.F.T)

    def apply_filter(self,point,confidence):
        self.kf.predict()
        variance = self.var_from_conf(confidence)
        self.kf.R = np.float32([[variance,0],
                                [0,variance]])
        self.kf.update(point)
        out_point = self.kf.x[:2].copy()
        #clip
        out_point[out_point>1]=1
        out_point[out_point<0]=0
        return out_point

    def var_from_conf(self,conf):
        """
        hueristic for getting measurement variance from neural-net confidence.
        the confidence is a number between 0 and 1. The x and y values are also
        between 0 and 1, because they're normalized to the image dimensions.

        intuitively:

        confidence of zero is highest possible variance: the point can be anywhere.
        set variance to 1.

        confidence of 1 is the lowest possible variance. but we should not trust the 
        neural net to be 100% right when it's 100% confident. so discount by factor

        """
        conf = conf * 0.99 #discount

        #var =  (1 - conf)**2 #if we interpret confidence as a normalized distance...
        #var = var**2
        epsilon = 1e-6
        var = 1/(conf+epsilon) - 1.0
        return var

    def __call__(self,point,confidence):
        return self.apply_filter(point,confidence)
        
class Pointwise2DSkelKF(object):
    """
    use the 2D kalman filter on a whole skeleton
    """
    CONF_MEMORY=10
    def __init__(self):
        self.ptf = [Kalman2DPoint(process_variance = self.get_pv(i))
                    for i in range(17)]#todo: smarter process var
        self.conf_history = np.zeros((17,self.CONF_MEMORY),dtype=np.float32)
        #based on skeletal joint index

    def apply_filter(self,skeleton,confidence):
        shape_assert(skeleton,[17,2])
        shape_assert(confidence,[17])
        self.conf_history[:,:self.CONF_MEMORY-1] = \
                self.conf_history[:,1:]
        self.conf_history[:,-1] = confidence
        avg_conf = np.mean(self.conf_history,axis=1)
        return np.float32([
            k(pt,c) for (k,pt,c) in zip(self.ptf,
                                        skeleton,
                                        confidence)
            ]),avg_conf
    def __call__(self,skeleton,confidence):
        return self.apply_filter(skeleton,confidence)

    def get_pv(self,joint_idx):
        import human_foo
        h = human_foo.Part
        if joint_idx in (h.LShoulder,h.RShoulder,h.LHip,h.RHip):
            #parts of body that tend to change direction less
            return 0.0001
        else:
            return 0.001
        

        

        
        
        
