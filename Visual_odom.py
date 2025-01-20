import os 
import numpy as np 
import cv2 
import random
from matplotlib import pyplot as plt

#import the visualization imports


#helps visualize the progress of the code with loading bars
from tqdm import tqdm 



class VisualOdometry():
    def ___init__(self,data_dir):
        self.K,self.P= self.load_calib(os.path.join(data_dir, 'calib.txt'))
        self.getposes=self.load_poses(os.path.join(data_dir,'poses.txtd'))
        self.images=self.load_images(os.path.join(data_dir,'image_l'))
       #creating orb object to extract keypoints
        self.orb=cv2.ORB_create(3000)
        #used for feature matching tasks in computer vision and ml 
        FLANN_INDEX_LSH =6 
    #FLANN_INDEX_LSH,  # Specifies the LSH algorithm for binary descriptors
    #table_number=6,             # Number of hash tables used (higher values increase search quality but are slower)
    #key_size=12,                # Length of the hash key in bits (determines how well data is separated in hash buckets)
    #multi_probe_level=1         # Level of multi-probe exploration (higher values increase recall but reduce speed)
        index_params= dict(algorithm=FLANN_INDEX_LSH,table_number=6, key_size=12, multi_probe_level=1 )
        search_params=dict(checks=50)
        self.flann=cv2.FlannBasedMatcher(indexParams=index_params,searchParams=search_params)

    @staticmethod
    def _load_calib(filepath):
        """
        Loads Camera Calib 
        params:
        ---------
        filepath(str):filepath of the camera file


        returns:
        ---------
        K(ndarray): Intrinsic parameters
        P(ndarray): Projection matrix

           
        
        
        """
        with open(filepath,'r') as f:
            params=np.fromstring(f.readline(),dtype=np.float64,sep=' ')
            P=np.reshape(params,(3,4))
            K=P[0:3,0:3]

        return K,P
    
    @staticmethod
    def _load_images(filepath):
        

        image_paths=[os.path.join(filepath,file) for file in sorted(os.listdir(filepath))]
        return[cv2.imread(path,cv2.IMREAD_GREYSCALE)for path in image_paths]
    @staticmethod
    def _form_transf(R,t):
        T=np.eye(4,dtype=np.float64)
        T[:3,:3]=R
        T[:3,3]=t
        return T
    
    @staticmethod
    def get_matches(self,i):
        keypoints1,descriptors1=self.orb.detectAndCompute(self.images[i-1],None)
        keypoints2,descriptors2=self.orb.detectAndCompute(self.images[i],None)


        matches=self.flann.knnMatch(descriptors1,descriptors2,k=2)


        good=[]

        for m,n in matches:
            if m.distance<0.8*n.distance:
                good.append(m)
        


        q1=np.float32([keypoints1[m.queryIDX].pt for m in good])
        q2=np.float32([keypoints2[m.trainIDX].pt for m in good])




        drawparams=dict(matchColor=-1,
                        singlePointColor=None,
                        matchesMask=None,
                        Flag=2)
        
        img3=cv2.drawMatches(self.images[i],keypoints1,self.images[i-1],keypoints2,good,None,**drawparams)
        cv2.imshow("image",img3)
        cv2.waitKey(750)


        return q1,q2
    
    def get_pose(self,q1,q2):



        Essential,mask=cv2.findEssentialMat(q1,q2,self.K)
        R,t=self.decomp_essential_mat(Essential,q1,q2)

        return self._form_transf(R,t)
    

    def decomp_essential_mat(self,E,q1,q2):




        R1,R2,t=cv2.decomposeEssentialMat(E)


        T1=self._form_transf(R1,np.ndarray.flatten(t))
        
        T2=self._form_transf(R2,np.ndarray.flatten(t))

        T3=self._form_transf(R1,np.ndarray.flatten(-t))

        T4=self._form_transf(R2,np.ndarray.flatten(-t))

        transformation=[T1,T2,T3,T4]


        K=np.concatenate((self.k,np.zeros((3,1))),axis=1)\
        

        projections=[K @ T1, K @ T2,K @T3,K @T4 ]


        np.set_printoptions(suppress=True)

        positives=[]
        for P,T in zip(projections,transformation):
            hom_Q1=cv2.triangulatePoints(self.P,P,q1.T,q2.T)
            hom_Q2= T@ hom_Q1

            Q1 =  hom_Q1[:3,:]/hom_Q1[3,:]
            
            Q2 =  hom_Q2[:3,:]/hom_Q2[3,:]


            total_sum= sum(Q2[2,:]>0)+sum(Q1[2,:]>0)

            relative_scale = np.mean(np.linalg.norm(Q1.T[:-1]-Q1.T[1:],axis=-1)/np.linalg.norm(Q2.T[:-1]-Q2.T[1:],axis=-1))

            positives.append(total_sum+relative_scale)



            max= np.argmax(positives)

            if (max==2):
                return R1,np.ndarray.flatten(-t)
            
            if(max==3):
                return R2,np.ndarray.flatten(-t)
            
            if (max==1):
                return R1,np.ndarray.flatten(t)
            
            if(max==0):
                return R2,np.ndarray.flatten(t)



def main():




    

        




