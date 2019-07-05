# Libraries #########################################################################

import cv2;\
import os;\
from compute_mapping_images import compute_mapping_images;\

# Main ################################################################### 

dir_path = os.path.dirname(os.path.realpath(__file__));

# read image
img_in=cv2.imread(dir_path+'\\img_input.png');\
              
img_t_22_5,img_t_45,img_t_67_5,img_t_90,\
img_t_112_5,img_t_135,img_t_157_5,img_t_180,\
img_t_202_5,img_t_225,img_t_247_5,img_t_270,\
img_t_292_5,img_t_315,img_t_337_5,img_t_360 = compute_mapping_images(img_in);

# end of codes #############################################################
