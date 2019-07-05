def compute_mapping_images(img_in):


    # Libraries #########################################################################

    import matplotlib.pyplot as plt;\
    import matplotlib;\
    import numpy;\
    import numpy as np;\
    import cv2;\
    from scipy import ndimage;\
    from skimage import morphology;\
    from skimage.color import rgb2hsv;\
    from copy import copy;\
    from skimage.morphology import square;\
    import scipy.io;\
    import matplotlib.transforms as mtransforms;\
    from scipy.ndimage import gaussian_filter;\
    import matplotlib.path as mplpath;\
    import os;\
    import scipy.misc;\

    ## def #########################################################################

    def im2double(im):
        info = np.iinfo(im.dtype) # Get the data type of the input image
        return im.astype(np.float) / info.max

    def imgradient(img):
        # Get x-gradient in "sx"
        sx = ndimage.sobel(img,axis=0,mode='constant')
        # Get y-gradient in "sy"
        sy = ndimage.sobel(img,axis=1,mode='constant')
        # Get square root of sum of squares
        img_gradient=np.hypot(sx,sy)
        return img_gradient

    def morpho_process(img_gray_in):

        img_double_in = im2double(img_gray_in)
        img_gauss_in = gaussian_filter(img_double_in, sigma=2);\
        img_grad_in = imgradient(img_gauss_in);\
        img_f= copy(img_grad_in);
        img_f[img_f<0.05] = 0 ;\
        img_f[img_f>0] = 1 ;\
        img_f1 = morphology.binary_dilation(img_f>0,square(2));
        img_f2 = morphology.remove_small_objects(img_f1>0, min_size=20000, connectivity=2)
        img_f3 = morphology.remove_small_holes(img_f2, 600000, connectivity=2)

        img_f3 = img_f3*1;

        xy_locs = np.array(np.where((img_f3 >= 1 )));

        y_loc =  xy_locs[0,:];              
        x_loc = xy_locs [1,:];

        return img_f3,y_loc,x_loc


    def four_point_transform(image,pts,dst):

            M = cv2.getPerspectiveTransform(pts, dst)
    ##	print(M)
            warped = cv2.warpPerspective(image, M, (1650, 820))

            # return the warped image
            return warped


    ## def for angle images #############################################################    

    ################################################################################
    def get_img_22_5(img_rgb_in,img_rgb_t,y_loc_t,x_loc_t):

        # polygon for input image
        pts_in = np.array([[228,106],[310,581],[1316,585],[1360,105]]);
        c_pts_in = pts_in.astype('float32');

        pts_in_top = np.array([[229,108],[237,147],[1355,150],[1359,108]]);
        c_pts_in_top = pts_in_top.astype('float32');

        pts_in_back1 = np.array([[229,106],[274,374],[325,375],[283,106]]);
        c_pts_in_back1 = pts_in_back1.astype('float32');

        pts_in_back2 = np.array([[278,376],[313,580],[365,580],[329,377]]);
        c_pts_in_back2 = pts_in_back2.astype('float32');

        # 22.5
        # polygon for target image
        pts_t = np.array([[837,123],[856,787],[1104,399],[1114,87]]);
        c_pts_t = pts_t.astype('float32');

        # poly gon for target image tyre
        pts_t_tyre1 = np.array([[898,614],[915,787],[982,664],[959,543]]);
        c_pts_t_tyre1 = pts_t_tyre1.astype('float32');

        pts_t_tyre2 = np.array([[1030,435],[1041,559],[1079,500],[1068,397]]);
        c_pts_t_tyre2 = pts_t_tyre2.astype('float32');

        pts_t_back1 = np.array([[245,111],[298,439],[853,508],[841,123]]);
        c_pts_t_back1 = pts_t_back1.astype('float32');

        pts_t_back2 = np.array([[298,433],[342,685],[861,788],[852,499]]);
        c_pts_t_back2 = pts_t_back2.astype('float32');

        pts_t_top = np.array([[236,114],[841,129],[1140,88],[735,66]]);
        c_pts_t_top = pts_t_top.astype('float32');

        ## get the warped image
        img_warped = four_point_transform(img_rgb_in,c_pts_in,c_pts_t)

        ## get the warped image
        img_warped_back1 = four_point_transform(img_rgb_in,c_pts_in_back1,c_pts_t_back1)

        ## get the warped image
        img_warped_back2 = four_point_transform(img_rgb_in,c_pts_in_back2,c_pts_t_back2)

        ## get the warped image
        img_warped_top = four_point_transform(img_rgb_in,c_pts_in_top,c_pts_t_top)

        # compute coordinates of polygon
        poly_big = mplpath.Path(c_pts_t)

        poly_tyre1 = mplpath.Path(c_pts_t_tyre1)

        poly_tyre2 = mplpath.Path(c_pts_t_tyre2)

        poly_back1 = mplpath.Path(c_pts_t_back1);

        poly_back2 = mplpath.Path(c_pts_t_back2);

        poly_top = mplpath.Path(c_pts_t_top);

        size_pixel = np.size(y_loc_t);

        img_rgb_t_process = copy(img_rgb_t);

        # map target image to template image
        n = 0
        for i in range(size_pixel):

            if poly_big.contains_point((x_loc_t[n],y_loc_t[n])) or\
                poly_tyre1.contains_point((x_loc_t[n],y_loc_t[n])) or\
                poly_tyre2.contains_point((x_loc_t[n],y_loc_t[n])):

                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped[y_loc_t[n],x_loc_t[n]];\

            if poly_top.contains_point((x_loc_t[n],y_loc_t[n])):
            
                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped_top[y_loc_t[n],x_loc_t[n]] 


            if poly_back1.contains_point((x_loc_t[n],y_loc_t[n])):
            
                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped_back1[y_loc_t[n],x_loc_t[n]]

            if poly_back2.contains_point((x_loc_t[n],y_loc_t[n])):
            
                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped_back2[y_loc_t[n],x_loc_t[n]]

       
            n +=1;

        return img_rgb_t_process



    ################################################################################
    def get_img_45(img_rgb_in,img_rgb_t,y_loc_t,x_loc_t):

        # polygon for input image
        pts_in = np.array([[228,106],[310,581],[1316,585],[1360,105]]);
        c_pts_in = pts_in.astype('float32');

        pts_in_top = np.array([[229,108],[237,147],[1355,150],[1359,108]]);
        c_pts_in_top = pts_in_top.astype('float32');

        pts_in_back1 = np.array([[229,106],[274,374],[325,375],[283,106]]);
        c_pts_in_back1 = pts_in_back1.astype('float32');

        pts_in_back2 = np.array([[278,376],[313,580],[365,580],[329,377]]);
        c_pts_in_back2 = pts_in_back2.astype('float32');

        # 45
        # polygon for target image
        pts_t = np.array([[597,126],[654,785],[1178,418],[1196,91]]);
        c_pts_t = pts_t.astype('float32');

        # poly gon for target image tyre
        pts_t_tyre1 = np.array([[761,783],[740,634],[868,540],[882,727]]);
        c_pts_t_tyre1 = pts_t_tyre1.astype('float32');

        pts_t_tyre2 = np.array([[1034,447],[1032,577],[1103,541],[1090,423]]);
        c_pts_t_tyre2 = pts_t_tyre2.astype('float32');

        pts_t_back1 = np.array([[216,105],[264,398],[634,511],[600,123]]);
        c_pts_t_back1 = pts_t_back1.astype('float32');

        pts_t_back2 = np.array([[263,393],[302,619],[655,791],[633,499]]);
        c_pts_t_back2 = pts_t_back2.astype('float32');

        pts_t_top = np.array([[201,108],[598,129],[1211,93],[705,60]]);
        c_pts_t_top = pts_t_top.astype('float32');

        ## get the warped image
        img_warped = four_point_transform(img_rgb_in,c_pts_in,c_pts_t)

        ## get the warped image
        img_warped_back1 = four_point_transform(img_rgb_in,c_pts_in_back1,c_pts_t_back1)

        ## get the warped image
        img_warped_back2 = four_point_transform(img_rgb_in,c_pts_in_back2,c_pts_t_back2)

        ## get the warped image
        img_warped_top = four_point_transform(img_rgb_in,c_pts_in_top,c_pts_t_top)

        # compute coordinates of polygon
        poly_big = mplpath.Path(c_pts_t)

        poly_tyre1 = mplpath.Path(c_pts_t_tyre1)

        poly_tyre2 = mplpath.Path(c_pts_t_tyre2)

        poly_back1 = mplpath.Path(c_pts_t_back1);

        poly_back2 = mplpath.Path(c_pts_t_back2);

        poly_top = mplpath.Path(c_pts_t_top);

        size_pixel = np.size(y_loc_t);

        img_rgb_t_process = copy(img_rgb_t);

        # map target image to template image
        n = 0
        for i in range(size_pixel):

            if poly_big.contains_point((x_loc_t[n],y_loc_t[n])) or\
                poly_tyre1.contains_point((x_loc_t[n],y_loc_t[n])) or\
                poly_tyre2.contains_point((x_loc_t[n],y_loc_t[n])):

                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped[y_loc_t[n],x_loc_t[n]];\

            if poly_top.contains_point((x_loc_t[n],y_loc_t[n])):
            
                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped_top[y_loc_t[n],x_loc_t[n]] 


            if poly_back1.contains_point((x_loc_t[n],y_loc_t[n])):
            
                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped_back1[y_loc_t[n],x_loc_t[n]]

            if poly_back2.contains_point((x_loc_t[n],y_loc_t[n])):
            
                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped_back2[y_loc_t[n],x_loc_t[n]]

       
            n +=1;

        return img_rgb_t_process

    ################################################################################

    def get_img_67_5(img_rgb_in,img_rgb_t,y_loc_t,x_loc_t):

        # polygon for input image
        pts_in = np.array([[228,106],[310,581],[1316,585],[1360,105]]);
        c_pts_in = pts_in.astype('float32');

        pts_in_top = np.array([[229,108],[237,147],[1355,150],[1359,108]]);
        c_pts_in_top = pts_in_top.astype('float32');

        pts_in_back1 = np.array([[229,106],[274,374],[325,375],[283,106]]);
        c_pts_in_back1 = pts_in_back1.astype('float32');

        pts_in_back2 = np.array([[278,376],[313,580],[365,580],[329,377]]);
        c_pts_in_back2 = pts_in_back2.astype('float32');

        ## 67.5
        # polygon for target image
        pts_t = np.array([[387,120],[471,754],[1246,453],[1272,91]]);
        c_pts_t = pts_t.astype('float32');

        # polygon for target image tyre
        pts_t_tyre1 = np.array([[604,585],[617,779],[777,718],[765,558]]);
        c_pts_t_tyre1 = pts_t_tyre1.astype('float32');

        pts_t_tyre2 = np.array([[998,474],[1013,608],[1117,567],[1100,442]]);
        c_pts_t_tyre2 = pts_t_tyre2.astype('float32');

        pts_t_back1 = np.array([[243,102],[283,357],[439,489],[390,118]]);
        c_pts_t_back1 = pts_t_back1.astype('float32');

        pts_t_back2 = np.array([[284,349],[317,556],[475,759],[438,474]]);
        c_pts_t_back2 = pts_t_back2.astype('float32');

        pts_t_top = np.array([[237,102],[ 393,126],[1327,96],[749,49]]);
        c_pts_t_top = pts_t_top.astype('float32');

        ## get the warped image
        img_warped = four_point_transform(img_rgb_in,c_pts_in,c_pts_t)

        ## get the warped image
        img_warped_back1 = four_point_transform(img_rgb_in,c_pts_in_back1,c_pts_t_back1)

        ## get the warped image
        img_warped_back2 = four_point_transform(img_rgb_in,c_pts_in_back2,c_pts_t_back2)

        ## get the warped image
        img_warped_top = four_point_transform(img_rgb_in,c_pts_in_top,c_pts_t_top)

        # compute coordinates of polygon
        poly_big = mplpath.Path(c_pts_t)

        poly_tyre1 = mplpath.Path(c_pts_t_tyre1)

        poly_tyre2 = mplpath.Path(c_pts_t_tyre2)

        poly_back1 = mplpath.Path(c_pts_t_back1);

        poly_back2 = mplpath.Path(c_pts_t_back2);

        poly_top = mplpath.Path(c_pts_t_top);

        size_pixel = np.size(y_loc_t);

        img_rgb_t_process = copy(img_rgb_t);

        # map target image to template image
        n = 0
        for i in range(size_pixel):

            if poly_big.contains_point((x_loc_t[n],y_loc_t[n])) or\
                poly_tyre1.contains_point((x_loc_t[n],y_loc_t[n])) or\
                poly_tyre2.contains_point((x_loc_t[n],y_loc_t[n])):

                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped[y_loc_t[n],x_loc_t[n]];\

            if poly_top.contains_point((x_loc_t[n],y_loc_t[n])):
            
                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped_top[y_loc_t[n],x_loc_t[n]] 


            if poly_back1.contains_point((x_loc_t[n],y_loc_t[n])):
            
                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped_back1[y_loc_t[n],x_loc_t[n]]

            if poly_back2.contains_point((x_loc_t[n],y_loc_t[n])):
            
                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped_back2[y_loc_t[n],x_loc_t[n]]

       
            n +=1;

        return img_rgb_t_process

    ################################################################################

    def get_img_90(img_rgb_in,img_rgb_t,y_loc_t,x_loc_t):

        # polygon for input image
        pts_in = np.array([[228,106],[310,581],[1316,585],[1360,105]]);
        c_pts_in = pts_in.astype('float32');

        pts_in_top = np.array([[229,108],[237,147],[1355,150],[1359,108]]);
        c_pts_in_top = pts_in_top.astype('float32');

        pts_in_front1 = np.array([[1313,108],[1283,408],[1324,408],[1355,108]]);
        c_pts_in_front1 = pts_in_front1.astype('float32');

        pts_in_front2 = np.array([[1297,411],[1280,583],[1312,582],[1329,411]]);
        c_pts_in_front2 = pts_in_front2.astype('float32');

        ## 90
        # polygon for target image
        pts_t = np.array([[222,102],[305,588],[1321,591],[1364,103]]);
        c_pts_t = pts_t.astype('float32');

        # polygon for target image tyre
        pts_t_tyre1 = np.array([[405,510],[413,652],[580,663],[578,513]]);
        c_pts_t_tyre1 = pts_t_tyre1.astype('float32');

        pts_t_tyre2 = np.array([[867,509],[876,660],[1044,664],[1032,502]]);
        c_pts_t_tyre2 = pts_t_tyre2.astype('float32');

        pts_t_top = np.array([[755,70],[186,105],[1378,109],[1320,87]]);
        c_pts_t_top = pts_t_top.astype('float32');

        ## get the warped image
        img_warped = four_point_transform(img_rgb_in,c_pts_in,c_pts_t);

        # compute coordinates of polygon
        poly_big = mplpath.Path(c_pts_t)

        poly_tyre1 = mplpath.Path(c_pts_t_tyre1)

        poly_tyre2 = mplpath.Path(c_pts_t_tyre2)

        poly_top = mplpath.Path(c_pts_t_top);

        size_pixel = np.size(y_loc_t);

        img_rgb_t_process = copy(img_rgb_t);

        # map target image to template image
        n = 0
        for i in range(size_pixel):

            if poly_big.contains_point((x_loc_t[n],y_loc_t[n])) or\
                poly_tyre1.contains_point((x_loc_t[n],y_loc_t[n])) or\
                poly_tyre2.contains_point((x_loc_t[n],y_loc_t[n])) or \
                poly_top.contains_point((x_loc_t[n],y_loc_t[n])):

                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped[y_loc_t[n],x_loc_t[n]];\


       
            n +=1;

        return img_rgb_t_process





    ################################################################################
    def get_img_112_5(img_rgb_in,img_rgb_t,y_loc_t,x_loc_t):

        # polygon for input image
        pts_in = np.array([[228,106],[310,581],[1316,585],[1360,105]]);
        c_pts_in = pts_in.astype('float32');

        pts_in_top = np.array([[229,108],[237,147],[1355,150],[1359,108]]);
        c_pts_in_top = pts_in_top.astype('float32');

        pts_in_front1 = np.array([[1313,108],[1283,408],[1324,408],[1355,108]]);
        c_pts_in_front1 = pts_in_front1.astype('float32');

        pts_in_front2 = np.array([[1297,411],[1280,583],[1312,582],[1329,411]]);
        c_pts_in_front2 = pts_in_front2.astype('float32');

        ## 112.5
        # polygon for target image
        pts_t = np.array([[329,96],[387,489],[1196,706],[1230,114]]);
        c_pts_t = pts_t.astype('float32');

        # polygon for target image tyre
        pts_t_tyre1 = np.array([[442,442],[451,564],[551,594],[546 ,460]]);
        c_pts_t_tyre1 = pts_t_tyre1.astype('float32');

        pts_t_tyre2 = np.array([[759,523],[759,656],[897,695],[898,534]]);
        c_pts_t_tyre2 = pts_t_tyre2.astype('float32');

        pts_t_front1 = np.array([[1225,114],[1196,499],[1321,355],[1343,96]]);
        c_pts_t_front1 = pts_t_front1.astype('float32');

        pts_t_front2 = np.array([[1197,487],[1187,707],[1308,506],[1321,348]]);
        c_pts_t_front2 = pts_t_front2.astype('float32');

        pts_t_top = np.array([[865,66],[296,97],[1228,123],[1353,99]]);
        c_pts_t_top = pts_t_top.astype('float32');

        ## get the warped image
        img_warped = four_point_transform(img_rgb_in,c_pts_in,c_pts_t)

        ## get the warped image
        img_warped_front1 = four_point_transform(img_rgb_in,c_pts_in_front1,c_pts_t_front1)

        ## get the warped image
        img_warped_front2 = four_point_transform(img_rgb_in,c_pts_in_front2,c_pts_t_front2)

        ## get the warped image
        img_warped_top = four_point_transform(img_rgb_in,c_pts_in_top,c_pts_t_top)

        # compute coordinates of polygon
        poly_big = mplpath.Path(c_pts_t)

        poly_tyre1 = mplpath.Path(c_pts_t_tyre1)

        poly_tyre2 = mplpath.Path(c_pts_t_tyre2)

        poly_front1 = mplpath.Path(c_pts_t_front1);

        poly_front2 = mplpath.Path(c_pts_t_front2);

        poly_top = mplpath.Path(c_pts_t_top);

        size_pixel = np.size(y_loc_t);

        img_rgb_t_process = copy(img_rgb_t);

        # map target image to template image
        n = 0
        for i in range(size_pixel):

            if poly_big.contains_point((x_loc_t[n],y_loc_t[n])) or\
                poly_tyre1.contains_point((x_loc_t[n],y_loc_t[n])) or\
                poly_tyre2.contains_point((x_loc_t[n],y_loc_t[n])):

                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped[y_loc_t[n],x_loc_t[n]];\

            if poly_top.contains_point((x_loc_t[n],y_loc_t[n])):
            
                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped_top[y_loc_t[n],x_loc_t[n]] 


            if poly_front1.contains_point((x_loc_t[n],y_loc_t[n])):
            
                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped_front1[y_loc_t[n],x_loc_t[n]]

            if poly_front2.contains_point((x_loc_t[n],y_loc_t[n])):
            
                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped_front2[y_loc_t[n],x_loc_t[n]]

       
            n +=1;

        return img_rgb_t_process




    ################################################################################
    def get_img_135(img_rgb_in,img_rgb_t,y_loc_t,x_loc_t):

        # polygon for input image
        pts_in = np.array([[228,106],[310,581],[1316,585],[1360,105]]);
        c_pts_in = pts_in.astype('float32');

        pts_in_top = np.array([[229,108],[237,147],[1355,150],[1359,108]]);
        c_pts_in_top = pts_in_top.astype('float32');

        pts_in_front1 = np.array([[1313,108],[1283,408],[1324,408],[1355,108]]);
        c_pts_in_front1 = pts_in_front1.astype('float32');

        pts_in_front2 = np.array([[1297,411],[1280,583],[1312,582],[1329,411]]);
        c_pts_in_front2 = pts_in_front2.astype('float32');

        ## 135
        # polygon for target image
        pts_t = np.array([[421,91],[463,447],[1030,761],[1045,120]]);
        c_pts_t = pts_t.astype('float32');

        # polygon for target image tyre
        pts_t_tyre1 = np.array([[495,414],[499,520],[570,556],[566,433]]);
        c_pts_t_tyre1 = pts_t_tyre1.astype('float32');

        pts_t_tyre2 = np.array([[699,514],[699 ,642],[792,697],[801,541]]);
        c_pts_t_tyre2 = pts_t_tyre2.astype('float32');

        pts_t_front1 = np.array([[1039,118],[1029,534],[1339,396],[1363,100]]);
        c_pts_t_front1 = pts_t_front1.astype('float32');

        pts_t_front2 = np.array([[1029,524],[1023,764],[1324,566],[1340,390]]);
        c_pts_t_front2 = pts_t_front2.astype('float32');

        pts_t_top = np.array([[885,69],[393,93],[1044,127],[1384,105]]);
        c_pts_t_top = pts_t_top.astype('float32');

        ## get the warped image
        img_warped = four_point_transform(img_rgb_in,c_pts_in,c_pts_t)

        ## get the warped image
        img_warped_front1 = four_point_transform(img_rgb_in,c_pts_in_front1,c_pts_t_front1)

        ## get the warped image
        img_warped_front2 = four_point_transform(img_rgb_in,c_pts_in_front2,c_pts_t_front2)

        ## get the warped image
        img_warped_top = four_point_transform(img_rgb_in,c_pts_in_top,c_pts_t_top)

        # compute coordinates of polygon
        poly_big = mplpath.Path(c_pts_t)

        poly_tyre1 = mplpath.Path(c_pts_t_tyre1)

        poly_tyre2 = mplpath.Path(c_pts_t_tyre2)

        poly_front1 = mplpath.Path(c_pts_t_front1);

        poly_front2 = mplpath.Path(c_pts_t_front2);

        poly_top = mplpath.Path(c_pts_t_top);

        size_pixel = np.size(y_loc_t);

        img_rgb_t_process = copy(img_rgb_t);

        # map target image to template image
        n = 0
        for i in range(size_pixel):

            if poly_big.contains_point((x_loc_t[n],y_loc_t[n])) or\
                poly_tyre1.contains_point((x_loc_t[n],y_loc_t[n])) or\
                poly_tyre2.contains_point((x_loc_t[n],y_loc_t[n])):

                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped[y_loc_t[n],x_loc_t[n]];\

            if poly_top.contains_point((x_loc_t[n],y_loc_t[n])):
            
                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped_top[y_loc_t[n],x_loc_t[n]] 


            if poly_front1.contains_point((x_loc_t[n],y_loc_t[n])):
            
                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped_front1[y_loc_t[n],x_loc_t[n]]

            if poly_front2.contains_point((x_loc_t[n],y_loc_t[n])):
            
                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped_front2[y_loc_t[n],x_loc_t[n]]

       
            n +=1;

        return img_rgb_t_process


    ################################################################################
    def get_img_157_5(img_rgb_in,img_rgb_t,y_loc_t,x_loc_t):

        # polygon for input image
        pts_in = np.array([[228,106],[310,581],[1316,585],[1360,105]]);
        c_pts_in = pts_in.astype('float32');

        pts_in_top = np.array([[229,108],[237,147],[1355,150],[1359,108]]);
        c_pts_in_top = pts_in_top.astype('float32');

        pts_in_front1 = np.array([[1313,108],[1283,408],[1324,408],[1355,108]]);
        c_pts_in_front1 = pts_in_front1.astype('float32');

        pts_in_front2 = np.array([[1297,411],[1280,583],[1312,582],[1329,411]]);
        c_pts_in_front2 = pts_in_front2.astype('float32');

        ## 157.5
        # polygon for target image
        pts_t = np.array([[504  ,  90],
       [537 ,  421],
       [861  , 791],
       [844   ,124]]);
        c_pts_t = pts_t.astype('float32');

        # polygon for target image tyre
        pts_t_tyre1 = np.array([[547,492],[561,388],[596,445],[584,523]]);
        c_pts_t_tyre1 = pts_t_tyre1.astype('float32');

        pts_t_tyre2 = np.array([[651,634],[669,499],[735,599],[707,685]]);
        c_pts_t_tyre2 = pts_t_tyre2.astype('float32');

        pts_t_front1 = np.array([[837,121],[852,553],[1319,433],[1346,106]]);
        c_pts_t_front1 = pts_t_front1.astype('float32');

        pts_t_front2 = np.array([[852,549],[858,785],[1302,617],[1316,427]]);
        c_pts_t_front2 = pts_t_front2.astype('float32');

        pts_t_top = np.array([[913,63],[467,91],[844,141],[1383,111]]);
        c_pts_t_top = pts_t_top.astype('float32');

        ## get the warped image
        img_warped = four_point_transform(img_rgb_in,c_pts_in,c_pts_t)

        ## get the warped image
        img_warped_front1 = four_point_transform(img_rgb_in,c_pts_in_front1,c_pts_t_front1)

        ## get the warped image
        img_warped_front2 = four_point_transform(img_rgb_in,c_pts_in_front2,c_pts_t_front2)

        ## get the warped image
        img_warped_top = four_point_transform(img_rgb_in,c_pts_in_top,c_pts_t_top)

        # compute coordinates of polygon
        poly_big = mplpath.Path(c_pts_t)

        poly_tyre1 = mplpath.Path(c_pts_t_tyre1)

        poly_tyre2 = mplpath.Path(c_pts_t_tyre2)

        poly_front1 = mplpath.Path(c_pts_t_front1);

        poly_front2 = mplpath.Path(c_pts_t_front2);

        poly_top = mplpath.Path(c_pts_t_top);

        size_pixel = np.size(y_loc_t);

        img_rgb_t_process = copy(img_rgb_t);

        # map target image to template image
        n = 0
        for i in range(size_pixel):

            if poly_big.contains_point((x_loc_t[n],y_loc_t[n])) or\
                poly_tyre1.contains_point((x_loc_t[n],y_loc_t[n])) or\
                poly_tyre2.contains_point((x_loc_t[n],y_loc_t[n])):

                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped[y_loc_t[n],x_loc_t[n]];\

            if poly_top.contains_point((x_loc_t[n],y_loc_t[n])):
            
                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped_top[y_loc_t[n],x_loc_t[n]] 


            if poly_front1.contains_point((x_loc_t[n],y_loc_t[n])):
            
                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped_front1[y_loc_t[n],x_loc_t[n]]

            if poly_front2.contains_point((x_loc_t[n],y_loc_t[n])):
            
                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped_front2[y_loc_t[n],x_loc_t[n]]

       
            n +=1;

        return img_rgb_t_process



    ################################################################################
    def get_img_180(img_rgb_in,img_rgb_t,y_loc_t,x_loc_t):

        # polygon for input image
        pts_in = np.array([[228,106],[310,581],[1316,585],[1360,105]]);
        c_pts_in = pts_in.astype('float32');

        pts_in_back1 = np.array([[229,106],[274,374],[325,375],[283,106]]);
        c_pts_in_back1 = pts_in_back1.astype('float32');

        pts_in_back2 = np.array([[278,376],[313,580],[365,580],[329,377]]);
        c_pts_in_back2 = pts_in_back2.astype('float32');


        pts_in_top = np.array([[229,108],[237,147],[1355,150],[1359,108]]);
        c_pts_in_top = pts_in_top.astype('float32');

        pts_in_front1 = np.array([[1313,108],[1283,408],[1324,408],[1355,108]]);
        c_pts_in_front1 = pts_in_front1.astype('float32');

        pts_in_front2 = np.array([[1297,411],[1280,583],[1312,582],[1329,411]]);
        c_pts_in_front2 = pts_in_front2.astype('float32');

        # 180
        # polygon for target image
        pts_t_top = np.array([[1082,78],[511,79],[396,132],[1164,130]]);

        c_pts_t_top = pts_t_top.astype('float32');

        pts_t_front1 = np.array([[426,120],[479,534],[1134,512],[1150,117]]);
        c_pts_t_front1 = pts_t_front1.astype('float32');

        pts_t_front2 = np.array([[478,530],[509,762],[1124,732],[1134,508]]);
        c_pts_t_front2 = pts_t_front2.astype('float32');
        
        ## get the warped image	
        img_warped_front1 = four_point_transform(img_rgb_in,c_pts_in_front1,c_pts_t_front1)

        ## get the warped image	
        img_warped_front2 = four_point_transform(img_rgb_in,c_pts_in_front2,c_pts_t_front2)

        ## get the warped image
        img_warped_top = four_point_transform(img_rgb_in,c_pts_in_top,c_pts_t_top)

        # compute coordinates of polygon

        poly_front1 = mplpath.Path(c_pts_t_front1);

        poly_front2 = mplpath.Path(c_pts_t_front2);

        poly_top = mplpath.Path(c_pts_t_top);

        size_pixel = np.size(y_loc_t);

        img_rgb_t_process = copy(img_rgb_t);

        # map target image to template image
        n = 0
        for i in range(size_pixel):

            if poly_front1.contains_point((x_loc_t[n],y_loc_t[n])):
            
                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped_front1[y_loc_t[n],x_loc_t[n]]

            if poly_front2.contains_point((x_loc_t[n],y_loc_t[n])):
            
                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped_front2[y_loc_t[n],x_loc_t[n]]

            if poly_top.contains_point((x_loc_t[n],y_loc_t[n])):
            
                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped_top[y_loc_t[n],x_loc_t[n]] 

            n +=1;

        return img_rgb_t_process


    ################################################################################
    def get_img_202_5(img_rgb_in,img_rgb_t,y_loc_t,x_loc_t):


        img_rgb_in = numpy.fliplr(img_rgb_in);

        # polygon for input image
        pts_in = np.array([[290,105],[336,587],[1346,587],[1428,105]]);
        c_pts_in = pts_in.astype('float32');

        pts_in_top = np.array([[293,108],[295,135],[1421,141],[1426,105]]);
        c_pts_in_top = pts_in_top.astype('float32');

        pts_in_front1 = np.array([[295,106],[323,408],[372,406],[345,108]]);
        c_pts_in_front1 = pts_in_front1.astype('float32');

        pts_in_front2 = np.array([[323,411],[342,584],[369,585],[353,411]]);
        c_pts_in_front2 = pts_in_front2.astype('float32');

        # 202.5
        # polygon for target image
        pts_t = np.array([[792,121],[819,794],[1119,402],[1129,87]]);
        c_pts_t = pts_t.astype('float32');

        # polygon for target image tyre
        pts_t_tyre1 = np.array([[944,526],[962,686],[1022,580],[996,486]]);
        c_pts_t_tyre1 = pts_t_tyre1.astype('float32');

        pts_t_tyre2 = np.array([[1071,402],[1074,505],[1101,468],[1106,370]]);
        c_pts_t_tyre2 = pts_t_tyre2.astype('float32');

        pts_t_top = np.array([[234,114],[795,127],[1131,90],[738,64]]);
        c_pts_t_top = pts_t_top.astype('float32');

        pts_t_front1 = np.array([[235,111],[292,466],[810,555],[795,124]]);
        c_pts_t_front1 = pts_t_front1.astype('float32');

        pts_t_front2 = np.array([[295,466],[329,673],[821,791],[813,551]]);
        c_pts_t_front2 = pts_t_front2.astype('float32');

        ## get the warped image
        img_warped = four_point_transform(img_rgb_in,c_pts_in,c_pts_t)

        ## get the warped image	
        img_warped_front1 = four_point_transform(img_rgb_in,c_pts_in_front1,c_pts_t_front1)

        ## get the warped image	
        img_warped_front2 = four_point_transform(img_rgb_in,c_pts_in_front2,c_pts_t_front2)

        ## get the warped image
        img_warped_top = four_point_transform(img_rgb_in,c_pts_in_top,c_pts_t_top)

        # compute coordinates of polygon
        poly_big = mplpath.Path(c_pts_t)

        poly_tyre1 = mplpath.Path(c_pts_t_tyre1)

        poly_tyre2 = mplpath.Path(c_pts_t_tyre2)

        poly_front1 = mplpath.Path(c_pts_t_front1);

        poly_front2 = mplpath.Path(c_pts_t_front2);

        poly_top = mplpath.Path(c_pts_t_top);

        size_pixel = np.size(y_loc_t);

        img_rgb_t_process = copy(img_rgb_t);

        # map target image to template image
        n = 0
        for i in range(size_pixel):

            if poly_big.contains_point((x_loc_t[n],y_loc_t[n])) or\
                poly_tyre1.contains_point((x_loc_t[n],y_loc_t[n])) or\
                poly_tyre2.contains_point((x_loc_t[n],y_loc_t[n])):

                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped[y_loc_t[n],x_loc_t[n]];\

            if poly_front1.contains_point((x_loc_t[n],y_loc_t[n])):
            
                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped_front1[y_loc_t[n],x_loc_t[n]]

            if poly_front2.contains_point((x_loc_t[n],y_loc_t[n])):
            
                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped_front2[y_loc_t[n],x_loc_t[n]]


            if poly_top.contains_point((x_loc_t[n],y_loc_t[n])):
            
                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped_top[y_loc_t[n],x_loc_t[n]] 

            n +=1;

        return img_rgb_t_process

    ################################################################################
    def get_img_225(img_rgb_in,img_rgb_t,y_loc_t,x_loc_t):


        img_rgb_in = numpy.fliplr(img_rgb_in);

        # polygon for input image
        pts_in = np.array([[290,105],[336,587],[1346,587],[1428,105]]);
        c_pts_in = pts_in.astype('float32');

        pts_in_top = np.array([[293,108],[295,135],[1421,141],[1426,105]]);
        c_pts_in_top = pts_in_top.astype('float32');

        pts_in_front1 = np.array([[295,106],[323,408],[372,406],[345,108]]);
        c_pts_in_front1 = pts_in_front1.astype('float32');

        pts_in_front2 = np.array([[323,411],[342,584],[369,585],[353,411]]);
        c_pts_in_front2 = pts_in_front2.astype('float32');

        # 225
        # polygon for target image
        pts_t = np.array([[581,120],[642,787],[1184,424],[1201,90]]);
        c_pts_t = pts_t.astype('float32');

        # polygon for target image tyre
        pts_t_tyre1 = np.array([[874,549],[891,686],[970,638],[972,489]]);
        c_pts_t_tyre1 = pts_t_tyre1.astype('float32');

        pts_t_tyre2 = np.array([[1098,411],[1102,536],[1157,486],[1155,390]]);
        c_pts_t_tyre2 = pts_t_tyre2.astype('float32');

        pts_t_top = np.array([[209,106],[585,127],[1213,91],[759,72]]);
        c_pts_t_top = pts_t_top.astype('float32');

        pts_t_front1 = np.array([[216,106],[269,426],[623,551],[586,123]]);
        c_pts_t_front1 = pts_t_front1.astype('float32');

        pts_t_front2 = np.array([[272,429],[305,616],[645,789],[627,552]]);
        c_pts_t_front2 = pts_t_front2.astype('float32');
        
        ## get the warped image
        img_warped = four_point_transform(img_rgb_in,c_pts_in,c_pts_t)

        ## get the warped image	
        img_warped_front1 = four_point_transform(img_rgb_in,c_pts_in_front1,c_pts_t_front1)

        ## get the warped image	
        img_warped_front2 = four_point_transform(img_rgb_in,c_pts_in_front2,c_pts_t_front2)

        ## get the warped image
        img_warped_top = four_point_transform(img_rgb_in,c_pts_in_top,c_pts_t_top)

        # compute coordinates of polygon
        poly_big = mplpath.Path(c_pts_t)

        poly_tyre1 = mplpath.Path(c_pts_t_tyre1)

        poly_tyre2 = mplpath.Path(c_pts_t_tyre2)

        poly_front1 = mplpath.Path(c_pts_t_front1);

        poly_front2 = mplpath.Path(c_pts_t_front2);

        poly_top = mplpath.Path(c_pts_t_top);

        size_pixel = np.size(y_loc_t);

        img_rgb_t_process = copy(img_rgb_t);

        # map target image to template image
        n = 0
        for i in range(size_pixel):

            if poly_big.contains_point((x_loc_t[n],y_loc_t[n])) or\
                poly_tyre1.contains_point((x_loc_t[n],y_loc_t[n])) or\
                poly_tyre2.contains_point((x_loc_t[n],y_loc_t[n])):

                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped[y_loc_t[n],x_loc_t[n]];\

            if poly_front1.contains_point((x_loc_t[n],y_loc_t[n])):
            
                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped_front1[y_loc_t[n],x_loc_t[n]]

            if poly_front2.contains_point((x_loc_t[n],y_loc_t[n])):
            
                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped_front2[y_loc_t[n],x_loc_t[n]]


            if poly_top.contains_point((x_loc_t[n],y_loc_t[n])):
            
                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped_top[y_loc_t[n],x_loc_t[n]] 

            n +=1;

        return img_rgb_t_process

    ################################################################################
    def get_img_247_5(img_rgb_in,img_rgb_t,y_loc_t,x_loc_t):


        img_rgb_in = numpy.fliplr(img_rgb_in);

        # polygon for input image
        pts_in = np.array([[290,105],[336,587],[1346,587],[1428,105]]);
        c_pts_in = pts_in.astype('float32');

        pts_in_top = np.array([[293,108],[295,135],[1421,141],[1426,105]]);
        c_pts_in_top = pts_in_top.astype('float32');

        pts_in_front1 = np.array([[295,106],[323,408],[372,406],[345,108]]);
        c_pts_in_front1 = pts_in_front1.astype('float32');

        pts_in_front2 = np.array([[323,411],[342,584],[369,585],[353,411]]);
        c_pts_in_front2 = pts_in_front2.astype('float32');

        # 247_5
        # polygon for target image
        pts_t = np.array([[365,118],[450,746],[1254,459],[1282,93]]);
        c_pts_t = pts_t.astype('float32');

        # polygon for target image tyre
        pts_t_tyre1 = np.array([[771,552],[782,688],[916,660],[909,502]]);
        c_pts_t_tyre1 = pts_t_tyre1.astype('float32');

        pts_t_tyre2 = np.array([[1118,441],[1118,558],[1206,535],[1204,418]]);
        c_pts_t_tyre2 = pts_t_tyre2.astype('float32');

        pts_t_top = np.array([[250,102],[368,123],[1288,96],[813,75]]);
        c_pts_t_top = pts_t_top.astype('float32');

        pts_t_front1 = np.array([[251,102],[296,384],[422,521],[368,118]]);
        c_pts_t_front1 = pts_t_front1.astype('float32');

        pts_t_front2 = np.array([[296,382],[322,547],[451,750],[425,524]]);
        c_pts_t_front2 = pts_t_front2.astype('float32');
        
        ## get the warped image
        img_warped = four_point_transform(img_rgb_in,c_pts_in,c_pts_t)

        ## get the warped image	
        img_warped_front1 = four_point_transform(img_rgb_in,c_pts_in_front1,c_pts_t_front1)

        ## get the warped image	
        img_warped_front2 = four_point_transform(img_rgb_in,c_pts_in_front2,c_pts_t_front2)

        ## get the warped image
        img_warped_top = four_point_transform(img_rgb_in,c_pts_in_top,c_pts_t_top)

        # compute coordinates of polygon
        poly_big = mplpath.Path(c_pts_t)

        poly_tyre1 = mplpath.Path(c_pts_t_tyre1)

        poly_tyre2 = mplpath.Path(c_pts_t_tyre2)

        poly_front1 = mplpath.Path(c_pts_t_front1);

        poly_front2 = mplpath.Path(c_pts_t_front2);

        poly_top = mplpath.Path(c_pts_t_top);

        size_pixel = np.size(y_loc_t);

        img_rgb_t_process = copy(img_rgb_t);

        # map target image to template image
        n = 0
        for i in range(size_pixel):

            if poly_big.contains_point((x_loc_t[n],y_loc_t[n])) or\
                poly_tyre1.contains_point((x_loc_t[n],y_loc_t[n])) or\
                poly_tyre2.contains_point((x_loc_t[n],y_loc_t[n])):

                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped[y_loc_t[n],x_loc_t[n]];\

            if poly_front1.contains_point((x_loc_t[n],y_loc_t[n])):
            
                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped_front1[y_loc_t[n],x_loc_t[n]]

            if poly_front2.contains_point((x_loc_t[n],y_loc_t[n])):
            
                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped_front2[y_loc_t[n],x_loc_t[n]]


            if poly_top.contains_point((x_loc_t[n],y_loc_t[n])):
            
                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped_top[y_loc_t[n],x_loc_t[n]] 

            n +=1;

        return img_rgb_t_process


    ################################################################################
    def get_img_270(img_rgb_in,img_rgb_t,y_loc_t,x_loc_t):


        img_rgb_in = numpy.fliplr(img_rgb_in);

        # polygon for input image
        pts_in = np.array([[290,105],[336,587],[1346,587],[1428,105]]);
        c_pts_in = pts_in.astype('float32');

        pts_in_top = np.array([[293,108],[295,135],[1421,141],[1426,105]]);
        c_pts_in_top = pts_in_top.astype('float32');


        # 270
        # polygon for target image
        pts_t = np.array([[227,103],[307,582],[1314,590],[1361,105]]);
        c_pts_t = pts_t.astype('float32');

        # poly gon for target image tyre
        pts_t_tyre1 = np.array([[ 589,509],[590,647],[746,658],[735,514]]);
        c_pts_t_tyre1 = pts_t_tyre1.astype('float32');

        pts_t_tyre2 = np.array([[1051,511],[1050,653],[1208,661],[1202,517]]);
        c_pts_t_tyre2 = pts_t_tyre2.astype('float32');

        pts_t_top = np.array([[226,105],[320,79],[1258,81],[1361,108]]);
        c_pts_t_top = pts_t_top.astype('float32');

        ## get the warped image
        img_warped = four_point_transform(img_rgb_in,c_pts_in,c_pts_t)

        # compute coordinates of polygon
        poly_big = mplpath.Path(c_pts_t)

        poly_tyre1 = mplpath.Path(c_pts_t_tyre1)

        poly_tyre2 = mplpath.Path(c_pts_t_tyre2)

        poly_top= mplpath.Path(c_pts_t_top)

        size_pixel = np.size(y_loc_t);

        img_rgb_t_process = copy(img_rgb_t);

        # map target image to template image
        n = 0
        for i in range(size_pixel):

            if poly_big.contains_point((x_loc_t[n],y_loc_t[n])) or\
                poly_tyre1.contains_point((x_loc_t[n],y_loc_t[n])) or\
                poly_tyre2.contains_point((x_loc_t[n],y_loc_t[n])) or\
                poly_top.contains_point((x_loc_t[n],y_loc_t[n])):

                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped[y_loc_t[n],x_loc_t[n]];\

            n +=1;

        return img_rgb_t_process

    ################################################################################
    def get_img_292_5(img_rgb_in,img_rgb_t,y_loc_t,x_loc_t):


        img_rgb_in = numpy.fliplr(img_rgb_in);

        # polygon for input image
        pts_in = np.array([[290,105],[336,587],[1346,587],[1428,105]]);
        c_pts_in = pts_in.astype('float32');

        pts_in_back1 = np.array([[1374,102],[1329,374],[1377,373],[1424,105]]);
        c_pts_in_back1 = pts_in_back1.astype('float32');

        pts_in_back2 = np.array([[1341,377],[1304,581],[1339,580],[1373,376]]);
        c_pts_in_back2 = pts_in_back2.astype('float32');

        pts_in_top = np.array([[293,108],[295,135],[1421,141],[1426,105]]);
        c_pts_in_top = pts_in_top.astype('float32');


        # 292.5
        # polygon for target image
        pts_t = np.array([[323,94],[383,496],[1202,704],[1241,111]]);
        c_pts_t = pts_t.astype('float32');

        # polygon for target image tyre
        pts_t_tyre1 = np.array([[545,476],[549,600],[670,623],[659,480]]);
        c_pts_t_tyre1 = pts_t_tyre1.astype('float32');

        pts_t_tyre2 = np.array([[917,556],[915,694],[1073,756],[1078,584]]);
        c_pts_t_tyre2 = pts_t_tyre2.astype('float32');

        pts_t_top = np.array([[822,75],[320,96],[1235,118],[1346,99]]);
        c_pts_t_top = pts_t_top.astype('float32');

        pts_t_back1 = np.array([[1233,114],[1216,450],[1321,327],[1341,96]]);
        c_pts_t_back1 = pts_t_back1.astype('float32');

        pts_t_back2 = np.array([[1214,447],[1194,698],[1305,505],[1321,320]]);
        c_pts_t_back2 = pts_t_back2.astype('float32');
        
        ## get the warped image
        img_warped = four_point_transform(img_rgb_in,c_pts_in,c_pts_t)

        ## get the warped image	
        img_warped_back1 = four_point_transform(img_rgb_in,c_pts_in_back1,c_pts_t_back1)

        ## get the warped image	
        img_warped_back2 = four_point_transform(img_rgb_in,c_pts_in_back2,c_pts_t_back2)

        ## get the warped image
        img_warped_top = four_point_transform(img_rgb_in,c_pts_in_top,c_pts_t_top)

        # compute coordinates of polygon
        poly_big = mplpath.Path(c_pts_t)

        poly_tyre1 = mplpath.Path(c_pts_t_tyre1)

        poly_tyre2 = mplpath.Path(c_pts_t_tyre2)

        poly_back1 = mplpath.Path(c_pts_t_back1);

        poly_back2 = mplpath.Path(c_pts_t_back2);

        poly_top = mplpath.Path(c_pts_t_top);

        size_pixel = np.size(y_loc_t);

        img_rgb_t_process = copy(img_rgb_t);

        # map target image to template image
        n = 0
        for i in range(size_pixel):

            if poly_big.contains_point((x_loc_t[n],y_loc_t[n])) or\
                poly_tyre1.contains_point((x_loc_t[n],y_loc_t[n])) or\
                poly_tyre2.contains_point((x_loc_t[n],y_loc_t[n])):

                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped[y_loc_t[n],x_loc_t[n]];\

            if poly_back1.contains_point((x_loc_t[n],y_loc_t[n])):
            
                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped_back1[y_loc_t[n],x_loc_t[n]]

            if poly_back2.contains_point((x_loc_t[n],y_loc_t[n])):
            
                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped_back2[y_loc_t[n],x_loc_t[n]]


            if poly_top.contains_point((x_loc_t[n],y_loc_t[n])):
            
                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped_top[y_loc_t[n],x_loc_t[n]] 

            n +=1;

        return img_rgb_t_process

    ################################################################################
    def get_img_315(img_rgb_in,img_rgb_t,y_loc_t,x_loc_t):


        img_rgb_in = numpy.fliplr(img_rgb_in);

        # polygon for input image
        pts_in = np.array([[290,105],[336,587],[1346,587],[1428,105]]);
        c_pts_in = pts_in.astype('float32');

        pts_in_back1 = np.array([[1374,102],[1329,374],[1377,373],[1424,105]]);
        c_pts_in_back1 = pts_in_back1.astype('float32');

        pts_in_back2 = np.array([[1341,377],[1304,581],[1339,580],[1373,376]]);
        c_pts_in_back2 = pts_in_back2.astype('float32');

        pts_in_top = np.array([[293,108],[295,135],[1421,141],[1426,105]]);
        c_pts_in_top = pts_in_top.astype('float32');

        pts_in_front1 = np.array([[295,106],[323,408],[372,406],[345,108]]);
        c_pts_in_front1 = pts_in_front1.astype('float32');

        pts_in_front2 = np.array([[323,411],[342,584],[369,585],[353,411]]);
        c_pts_in_front2 = pts_in_front2.astype('float32');

        # 315
        # polygon for target image
        pts_t = np.array([[413,90],[456,450],[1047,758],[1065,117]]);
        c_pts_t = pts_t.astype('float32');

        # polygon for target image tyre
        pts_t_tyre1 = np.array([[564,442],[563,552],[632,607],[648,486]]);
        c_pts_t_tyre1 = pts_t_tyre1.astype('float32');

        pts_t_tyre2 = np.array([[810,563],[815,718],[942,774],[945,595]]);
        c_pts_t_tyre2 = pts_t_tyre2.astype('float32');

        pts_t_top = np.array([[903,72],[407,91],[1064,123],[1372,105]]);
        c_pts_t_top = pts_t_top.astype('float32');

        pts_t_back1 = np.array([[1057,117],[1050,484],[1342,363],[1365,100]]);
        c_pts_t_back1 = pts_t_back1.astype('float32');

        pts_t_back2 = np.array([[1050,477],[1040,759],[1323,560],[1341,353]]);
        c_pts_t_back2 = pts_t_back2.astype('float32');
        
        ## get the warped image
        img_warped = four_point_transform(img_rgb_in,c_pts_in,c_pts_t)

        ## get the warped image	
        img_warped_back1 = four_point_transform(img_rgb_in,c_pts_in_back1,c_pts_t_back1)

        ## get the warped image	
        img_warped_back2 = four_point_transform(img_rgb_in,c_pts_in_back2,c_pts_t_back2)

        ## get the warped image
        img_warped_top = four_point_transform(img_rgb_in,c_pts_in_top,c_pts_t_top)

        # compute coordinates of polygon
        poly_big = mplpath.Path(c_pts_t)

        poly_tyre1 = mplpath.Path(c_pts_t_tyre1)

        poly_tyre2 = mplpath.Path(c_pts_t_tyre2)

        poly_back1 = mplpath.Path(c_pts_t_back1);

        poly_back2 = mplpath.Path(c_pts_t_back2);

        poly_top = mplpath.Path(c_pts_t_top);

        size_pixel = np.size(y_loc_t);

        img_rgb_t_process = copy(img_rgb_t);

        # map target image to template image
        n = 0
        for i in range(size_pixel):

            if poly_big.contains_point((x_loc_t[n],y_loc_t[n])) or\
                poly_tyre1.contains_point((x_loc_t[n],y_loc_t[n])) or\
                poly_tyre2.contains_point((x_loc_t[n],y_loc_t[n])):

                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped[y_loc_t[n],x_loc_t[n]];\

            if poly_back1.contains_point((x_loc_t[n],y_loc_t[n])):
            
                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped_back1[y_loc_t[n],x_loc_t[n]]

            if poly_back2.contains_point((x_loc_t[n],y_loc_t[n])):
            
                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped_back2[y_loc_t[n],x_loc_t[n]]


            if poly_top.contains_point((x_loc_t[n],y_loc_t[n])):
            
                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped_top[y_loc_t[n],x_loc_t[n]] 

            n +=1;

        return img_rgb_t_process

    ################################################################################
    def get_img_337_5(img_rgb_in,img_rgb_t,y_loc_t,x_loc_t):


        img_rgb_in = numpy.fliplr(img_rgb_in);

        # polygon for input image
        pts_in = np.array([[290,105],[336,587],[1346,587],[1428,105]]);
        c_pts_in = pts_in.astype('float32');

        pts_in_back1 = np.array([[1374,102],[1329,374],[1377,373],[1424,105]]);
        c_pts_in_back1 = pts_in_back1.astype('float32');

        pts_in_back2 = np.array([[1341,377],[1304,581],[1339,580],[1373,376]]);
        c_pts_in_back2 = pts_in_back2.astype('float32');

        pts_in_top = np.array([[293,108],[295,135],[1421,141],[1426,105]]);
        c_pts_in_top = pts_in_top.astype('float32');

        pts_in_front1 = np.array([[295,106],[323,408],[372,406],[345,108]]);
        c_pts_in_front1 = pts_in_front1.astype('float32');

        pts_in_front2 = np.array([[323,411],[342,584],[369,585],[353,411]]);
        c_pts_in_front2 = pts_in_front2.astype('float32');

        # 337.5
        # polygon for target image
        pts_t = np.array([[507,90],[540,418],[853,791],[835,121]]);
        c_pts_t = pts_t.astype('float32');

        # polygon for target image tyre
        pts_t_tyre1 = np.array([[ 584,423],[593,545],[635,580],[626,438]]);
        c_pts_t_tyre1 = pts_t_tyre1.astype('float32');

        pts_t_tyre2 = np.array([[717,574],[719,727],[795,777],[789,598]]);
        c_pts_t_tyre2 = pts_t_tyre2.astype('float32');

        pts_t_top = np.array([[835,63],[492,90],[835,127],[1357,111]]);
        c_pts_t_top = pts_t_top.astype('float32');

        pts_t_back1 = np.array([[831,120],[843,506],[1321,400],[1347,106]]);
        c_pts_t_back1 = pts_t_back1.astype('float32');

        pts_t_back2 = np.array([[843,497],[850,784],[1298,617],[1319,392]]);
        c_pts_t_back2 = pts_t_back2.astype('float32');
        
        ## get the warped image
        img_warped = four_point_transform(img_rgb_in,c_pts_in,c_pts_t)

        ## get the warped image	
        img_warped_back1 = four_point_transform(img_rgb_in,c_pts_in_back1,c_pts_t_back1)

        ## get the warped image	
        img_warped_back2 = four_point_transform(img_rgb_in,c_pts_in_back2,c_pts_t_back2)

        ## get the warped image
        img_warped_top = four_point_transform(img_rgb_in,c_pts_in_top,c_pts_t_top)

        # compute coordinates of polygon
        poly_big = mplpath.Path(c_pts_t)

        poly_tyre1 = mplpath.Path(c_pts_t_tyre1)

        poly_tyre2 = mplpath.Path(c_pts_t_tyre2)

        poly_back1 = mplpath.Path(c_pts_t_back1);

        poly_back2 = mplpath.Path(c_pts_t_back2);

        poly_top = mplpath.Path(c_pts_t_top);

        size_pixel = np.size(y_loc_t);

        img_rgb_t_process = copy(img_rgb_t);

        # map target image to template image
        n = 0
        for i in range(size_pixel):

            if poly_big.contains_point((x_loc_t[n],y_loc_t[n])) or\
                poly_tyre1.contains_point((x_loc_t[n],y_loc_t[n])) or\
                poly_tyre2.contains_point((x_loc_t[n],y_loc_t[n])):

                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped[y_loc_t[n],x_loc_t[n]];\

            if poly_back1.contains_point((x_loc_t[n],y_loc_t[n])):
            
                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped_back1[y_loc_t[n],x_loc_t[n]]

            if poly_back2.contains_point((x_loc_t[n],y_loc_t[n])):
            
                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped_back2[y_loc_t[n],x_loc_t[n]]


            if poly_top.contains_point((x_loc_t[n],y_loc_t[n])):
            
                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped_top[y_loc_t[n],x_loc_t[n]] 

            n +=1;

        return img_rgb_t_process

    ################################################################################
    def get_img_360(img_rgb_in,img_rgb_t,y_loc_t,x_loc_t):

        # polygon for input image
        pts_in = np.array([[228,106],[310,581],[1316,585],[1360,105]]);
        c_pts_in = pts_in.astype('float32');

        pts_in_back1 = np.array([[229,106],[274,374],[325,375],[283,106]]);
        c_pts_in_back1 = pts_in_back1.astype('float32');

        pts_in_back2 = np.array([[278,376],[313,580],[365,580],[329,377]]);
        c_pts_in_back2 = pts_in_back2.astype('float32');


        pts_in_top = np.array([[229,108],[237,147],[1355,150],[1359,108]]);
        c_pts_in_top = pts_in_top.astype('float32');

        pts_in_front1 = np.array([[1313,108],[1283,408],[1324,408],[1355,108]]);
        c_pts_in_front1 = pts_in_front1.astype('float32');

        pts_in_front2 = np.array([[1297,411],[1280,583],[1312,582],[1329,411]]);
        c_pts_in_front2 = pts_in_front2.astype('float32');

        # 360
        # polygon for target image
        pts_t_top = np.array([[420,124],[1174,123],[1013,75],[612,72]]);
        c_pts_t_top = pts_t_top.astype('float32');

        pts_t_back1 = np.array([[430,120],[475,492],[1141,471],[1157,115]]);
        c_pts_t_back1 = pts_t_back1.astype('float32');

        pts_t_back2 = np.array([[475,486],[512,764],[1132,732],[1141,463]]);
        c_pts_t_back2 = pts_t_back2.astype('float32');
        
        ## get the warped image	
        img_warped_back1 = four_point_transform(img_rgb_in,c_pts_in_back1,c_pts_t_back1)

        ## get the warped image	
        img_warped_back2 = four_point_transform(img_rgb_in,c_pts_in_back2,c_pts_t_back2)

        ## get the warped image
        img_warped_top = four_point_transform(img_rgb_in,c_pts_in_top,c_pts_t_top)

        # compute coordinates of polygon

        poly_back1 = mplpath.Path(c_pts_t_back1);

        poly_back2 = mplpath.Path(c_pts_t_back2);

        poly_top = mplpath.Path(c_pts_t_top);

        size_pixel = np.size(y_loc_t);

        img_rgb_t_process = copy(img_rgb_t);

        # map target image to template image
        n = 0
        for i in range(size_pixel):

            if poly_back1.contains_point((x_loc_t[n],y_loc_t[n])):
            
                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped_back1[y_loc_t[n],x_loc_t[n]]

            if poly_back2.contains_point((x_loc_t[n],y_loc_t[n])):
            
                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped_back2[y_loc_t[n],x_loc_t[n]]

            if poly_top.contains_point((x_loc_t[n],y_loc_t[n])):
            
                img_rgb_t_process[y_loc_t[n],x_loc_t[n]] = img_warped_top[y_loc_t[n],x_loc_t[n]] 

            n +=1;

        return img_rgb_t_process

        
    ## Main ####################################################################################

    print('Reading input image ...');\
    print(' ');\

    ## Input image #############################################################################

    dir_path = os.path.dirname(os.path.realpath(__file__));

    # read image
##    img_in=cv2.imread(dir_path+'\\img_input.png');\
                  
    # get an original copy of image
    img_ori_in = copy(img_in);

    # convert image to grayscale
    img_gray_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY);\

    # get rgb channel
    b_in,g_in,r_in = cv2.split(img_in);\

    # get rgb image
    img_rgb_in = copy(cv2.merge([r_in,g_in,b_in]));\

    # get mask & coordinate of item
    [img_f_in,y_loc_in,x_loc_in] = morpho_process(img_gray_in);

    ## Target image ###########################################################################

    print('Reading and Processing template images ...');\
    print(' ');\

    img_target = numpy.zeros((820,1650,3,16));
    # read image
    img_target[:,:,:,0]=cv2.imread(dir_path+'\\template_images\\img_22_5.png');\
    img_target[:,:,:,1]=cv2.imread(dir_path+'\\template_images\\img_45.png');\
    img_target[:,:,:,2]=cv2.imread(dir_path+'\\template_images\\img_67_5.png');\
    img_target[:,:,:,3]=cv2.imread(dir_path+'\\template_images\\img_90.png');\
    img_target[:,:,:,4]=cv2.imread(dir_path+'\\template_images\\img_112_5.png');\
    img_target[:,:,:,5]=cv2.imread(dir_path+'\\template_images\\img_135.png');\
    img_target[:,:,:,6]=cv2.imread(dir_path+'\\template_images\\img_157_5.png');\
    img_target[:,:,:,7]=cv2.imread(dir_path+'\\template_images\\img_180.png');\
    img_target[:,:,:,8]=cv2.imread(dir_path+'\\template_images\\img_202_5.png');\
    img_target[:,:,:,9]=cv2.imread(dir_path+'\\template_images\\img_225.png');\
    img_target[:,:,:,10]=cv2.imread(dir_path+'\\template_images\\img_247_5.png');\
    img_target[:,:,:,11]=cv2.imread(dir_path+'\\template_images\\img_270.png');\
    img_target[:,:,:,12]=cv2.imread(dir_path+'\\template_images\\img_292_5.png');\
    img_target[:,:,:,13]=cv2.imread(dir_path+'\\template_images\\img_315.png');\
    img_target[:,:,:,14]=cv2.imread(dir_path+'\\template_images\\img_337_5.png');\
    img_target[:,:,:,15]=cv2.imread(dir_path+'\\template_images\\img_360.png');\

    img_t = copy(img_target.astype('uint8'));
                  
    # get an original copy of image
    img_ori_t = copy(img_t);

    # convert image to grayscale
    img_gray_t = numpy.zeros((820,1650,16));
    img_rgb_t = numpy.zeros((820,1650,3,16));
    for i in range(img_t.shape[3]):
        img_gray_t[:,:,i] = cv2.cvtColor(img_t[:,:,:,i], cv2.COLOR_BGR2GRAY);\

        # get rgb channel
        b_t,g_t,r_t = cv2.split(img_t[:,:,:,i]);\

        # get rgb image
        img_rgb_t[:,:,:,i] = copy(cv2.merge([r_t,g_t,b_t]));\

    # convert grayscale imge to uint8                         
    img_gray_t = img_gray_t.astype('uint8');
    img_rgb_t = img_rgb_t.astype('uint8');

    # get mask & coordinates of item
    [img_f_t_22_5,y_loc_t_22_5,x_loc_t_22_5] = morpho_process(img_gray_t[:,:,0]);
    [img_f_t_45,y_loc_t_45,x_loc_t_45] = morpho_process(img_gray_t[:,:,1]);
    [img_f_t_67_5,y_loc_t_67_5,x_loc_t_67_5] = morpho_process(img_gray_t[:,:,2]);
    [img_f_t_90,y_loc_t_90,x_loc_t_90] = morpho_process(img_gray_t[:,:,3]);
    [img_f_t_112_5,y_loc_t_112_5,x_loc_t_112_5] = morpho_process(img_gray_t[:,:,4]);
    [img_f_t_135,y_loc_t_135,x_loc_t_135] = morpho_process(img_gray_t[:,:,5]);
    [img_f_t_157_5,y_loc_t_157_5,x_loc_t_157_5] = morpho_process(img_gray_t[:,:,6]);
    [img_f_t_180,y_loc_t_180,x_loc_t_180] = morpho_process(img_gray_t[:,:,7]);
    [img_f_t_202_5,y_loc_t_202_5,x_loc_t_202_5] = morpho_process(img_gray_t[:,:,8]);
    [img_f_t_225,y_loc_t_225,x_loc_t_225] = morpho_process(img_gray_t[:,:,9]);
    [img_f_t_247_5,y_loc_t_247_5,x_loc_t_247_5] = morpho_process(img_gray_t[:,:,10]);
    [img_f_t_270,y_loc_t_270,x_loc_t_270] = morpho_process(img_gray_t[:,:,11]);
    [img_f_t_292_5,y_loc_t_292_5,x_loc_t_292_5] = morpho_process(img_gray_t[:,:,12]);
    [img_f_t_315,y_loc_t_315,x_loc_t_315] = morpho_process(img_gray_t[:,:,13]);
    [img_f_t_337_5,y_loc_t_337_5,x_loc_t_337_5] = morpho_process(img_gray_t[:,:,14]);
    [img_f_t_360,y_loc_t_360,x_loc_t_360] = morpho_process(img_gray_t[:,:,15]);


    ## Compute images of different angle ######################################################

    print('Computing image for angle 22.5 ...');\
    print(' ');\
    img_t_22_5 = get_img_22_5(img_rgb_in,img_rgb_t[:,:,:,0],y_loc_t_22_5,x_loc_t_22_5)
    print('Computing image for angle 45 ...');\
    print(' ');\
    img_t_45 = get_img_45(img_rgb_in,img_rgb_t[:,:,:,1],y_loc_t_45,x_loc_t_45)
    print('Computing image for angle 67.5 ...');\
    print(' ');\
    img_t_67_5 = get_img_67_5(img_rgb_in,img_rgb_t[:,:,:,2],y_loc_t_67_5,x_loc_t_67_5)
    print('Computing image for angle 90 ...');\
    print(' ');\
    img_t_90 = get_img_90(img_rgb_in,img_rgb_t[:,:,:,3],y_loc_t_90,x_loc_t_90)
    print('Computing image for angle 112.5 ...');\
    print(' ');\
    img_t_112_5 = get_img_112_5(img_rgb_in,img_rgb_t[:,:,:,4],y_loc_t_112_5,x_loc_t_112_5)
    print('Computing image for angle 135 ...');\
    print(' ');\
    img_t_135 = get_img_135(img_rgb_in,img_rgb_t[:,:,:,5],y_loc_t_135,x_loc_t_135)
    print('Computing image for angle 157.5 ...');\
    print(' ');\
    img_t_157_5 = get_img_157_5(img_rgb_in,img_rgb_t[:,:,:,6],y_loc_t_157_5,x_loc_t_157_5)
    print('Computing image for angle 180 ...');\
    print(' ');\
    img_t_180 = get_img_180(img_rgb_in,img_rgb_t[:,:,:,7],y_loc_t_180,x_loc_t_180)
    print('Computing image for angle 202.5 ...');\
    print(' ');\
    img_t_202_5 = get_img_202_5(img_rgb_in,img_rgb_t[:,:,:,8],y_loc_t_202_5,x_loc_t_202_5)
    print('Computing image for angle 225 ...');\
    print(' ');\
    img_t_225 = get_img_225(img_rgb_in,img_rgb_t[:,:,:,9],y_loc_t_225,x_loc_t_225)
    print('Computing image for angle 247.5 ...');\
    print(' ');\
    img_t_247_5 = get_img_247_5(img_rgb_in,img_rgb_t[:,:,:,10],y_loc_t_247_5,x_loc_t_247_5)
    print('Computing image for angle 270 ...');\
    print(' ');\
    img_t_270 = get_img_270(img_rgb_in,img_rgb_t[:,:,:,11],y_loc_t_270,x_loc_t_270)
    print('Computing image for angle 292.5 ...');\
    print(' ');\
    img_t_292_5 = get_img_292_5(img_rgb_in,img_rgb_t[:,:,:,12],y_loc_t_292_5,x_loc_t_292_5)
    print('Computing image for angle 315 ...');\
    print(' ');\
    img_t_315 = get_img_315(img_rgb_in,img_rgb_t[:,:,:,13],y_loc_t_315,x_loc_t_315)
    print('Computing image for angle 337.5 ...');\
    print(' ');\
    img_t_337_5 = get_img_337_5(img_rgb_in,img_rgb_t[:,:,:,14],y_loc_t_337_5,x_loc_t_337_5)
    print('Computing image for angle 360 ...');\
    print(' ');\
    img_t_360 = get_img_360(img_rgb_in,img_rgb_t[:,:,:,15],y_loc_t_360,x_loc_t_360)
    
    print('Saving transformed images to folder...');\
    print(' ');\
    # save to images
    
    scipy.misc.imsave(dir_path+'\\transformed_images\\img_t_22_5.png',img_t_22_5);\
    scipy.misc.imsave(dir_path+'\\transformed_images\\img_t_45.png',img_t_45);\
    scipy.misc.imsave(dir_path+'\\transformed_images\\img_t_67_5.png',img_t_67_5);\
    scipy.misc.imsave(dir_path+'\\transformed_images\\img_t_90.png',img_t_90);\
    scipy.misc.imsave(dir_path+'\\transformed_images\\img_t_112_5.png',img_t_112_5);\
    scipy.misc.imsave(dir_path+'\\transformed_images\\img_t_135.png',img_t_135);\
    scipy.misc.imsave(dir_path+'\\transformed_images\\img_t_157_5.png',img_t_157_5);\
    scipy.misc.imsave(dir_path+'\\transformed_images\\img_t_180.png',img_t_180);\
    scipy.misc.imsave(dir_path+'\\transformed_images\\img_t_202_5.png',img_t_202_5);\
    scipy.misc.imsave(dir_path+'\\transformed_images\\img_t_225.png',img_t_225);\
    scipy.misc.imsave(dir_path+'\\transformed_images\\img_t_247_5.png',img_t_247_5);\
    scipy.misc.imsave(dir_path+'\\transformed_images\\img_t_270.png',img_t_270);\
    scipy.misc.imsave(dir_path+'\\transformed_images\\img_t_292_5.png',img_t_292_5);\
    scipy.misc.imsave(dir_path+'\\transformed_images\\img_t_315.png',img_t_315);\
    scipy.misc.imsave(dir_path+'\\transformed_images\\img_t_337_5.png',img_t_337_5);\
    scipy.misc.imsave(dir_path+'\\transformed_images\\img_t_360.png',img_t_360);\

    print('Done');\

    return img_t_22_5,img_t_45,img_t_67_5,img_t_90,\
           img_t_112_5,img_t_135,img_t_157_5,img_t_180,\
           img_t_202_5,img_t_225,img_t_247_5,img_t_270,\
           img_t_292_5,img_t_315,img_t_337_5,img_t_360








    ## end of codes #######################################################################




