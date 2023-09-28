import os
import sys
from distutils.version import LooseVersion
# Numerical libs
import torch
from torchvision import transforms as T
import argparse 
from torch.distributed import get_rank
import cv2
import numpy as np
import time
import timeit
from config import load_cfg_from_cfg_file

from torch.distributed import get_rank

class UNETInference():
    def __init__(self,cfg_file,img_ori,model_ckpt,device):
        self.cfg =load_cfg_from_cfg_file(cfg_file)
        self.model_path= model_ckpt
        self.img_ori= img_ori
        self.device=device
        self.model = self.load_model(self.model_path, self.device)

    def color_class(self,preds):
    # fonction pour afficher que la route
        #print("--vigne_color")
        preds[preds==1]=255 # mettre classe route vigne = 255
        preds[preds!=255]=0 # mettre le reste des classes = 0
        return preds

    def resize(self,img):
        H_RESIZE=384#318
        # resize the image and masks
        img = cv2.resize(img, dsize=(self.cfg.TEST_W, H_RESIZE))
        return img

    def add_pad(self,img):
        global pad_h,pad_h_half,pad_w, pad_w_half, ori_h,ori_w
        #print(mask.shape, img.shape)
        ori_h, ori_w, _ = img.shape
        pad_h = max(self.cfg.TEST_H - ori_h, 0)
        pad_w = max(self.cfg.TEST_W - ori_w, 0)
        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)
        if pad_h > 0 or pad_w > 0:
            img = cv2.copyMakeBorder(img, pad_h_half, pad_h - pad_h_half,pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=0)
        return img

    def load_model(self,model_ckpt, device):
    # fonction for loading the pretrained_model
        # load the modele in the device
        model = torch.load(model_ckpt,map_location=device)
        model.eval()
        return model

    def pre_process(self,img, device):
        #resize image
        img=self.resize(img)
        img=self.add_pad(img) # with padding
        # normalisation and standardization
        t = T.Compose([T.ToTensor(), T.Normalize(self.cfg.MEAN,self.cfg.STD)])
        img=t(img)
        # image to device
        img=img.to(device)
        img = img.unsqueeze(0)
        return img

    def inference(self,model,img):
        # fonction for prediction
        with torch.no_grad():
            prediction = model(img)
        return prediction

    def post_process(self,prediction,image_ori):

        # fonction to treat image after inference
        pred_mask = torch.argmax(prediction, dim=1)
        pred_mask = pred_mask.detach().cpu().numpy().copy()
        # ----------choose the color of class  and the 3 channel image
        pred =pred_mask.astype(np.uint8).transpose((1, 2, 0))
        #crop pred
        pred = pred[pad_h_half:pad_h_half+ori_h, pad_w_half:pad_w_half+ori_w]
        pred = cv2.resize(pred, ( image_ori.shape[1], image_ori.shape[0]), interpolation=cv2.INTER_LINEAR)
        pred=self.color_class(pred)
        # takes single channel images and combines them to make a multi-channel image: get tree chanel (355, 473, 3)
        pred = cv2.merge([pred, pred, pred]) 
        # add prediction and image 
        #print("pred",pred.shape, type(pred))
        #print("pred",image_ori.shape, type (image_ori))
        #image_merge = cv2.addWeighted(image_ori, 0.5, pred, 0.8, 0)
        #---------if we do inference with ros comment 
        #cv2.imshow("affiche", image_merge)
        #cv2.waitKey(5000)
        return pred
    def get_contour(self,image):
        # Convert image to gray and blur it
        src_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        src_gray = cv2.blur(src_gray, (9,9))
        contours, hierarchy = cv2.findContours(src_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        drawing = np.zeros((src_gray .shape[0], src_gray .shape[1], 3), dtype=np.uint8)
        #print("contours",len(contours))
        for i in range(len(contours)):
            color =  255,0,0
            cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)
        vecteur_contour=np.array(contours).squeeze(0)
        return vecteur_contour
   
    def img_infer(self):
        input=self.pre_process(self.img_ori,self.device)
        output=self.inference(self.model,input)
        output=self.post_process(output,self.img_ori)
        contour= self.get_contour(output)
        return output,contour
