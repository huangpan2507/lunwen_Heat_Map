import torch
import glob
import os
import argparse
import cv2
import numpy as np
import torch
from pyarrow.lib import string
from torch.autograd import Function
from torchvision import models, transforms


class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0), -1)
            else:
                x = module(x)

        return target_activations, x


def preprocess_image(img):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    preprocessing = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return preprocessing(img.copy()).unsqueeze(0)


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, target_category=None):
        if self.cuda:
            input_img = input_img.cuda()

        features, output = self.extractor(input_img)

        if target_category == None:
            target_category = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = one_hot.cuda()

        one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input_img.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


class GuidedBackpropReLU(Function):
    @staticmethod
    def forward(self, input_img):
        positive_mask = (input_img > 0).type_as(input_img)
        output = torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), input_img, positive_mask)
        self.save_for_backward(input_img, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input_img, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input_img > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input_img.size()).type_as(input_img),
                                   torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), grad_output,
                                                 positive_mask_1), positive_mask_2)
        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply

        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, target_category=None):
        if self.cuda:
            input_img = input_img.cuda()

        input_img = input_img.requires_grad_(True)

        output = self.forward(input_img)

        if target_category == None:
            target_category = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = one_hot.cuda()

        one_hot = torch.sum(one_hot * output)
        one_hot.backward(retain_graph=True)

        output = input_img.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)


if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    args = get_args()
    str_pre = args.image_path.split(".")[0]
    _,_1,_2,str_cam, str_num,str_name = str(str_pre).split("/",5)     #将给定路径以／符号为分割副进行拆分．
    print("...: ", _,_1,_2, str_cam, str_num,str_name)
    cam_path = _ +"/"+ _1+"/" +_2+"/"+ str_cam+"/"                    #形成　类似/mnt/sysu/cam2/　的文件路径
    print("cam_path: ",cam_path)     #/mnt/sysu/cam2/　　　　　　　
    files = os.listdir(cam_path)                                      #列出所有的文件
    i = 0
    print("files len :", len(files))                                  # 统计类似/mnt/sysu/cam2/　下总共有多少文件夹
    for _ in range(len(files)):
        i+=1                                                          # 因为原数据集路径为/mnt/SYSU-MM01/cam5/0002/0004.jpg
        pic_path = cam_path+str(i).zfill(4)                           #　这一步是为了形成0002这样的文件夹，若不存在则加１，zfill(4)前面补０成四位数
        print("pic_path: ", pic_path)

        while(os.path.exists(pic_path)==False):
            i+=1
            pic_path = cam_path + str(i).zfill(4)

        files_pic = os.listdir(pic_path)
        print("len files_pic: ", len(files_pic))
        for j in range(len(files_pic)):

            path = pic_path+"/"+str(j+1).zfill(4)+".jpg"
            print("path: ",path)
            while (os.path.exists(path) == False):                 #这一步是为了形成0002.jpg这样的图片，若不存在则加１，zfill(4)前面补０成四位数
                j += 1
                path = pic_path+"/"+str(j).zfill(4)+".jpg"
            #path = path.strip()

            img = cv2.imread(path, 1)
            img = np.float32(img) / 255
            #print("img ",img)
            # Opencv loads as BGR:
            img = img[:, :, ::-1]
            input_img = preprocess_image(img)

            # If None, returns the map for the highest scoring category.
            # Otherwise, targets the requested category.
            target_category = None

            model = models.resnet50(pretrained=True)
            grad_cam = GradCam(model=model, feature_module=model.layer4, \
                               target_layer_names=["2"], use_cuda=args.use_cuda)
            grayscale_cam = grad_cam(input_img, target_category)

            grayscale_cam = cv2.resize(grayscale_cam, (img.shape[1], img.shape[0]))
            cam = show_cam_on_image(img, grayscale_cam)

            gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
            gb = gb_model(input_img, target_category=target_category)
            gb = gb.transpose((1, 2, 0))

            cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
            cam_gb = deprocess_image(cam_mask * gb)
            gb = deprocess_image(gb)
            result_pre = path.split('.')[0]
            _, _,_,result_cam,result_num,result_name = str(result_pre).split('/',5)
            out_path = './pic/'+str(result_cam)+'/'+str(result_num)+'/'+str(result_name)    # 在当前目录下新建pic文件夹，再该文件夹下新建与数据集相同的文件目录

            os.makedirs(out_path)                                                           # 新建与原数据集一样的目录级．
            #print("hhh: ",os.makedirs(out_path))
            cv2.imwrite(out_path+'.jpg', cam)                                               # 生成对应目录下对应文件名的热力图
            #cv2.imwrite("./pic/"+str(result_cam)+"/"+str(result_num)+"/"+str(result_name)+".jpg", gb)
            #cv2.imwrite('./pic/result_num/result_name.jpg.jpg', cam_gb)

        """try:
            files_pic = os.listdir(pic_path)
        except FileNotFoundError as err:
            while (not files_pic):

                pic_path = cam_path + str(i).zfill(4)
                print("文件夹不存在，具体信息为{}".format(err))
                i += 1
                while ()"""
