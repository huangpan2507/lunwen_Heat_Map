# lunwen_Heat_Map
这是cam画论文的热力图，在原来代码的if __nnme__=='main'处，增加了，自动查找每个文件夹下的图片，并生成热力图．

原github地址：https://github.com/jacobgil/pytorch-grad-cam

使用方法：python gradcam.py --image-path /mnt/SYSU-MM01/cam5/0002/0004.jpg --use-cuda　　


Usage: python gradcam.py --image-path <path_to_image>

To use with CUDA: python gradcam.py --image-path <path_to_image> --use-cuda



增加的功能：自动从指定的cam文件夹下寻找所有文件夹里所有的图片，并画出对应的热力图，新建同样的目录级并将图片保存在其中．
