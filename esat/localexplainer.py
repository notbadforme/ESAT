import glob
import os
from PIL import Image
from torchvision import transforms
from model import siameseexplainer as s
from utils import explainer_utils as m
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
Image.MAX_IMAGE_PIXELS = 10000000000
tensor_to_pil =  transforms.ToPILImage()

t_transforms = transforms.Compose(
    [
        transforms.Resize((6144, 6144)),
        transforms.ToTensor(),
    ]
)
train_dir="../NLST2/"
all_list = glob.glob(os.path.join(train_dir,'*/*.png'))
surv_time_list=[]
status_list=[]
S=s.SiameseNet()
E=s.Explainer(S)


for i in all_list:
    status= i.split('/')[-1].split('.')[0].split('_')[-2]
    if int(status)!= -1:
        surv_time=i.split('/')[-1].split('.')[0].split('_')[-1]
        img = Image.open(i).convert("RGB")
        img_transformed = t_transforms(img)
        pil_img = tensor_to_pil(img_transformed)
        x=img_transformed.unsqueeze(0)
        surv_time_list.append(int(surv_time))
        status_list.append(int(status))
        a=E.globalexplainer(x)
        explainer_png1=m.mask2png1(a,pil_img,12,5)
        explainer_png2=m.mask2png2(a,pil_img,12,5)
        for k in range(5):
            name1="NLSTexplainer/"+status+"_"+surv_time+"_"+"_"+str(k)+"_pos.png"
            name2="NLSTexplainer/"+status+"_"+surv_time+"_"+"_"+str(k)+"_neg.png"
            explainer_png1[k].save(name1)
            explainer_png2[k].save(name2)
