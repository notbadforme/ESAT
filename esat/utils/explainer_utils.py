import torch
def intialize_mask(x,mask,size):
    empty_array=torch.ones((size,size))
    for i in range(len(mask[0])):
        for j in range(len(mask[0])):
            new_mask=torch.sigmoid(mask[i][j])*empty_array
            if j == 0:
                line_mask=new_mask
            else:
                line_mask=torch.cat((line_mask,new_mask),1)
        if i == 0:
            all_mask=line_mask
        else:
            all_mask=torch.cat((all_mask,line_mask),0)
    return x*all_mask

def mask2png1(mask,image,mask_size,k):
    patch_list=[]
    while k>0:
        maxvalue,line=torch.max(mask,0)
        _,cow=torch.max(maxvalue,0)
        i,j=line[cow].numpy(),cow.numpy()
        mask[i][j]=-1
        width, height = image.size
        item_width = int(width / mask_size)
        print(j*item_width,i*item_width,(j+1)*item_width,(i+1)*item_width)
        box = (j*item_width,i*item_width,(j+1)*item_width,(i+1)*item_width)
        patch = image.crop(box)
        patch = patch.resize((512,512))
        patch_list.append(patch)
        k-=1
    return patch_list

def mask2png2(mask,image,mask_size,k):
    patch_list=[]
    while k>0:
        minvalue,line=torch.min(mask,0)
        _,cow=torch.min(minvalue,0)
        i,j=line[cow].numpy(),cow.numpy()
        mask[i][j]=2
        width, height = image.size
        item_width = int(width / mask_size)
        print(j*item_width,i*item_width,(j+1)*item_width,(i+1)*item_width)
        box = (j*item_width,i*item_width,(j+1)*item_width,(i+1)*item_width)

        patch = image.crop(box)
        patch = patch.resize((512,512))
        patch_list.append(patch)
        k-=1
    return patch_list