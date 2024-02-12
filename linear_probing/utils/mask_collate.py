from multiprocessing import context
from re import T
import torch
import torch.nn as nn





class MaskGenerator(object):

    def __init__(self, 
                img_size = (224,224), 
                patch_size = (16,16), 
                tgt_scale = (0.15, 1.0), 
                tgt_aspect = (0.75, 1.5),
                context_scale = (0.15, 1.0),
                context_aspect = (0.75, 1.5),
                mask_rate = 0.0):


        super(MaskGenerator, self).__init__()

        self.img_size = img_size
        self.patch_size = patch_size


        self.h = img_size[0] // patch_size[0]
        self.w = img_size[1] // patch_size[1]

        self.tgt_scale = tgt_scale
        self.tgt_aspect = tgt_aspect
        self.context_scale = context_scale
        self.context_aspect = context_aspect

        self.sample_ratio = 1.0 - mask_rate


    def get_size(self, scale, aspect):
        
        min_s, max_s = scale
        min_a, max_a = aspect

        s = torch.rand(1)
        a = torch.rand(1)

        s = min_s + s * (max_s - min_s)
        a = min_a + a * (max_a - min_a)

        s, a = s.item(), a.item()

        lenght = self.h * self.w * s

        h = int(round((lenght*a)**0.5))
        w = int(round((lenght/a)**0.5))

        #check size

        h = min(self.h, h)
        w = min(self.w, w)

        return h,w


    def get_mask(self, sizes):
        
        h,w = sizes

        top_range = self.h - h + 1
        left_range = self.w - w + 1

        top = torch.randint(0, top_range, (1,))
        left = torch.randint(0, left_range, (1,))

        mask = torch.zeros((self.h, self.w))
        mask[top:top+h, left:left+w] = 1
        
        return mask


    def get_indices(self, mask):

        indices = torch.nonzero(mask.view(-1))

        return indices

    def sample(self, indices, n):

        idx = torch.randperm(len(indices))
        sampled = indices[idx[:n]]

        return sampled
    

    def __call__(self, batch):

        B = len(batch)
        collated_batch = torch.utils.data.default_collate(batch)[0] #pas les classes

        #on fixe les tailles du ctx/tgt sur le batch

        ctx_size = self.get_size(self.context_scale, self.context_aspect)
        tgt_size = self.get_size(self.tgt_scale, self.tgt_aspect)

        #sampling des position, génération de B * n_targets combinaisons

        ctx_masks = []
        tgt_masks = []

        for i in range(B):
            
            valid_loc = False

            ctx_mask = self.get_mask(ctx_size)
            tgt_mask = self.get_mask(tgt_size)
            
            ctx_masks.append(ctx_mask)
            tgt_masks.append(tgt_mask)

        #conversion masks en indices

        ctx_indices = [self.get_indices(m) for m in ctx_masks]
        tgt_indices = [self.get_indices(m) for m in tgt_masks]

        ctx_sampled = [self.sample(idx, int(len(idx) * self.sample_ratio)) for idx in ctx_indices]
        tgt_sampled = [self.sample(idx, int(len(idx) * self.sample_ratio)) for idx in tgt_indices]
        
        ctx = torch.stack((ctx_sampled), dim = 0).squeeze(2)
        tgt = torch.stack((tgt_sampled), dim = 0).squeeze(2)

        patchs = torch.nn.functional.unfold(collated_batch, kernel_size=self.patch_size, stride=self.patch_size).permute(0,2,1)

        patchs1 = torch.gather(patchs, dim=1, index=ctx.unsqueeze(-1).repeat(1, 1, patchs.size(2)))
        patchs2 = torch.gather(patchs, dim=1, index=tgt.unsqueeze(-1).repeat(1, 1, patchs.size(2)))

        return patchs1, patchs2, ctx, tgt

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    import torchvision
    import time
    transforms = torchvision.transforms.Compose([torchvision.transforms.Resize(224),
                                                 torchvision.transforms.ToTensor()])

    cifar100 = torchvision.datasets.CIFAR100(root='./data', transform=transforms, download=True)
    
    m = MaskGenerator(mask_rate=0.0)

    x = [cifar100[i] for i in range(128)]

    s = time.time()
    a,b,c = m(x)
    e = time.time()

    print('time : {}'.format(e - s))

    print(a.size())
    print(b.size())
    print(c.size())

    #plot des masks et images:

    for i in range(16):
        
        

            
        img = torch.clone(a[i])
            

        #supression pixels sur ctx/tgt

        img[b[i]] += 0.2
        img[c[i]] += 0.2

        #remise sous forme d'image
        img = img.unsqueeze(0)
        img = img.permute(0,2,1)
        img = torch.nn.functional.fold(img, (224,224), kernel_size=(16,16), stride=(16,16))
        img = img.squeeze(0)

        img = img.permute(1,2,0)

        plt.subplot(4,4,i + 1)
        plt.imshow(img)
    
    plt.show()

