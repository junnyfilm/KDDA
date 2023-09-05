import itertools
from torch.utils import data

# def distill_loss(output, target, teacher_output, intermediate, opt=optimizer):
#     if intermediate is not None:
        

        
#         distillation_loss=intermediate
#     else:
#         distillation_loss = 0
        
#     student_loss = nn.CrossEntropyLoss().to(device)(output, target)
#     alpha=0.5
#     loss_b=alpha * student_loss + (1-alpha) * (distillation_loss)

#     if opt is not None:
#         opt.zero_grad()
#         loss_b.backward()
#         opt.step()

#     return loss_b.item()


def l2diff(x1, x2):
    """
    standard euclidean norm
    """
    return torch.sum((x1-x2).pow(2)).pow(0.5)


def moment_diff(xx1, xx2, k):
    """
    difference between moments
    """
    ss1 = (xx1.pow(k)).mean(dim=1)
    ss2 = (xx2.pow(k)).mean(dim=1)
    return l2diff(ss1, ss2)

def CMD(x1,x2):
    mx1 = x1.mean(dim=1)
    mx2 = x2.mean(dim=1)
    sx1=x1.clone()
    sx2=x2.clone()
    
    for i in range(x1.shape[0]):
        for j in range(x1.shape[1]):
            sx1[i][j]=x1[i][j]-mx1[i]    
            sx2[i][j]=x2[i][j]-mx2[i]    

    scms1 = l2diff(mx1, mx2)
    scms2 = moment_diff(sx1, sx2, 2)
    scms3 = moment_diff(sx1, sx2, 3)
    # scms4 = moment_diff(sx1, sx2, 4)
    # scms5 = moment_diff(sx1, sx2, 5)
    scms=scms1+scms2+scms3
    return scms


def gaussian_kernel(a, b):
    dim1_1, dim1_2 = a.shape[0], b.shape[0]
    depth = a.shape[1]
    a = a.view(dim1_1, 1, depth)
    b = b.view(1, dim1_2, depth)
    a_core = a.expand(dim1_1, dim1_2, depth)
    b_core = b.expand(dim1_1, dim1_2, depth)
    numerator = (a_core - b_core).pow(2).mean(2)/depth
    return torch.exp(-numerator)

def MMD(a, b):
    return gaussian_kernel(a, a).mean() + gaussian_kernel(b, b).mean() - 2*gaussian_kernel(a, b).mean()