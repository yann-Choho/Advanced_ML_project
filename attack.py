import torch
import torch.nn.functional as F

class FastGradientSignUntargeted():
    def __init__(self, model, epsilon, alpha, min_val, max_val, max_iters, _type='linf'):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.min_val = min_val
        self.max_val = max_val
        self.max_iters = max_iters
        self._type = _type
        
    def perturb(self, original_images, labels, reduction4loss='mean', random_start=False):
        x = original_images.clone()
        x.requires_grad = True 

        for _iter in range(self.max_iters):
            outputs = self.model(x, _eval=True)
            loss = F.cross_entropy(outputs, labels, reduction=reduction4loss)
            grad_outputs = torch.ones(loss.shape).to(x.device) if reduction4loss == 'none' else None
            grads = torch.autograd.grad(loss, x, grad_outputs=grad_outputs, only_inputs=True)[0]
            x.data += self.alpha * torch.sign(grads.data)
            x = project(x, original_images, self.epsilon, self._type)
            x.clamp_(self.min_val, self.max_val)

        return x

def project(x, original_x, epsilon, _type='linf'):
    if _type == 'linf':
        max_x = original_x + epsilon
        min_x = original_x - epsilon
        x = torch.max(torch.min(x, max_x), min_x)
    elif _type == 'l2':
        dist = (x - original_x).view(x.shape[0], -1)
        dist_norm = torch.norm(dist, dim=1, keepdim=True)
        mask = (dist_norm > epsilon).unsqueeze(2).unsqueeze(3)
        dist = dist / dist_norm * epsilon
        dist = dist.view(x.shape)
        x = (original_x + dist) * mask.float() + x * (1 - mask.float())
    else:
        raise NotImplementedError

    return x
