import torch
import numpy as np


# compute rollout between attention layers
def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration- code adapted from https://github.com/samiraabnar/attention_flow
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                          for i in range(len(all_layer_matrices))]
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer+1, len(matrices_aug)):
        joint_attention = matrices_aug[i].bmm(joint_attention)
    return joint_attention

class LRP:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_LRP(self, input, index=None, method="transformer_attribution", is_ablation=False, start_layer=0):
        output = self.model(input)
        kwargs = {"alpha": 1}
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True).to(input.device)
        one_hot = torch.sum(one_hot * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        return self.model.relprop(torch.tensor(one_hot_vector).to(input.device), method=method, is_ablation=is_ablation,
                                  start_layer=start_layer, **kwargs)



class Baselines:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    #CAM
    def generate_cam_attn(self, input, index=None, mae=False):
        output = self.model(input.to(self.model.device), register_hook=True)
        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(self.model.device) * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        #################### attn
        grad = self.model.blocks[-1].attn.get_attn_gradients()
        cam = self.model.blocks[-1].attn.get_attention_map()
        if mae:
            cam = cam[0, :, 1:, 1:]
            grad = grad[0, :, 1:, 1:]
            grad = grad.mean(dim=[0, 2], keepdim=True)
            cam = (cam * grad).mean(0).mean(1).clamp(min=0)
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = cam[0, :, 0, 1:].reshape(-1, 14, 14)
            grad = grad[0, :, 0, 1:].reshape(-1, 14, 14)
            grad = grad.mean(dim=[1, 2], keepdim=True)
            cam = (cam * grad).mean(0).clamp(min=0)
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam
        #################### attn

    #RAM
    def generate_attn(self, input, mae=False, dino=False):
        output = self.model(input.to(self.model.device), register_hook=True)
        cam = self.model.blocks[-1].attn.get_attention_map()
        if mae:
            cam = cam[0, :, 1:, 1:]
            cam = cam.mean(0).mean(1).clamp(min=0)
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = cam[0, :, 0, 1:].reshape(-1, 14, 14)
            cam = cam.mean(0).clamp(min=0)
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam

    def generate_rollout(self, input, start_layer=0, mae=False):
        self.model(input)
        blocks = self.model.blocks
        all_layer_attentions = []
        for blk in blocks:
            attn_heads = blk.attn.get_attention_map()
            avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
            all_layer_attentions.append(avg_heads)
        rollout = compute_rollout_attention(all_layer_attentions, start_layer=start_layer)
        if mae:
            return rollout[:,1:,1:].mean(1)
        else:
            return rollout[:,0, 1:]

    # Head-wise
    def generate_BIH(self, input, index=None, steps=20, start_layer=4, mae=False, dino=False, ssl=False):
        b = input.shape[0]
        output = self.model(input, register_hook=True)
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((b, output.size()[-1]), dtype=np.float32)
        one_hot[np.arange(b), index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(self.model.device) * output)
        
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        
        _, num_head, num_tokens, _ = self.model.blocks[-1].attn.get_attention_map().shape

        R = torch.eye(num_tokens, num_tokens).expand(b, num_tokens, num_tokens).to(self.model.device)
        for nb, blk in enumerate(self.model.blocks):
            if nb < start_layer-1:
                continue
            
            grad = blk.attn.get_attn_gradients()
            grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
            cam = blk.attn.get_attention_map()
            cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
            
            Ih = torch.mean(torch.matmul(cam.transpose(-1,-2), grad).abs(), dim=(-1,-2))
            Ih = Ih/torch.sum(Ih)
            cam = torch.matmul(Ih,cam.reshape(num_head,-1)).reshape(num_tokens,num_tokens)
            
            R = R + torch.matmul(cam.to(self.model.device), R.to(self.model.device))
        
        if ssl:
            if mae:
                return R[:, 1:, 1:].abs().mean(axis=1)
            elif dino:
                return (R[:, 1:, 1:].abs().mean(axis=1)+R[:, 0, 1:].abs())
            else:
                return R[:, 0, 1:].abs() 
        
        total_gradients = torch.zeros(b, num_head, num_tokens, num_tokens).to(self.model.device)
        for alpha in np.linspace(0, 1, steps):        
            # forward propagation
            data_scaled = input * alpha

            # backward propagation
            output = self.model(data_scaled, register_hook=True)
            one_hot = np.zeros((b, output.size()[-1]), dtype=np.float32)
            one_hot[np.arange(b), index] = 1
            one_hot_vector = one_hot
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            one_hot = torch.sum(one_hot.to(self.model.device) * output)

            self.model.zero_grad()
            one_hot.backward(retain_graph=True)

            # cal grad
            gradients = self.model.blocks[-1].attn.get_attn_gradients()
            total_gradients += gradients        
       
        W_state = (total_gradients / steps).clamp(min=0).mean(1).reshape(b, num_tokens, num_tokens)
        R = W_state * R
        
        if mae:
            return R[:, 1:, 1:].mean(axis=1)
        elif dino:
            return (R[:, 1:, 1:].mean(axis=1) + R[:, 0, 1:])
        else:
            return R[:, 0, 1:]
    
    #token wise
    def generate_BIT(self, input, index=None, steps=20, start_layer=6, mae=False, ssl=False, dino=False):
        b = input.shape[0]
        output = self.model(input, register_hook=True)
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((b, output.size()[-1]), dtype=np.float32)
        one_hot[np.arange(b), index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(self.model.device) * output)
        
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        
        _, num_head, num_tokens, _ = self.model.blocks[-1].attn.get_attention_map().shape

        R = torch.eye(num_tokens, num_tokens).expand(b, num_tokens, num_tokens).to(self.model.device)
        for nb, blk in enumerate(self.model.blocks):
            if nb < start_layer-1:
                continue
            z = blk.get_input()
            vproj = blk.attn.get_vproj()
    
            order = torch.linalg.norm(vproj, dim=-1).squeeze()/torch.linalg.norm(z, dim=-1).squeeze()
            m = torch.diag_embed(order)
            cam = blk.attn.get_attention_map()
            cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1]).mean(0)
            
            R = R + torch.matmul(torch.matmul(cam.to(self.model.device), m.to(self.model.device)), R.to(self.model.device))
        
        if ssl:
            if mae:
                return R[:, 1:, 1:].abs().mean(axis=1)
            elif dino:
                return (R[:, 1:, 1:].abs().mean(axis=1)+R[:, 0, 1:].abs())
            else:
                return R[:, 0, 1:].abs()
        
        total_gradients = torch.zeros(b, num_head, num_tokens, num_tokens).to(self.model.device)
        for alpha in np.linspace(0, 1, steps):        
            # forward propagation
            data_scaled = input * alpha

            # backward propagation
            output = self.model(data_scaled, register_hook=True)
            one_hot = np.zeros((b, output.size()[-1]), dtype=np.float32)
            one_hot[np.arange(b), index] = 1
            one_hot_vector = one_hot
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            one_hot = torch.sum(one_hot.to(self.model.device) * output)

            self.model.zero_grad()
            one_hot.backward(retain_graph=True)

            # cal grad
            gradients = self.model.blocks[-1].attn.get_attn_gradients()
            total_gradients += gradients        
       
        W_state = (total_gradients / steps).clamp(min=0).mean(1).reshape(b, num_tokens, num_tokens)
            
        R = W_state * R.abs()     
        
        if mae:
            return R[:, 1:, 1:].mean(axis=1)
        elif dino:
            return (R[:, 1:, 1:].mean(axis=1) + R[:, 0, 1:])
        else:
            return R[:, 0, 1:]
    
    
    
    #GA
    def generate_genattr(self, input, start_layer=1, index=None, mae=False):
        b = input.shape[0]
        output = self.model(input, register_hook=True)
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((b, output.size()[-1]), dtype=np.float32)
        one_hot[np.arange(b), index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(self.model.device) * output)
        
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        
        _, num_head, num_tokens, _ = self.model.blocks[-1].attn.get_attention_map().shape

        R = torch.eye(num_tokens, num_tokens).expand(b, num_tokens, num_tokens).to(self.model.device)
        for nb, blk in enumerate(self.model.blocks):
            if nb < start_layer-1:
                continue
                
            cam = blk.attn.get_attention_map()
            cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
            grad = blk.attn.get_attn_gradients()
            grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
            cam = (grad*cam).mean(0).clamp(min=0)
            R = R + torch.matmul(cam, R)  
        
        if mae:
            return R[:, 1:, 1:].mean(axis=1)
        else:
            return R[:, 0, 1:]



    #TAM
    def generate_transition_attention_maps(self, input, index=None, start_layer=0, steps=20, with_integral=True, first_state=False, mae=False, dino=False):
        b = input.shape[0]
        output = self.model(input, register_hook=True)
        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((b, output.size()[-1]), dtype=np.float32)
        one_hot[np.arange(b), index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(self.model.device) * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        b, h, s, _ = self.model.blocks[-1].attn.get_attention_map().shape

        num_blocks = len(self.model.blocks)

        states = self.model.blocks[-1].attn.get_attention_map().mean(1)[:, 0, :].reshape(b, 1, s)
        for i in range(start_layer, num_blocks-1)[::-1]:
            attn = self.model.blocks[i].attn.get_attention_map().mean(1)

            states_ = states
            states = states.bmm(attn)
            # add residual
            states += states_

        total_gradients = torch.zeros(b, h, s, s).to(self.model.device)
        for alpha in np.linspace(0, 1, steps):        
            # forward propagation
            data_scaled = input * alpha

            # backward propagation
            output = self.model(data_scaled, register_hook=True)
            one_hot = np.zeros((b, output.size()[-1]), dtype=np.float32)
            one_hot[np.arange(b), index] = 1
            one_hot_vector = one_hot
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            one_hot = torch.sum(one_hot.to(self.model.device) * output)

            self.model.zero_grad()
            one_hot.backward(retain_graph=True)

            # cal grad
            gradients = self.model.blocks[-1].attn.get_attn_gradients()
            total_gradients += gradients
        
        if with_integral:
            W_state = (total_gradients / steps).clamp(min=0).mean(1)[:, 0, :].reshape(b, 1, s)
        else:
            W_state = self.model.blocks[-1].attn.get_attn_gradients().clamp(min=0).mean(1)[:, 0, :].reshape(b, 1, s)
        
        if first_state:
            states = self.model.blocks[-1].attn.get_attention_map().mean(1)[:, 0, :].reshape(b, 1, s)
        
        states = states * W_state
        
        if mae:
            return states[:, 1:, 1:].mean(axis=1)
        elif dino:
            return (states[:, 1:, 1:].mean(axis=1) + states[:, 0, 1:])
        else:
            return states[:, 0, 1:]
    
    
    
    #Mult
    # def generate_mult(self,input, index=None):
    #     output = self.model(input.to(self.model.device), register_hook=True)
    #     if index == None:
    #         index = np.argmax(output.cpu().data.numpy(), axis=-1)
        
    #     one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
    #     one_hot[0][index] = 1
    #     one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    #     one_hot = torch.sum(one_hot.to(self.model.device) * output)

    #     self.model.zero_grad()
    #     one_hot.backward(retain_graph=True)

    #     num_heads = self.model.blocks[-1].attn.num_heads
    #     num_patches = self.model.patch_embed.num_patches + 1
    #     ans = torch.zeros_like(self.model.blocks[-1].attn.get_attention_map().detach())

    #     H = torch.zeros(input.shape[0],len(self.model.blocks), num_heads, num_patches, num_patches)
    #     A = torch.zeros(input.shape[0],len(self.model.blocks), num_heads, num_patches, num_patches)
    #     for i,block in enumerate(self.model.blocks):
    #         A[:,i] = block.attn.get_attention_map().detach()
    #         H[:,i] = block.attn.get_attn_gradients().detach()

        
    #     ans1 = self.generate_gradsamplusplusv1wrong(input, index=index)
        
    #     ans2 = A.sum(dim=[1,2]) * H[:,-1].mean(1).clamp(min=0)

    #     ans2 = ans2[:,0,1:]

    #     return ans1*ans2


    # def generate_gradsamplusplusv1wrong(self, input, index=None):
    #     output = self.model(input.to(self.model.device), register_hook=True)
        
    #     if index == None:
    #         index = np.argmax(output.cpu().data.numpy())

    #     one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
    #     one_hot[0][index] = 1
    #     one_hot = torch.from_numpy(one_hot).requires_grad_(True).to(self.model.device)
    #     one_hot = torch.sum(one_hot * output)

    #     self.model.zero_grad()
    #     one_hot.backward(retain_graph=True)

    #     num_heads = self.model.blocks[-1].attn.num_heads
    #     num_patches = self.model.patch_embed.num_patches + 1
    #     H = torch.zeros(input.shape[0],len(self.model.blocks), num_heads, num_patches, num_patches)
    #     A = torch.zeros(input.shape[0],len(self.model.blocks), num_heads, num_patches, num_patches)
    #     for i,block in enumerate(self.model.blocks):
    #         A[:,i] = block.attn.get_attention_map().detach()
    #         H[:,i] = block.attn.get_attn_gradients().detach()

    #     # print(A)
    #     grads_power_2 = A**2
    #     sum_activations = torch.sum(A, dim=(-1,-2))
    #     res = einsum('b i j, b i j h w -> b i j h w',sum_activations, grads_power_2)
    #     aij = grads_power_2 / (2*grads_power_2  + res)
    #     w = aij * F.relu(H[:,-1])
    #     w = w.mean(dim=[1,2])
        
    #     return w[:,0,1:]