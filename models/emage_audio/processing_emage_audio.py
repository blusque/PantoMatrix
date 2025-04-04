import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

def _copysign(a, b):
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)

def _sqrt_positive_part(x):
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def matrix_to_quaternion(matrix):
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError
    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    o0 = 0.5 * _sqrt_positive_part(1 + m00 + m11 + m22)
    x = 0.5 * _sqrt_positive_part(1 + m00 - m11 - m22)
    y = 0.5 * _sqrt_positive_part(1 - m00 + m11 - m22)
    z = 0.5 * _sqrt_positive_part(1 - m00 - m11 + m22)
    o1 = _copysign(x, matrix[..., 2, 1] - matrix[..., 1, 2])
    o2 = _copysign(y, matrix[..., 0, 2] - matrix[..., 2, 0])
    o3 = _copysign(z, matrix[..., 1, 0] - matrix[..., 0, 1])
    return torch.stack((o0, o1, o2, o3), -1)

def quaternion_to_axis_angle(quaternions):
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles]*angles[small_angles])/48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles

def matrix_to_axis_angle(matrix):
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def rotation_6d_to_axis_angle(rot6d):
    return matrix_to_axis_angle(rotation_6d_to_matrix(rot6d))

def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    return matrix[..., :2, :].clone().reshape(*matrix.size()[:-2], 6)

def axis_angle_to_quaternion(axis_angle):
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions

def quaternion_to_matrix(quaternions):
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def axis_angle_to_matrix(axis_angle):
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))

def axis_angle_to_rotation_6d(axis_angle):
    return matrix_to_rotation_6d(axis_angle_to_matrix(axis_angle))


def velocity2position(data_seq, dt, init_pos):
    res_trans = []
    for i in range(data_seq.shape[1]):
        if i == 0:
            res_trans.append(init_pos.unsqueeze(1))
        else:
            res = data_seq[:, i-1:i] * dt + res_trans[-1]
            res_trans.append(res)
    return torch.cat(res_trans, dim=1)


def recover_from_mask_ts(selected_motion: torch.Tensor, mask: list) -> torch.Tensor:
    device = selected_motion.device
    dtype = selected_motion.dtype
    mask_arr = torch.tensor(mask, dtype=torch.bool, device=device)
    j = len(mask_arr)
    sum_mask = mask_arr.sum().item()
    c_channels = selected_motion.shape[-1] // sum_mask
    new_shape = selected_motion.shape[:-1] + (sum_mask, c_channels)
    selected_motion = selected_motion.reshape(new_shape)
    out_shape = list(selected_motion.shape[:-2]) + [j, c_channels]
    recovered = torch.zeros(out_shape, dtype=dtype, device=device)
    recovered[..., mask_arr, :] = selected_motion
    final_shape = list(recovered.shape[:-2]) + [j*c_channels]
    recovered = recovered.reshape(final_shape)
    return recovered

# Code from https://zhuanlan.zhihu.com/p/68748778
class EMA():
    def __init__(self, model, decay, warmup_steps=0):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                device = param.data.device
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name].to(device)
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                device = param.data.device
                param.data = self.shadow[name].to(device)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# # 初始化
# ema = EMA(model, 0.999)
# ema.register()

# # 训练过程中，更新完参数后，同步update shadow weights
# def train():
#     optimizer.step()
#     ema.update()

# # eval前，apply shadow weights；eval之后，恢复原来模型的参数
# def evaluate():
#     ema.apply_shadow()
#     # evaluate
#     ema.restore()


class Quantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta):
        super().__init__()
        self.e_dim = e_dim
        self.n_e = n_e
        self.beta = beta
        self.ema_lambda = 0.99
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.requires_grad_(False)
        self.init_weights()
        self.register_buffer('N_t', torch.zeros(self.n_e, dtype=torch.float32, requires_grad=False))
        self.register_buffer('m_t', torch.zeros(self.n_e, self.e_dim, dtype=torch.float32, requires_grad=False))

    def forward(self, z):
        assert z.shape[-1] == self.e_dim
        z_flattened = z.contiguous().view(-1, self.e_dim)
        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2*torch.matmul(z_flattened, self.embedding.weight.t())
        min_encoding_indices = torch.argmin(d, dim=1)
        indices_selected, encoding_counts = torch.unique(min_encoding_indices, return_counts=True)
        with torch.no_grad():
            self.N_t[indices_selected] = self.N_t[indices_selected] * self.ema_lambda + encoding_counts * (1 - self.ema_lambda)
            E_x = []
            for idx in indices_selected:
                E_x.append(torch.sum((min_encoding_indices == idx).to(torch.float32).unsqueeze(-1) * z_flattened, dim=0)[None, ...])
            E_x = torch.cat(E_x, dim=0)
            self.m_t[indices_selected] = self.m_t[indices_selected] * self.ema_lambda + E_x * (1 - self.ema_lambda)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        # loss = torch.mean((z_q - z.detach())**2) + self.beta*torch.mean((z_q.detach() - z)**2)
        loss = self.beta*torch.mean((z_q.detach() - z)**2)  # the first term is replaced by ema + meanshift
        z_q = z + (z_q - z).detach()
        min_encodings = F.one_hot(min_encoding_indices, self.n_e).type(z.dtype)
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean*torch.log(e_mean+1e-10)))
        return loss, z_q, min_encoding_indices, perplexity
    
    @torch.no_grad()
    def update_codebook(self):
        self.embedding.weight.data = self.m_t / self.N_t.unsqueeze(-1)   # update the codebook

    def map2index(self, z):
        assert z.shape[-1] == self.e_dim
        z_flattened = z.contiguous().view(-1, self.e_dim)
        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2*torch.matmul(z_flattened, self.embedding.weight.t())
        min_encoding_indices = torch.argmin(d, dim=1)
        return min_encoding_indices.reshape(z.shape[0], -1)

    def get_codebook_entry(self, indices):
        index_flattened = indices.view(-1)
        z_q = self.embedding(index_flattened)
        z_q = z_q.view(indices.shape+(self.e_dim,)).contiguous()
        return z_q

    def init_weights(self, method='uniform'):
        if method == 'uniform':
            self.embedding.weight.data.uniform_(-1.0/self.n_e, 1.0/self.n_e)
        else:
            raise ValueError(f'Method {method} not implemented')


class ResidualQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta, layers=4, init_method='uniform'):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.layers = layers
        rvq = []
        for _ in range(layers):
            rvq.append(Quantizer(self.n_e, self.e_dim, self.beta))
        self.quantizers = nn.ModuleList(rvq)
        for quantizer in self.quantizers:
            quantizer.init_weights(init_method)


    def forward(self, z):
        assert z.shape[-1] == self.e_dim
        total_loss = []
        residual = z
        total_z_q = []
        total_indices = []
        total_perplexity = []
        loss, z_q, indices, perplexity = self.quantizers[0](residual)
        for quantizer in self.quantizers:
            loss, z_q, indices, perplexity = quantizer(residual)
            total_loss.append(loss)
            residual = residual.clone() - z_q
            total_z_q.append(z_q)
            total_indices.append(indices)
            total_perplexity.append(perplexity)
        z_q = torch.zeros_like(total_z_q[0])
        for r in total_z_q:
            z_q += r
        loss = torch.zeros_like(total_loss[0])
        for l in total_loss:
            loss += l
        return loss, z_q, total_indices, total_perplexity

    def map2index(self, z):
        assert z.shape[-1] == self.e_dim
        residual = z.clone()
        indices = []
        for quantizer in self.quantizers:
            indices.append(quantizer.map2index(residual))
        return indices

    def get_codebook_entry(self, indices_list):
        z_qs = []
        for i, indices in enumerate(indices_list):
            index_flattened = indices.view(-1)
            z_q = self.quantizers[i].embedding(index_flattened)
            z_q = z_q.view(indices.shape+(self.e_dim,)).contiguous()
            z_qs.append(z_q)
        return z_qs

    def cumsum(self, residuals):
        result = torch.zeros_like(residuals[0]).to(residuals[0].device)
        for residual in residuals:
            result += residual
        return result
    
    def update_codebook(self):
        for quantizer in self.quantizers:
            quantizer.update_codebook()
        

def init_weight(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class ResBlock(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(channel, channel, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(channel, channel, 3, 1, 1),
        )
    def forward(self, x):
        return self.model(x)+x

class VQEncoderV5(nn.Module):
    def __init__(self, args):
        super().__init__()
        n_down = args.vae_layer
        channels = [args.vae_length]*(n_down)
        input_size = args.vae_test_dim
        layers = [
            nn.Conv1d(input_size, channels[0], 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            ResBlock(channels[0]),
        ]
        for i in range(1, n_down):
            layers += [
                nn.Conv1d(channels[i-1], channels[i], 3, 1, 1),
                nn.LeakyReLU(0.2, True),
                ResBlock(channels[i]),
            ]
        self.main = nn.Sequential(*layers)
        self.main.apply(init_weight)
    def forward(self, inputs):
        inputs = inputs.permute(0,2,1)
        outputs = self.main(inputs).permute(0,2,1)
        return outputs

class VQEncoderV6(nn.Module):
    def __init__(self, args):
        super().__init__()
        n_down = args.vae_layer
        channels = [args.vae_length]*(n_down)
        input_size = args.vae_test_dim
        layers = [
            nn.Conv1d(input_size, channels[0], 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            ResBlock(channels[0]),
        ]
        for i in range(1, n_down):
            layers += [
                nn.Conv1d(channels[i-1], channels[i], 3, 1, 1),
                nn.LeakyReLU(0.2, True),
                ResBlock(channels[i]),
            ]
        self.main = nn.Sequential(*layers)
        self.main.apply(init_weight)
    def forward(self, inputs):
        inputs = inputs.permute(0,2,1)
        outputs = self.main(inputs).permute(0,2,1)
        return outputs

class VQDecoderV5(nn.Module):
    def __init__(self, args):
        super().__init__()
        n_up = args.vae_layer
        channels = [args.vae_length]*(n_up)+[args.vae_test_dim]
        input_size = args.vae_length
        n_resblk = 2
        if input_size == channels[0]:
            layers = []
        else:
            layers = [nn.Conv1d(input_size, channels[0], 3, 1, 1)]
        for i in range(n_resblk):
            layers += [ResBlock(channels[0])]
        for i in range(n_up):
            layers += [
                nn.Conv1d(channels[i], channels[i+1], 3, 1, 1),
                nn.LeakyReLU(0.2, True)
            ]
        layers += [nn.Conv1d(channels[-1], channels[-1], 3, 1, 1)]
        self.main = nn.Sequential(*layers)
        self.main.apply(init_weight)
    def forward(self, inputs):
        inputs = inputs.permute(0,2,1)
        outputs = self.main(inputs).permute(0,2,1)
        return outputs

class BasicBlock(nn.Module):
    """ based on timm: https://github.com/rwightman/pytorch-image-models """
    def __init__(self, inplanes, planes, ker_size, stride=1, downsample=None, dilation=1, first_dilation=None, act_layer=nn.LeakyReLU, norm_layer=nn.BatchNorm1d):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            inplanes, planes, kernel_size=ker_size, stride=stride, padding=first_dilation,
            dilation=dilation, bias=True)
        self.bn1 = norm_layer(planes)
        self.act1 = act_layer(inplace=True)
        self.conv2 = nn.Conv1d(
            planes, planes, kernel_size=ker_size, padding=ker_size//2, dilation=dilation, bias=True)
        self.bn2 = norm_layer(planes)
        self.act2 = act_layer(inplace=True)
        if downsample is not None:
            self.downsample = nn.Sequential(
                nn.Conv1d(inplanes, planes,  stride=stride, kernel_size=ker_size, padding=first_dilation, dilation=dilation, bias=True),
                norm_layer(planes), 
            )
        else: self.downsample=None

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act2(x)
        return x

class WavEncoder(nn.Module):
    def __init__(self, out_dim, audio_in=1):
        super().__init__() 
        self.out_dim = out_dim
        self.feat_extractor = nn.Sequential( 
                BasicBlock(audio_in, out_dim//4, 15, 5, first_dilation=1600, downsample=True),
                BasicBlock(out_dim//4, out_dim//4, 15, 6, first_dilation=0, downsample=True),
                BasicBlock(out_dim//4, out_dim//4, 15, 1, first_dilation=7, ),
                BasicBlock(out_dim//4, out_dim//2, 15, 6, first_dilation=0, downsample=True),
                BasicBlock(out_dim//2, out_dim//2, 15, 1, first_dilation=7),
                BasicBlock(out_dim//2, out_dim, 15, 3,  first_dilation=0,downsample=True),     
            )
    def forward(self, wav_data):
        if wav_data.dim() == 2:
            wav_data = wav_data.unsqueeze(1) 
        else:
            wav_data = wav_data.transpose(1, 2)
        out = self.feat_extractor(wav_data)
        return out.transpose(1, 2)

class MLP(nn.Module):
    def __init__(self, in_dim, middle_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, middle_dim)
        self.fc2 = nn.Linear(middle_dim, out_dim)
        self.act = nn.LeakyReLU(0.1, True)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=15, max_seq_len=60): 
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0)
        repeat_num = (max_seq_len//period)+1
        pe = pe.repeat(1, repeat_num, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)