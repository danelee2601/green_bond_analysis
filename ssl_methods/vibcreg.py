import torch
from torch import nn
from torch.nn import Parameter
from torch import relu
import torch.nn.functional as F
import numpy as np

from torch import Tensor


class iterative_normalization_py(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        X, running_mean, running_wmat, nc, ctx.T, eps, momentum, training = args

        # change NxCxHxW to (G x D) x(NxHxW), i.e., g*d*m
        ctx.g = X.size(1) // nc
        x = X.transpose(0, 1).contiguous().view(ctx.g, nc, -1)
        _, d, m = x.size()
        saved = []
        if training:
            # calculate centered activation by subtracted mini-batch mean
            mean = x.mean(-1, keepdim=True)
            xc = x - mean
            saved.append(xc)
            # calculate covariance matrix
            P = [None] * (ctx.T + 1)
            P[0] = torch.eye(d).to(X).expand(ctx.g, d, d)
            Sigma = torch.baddbmm(beta=eps,
                                  input=P[0],
                                  alpha=1. / m,
                                  batch1=xc,
                                  batch2=xc.transpose(1, 2))
            # reciprocal of trace of Sigma: shape [g, 1, 1]
            rTr = (Sigma * P[0]).sum((1, 2), keepdim=True).reciprocal_()
            saved.append(rTr)
            Sigma_N = Sigma * rTr
            saved.append(Sigma_N)
            for k in range(ctx.T):
                P[k + 1] = torch.baddbmm(beta=1.5,
                                         input=P[k],
                                         alpha=-0.5,
                                         batch1=torch.matrix_power(P[k], 3),
                                         batch2=Sigma_N)
            saved.extend(P)
            wm = P[ctx.T].mul_(rTr.sqrt())  # whiten matrix: the matrix inverse of Sigma, i.e., Sigma^{-1/2}

            running_mean.copy_(momentum * mean + (1. - momentum) * running_mean)
            running_wmat.copy_(momentum * wm + (1. - momentum) * running_wmat)
        else:
            xc = x - running_mean
            wm = running_wmat
        xn = wm.matmul(xc)
        Xn = xn.view(X.size(1), X.size(0), *X.size()[2:]).transpose(0, 1).contiguous()
        ctx.save_for_backward(*saved)
        return Xn

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad, = grad_outputs
        saved = ctx.saved_tensors
        if len(saved) == 0:
            return None, None, None, None, None, None, None, None

        xc = saved[0]  # centered input
        rTr = saved[1]  # trace of Sigma
        sn = saved[2].transpose(-2, -1)  # normalized Sigma
        P = saved[3:]  # middle result matrix,
        g, d, m = xc.size()

        g_ = grad.transpose(0, 1).contiguous().view_as(xc)
        g_wm = g_.matmul(xc.transpose(-2, -1))
        g_P = g_wm * rTr.sqrt()
        wm = P[ctx.T]
        g_sn = 0
        for k in range(ctx.T, 1, -1):
            P[k - 1].transpose_(-2, -1)
            P2 = P[k - 1].matmul(P[k - 1])
            g_sn += P2.matmul(P[k - 1]).matmul(g_P)
            g_tmp = g_P.matmul(sn)
            g_P.baddbmm_(beta=1.5, alpha=-0.5, batch1=g_tmp, batch2=P2)
            g_P.baddbmm_(beta=1, alpha=-0.5, batch1=P2, batch2=g_tmp)
            g_P.baddbmm_(beta=1, alpha=-0.5, batch1=P[k - 1].matmul(g_tmp), batch2=P[k - 1])
        g_sn += g_P
        # g_sn = g_sn * rTr.sqrt()
        g_tr = ((-sn.matmul(g_sn) + g_wm.transpose(-2, -1).matmul(wm)) * P[0]).sum((1, 2), keepdim=True) * P[0]
        g_sigma = (g_sn + g_sn.transpose(-2, -1) + 2. * g_tr) * (-0.5 / m * rTr)
        # g_sigma = g_sigma + g_sigma.transpose(-2, -1)
        g_x = torch.baddbmm(wm.matmul(g_ - g_.mean(-1, keepdim=True)), g_sigma, xc)
        grad_input = g_x.view(grad.size(1), grad.size(0), *grad.size()[2:]).transpose(0, 1).contiguous()
        return grad_input, None, None, None, None, None, None, None


class IterNorm(torch.nn.Module):
    def __init__(self, num_features, num_groups=64, num_channels=None, T=5, dim=2, eps=1e-5, momentum=0.1, affine=True,
                 *args, **kwargs):
        super(IterNorm, self).__init__()
        # assert dim == 4, 'IterNorm is not support 2D'
        self.T = T
        self.eps = eps
        self.momentum = momentum
        self.num_features = num_features
        self.affine = affine
        self.dim = dim
        if num_channels is None:
            num_channels = (num_features - 1) // num_groups + 1
        num_groups = num_features // num_channels
        while num_features % num_channels != 0:
            num_channels //= 2
            num_groups = num_features // num_channels
        assert num_groups > 0 and num_features % num_groups == 0, "num features={}, num groups={}".format(num_features,
                                                                                                          num_groups)
        self.num_groups = num_groups
        self.num_channels = num_channels
        shape = [1] * dim
        shape[1] = self.num_features
        if self.affine:
            self.weight = Parameter(torch.Tensor(*shape))
            self.bias = Parameter(torch.Tensor(*shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.register_buffer('running_mean', torch.zeros(num_groups, num_channels, 1))
        # running whiten matrix
        self.register_buffer('running_wm',
                             torch.eye(num_channels).expand(num_groups, num_channels, num_channels).clone())
        self.reset_parameters()

    def reset_parameters(self):
        # self.reset_running_stats()
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, X: torch.Tensor):
        X_hat = iterative_normalization_py.apply(X, self.running_mean, self.running_wm, self.num_channels, self.T,
                                                 self.eps, self.momentum, self.training)
        # affine
        if self.affine:
            return X_hat * self.weight + self.bias
        else:
            return X_hat

    def extra_repr(self):
        return '{num_features}, num_channels={num_channels}, T={T}, eps={eps}, ' \
               'momentum={momentum}, affine={affine}'.format(**self.__dict__)


class Projector(nn.Module):
    def __init__(self, out_size_enc: int, proj_hid: int, proj_out: int):
        super().__init__()

        # define layers
        self.linear1 = nn.Linear(out_size_enc, proj_hid)
        self.nl1 = nn.BatchNorm1d(proj_hid)
        self.linear2 = nn.Linear(proj_hid, proj_hid)
        self.nl2 = nn.BatchNorm1d(proj_hid)
        self.linear3 = nn.Linear(proj_hid, proj_out)
        self.nl3 = IterNorm(proj_out, num_groups=64, T=5, dim=2, affine=True)

    def forward(self, x: Tensor) -> Tensor:
        out = relu(self.nl1(self.linear1(x)))
        out = relu(self.nl2(self.linear2(out)))
        out = self.linear3(out)
        out = self.nl3(out)  # [B, F]
        return out


def cos_sim_loss(z1: Tensor, z2: Tensor) -> Tensor:
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    return 1 - (z1 * z2).sum(dim=1).mean()


def vibcreg_invariance_loss(z1: Tensor, z2: Tensor, loss_type_vibcreg: str) -> Tensor:
    sim_loss = 0.
    if loss_type_vibcreg == 'mse':
        sim_loss = F.mse_loss(z1, z2)
    elif loss_type_vibcreg == 'huber':
        sim_loss = F.huber_loss(z1, z2)
    elif loss_type_vibcreg == 'cos_sim':
        sim_loss = cos_sim_loss(z1, z2)
    elif loss_type_vibcreg == 'hybrid':
        sim_loss = 0.5 * F.mse_loss(z1, z2) + 0.5 * cos_sim_loss(z1, z2)
    return sim_loss


def vibcreg_var_loss(z1: Tensor, z2: Tensor) -> Tensor:
    std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4)
    std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4)
    var_loss = torch.mean(relu(1. - std_z1)) + torch.mean(relu(1. - std_z2))
    return var_loss


def fxf_corr_mat1(z: Tensor) -> Tensor:
    norm_z = (z - z.mean(dim=0))
    norm_z = F.normalize(norm_z, p=2, dim=0)  # (B x D); l2-norm
    corr_mat_z = torch.mm(norm_z.T, norm_z)  # (D x D)
    return corr_mat_z


def vibcreg_cov_loss(z1: Tensor, z2: Tensor) -> Tensor:
    corr_mat_z1 = fxf_corr_mat1(z1)
    corr_mat_z2 = fxf_corr_mat1(z2)

    ind = np.diag_indices(corr_mat_z1.shape[0])
    corr_mat_z1[ind[0], ind[1]] = torch.zeros(corr_mat_z1.shape[0]).to(z1.get_device())
    corr_mat_z2[ind[0], ind[1]] = torch.zeros(corr_mat_z2.shape[0]).to(z2.get_device())
    cov_loss = (corr_mat_z1 ** 2).mean() + (corr_mat_z2 ** 2).mean()
    return cov_loss


def compute_vibcreg_loss(z1: Tensor, z2: Tensor, params: dict, loss_hist: dict):
    sim_loss = vibcreg_invariance_loss(z1, z2, 'mse')
    var_loss = vibcreg_var_loss(z1, z2)  # Feature-component Expressiveness (FcE) loss
    cov_loss = vibcreg_cov_loss(z1, z2)  # Feature Decorrelation (FD) loss)
    loss = params['lambda_'] * sim_loss + params['mu'] * var_loss + params['nu'] * cov_loss

    loss_hist[f'sim_loss'] = sim_loss
    loss_hist[f'var_loss'] = var_loss
    loss_hist[f'cov_loss'] = cov_loss

    return loss


class VIbCReg(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 out_size_enc: int,
                 proj_hid: int = 1024,
                 proj_out: int = 1024,
                 **kwargs):
        super().__init__()
        self.out_size_enc = out_size_enc

        self.encoder = encoder
        self.projector = Projector(out_size_enc, proj_hid, proj_out)

    def forward(self, input: Tensor) -> (Tensor, Tensor):
        y = self.encoder(input)
        z = self.projector(y)
        return y, z

    def loss_function(self,
                      z1: Tensor,
                      z2: Tensor,
                      params: dict = None,
                      loss_hist: dict = None) -> Tensor:
        vibcreg_loss = compute_vibcreg_loss(z1, z2, params, loss_hist)
        return vibcreg_loss
