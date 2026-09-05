"""
@author: Ziyan Chen
@version:42.9.22
@Date: 20250930_18:48

case1：droplet coalescence hydrophilic case (\theta_s = 50)
"""

import os
import time
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.autograd import Variable
from pydoe import lhs
import argparse
from datetime import datetime
import matplotlib
from collections import deque
import scipy.io
import soap
from pathlib import Path

matplotlib.use("Agg")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
seed = 1234
torch.set_default_dtype(torch.float)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

use_gpu = torch.cuda.is_available()
print('GPU:', use_gpu)

current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M")

ver_num = "42_9_22"  # 只改数字
version_name = "42_9_22"  # 文件名，可加其他后缀
folder_path = f"./{version_name}_{current_datetime}/"

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
os.chdir(folder_path)

# ------------------------ 配置保存路径 ------------------------
SAVE_DIR = Path("checkpoints")
SAVE_DIR.mkdir(exist_ok=True)

# ------------------------设置字体----------------------------
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'STIXGeneral', 'DejaVu Serif', 'Computer Modern Roman']
plt.rcParams['mathtext.fontset'] = 'stix'

parser = argparse.ArgumentParser()

parser.add_argument('--hid_layers', help='number of hidden layers', type=int, default=6)
parser.add_argument('--hid_neurons', help='number of neurons per layer', type=int, default=128)
parser.add_argument('--num_x', help='sample points of x direction', type=int, default=41)
parser.add_argument('--num_y', help='sample points of y direction', type=int, default=21)
parser.add_argument('--num_x_col', help='sample col points of x direction', type=int,
                    default=50)  # 从50*50组成的点里抽取符合条件的点(界面点)当固定点
parser.add_argument('--num_y_col', help='sample col points of y direction', type=int, default=50)
parser.add_argument('--M', help='number of movable collocation points', type=int, default=1000)  # 随机了1000个可移动的点
parser.add_argument('--M_b', help='number of movable boundary points', type=int, default=500)
parser.add_argument('--adam_iter', help='number of adam iteration', type=int, default=10000)  # 10000
parser.add_argument('--lbfgs_iter', help='number of lbfgs iteration', type=int, default=0)
parser.add_argument('--adam_lr', help='learning rate of adam', type=float, default=0.001)
parser.add_argument('--lbfgs_lr', help='learning rate of lbfgs', type=float, default=0.5)
parser.add_argument('--model_type', help='0:baseline  1:AM  2:AM_AW', type=int, default=2)
parser.add_argument('--AM_type', help='0:RAM  1:WAM 2:new_RAM 3:new_new_RAM 4:new_WAM', type=int, default=4)
parser.add_argument('--AM_K', help='parameter k of the probability density function', type=float,
                    default=1)  # the degree of concentration of sampling points; as k increases, the sampling points will be more concentrated in the high-gradient areas
parser.add_argument('--AM_count', help='number of rounds for adaptive collocation point movement', type=int, default=1)
parser.add_argument('--AW_lr', help='learning rate of adaptive weights', type=float, default=0.001)
parser.add_argument('--q', help='stage of IRK', type=int, default=20)
parser.add_argument('--move_type', help='0:RAM_Residual  1： RAM 2:WAM 3:norm', type=int, default=1)  # 2 3出bug,clone后好了
parser.add_argument('--Nx_scale', help='scaling for x direction', type=float, default=20)  # 1 / (epsilon^2 * L_d)
parser.add_argument('--Ny_scale', help='scaling for y direction', type=float, default=20)
parser.add_argument('--Nt_scale', help='scaling for t direction', type=float, default=1)
parser.add_argument('--epsilon', help='eps', type=float, default=0.02)
parser.add_argument('--L_d', help='L_d', type=float, default=1)
parser.add_argument('--t_final', help='t_final', type=float, default=10)
parser.add_argument('--delta_t', help='delta_t', type=float, default=0.1)
parser.add_argument('--threshold', help='红色区域', type=float, default=0.0)
parser.add_argument('--theta_s', help='theta', type=float, default=(5 * np.pi) / 18)  # 50度
parser.add_argument('--alpha_var', help='alpha', type=float, default=100)  # 1000000， 100
parser.add_argument('--debug_val', help='0 代表 False （不画图，但是效率高），1 代表 True（画图）', type=float,
                    default=0)  


def generate_layers(hid_neurons: int, hid_layer: int, q: int):
    nn_layers = [2] + [hid_neurons] * hid_layer + [2 * q + 2]  # in: x,y out:前面q+1个h(index:0到q)，后面q+1个mu(index:q+1到2q+1)
    return nn_layers


def random_fun(num):
    temp = torch.from_numpy(lb + (ub - lb) * lhs(2, num)).float()
    if use_gpu:
        temp = temp.cuda()
    return temp


def is_cuda(data):
    if use_gpu:
        data = data.cuda()  # 将张量（Tensor）从CPU内存移动到GPU内存
    return data


def np_tensor(data_var):
    # 将传入的 data_var（预期是一个NumPy数组）转换为一个PyTorch张量（Tensor），并确保这个张量的数据类型是浮点数（即 .float()）。
    return torch.from_numpy(data_var).float()


def np_to_tensor_to_cuda(var):
    # 先把参数从numpy变成tensor再将Tensor从CPU内存移动到GPU内存
    return is_cuda(np_tensor(var))


def hstack_data(var1, var2):
    var_hstack = np.hstack((var1.flatten(order='F')[:, None],
                            var2.flatten(order='F')[:, None]))
    return var_hstack


def get_bc_data(num_x, num_y, x, y):
    # bc data
    # 均匀采样
    index_b_x = np.expand_dims(np.linspace(0, x.shape[0] - 1, num_x), axis=1).astype(int)
    index_b_y = np.expand_dims(np.linspace(0, y.shape[0] - 1, num_y), axis=1).astype(int)

    index_b_x0 = 0
    index_b_x1 = len(x) - 1
    index_b_y0 = 0
    index_b_y1 = len(y) - 1

    index_b_x_bot_meshgrid, index_b_y_bot_meshgrid = np.meshgrid(index_b_x, index_b_y0)  # (x,-1,t)的index
    index_b_x_top_meshgrid, index_b_y_top_meshgrid = np.meshgrid(index_b_x, index_b_y1)  # (x,1,t)的index
    index_b_x_left_meshgrid, index_b_y_left_meshgrid = np.meshgrid(index_b_x0, index_b_y)  # (0,y,t)的index
    index_b_x_right_meshgrid, index_b_y_right_meshgrid = np.meshgrid(index_b_x1, index_b_y)  # (1,y, t)的index

    idx_xyt_b_bot_meshgrid_train = hstack_data(index_b_x_bot_meshgrid, index_b_y_bot_meshgrid)
    idx_xyt_b_top_meshgrid_train = hstack_data(index_b_x_top_meshgrid, index_b_y_top_meshgrid)
    idx_xyt_b_left_meshgrid_train = hstack_data(index_b_x_left_meshgrid, index_b_y_left_meshgrid)
    idx_xyt_b_right_meshgrid_train = hstack_data(index_b_x_right_meshgrid, index_b_y_right_meshgrid)
    # 上面都是index，下面才是对应的数值。
    # 如果坐标轴的原点在左上角，x轴往右，y轴往前，则这是上边那一条线。

    xyt_b_bot_meshgrid_train_np = np.column_stack((x[idx_xyt_b_bot_meshgrid_train[:, 0]],
                                                   y[idx_xyt_b_bot_meshgrid_train[:, 1]]))
    xyt_b_top_meshgrid_train_np = np.column_stack((x[idx_xyt_b_top_meshgrid_train[:, 0]],
                                                   y[idx_xyt_b_top_meshgrid_train[:, 1]]))
    xyt_b_left_meshgrid_train_np = np.column_stack((x[idx_xyt_b_left_meshgrid_train[:, 0]],
                                                    y[idx_xyt_b_left_meshgrid_train[:, 1]]
                                                    ))
    xyt_b_right_meshgrid_train_np = np.column_stack((x[idx_xyt_b_right_meshgrid_train[:, 0]],
                                                     y[idx_xyt_b_right_meshgrid_train[:, 1]]))

    xyt_b_bot_meshgrid_train = np_to_tensor_to_cuda(xyt_b_bot_meshgrid_train_np)
    xyt_b_top_meshgrid_train = np_to_tensor_to_cuda(xyt_b_top_meshgrid_train_np)
    xyt_b_left_meshgrid_train = np_to_tensor_to_cuda(xyt_b_left_meshgrid_train_np)
    xyt_b_right_meshgrid_train = np_to_tensor_to_cuda(xyt_b_right_meshgrid_train_np)

    # 不合并
    return xyt_b_bot_meshgrid_train, xyt_b_top_meshgrid_train, xyt_b_left_meshgrid_train, xyt_b_right_meshgrid_train


def get_test_data(x, y):
    X, Y = np.meshgrid(x, y)
    x_test_np = hstack_data(X, Y)

    x_test = is_cuda(torch.from_numpy(x_test_np).float())

    return x_test


def data_generate(x, y, num_x, num_y):
    x_test = get_test_data(x, y)

    (xyt_b_bot_meshgrid_train, xyt_b_top_meshgrid_train, xyt_b_left_meshgrid_train,
     xyt_b_right_meshgrid_train) = get_bc_data(num_x, num_y, x, y)

    return (x_test, xyt_b_bot_meshgrid_train, xyt_b_top_meshgrid_train, xyt_b_left_meshgrid_train,
            xyt_b_right_meshgrid_train)


def get_initial_interface(lb2, ub2, h_pred_b4_time_interval, x_test, time_block):
    # 如果是属于第一个时间块，就在given initial condition这里抽取选点
    if time_block == 0:
        x0 = np.linspace(lb2[0], ub2[0], num_x_col).reshape(1, -1)
        y0 = np.linspace(lb2[1], ub2[1], num_y_col).reshape(1, -1)

        xy_meshgrid = np.meshgrid(x0, y0)

        x0_train = xy_meshgrid[0]
        y0_train = xy_meshgrid[1]

        x_f = np.stack([x0_train.ravel(), y0_train.ravel()], axis=1)

        h00 = get_initial_h(x_f)  # given initial condition

        if isinstance(h00, torch.Tensor):
            h00 = h00.detach().cpu().numpy()
            # 设置容差，判断h_0不属于1或-1(AS的criterion)
        tol = 1e-1
        mask = (np.abs(h00 - 1) > tol) & (np.abs(h00 + 1) > tol)

        # 把 mask 展平为一维
        mask = mask.ravel()

        # 展平后再筛选
        x_flat = x0_train.ravel()
        y_flat = y0_train.ravel()

        selected_x = x_flat[mask]
        selected_y = y_flat[mask]

        # 或者合并成坐标对
        x_f_N = np.stack([selected_x, selected_y], axis=1)
        x_f_N = np_to_tensor_to_cuda(x_f_N)

    # 如果不是属于第一个时间块，就在上一个时间块预测出来的值那里抽取选点
    else:
        x_coords = x_test[:, 0:1].detach().cpu().numpy()
        y_coords = x_test[:, 1:2].detach().cpu().numpy()
        h00 = h_pred_b4_time_interval

        if isinstance(h00, torch.Tensor):
            h00 = h00.detach().cpu().numpy()
            # 设置容差，判断h_0不属于1或-1
        tol = 1e-1
        mask = (np.abs(h00 - 1) > tol) & (np.abs(h00 + 1) > tol)

        # 把 mask 展平为一维
        mask = mask.ravel()

        # 展平后再筛选
        x_flat = x_coords.ravel()
        y_flat = y_coords.ravel()

        selected_x = x_flat[mask]
        selected_y = y_flat[mask]

        # 或者合并成坐标对
        x_f_N = np.stack([selected_x, selected_y], axis=1)
        x_f_N = np_to_tensor_to_cuda(x_f_N)

    return x_f_N


class Net(nn.Module):
    def __init__(self, layers, q):
        super(Net, self).__init__()
        self.q = q
        self.layers = layers
        self.iter = 0
        self.activation = nn.Tanh()
        self.linear = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        for i in range(len(layers) - 1):
            nn.init.xavier_normal_(self.linear[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linear[i].bias.data)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)

        x_part = x[:, 0]
        y_part = x[:, 1]

        x_part = Nx_scale * torch.cos(x_part / Nx_scale)
        x = torch.vstack([x_part, y_part]).T

        a = self.activation(self.linear[0](x))
        for i in range(1, len(self.layers) - 2):
            z = self.linear[i](a)
            a = self.activation(z)

        # 输出层
        output = self.linear[-1](a)

        # 分割输出
        h_part = output[:, :self.q + 1]
        mu_part = output[:, self.q + 1:]

        # 对 h_part 应用 tanh
        h_part = torch.tanh(h_part) / 0.90

        # 合并输出
        final_output = torch.cat([h_part, mu_part], dim=1)

        return final_output


class Model:
    def __init__(self, net,
                 xyt_b_bot_meshgrid_train, xyt_b_top_meshgrid_train,
                 xyt_b_left_meshgrid_train, xyt_b_right_meshgrid_train,
                 x_f_loss_fun_h, x_f_loss_fun_mu,
                 x_test, dt, tStart, tFinal, q,
                 x_f_N, x_f_M, current_t_num, debug_val):
        torch.set_default_dtype(torch.float32)
        self.current_t_num = current_t_num
        self.AM_count_num = 0

        self.x_b_s_h = None
        self.x_f_s_h = None
        self.x_b_s_mu = None
        self.x_f_s_mu = None

        self.s_collect = []

        self.net = net

        # 边界点
        self.xyt_b_bot_meshgrid_train = xyt_b_bot_meshgrid_train
        self.xyt_b_top_meshgrid_train = xyt_b_top_meshgrid_train
        self.xyt_b_left_meshgrid_train = xyt_b_left_meshgrid_train
        self.xyt_b_right_meshgrid_train = xyt_b_right_meshgrid_train

        # 内部点
        self.x_f_N = x_f_N
        self.x_f_M = x_f_M
        self.x_f = None

        # residual
        self.x_f_loss_fun_h = x_f_loss_fun_h
        self.x_f_loss_fun_mu = x_f_loss_fun_mu

        # 测试集
        self.x_test = x_test

        # 储存loss
        self.x_b_loss_collect = []
        self.x_f_loss_collect = []

        # 储存自适应权重
        self.x_f_weight_collect = []  # 记录内部点权重 [iter, w_f_h, w_f_mu, w_f_total]
        self.x_b_weight_collect = []  # 记录边界点权重 [iter, w_b_h, w_b_mu, w_b_total]

        self.dt = dt
        self.tStart = tStart
        self.tFinal = tFinal
        self.q = q

        # 0 代表 False （不画图，但是效率高），1 代表 True（画图）
        self.debug_val = debug_val

        # V100(8GPUs)
        # #tmp = np.float64(np.loadtxt('/home/zychen/com_b/IRK_weights/Butcher_IRK%d.txt' % q, ndmin=2))

        # A100
        # tmp = np.float64(np.loadtxt('/home/zychen/complex_b_9_3_33/IRK_weights/Butcher_IRK%d.txt' % q, ndmin=2))

        # 4090
        tmp = np.float64(np.loadtxt('/media/czy/Disk2/DiskE/AMAW_20240219/Dis离散模型/IRK_weights/Butcher_IRK%d.txt' % q,ndmin=2))


        self.IRKWeights = np_to_tensor_to_cuda(np.reshape(tmp[0: q ** 2 + q], (q + 1, q)))
        self.IRKTimes = np_to_tensor_to_cuda(tmp[q ** 2 + q:]) * (self.tFinal - self.tStart) + self.tStart
        self.IRKTimesExtended = np_to_tensor_to_cuda((np.append(tmp[q ** 2 + q:], 1).reshape(q + 1, 1))) * (
                    self.tFinal - self.tStart) + self.tStart

    def train_U(self, x):
        return self.net(x)

    def predict_H(self, x):
        return self.train_U(x)[:, self.q:(self.q + 1)]  # h

    def likelihood_loss(self, loss_f_h, loss_b_h, loss_f_mu, loss_b_mu):
        loss = (torch.exp(-self.x_f_s_h) * loss_f_h.detach() + self.x_f_s_h
                + torch.exp(-self.x_b_s_h) * loss_b_h.detach() + self.x_b_s_h
                + torch.exp(-self.x_f_s_mu) * loss_f_mu.detach() + self.x_f_s_mu
                + torch.exp(-self.x_b_s_mu) * loss_b_mu.detach() + self.x_b_s_mu)

        return loss

    def true_loss(self, loss_f_h, loss_b_h, loss_f_mu, loss_b_mu):
        true_loss_value = (torch.exp(-self.x_f_s_h.detach()) * loss_f_h
                           + torch.exp(-self.x_b_s_h.detach()) * loss_b_h
                           + torch.exp(-self.x_f_s_mu.detach()) * loss_f_mu
                           + torch.exp(-self.x_b_s_mu.detach()) * loss_b_mu)

        return true_loss_value

    def ub_x_h(self, boundary_data):
        if not boundary_data.requires_grad:
            boundary_data = Variable(boundary_data, requires_grad=True)
        ub_pred = self.train_U(boundary_data)[:, 0:self.q + 1]  # h^{n+c1},...,h 包含h
        # fwd_gradients_1
        zx = torch.ones_like(ub_pred, requires_grad=True)
        gy = torch.autograd.grad(ub_pred, boundary_data, grad_outputs=zx, retain_graph=True, create_graph=True)[0][
            :, 0].unsqueeze(-1)
        ub_bot_pred_x = \
        torch.autograd.grad(gy, zx, grad_outputs=is_cuda(torch.ones(gy.shape)), retain_graph=True, create_graph=True)[0]
        return ub_bot_pred_x

    def ub_y_h(self, boundary_data):
        if not boundary_data.requires_grad:
            boundary_data = Variable(boundary_data, requires_grad=True)
        ub_pred = self.train_U(boundary_data)[:, 0:self.q + 1]  # h^{n+c1},...,h 包含h
        # fwd_gradients_1
        zy = is_cuda((torch.ones(ub_pred.shape, dtype=torch.float64)).requires_grad_(True))
        gy = torch.autograd.grad(ub_pred, boundary_data, grad_outputs=zy, retain_graph=True, create_graph=True)[0][
            :, 1].unsqueeze(-1)
        ub_bot_pred_y = \
        torch.autograd.grad(gy, zy, grad_outputs=is_cuda(torch.ones(gy.shape)), retain_graph=True, create_graph=True)[0]
        return ub_bot_pred_y

    def ub_x_mu(self, boundary_data):
        if not boundary_data.requires_grad:
            boundary_data = Variable(boundary_data, requires_grad=True)
        ub_pred = self.train_U(boundary_data)[:, (self.q + 1):(2 * self.q + 2)]  # mu^{n+c1},...,mu 包含mu
        # fwd_gradients_1
        zx = torch.ones_like(ub_pred, requires_grad=True)
        gy = torch.autograd.grad(ub_pred, boundary_data, grad_outputs=zx, retain_graph=True, create_graph=True)[0][
            :, 0].unsqueeze(-1)
        ub_bot_pred_x = \
        torch.autograd.grad(gy, zx, grad_outputs=is_cuda(torch.ones(gy.shape)), retain_graph=True, create_graph=True)[0]
        return ub_bot_pred_x

    def ub_y_mu(self, boundary_data):
        if not boundary_data.requires_grad:
            boundary_data = Variable(boundary_data, requires_grad=True)
        ub_pred = self.train_U(boundary_data)[:, (self.q + 1):(2 * self.q + 2)]  # mu^{n+c1},...,mu 包含mu
        # fwd_gradients_1
        zy = is_cuda((torch.ones(ub_pred.shape, dtype=torch.float64)).requires_grad_(True))
        gy = torch.autograd.grad(ub_pred, boundary_data, grad_outputs=zy, retain_graph=True, create_graph=True)[0][
            :, 1].unsqueeze(-1)
        ub_bot_pred_y = \
        torch.autograd.grad(gy, zy, grad_outputs=is_cuda(torch.ones(gy.shape)), retain_graph=True, create_graph=True)[0]
        return ub_bot_pred_y

    def get_b_pred(self, boundary_data):
        if not boundary_data.requires_grad:
            boundary_data = Variable(boundary_data, requires_grad=True)
        ub_pred = self.train_U(boundary_data)  # h^{n+c1},...,h, mu^{n+c1},...,mu

        return ub_pred

    # IRKtimes连续时间步之间的差分
    def compute_phi_t_backward(self, phi, h_0_train):
        t_vec = self.IRKTimesExtended.squeeze()  # (51,)
        t_start = self.tStart

        # Ensure all inputs are on the same device and dtype
        device = phi.device
        dtype = phi.dtype

        t_start = torch.as_tensor(t_start, dtype=dtype, device=device)
        t_vec = torch.as_tensor(t_vec, dtype=dtype, device=device).squeeze()
        h_0_train = torch.as_tensor(h_0_train, dtype=dtype, device=device).squeeze()

        num_points, num_times = phi.shape

        # Step 1: Compute delta_t for each time step
        delta_t = torch.zeros(num_times, dtype=dtype, device=device)
        delta_t[0] = t_vec[0] - t_start  # from initial time to first IRK point
        if num_times > 1:
            delta_t[1:] = t_vec[1:] - t_vec[:-1]  # consecutive differences

        # Step 2: Initialize phi_t
        phi_t = torch.zeros_like(phi)  # (num_points, num_times)

        # Step 3: Backward difference
        # At first time step: use h_0_train
        phi_t[:, 0] = (phi[:, 0] - h_0_train) / delta_t[0]

        # At subsequent steps: phi[:,n] - phi[:,n-1]
        if num_times > 1:
            phi_t[:, 1:] = (phi[:, 1:] - phi[:, :-1]) / delta_t[1:].unsqueeze(0)  # broadcasting

        return phi_t  # shape (num_points, num_times)

    def partial_derivative_phi(self, boundary_points, phi, epsilon, theta_s):
        # 这里的phi指的是h^{n+c1},...,h 包含h
        # 给出的
        epsilon = torch.tensor(epsilon)
        theta_s = torch.tensor(theta_s)
        constant_factor = torch.tensor((np.sqrt(2) * np.pi) / 6)
        cos_1 = torch.cos(theta_s)
        cos_2 = torch.cos(torch.pi / 2 * phi)

        if self.current_t_num == 0:
            # 离散型要增加对应的h，这里用初值是因为没有准确解的公式
            h_0_train = get_initial_h(boundary_points)  # 这里只能是用0开始预测才能用，如果是0.1或者其他就不可以。要改
        else:
            h_0_train = models[self.current_t_num - 1].predict_H(boundary_points)
            h_0_train = h_0_train[:, -1]

        phi_t = self.compute_phi_t_backward(
            phi=phi,
            h_0_train=h_0_train
        )

        # 中间项：要保存的部分
        middle_term = (1 / alpha_var) * Nt_scale * phi_t

        # 计算偏导数的值（这里其实只是关于phi的函数）
        derivative_value = (constant_factor * cos_1 * cos_2 - middle_term) / epsilon

        if self.net.iter % 1000 == 0:
            self.save_phi_t(middle_term)

        return derivative_value

    def save_phi_t(self, middle_term):
        filename = "middle_term_phi_t_history.txt"

        # 获取当前状态信息
        iter_step = self.net.iter
        current_t_num = self.current_t_num
        am_count_num = self.AM_count_num

        # 将 middle_term 转为 numpy 并展平
        data_flat = middle_term.cpu().detach().numpy().flatten()  # shape: (2101,)
        row_to_save = np.hstack([iter_step, data_flat])  # 第一个为 iter

        # 计算统计量
        max_val = np.max(data_flat)
        min_val = np.min(data_flat)
        mean_val = np.mean(data_flat)
        mse_val = np.mean(data_flat ** 2)  # MSE = mean(x_i^2)
        std_val = np.std(data_flat)

        # 构造统计信息字符串
        stats_line = (
            f"# Stats: max={max_val:.6e}, min={min_val:.6e}, "
            f"mean={mean_val:.6e}, MSE={mse_val:.6e}, std={std_val:.6e}"
        )

        # 注释头
        header = f"# TimeBlock={current_t_num}, AM_count={am_count_num}, Iter={iter_step}"

        # 写入文件（追加模式）
        with open(filename, 'a') as f:
            f.write("\n\n")  # 两个空行分隔
            f.write(header + "\n")  # 第一行：时间块与迭代信息
            f.write(stats_line + "\n")  # 第二行：统计信息
            # 第三行：数据行
            np.savetxt(f, [row_to_save], fmt=['%d'] + ['%.6e'] * len(data_flat), delimiter=' ')

    def loss_boundary_h(self):
        xyt_b_left_x = self.ub_x_h(self.xyt_b_left_meshgrid_train)  # 左边
        xyt_b_right_x = self.ub_x_h(self.xyt_b_right_meshgrid_train)  # 右边 u_x(t,1,y)

        xyt_b_bot_y = self.ub_y_h(self.xyt_b_bot_meshgrid_train)  # 下边那条线, u_y(t, x, 0)
        xyt_b_top_y = self.ub_y_h(self.xyt_b_top_meshgrid_train)  # 上边

        xyt_b_bot = self.get_b_pred(self.xyt_b_bot_meshgrid_train)[
            :, 0:self.q + 1]  # h^{n+c1},...,h 包含h  # 下边 维度是（41，51) 即（41个点，q是51)
        xyt_b_top = self.get_b_pred(self.xyt_b_top_meshgrid_train)[:, 0:self.q + 1]  # h^{n+c1},...,h 包含h  # 上边
        xyt_b_left = self.get_b_pred(self.xyt_b_left_meshgrid_train)[:, 0:self.q + 1]  # h^{n+c1},...,h 包含h  # 左边
        xyt_b_right = self.get_b_pred(self.xyt_b_right_meshgrid_train)[:, 0:self.q + 1]  # h^{n+c1},...,h 包含h  # 右边

        h_bot_y = -1 * self.partial_derivative_phi(self.xyt_b_bot_meshgrid_train, xyt_b_bot, epsilon, theta_s) * (
                    1 / Ny_scale)  # 下边
        h_top_y = self.partial_derivative_phi(self.xyt_b_top_meshgrid_train, xyt_b_top, epsilon, theta_s) * (
                    1 / Ny_scale)  # 上边
        h_left_x = -1 * self.partial_derivative_phi(self.xyt_b_left_meshgrid_train, xyt_b_left, epsilon, theta_s) * (
                    1 / Nx_scale)  # 左边
        h_right_x = self.partial_derivative_phi(self.xyt_b_right_meshgrid_train, xyt_b_right, epsilon, theta_s) * (
                    1 / Nx_scale)  # 右边

        loss_b_h = (torch.mean((xyt_b_left_x - h_left_x) ** 2)
                    + torch.mean((xyt_b_right_x - h_right_x) ** 2)
                    + torch.mean((xyt_b_bot_y - h_bot_y) ** 2)
                    + torch.mean((xyt_b_top_y - h_top_y) ** 2))

        return loss_b_h

    def loss_boundary_mu(self):
        xyt_b_left_x = self.ub_x_mu(self.xyt_b_left_meshgrid_train)  # 左边
        xyt_b_right_x = self.ub_x_mu(self.xyt_b_right_meshgrid_train)  # 右边 u_x(t,1,y)

        xyt_b_bot_y = self.ub_y_mu(self.xyt_b_bot_meshgrid_train)
        xyt_b_top_y = self.ub_y_mu(self.xyt_b_top_meshgrid_train)

        loss_b_mu = (torch.mean((xyt_b_left_x - 0) ** 2)
                     + torch.mean((xyt_b_right_x - 0) ** 2)
                     + torch.mean((xyt_b_bot_y - 0) ** 2)
                     + torch.mean((xyt_b_top_y - 0) ** 2))

        return loss_b_mu

    # (debug_val==1才画)
    def draw_loss_f(self, current_t_num, x_f, U00, U11, AM_count):
        x_vals = (x_f[:, 0] / Nx_scale).detach().cpu().numpy()
        y_vals = (x_f[:, 1] / Ny_scale).detach().cpu().numpy()
        u00 = U00.detach().cpu().numpy().flatten()
        u11 = U11.detach().cpu().numpy().flatten()
        error = np.abs(u11 - u00)

        data_list = [u00, u11, error]
        titles = [r'$u_{f\_pred}$', r'$u_{0\_train}$', r'$Error$']

        fig, axes = plt.subplots(3, 1, figsize=(10, 14))

        for i, ax in enumerate(axes):
            sc = ax.scatter(x_vals, y_vals, c=data_list[i], cmap='jet', s=10)

            fig.colorbar(sc, ax=ax, aspect=15)

            ax.set_xlabel(r'$x$')
            ax.set_ylabel(r'$y$')
            ax.set_title(titles[i])

            ax.set_aspect('equal', adjustable='box')

            ax.set_xlim(x_vals.min(), x_vals.max())
            ax.set_ylim(y_vals.min(), y_vals.max())

            ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='black', alpha=0.5)

        plt.tight_layout()

        subfolder_path = os.path.join("./debug/", f"t_{current_t_num}", f"count_{AM_count}")
        os.makedirs(subfolder_path, exist_ok=True)

        filename = f"Sol_u_f_pred_t_{current_t_num}_count_{AM_count}_iter_{self.net.iter}.png"
        filepath = os.path.join(subfolder_path, filename)

        fig.savefig(filepath, dpi=500, bbox_inches='tight')
        plt.close(fig)

    def epoch_loss(self):
        x_f = torch.cat((self.x_f_N, self.x_f_M), dim=0)
        if self.current_t_num == 0:
            # 离散型要增加对应的h，这里用初值是因为没有准确解的公式
            h_0_train = get_initial_h(x_f)  # 这里只能是用0开始预测才能用，如果是0.1或者其他就不可以。要改！
        else:
            h_0_train = models[self.current_t_num - 1].predict_H(x_f)

        u_f_pred_h = self.x_f_loss_fun_h(x_f, self.train_U)
        u_f_pred_mu = self.x_f_loss_fun_mu(x_f, self.train_U)

        # debug的时候用, 用来检查内部点
        if self.net.iter % 1000 == 0 and debug_val == 1:
            U00 = u_f_pred_h[:, -1]  # 这里是只看输出的最后一个，即h^{n+1}
            U11 = h_0_train[:, -1]  # 这里是只看输出的最后一个，即h^{n+1},固定
            self.draw_loss_f(self.current_t_num, x_f, U00, U11, self.AM_count_num)
            draw_exact_iter_count(self.net.iter, self.AM_count_num, self.current_t_num)

        loss_f_h = torch.mean((u_f_pred_h - h_0_train) ** 2)
        loss_f_mu = torch.mean(u_f_pred_mu ** 2)

        loss_b_h = self.loss_boundary_h()
        loss_b_mu = self.loss_boundary_mu()

        loss_f = loss_f_h + loss_f_mu
        loss_b = loss_b_h + loss_b_mu

        # ========== 收集loss用来画图用 ==========
        self.x_f_loss_collect.append([self.net.iter, loss_f.item(), loss_f_h.item(), loss_f_mu.item()])
        self.x_b_loss_collect.append([self.net.iter, loss_b.item(), loss_b_h.item(), loss_b_mu.item()])

        # ========== 收集自适应权重 ==========
        w_f_h = torch.exp(-self.x_f_s_h.detach()).item()
        w_f_mu = torch.exp(-self.x_f_s_mu.detach()).item()
        w_b_h = torch.exp(-self.x_b_s_h.detach()).item()
        w_b_mu = torch.exp(-self.x_b_s_mu.detach()).item()
        self.x_f_weight_collect.append([self.net.iter, w_f_h, w_f_mu, w_f_h + w_f_mu])
        self.x_b_weight_collect.append([self.net.iter, w_b_h, w_b_mu, w_b_h + w_b_mu])
        # =========================================

        return loss_f_h, loss_f_mu, loss_b_h, loss_b_mu

    def run_mcl_pinns(self, current_t_num):
        self.x_f_s_h = nn.Parameter(self.x_f_s_h, requires_grad=True)
        self.x_b_s_h = nn.Parameter(self.x_b_s_h, requires_grad=True)

        self.x_f_s_mu = nn.Parameter(self.x_f_s_mu, requires_grad=True)
        self.x_b_s_mu = nn.Parameter(self.x_b_s_mu, requires_grad=True)

        for move_count in range(AM_count):
            if debug_val == 1:  # 画出每次AM前的内部点
                draw_exact_points_count_b4_AM(move_count, current_t_num, self.x_f_N, self.x_f_M)
                # 这里随着AS选出来的界面点self.x_f_N是绿色的点, 固定的内部点self.x_f_M是黑色

            self.AM_count_num = move_count
            # 初始化参数
            last_losses = deque(maxlen=100)  # 自动维护最近100个loss
            loss_percentage_threshold = 0.995
            continue_training = True
            early_stop_window = 100
            num_epochs = adam_iter  # 初始设定的最大迭代次数
            max_epochs = num_epochs + early_stop_window  # 最大允许训练次数

            optimizer_adam = soap.SOAP(self.net.parameters(), lr=adam_lr)  # 更新神经网络的参数
            optimizer_adam_weight = torch.optim.Adam([self.x_f_s_h] + [self.x_b_s_h]
                                                     + [self.x_f_s_mu] + [self.x_b_s_mu],
                                                     lr=AW_lr)  # 更新自适应权重

            epoch = 0
            while continue_training and epoch < max_epochs:
                self.s_collect.append([self.net.iter, self.x_f_s_h.item(),
                                       self.x_b_s_h.item(), self.x_f_s_mu.item(), self.x_b_s_mu.item()])

                loss_f_h, loss_f_mu, loss_b_h, loss_b_mu = self.epoch_loss()

                optimizer_adam.zero_grad()

                loss = self.true_loss(loss_f_h, loss_f_mu, loss_b_h, loss_b_mu)
                loss_b = loss_b_h + loss_b_mu
                loss_f = loss_f_h + loss_f_mu
                total_loss = loss_b + loss_f

                loss.backward()
                optimizer_adam.step()
                self.net.iter += 1
                # 合并条件判断：当 debug_val 为 0 且步数为 1000 的倍数，或 debug_val 为 1 且步数为 100 的倍数时打印
                if (debug_val == 0 and self.net.iter % 1000 == 0) or (debug_val == 1 and self.net.iter % 100 == 0):
                    print(
                        f"\nIter: {self.net.iter}"
                        f"\nTotal Loss: {total_loss.item():0.2e} "
                        f"Total BC Loss: {loss_b.item():0.2e} "
                        f"Total PDE Loss: {loss_f.item():0.2e}"
                        f"\nloss_b_h: {loss_b_h.item():0.2e} "
                        f"loss_b_mu: {loss_b_mu.item():0.2e} "
                        f"\nloss_f_h: {loss_f_h.item():0.2e} "
                        f"loss_f_mu: {loss_f_mu.item():0.2e}"
                    )

                optimizer_adam_weight.zero_grad()
                loss2 = self.likelihood_loss(loss_f_h, loss_f_mu, loss_b_h, loss_b_mu)
                loss2.backward()
                optimizer_adam_weight.step()

                # ========== 往前多跑100调停机制 ==========
                current_loss = total_loss.item()
                if epoch >= num_epochs - early_stop_window:
                    last_losses.append(current_loss)

                if continue_training and epoch >= num_epochs:
                    if len(last_losses) == early_stop_window:
                        max_loss = max(last_losses)
                        min_loss = min(last_losses)
                        diff = max_loss - min_loss

                        if diff < 1e-5:
                            print(f"[AM Count {move_count}] Early stopping at epoch {epoch}: max - min < 1e-5")
                            continue_training = False
                            break

                        threshold_value = (1 - loss_percentage_threshold) * diff + min_loss
                        if current_loss < threshold_value or epoch >= max_epochs - 1:
                            print(
                                f"[AM Count {move_count}] Early stopping at epoch {epoch}: "
                                f"current_loss {current_loss} < threshold {threshold_value} or reached max_epochs {max_epochs}."
                            )
                            continue_training = False
                            break

                epoch += 1
                # ==================================

            print('Adam done!')

            # 把loss和自适应权的数据保存起来
            save_loss_aw_data(move_count, current_t_num)

            if debug_val == 1:
                draw_exact_count(move_count, current_t_num)  # 画出每次选点前的结果
                draw_pred_q_count(move_count, current_t_num)
                draw_epoch_loss_count(move_count, current_t_num)

            # 重新选self.x_f_M之前，先把一些有用的可移动的内部点放到固定的内部点
            ## self.move_some_movable_points_to_fixed_points()

            # --- 只有在 AM_count > 1 且不是第一次时，才需要选点和绘图 ---
            if AM_count > 1 or move_count > 0:  # 即：只有多次 AM 才需要后续操作
                print(f'change_counts: from {move_count} to {move_count + 1}')
                self.choose_move_pts_and_draw(move_count, current_t_num)

    # 这个版本AM_count默认为1，skip这里的代码
    def choose_move_pts_and_draw(self, move_count, current_t_num):
        if AM_type == 0:
            x_init = random_fun(10000)
            x_init_residual = abs(self.x_f_loss_fun_h(x_init, self.train_U))
            x_init_residual = x_init_residual.cpu().detach().numpy()
            err_eq = np.power(x_init_residual, AM_K) / np.power(x_init_residual, AM_K).mean()
            err_eq_normalized = (err_eq / sum(err_eq))[:, 0]
            X_ids = np.random.choice(a=len(x_init), size=M, replace=False, p=err_eq_normalized)
            self.x_f_M = x_init[X_ids]

        elif AM_type == 1:
            x_init = random_fun(10000)
            x = Variable(x_init, requires_grad=True)
            u = self.train_U(x)
            dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]  # ∇u=∂u/∂x
            grad_x1 = dx[:, [0]].squeeze()  # ∂u/∂x1
            grad_x2 = dx[:, [1]].squeeze()  # ∂u/∂x2
            dx = torch.sqrt(
                1 + grad_x1 ** 2 + grad_x2 ** 2).cpu().detach().numpy()  # dx=sqrt(1+(grad_x1)^2+(grad_x2)^2) = sqrt(1+||∇u||^2）=omega(x)
            err_dx = np.power(dx, AM_K) / np.power(dx, AM_K).mean()
            p = (err_dx / sum(err_dx))  # p(x)=omega(x)^k / sum(omega(x)^k)
            X_ids = np.random.choice(a=len(x_init), size=M, replace=False, p=p)
            self.x_f_M = x_init[X_ids]

        elif AM_type == 2:
            x_init = random_fun(10000)
            u_0_train_init = get_initial_h(x_init)
            u_0_train_init = np_to_tensor_to_cuda(u_0_train_init)
            x_init_residual = abs(self.x_f_loss_fun_h(x_init, self.train_U) - u_0_train_init)  # 先和初值作差
            x_init_residual = x_init_residual.cpu().detach().numpy()
            err_eq = np.power(x_init_residual, AM_K) / np.power(x_init_residual, AM_K).mean()
            err_eq_normalized = (err_eq / sum(err_eq))[:, 0]
            X_ids = np.random.choice(a=len(x_init), size=M, replace=False, p=err_eq_normalized)
            self.x_f_M = x_init[X_ids]

        elif AM_type == 3:
            x_init = random_fun(10000)  # 选点在给定区域
            u_0_train_init = get_initial_h(x_init)
            u_0_train_init = np_to_tensor_to_cuda(u_0_train_init)
            x_init_residual = abs(self.x_f_loss_fun_h(x_init, self.train_U) - u_0_train_init)
            x_init_residual = x_init_residual.cpu().detach().numpy()
            err_eq = np.power(x_init_residual, AM_K) / np.power(x_init_residual, AM_K).mean()
            err_eq_normalized = (err_eq / sum(err_eq))[:, 0]
            X_ids = np.random.choice(a=len(x_init), size=M, replace=False, p=err_eq_normalized)
            self.x_f_M = x_init[X_ids]

        elif AM_type == 4:
            x_init = random_fun(10000)  # 和1的区别
            x4 = x_init.clone().detach().requires_grad_(True)
            u = self.train_U(x4)[:, self.q:(self.q + 1)]  # 和1的区别
            dx = torch.autograd.grad(u, x4, grad_outputs=torch.ones_like(u), create_graph=True)[0]  # ∇u=∂u/∂x
            grad_x1 = dx[:, [0]].squeeze()  # ∂u/∂x1
            grad_x2 = dx[:, [1]].squeeze()  # ∂u/∂x2
            dx = torch.sqrt(
                1 + grad_x1 ** 2 + grad_x2 ** 2).cpu().detach().numpy()  # dx=sqrt(1+(grad_x1)^2+(grad_x2)^2) = sqrt(1+||∇u||^2）=omega(x)
            err_dx = np.power(dx, AM_K) / np.power(dx, AM_K).mean()
            p = (err_dx / sum(err_dx))  # p(x)=omega(x)^k / sum(omega(x)^k)
            X_ids = np.random.choice(a=len(x_init), size=M, replace=False, p=p)
            self.x_f_M = x_init[X_ids]

        if debug_val == 1:
            draw_exact_points_after_AM_count(move_count, current_t_num, self.x_f_N, self.x_f_M)  # 画出每次AM选点后的内部点
            draw_epoch_w_count(move_count, current_t_num)

    def train(self, current_t_num):
        self.x_f_s_h = is_cuda(-torch.log(torch.tensor(1000.).float()))
        self.x_b_s_h = is_cuda(torch.tensor(0.).float())

        self.x_f_s_mu = is_cuda(torch.tensor(0.).float())
        self.x_b_s_mu = is_cuda(torch.tensor(0.).float())

        start_time = time.time()
        self.run_mcl_pinns(current_t_num)
        elapsed = time.time() - start_time
        print('Training time: %.2f' % elapsed)


# given initial condition
def get_initial_h(x_f):
    # 确保输入是 tensor 且在 GPU 上
    if not isinstance(x_f, torch.Tensor):
        x_f = torch.from_numpy(x_f).float()
    if use_gpu:
        x_f = x_f.cuda()

    x_i = x_f[:, 0] / Nx_scale
    y_i = x_f[:, 1] / Ny_scale

    r = 0.4
    R1 = torch.sqrt((x_i - 0.7 * r) ** 2 + (y_i + 1) ** 2)
    R2 = torch.sqrt((x_i + 0.7 * r) ** 2 + (y_i + 1) ** 2)

    phi1 = torch.tanh((r - R1) / (2 * epsilon))
    phi2 = torch.tanh((r - R2) / (2 * epsilon))

    u_0_train = torch.maximum(phi1, phi2).unsqueeze(-1)
    return u_0_train


# fwd_gradients_0
def uf_x_h(h, col_pts):
    zx = torch.ones_like(h, requires_grad=True)  # zx:(1270,20)
    gx = torch.autograd.grad(h, col_pts, grad_outputs=zx, retain_graph=True, create_graph=True)[0][:, 0].unsqueeze(
        -1)  # gx;(1270,1)
    h_x = torch.autograd.grad(gx, zx, grad_outputs=is_cuda(torch.ones(gx.shape)), retain_graph=True, create_graph=True)[
        0]  # h_x:(1270,20)
    return h_x

def uf_y_h(h, col_pts):
    zx = torch.ones_like(h, requires_grad=True)
    gx = torch.autograd.grad(h, col_pts, grad_outputs=zx, retain_graph=True, create_graph=True)[0][:, 1].unsqueeze(-1)
    h_x = torch.autograd.grad(gx, zx, grad_outputs=is_cuda(torch.ones(gx.shape)), retain_graph=True, create_graph=True)[
        0]
    return h_x


def x_f_loss_fun_h(x_f, train_U):
    if not x_f.requires_grad:
        x_f = Variable(x_f, requires_grad=True)

    h_mu = train_U(x_f)  # h^{n+c_1}, h^{n+c_2}, ..., h^{n+c_q}, h ,  mu^{n+c_1}, mu^{n+c_2}, ..., mu^{n+c_q}, mu
    h1 = h_mu[:, 0:(args.q + 1)]  # h^{n+c_1}, h^{n+c_2}, ..., h^{n+c_q}, h
    mu1 = h_mu[:, (args.q + 1):(2 * args.q + 2)]  # mu^{n+c_1}, mu^{n+c_2}, ..., mu^{n+c_q}, mu
    mu = mu1[:, :-1]  # mu^{n+c_1}, mu^{n+c_2}, ..., mu^{n+c_q}

    mu_x = uf_x_h(mu, x_f)
    mu_y = uf_y_h(mu, x_f)
    mu_xx = uf_x_h(mu_x, x_f)
    mu_yy = uf_y_h(mu_y, x_f)
    delta_mu = ((Nx_scale * Nx_scale) / Nt_scale) * mu_xx + ((Ny_scale * Ny_scale) / Nt_scale) * mu_yy
    # F = delta_mu
    F = delta_mu

    U0 = h1 - model.dt * torch.matmul(F, model.IRKWeights.T)  # h1

    return U0  # 这个是要跟初值h比较


def x_f_loss_fun_mu(x_f, train_U):
    if not x_f.requires_grad:
        x_f = Variable(x_f, requires_grad=True)

    h_mu = train_U(x_f)  # h^{n+c_1}, h^{n+c_2}, ..., h^{n+c_q}, h ,  mu^{n+c_1}, mu^{n+c_2}, ..., mu^{n+c_q}, mu
    h1 = h_mu[:, 0:(args.q + 1)]  # h^{n+c_1}, h^{n+c_2}, ..., h^{n+c_q}, h
    mu1 = h_mu[:, (args.q + 1):(2 * args.q + 2)]  # mu^{n+c_1}, mu^{n+c_2}, ..., mu^{n+c_q}, mu
    mu = mu1[:, :-1]  # mu^{n+c_1}, mu^{n+c_2}, ..., mu^{n+c_q}
    h = h1[:, :-1]  # h^{n+c_1}, h^{n+c_2}, ..., h^{n+c_q}

    h_x = uf_x_h(h, x_f)
    h_y = uf_y_h(h, x_f)
    h_xx = uf_x_h(h_x, x_f)
    h_yy = uf_y_h(h_y, x_f)

    # 把scale放里
    right_hand_side2 = (((-(L_d * epsilon ** 2)) * Nx_scale * Nx_scale) * h_xx
                        + ((-(L_d * epsilon ** 2)) * Ny_scale * Ny_scale) * h_yy
                        + L_d * (h ** 3) - L_d * h)
    residual = mu - right_hand_side2

    return residual


# 画出每1000iter的h(debug_val==1才画)
def draw_exact_iter_count(iter_num, AM_count, current_t_num):
    predict_np = model.predict_H(x_test).cpu().detach().numpy().T
    XX1, XX2 = np.meshgrid(x / Nx_scale, y / Ny_scale)
    h_pred = np.reshape(predict_np, (XX1.shape[0], XX1.shape[1]), order='F')

    fig, ax = plt.subplots(figsize=(9, 4.5))

    im = ax.pcolormesh(XX1, XX2, h_pred, shading='auto', cmap='jet')

    fig.colorbar(im, ax=ax, aspect=10)

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(XX1.min(), XX1.max())
    ax.set_ylim(XX2.min(), XX2.max())

    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='black', alpha=0.5)
    ax.set_title(r'$\phi$ at t=' + f"{tFinal.item():g}")

    plt.tight_layout()

    subfolder_path = os.path.join("./debug_figures_count/", f"t_{current_t_num}", f"count_{AM_count}")
    os.makedirs(subfolder_path, exist_ok=True)

    filename = f"Sol_disc_t_{current_t_num}_count_{AM_count}_iter_{iter_num}.png"
    filepath = os.path.join(subfolder_path, filename)

    fig.savefig(filepath, dpi=500, bbox_inches='tight')
    plt.close(fig)


# 画出每一次AM_count计算出来的h(debug_val==1才画)
def draw_exact_count(AM_count, current_t_num):
    predict_np = model.predict_H(x_test).cpu().detach().numpy().T
    XX1, XX2 = np.meshgrid(x / Nx_scale, y / Ny_scale)
    u_pred = np.reshape(predict_np, (XX1.shape[0], XX1.shape[1]), order='F')

    fig, ax = plt.subplots(figsize=(9, 4.5))

    im = ax.pcolormesh(XX1, XX2, u_pred, shading='auto', cmap='jet')

    fig.colorbar(im, ax=ax, aspect=10)

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')

    ax.set_aspect('equal', adjustable='box')

    ax.set_xlim(XX1.min(), XX1.max())
    ax.set_ylim(XX2.min(), XX2.max())

    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='black', alpha=0.5)
    ax.set_title(r'$\phi$ at t=' + f"{tFinal.item():g}")

    plt.tight_layout()

    save_dir = "./figures_count"
    os.makedirs(save_dir, exist_ok=True)

    filename = f"Sol_disc_t_{current_t_num}_count_{AM_count}.png"
    filepath = os.path.join(save_dir, filename)

    fig.savefig(filepath, dpi=500, bbox_inches='tight')
    plt.close(fig)


# (debug_val==1才画)
def draw_pred_q_count(AM_count, current_t_num):
    XX1, XX2 = np.meshgrid(x / Nx_scale, y / Ny_scale)
    # q_preds = []  # 存储所有的 q_pred
    # q_pred_plots = {}  # 存储所有的 q_pred_plotting

    for j in range(0, args.q + 1):
        q_pred_plotting = model.net(x_test)[:, j:j + 1].cpu().detach().numpy().reshape((XX1.shape[0], XX1.shape[1]),
                                                                                       order='F')

        plt.figure(figsize=(8, 4))
        plt.pcolor(XX1, XX2, q_pred_plotting, shading='auto', cmap='jet')
        plt.colorbar()
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.axis('equal')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='black', alpha=0.5)
        plt.title(r'q pred ' + str(model.IRKTimesExtended[j].detach().cpu().numpy()))

        os.makedirs("./figures_q_count/", exist_ok=True)
        plt.savefig("figures_q_count/q_disc_" + "t_" + str(current_t_num)
                    + "_count_" + str(AM_count) + "_IRK_" + str(
            model.IRKTimesExtended[j].detach().cpu().numpy()) + ".png",
                    dpi=500, bbox_inches='tight')
        plt.close()


# 画出当前时间段IRKTimes (debug_val==1才画)
def draw_pred_q_current(time_current):
    XX1, XX2 = np.meshgrid(x / Nx_scale, y / Ny_scale)
    # q_preds = []  # 存储所有的 q_pred
    # q_pred_plots = {}  # 存储所有的 q_pred_plotting

    for j in range(0, args.q + 1):
        q_pred_plotting = model.net(x_test)[:, j:j + 1].cpu().detach().numpy().reshape((XX1.shape[0], XX1.shape[1]),
                                                                                       order='F')
        plt.figure(figsize=(8, 4))
        plt.pcolor(XX1, XX2, q_pred_plotting, shading='auto', cmap='jet')
        plt.colorbar()
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.axis('equal')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='black', alpha=0.5)
        plt.title(r'q pred ' + str(model.IRKTimesExtended[j].detach().cpu().numpy()))

        os.makedirs("./current_figures_q/", exist_ok=True)
        plt.savefig("current_figures_q/q_disc_" + "_t_" + str(time_current) + "_IRK_" + str(
            model.IRKTimesExtended[j].detach().cpu().numpy()) + ".png",
                    dpi=500, bbox_inches='tight')
        plt.close()


# 画出IRKTimes (最后画图)
def draw_pred_q():
    XX1, XX2 = np.meshgrid(x / Nx_scale, y / Ny_scale)
    # q_preds = []  # 存储所有的 q_pred
    # q_pred_plots = {}  # 存储所有的 q_pred_plotting

    for j in range(0, args.q + 1):
        q_pred_plotting = model.net(x_test)[:, j:j + 1].cpu().detach().numpy().reshape((XX1.shape[0], XX1.shape[1]),
                                                                                       order='F')

        plt.figure(figsize=(8, 4))
        plt.pcolor(XX1, XX2, q_pred_plotting, shading='auto', cmap='jet')
        plt.colorbar()
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.axis('equal')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='black', alpha=0.5)
        plt.title(r'q pred ' + str(model.IRKTimesExtended[j].detach().cpu().numpy()))

        os.makedirs("./figures_q/", exist_ok=True)
        plt.savefig("figures_q/q_disc_" "IRK_" + str(model.IRKTimesExtended[j].detach().cpu().numpy()) + ".png",
                    dpi=500, bbox_inches='tight')
        plt.close()


def draw_current_exact(time_block):
    predict_np = model.predict_H(x_test).cpu().detach().numpy()
    XX1, XX2 = np.meshgrid(x / Nx_scale, y / Ny_scale)
    u_pred = np.reshape(predict_np, (XX1.shape[0], XX1.shape[1]), order='F')

    fig, ax = plt.subplots(figsize=(8, 4))

    plt.pcolormesh(XX1, XX2, u_pred, shading='auto', cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')

    ax.set_aspect('equal')

    plt.xlim(XX1.min(), XX1.max())
    plt.ylim(XX2.min(), XX2.max())

    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='black', alpha=0.5)
    plt.title(r'$\phi$ at t=' + f"{tFinal.item():g}")

    plt.tight_layout()

    os.makedirs("./current_figures/", exist_ok=True)
    fig.savefig("current_figures/Sol" + "_t_" + str(time_block) + ".png", dpi=500, bbox_inches='tight')
    plt.close(fig)

    # 储存数值结果数据
    file_name = "./current_figures_matlab_data/"
    os.makedirs(file_name, exist_ok=True)

    scipy.io.savemat(file_name + "Sol_" + str(time_block) + "_" + str(ver_num) + ".mat",
                     {'x': x, 'y': y, 'XX': XX1, 'YY': XX2, 'predict_np_h': predict_np})


# 画出最后一次AM_count计算出来的h (最后画图)
def draw_exact():
    predict_np = model.predict_H(x_test).cpu().detach().numpy()

    XX1, XX2 = np.meshgrid(x / Nx_scale, y / Ny_scale)

    u_pred = np.reshape(predict_np, (XX1.shape[0], XX1.shape[1]), order='F')

    fig_1 = plt.figure(1, figsize=(8, 4))

    plt.pcolormesh(XX1, XX2, u_pred, shading='auto', cmap='jet')
    plt.colorbar()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.axis('equal')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='black', alpha=0.5)
    plt.title(r'$\phi$ at t=' + f"{tFinal.item():g}")
    plt.subplots_adjust(hspace=0.4)

    os.makedirs("./figures/", exist_ok=True)
    fig_1.savefig("figures/Sol_disc_" + params_name + ".png", dpi=500, bbox_inches='tight')
    plt.close()


# 画出每一次AM_count移动的内部点 (debug_val为1才画)
# N_points = self.x_f_N, points = self.x_f_M
def draw_exact_points_after_AM_count(AM_count, current_t_num, N_points, points=None):
    fig, ax = plt.subplots()

    if points is not None:
        adds = points.cpu().detach().numpy()
        ax.plot(adds[:, 0], adds[:, 1], 'kx', markersize=1)

    N_points_np = N_points.cpu().detach().numpy()
    ax.plot(N_points_np[:, 0], N_points_np[:, 1], 'gx', markersize=1)

    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='black', alpha=0.5)
    ax.set_xlabel(r'$x$', fontsize=20)
    ax.set_ylabel(r'$y$', fontsize=20)

    plt.margins(0.01)

    plt.tight_layout()

    save_dir = "./col_pts_count"
    os.makedirs(save_dir, exist_ok=True)

    filename = f"Colpt_after_AM_t_{current_t_num}_count_{AM_count}.png"
    filepath = os.path.join(save_dir, filename)

    fig.savefig(filepath, dpi=500, bbox_inches='tight')
    plt.close(fig)


# 画出每一次AM_count移动前的内部点 (debug_val为1才画)
# move_points = self.x_f_N, fix_points = self.x_f_M
def draw_exact_points_count_b4_AM(AM_count, current_t_num, move_points, fix_points=None):
    fig, ax = plt.subplots()
    adds = fix_points.cpu().detach().numpy()
    ax.plot(adds[:, 0], adds[:, 1], 'kx', markersize=1, label='Fixed')

    move_pts_np = move_points.cpu().detach().numpy()
    ax.plot(move_pts_np[:, 0], move_pts_np[:, 1], 'gx', markersize=1, label='Moving')

    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='black', alpha=0.5)
    ax.set_xlabel(r'$x$', fontsize=20)
    ax.set_ylabel(r'$y$', fontsize=20)

    plt.margins(0.01)

    plt.tight_layout()

    save_dir = "./col_pts_count"
    os.makedirs(save_dir, exist_ok=True)

    filename = f"Colpt_b4_AM_t_{current_t_num}_count_{AM_count}.png"
    filepath = os.path.join(save_dir, filename)

    fig.savefig(filepath, dpi=500, bbox_inches='tight')

    plt.close(fig)


# (debug_val==1才画)
def draw_epoch_loss_count(AM_count, current_t_num):
    x_b_loss_collect = np.array(model.x_b_loss_collect)
    x_f_loss_collect = np.array(model.x_f_loss_collect)

    plot_configs = [
        (x_b_loss_collect, 1, '#859ED7', r'$\mathcal{L}_b$'),
        (x_b_loss_collect, 2, '#2DA248', r'$\mathcal{L}_{b_{\phi}}$'),
        (x_b_loss_collect, 3, '#9368AB', r'$\mathcal{L}_{b_{\mu}}$'),
        (x_f_loss_collect, 1, '#CDD9EC', r'$\mathcal{L}_f$'),
        (x_f_loss_collect, 2, '#F47F1E', r'$\mathcal{L}_{f_{\phi}}$'),
        (x_f_loss_collect, 3, '#22BDD2', r'$\mathcal{L}_{f_{\mu}}$'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    axes = axes.flatten()

    for i, (data, col_idx, color, ylabel) in enumerate(plot_configs):
        ax = axes[i]
        ax.set_yscale('log')
        ax.plot(data[:, 0], data[:, col_idx], color=color, label=ylabel, linewidth=1.5)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=10)

        ax.legend(fontsize=11, loc='upper right', framealpha=0.9)

    os.makedirs("./loss_count/", exist_ok=True)

    save_path = f"./loss_count/Loss_disc_t_{current_t_num}_count_{AM_count}.png"

    fig.savefig(save_path, dpi=500, bbox_inches='tight')
    plt.close(fig)


def save_loss_aw_data(AM_count, current_t_num):
    data_dict = {
        "loss_b": np.array(model.x_b_loss_collect),
        "loss_f": np.array(model.x_f_loss_collect),
        "weight_b": np.array(model.x_b_weight_collect),
        "weight_f": np.array(model.x_f_weight_collect)
    }

    save_dir = "./data_loss_txt_file"
    os.makedirs(save_dir, exist_ok=True)
    file_prefix = f"{AM_count}_{current_t_num}"

    for name, data in data_dict.items():
        filepath = os.path.join(save_dir, f"{name}_{file_prefix}.npy")
        np.save(filepath, data)


# (debug_val==1才画)
def draw_exact_points_current(current_t_num, points, N_points):
    fig, ax = plt.subplots()

    adds = points.cpu().detach().numpy()
    ax.plot(adds[:, 0], adds[:, 1], 'kx', markersize=4)
    N_points_np = N_points.cpu().detach().numpy()
    ax.plot(N_points_np[:, 0], N_points_np[:, 1], 'gx', markersize=4)

    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='black', alpha=0.5)
    ax.set_xlabel(r'$x$', fontsize=20)
    ax.set_ylabel(r'$y$', fontsize=20)

    plt.tight_layout()

    save_dir = "./current_col_pt"
    os.makedirs(save_dir, exist_ok=True)
    filename = f"Colpt_disc_t_{current_t_num}.png"
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath, dpi=500, bbox_inches='tight')
    plt.close(fig)


# (debug_val==1才画)
def draw_epoch_loss_current(loss_current_num):
    x_b_loss_collect = np.array(model.x_b_loss_collect)
    x_f_loss_collect = np.array(model.x_f_loss_collect)

    plot_configs = [
        (x_b_loss_collect, 1, '#859ED7', r'$\mathcal{L}_b$'),
        (x_b_loss_collect, 2, '#2DA248', r'$\mathcal{L}_{b_{\phi}}$'),
        (x_b_loss_collect, 3, '#9368AB', r'$\mathcal{L}_{b_{\mu}}$'),
        (x_f_loss_collect, 1, '#CDD9EC', r'$\mathcal{L}_f$'),
        (x_f_loss_collect, 2, '#F47F1E', r'$\mathcal{L}_{f_{\phi}}$'),
        (x_f_loss_collect, 3, '#22BDD2', r'$\mathcal{L}_{f_{\mu}}$'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    axes = axes.flatten()

    for i, (data, col_idx, color, ylabel) in enumerate(plot_configs):
        ax = axes[i]
        ax.set_yscale('log')
        ax.plot(data[:, 0], data[:, col_idx], color=color, label=ylabel, linewidth=1.5)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=10)

        ax.legend(fontsize=11, loc='upper right', framealpha=0.9)

    os.makedirs("./current_loss_figures/", exist_ok=True)
    save_path = f"./current_loss_figures/Loss_disc_{loss_current_num}.png"

    fig.savefig(save_path, dpi=500, bbox_inches='tight')
    plt.close(fig)


# (最后画图)
def draw_epoch_loss():
    x_b_loss_collect = np.array(model.x_b_loss_collect)
    x_f_loss_collect = np.array(model.x_f_loss_collect)

    plot_configs = [
        (x_b_loss_collect, 1, '#859ED7', r'$\mathcal{L}_b$'),
        (x_b_loss_collect, 2, '#2DA248', r'$\mathcal{L}_{b_{\phi}}$'),
        (x_b_loss_collect, 3, '#9368AB', r'$\mathcal{L}_{b_{\mu}}$'),
        (x_f_loss_collect, 1, '#CDD9EC', r'$\mathcal{L}_f$'),
        (x_f_loss_collect, 2, '#F47F1E', r'$\mathcal{L}_{f_{\phi}}$'),
        (x_f_loss_collect, 3, '#22BDD2', r'$\mathcal{L}_{f_{\mu}}$'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    axes = axes.flatten()

    for i, (data, col_idx, color, ylabel) in enumerate(plot_configs):
        ax = axes[i]
        ax.set_yscale('log')
        ax.plot(data[:, 0], data[:, col_idx], color=color, label=ylabel, linewidth=1.5)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.legend(fontsize=11, loc='upper right', framealpha=0.9)

    os.makedirs("./figures/", exist_ok=True)
    save_path = f"./figures/Loss_disc{params_name}.png"

    fig.savefig(save_path, dpi=500, bbox_inches='tight')
    plt.close(fig)


# (debug_val为1才画)
def draw_epoch_w_count(AM_count, time_block):
    s_collect = np.array(model.s_collect)
    plt.yscale('log')
    plt.plot(s_collect[:, 0], np.exp(-s_collect[:, 1]), '#F47F1E', label='e^{-s_{f_h}}')
    plt.plot(s_collect[:, 0], np.exp(-s_collect[:, 2]), '#2DA248', label='e^{-s_{b_h}}')
    plt.plot(s_collect[:, 0], np.exp(-s_collect[:, 3]), '#22BDD2', label='e^{-s_{f_mu}}')
    plt.plot(s_collect[:, 0], np.exp(-s_collect[:, 4]), '#9368AB', label='e^{-s_{b_mu}}')
    plt.xlabel('$Iters$')
    plt.ylabel(r'$\lambda$')
    plt.legend()
    os.makedirs("./aw/", exist_ok=True)
    plt.savefig("aw/RAM-AW_S_disc_t_" + str(time_block) + "_count_" + str(AM_count) + ".png")
    plt.close()


# (最后画图)
def draw_epoch_w():
    s_collect = np.array(model.s_collect)
    plt.yscale('log')
    plt.plot(s_collect[:, 0], np.exp(-s_collect[:, 1]), 'b-', label=r'$e^{-s_{f_{\phi}}}$')
    plt.plot(s_collect[:, 0], np.exp(-s_collect[:, 2]), 'r-', label=r'$e^{-s_{b_{\phi}}}$')
    plt.plot(s_collect[:, 0], np.exp(-s_collect[:, 3]), 'k--', label=r'$e^{-s_{f_{\mu}}}$')
    plt.plot(s_collect[:, 0], np.exp(-s_collect[:, 4]), 'g-', label=r'$e^{-s_{b_{\mu}}}$')
    plt.xlabel('Epoch')
    plt.ylabel(r'$\lambda$')
    plt.legend()
    os.makedirs("./figures/", exist_ok=True)
    plt.savefig("figures/RAM-AW_S_disc" + params_name + ".png")
    plt.close()


def save_checkpoint(i, net, h_pred_current, x_test_shape, all_predictions, models):
    ckpt_path = SAVE_DIR / f"ckpt_{i}.pt"
    torch.save({
        'i': i,
        'model_state_dict': net.state_dict(),
        'h_pred_current': h_pred_current,
        'x_test_shape': x_test_shape,
        'all_predictions': all_predictions,
        'models': models,  # 保存所有已训练的 Model 实例
        'steps_lb': steps_lb.tolist(),  # 保证恢复时 steps_lb 一致
        'args': {k: v for k, v in vars(args).items() if k not in ['self']}  # 可选：保存参数
    }, ckpt_path)
    print(f"Saved checkpoint {i} at {ckpt_path}")


def load_latest_checkpoint(nets):
    if not SAVE_DIR.exists():
        return 0, None, None, []

    ckpts = sorted(SAVE_DIR.glob("ckpt_*.pt"), key=lambda x: int(x.stem.split('_')[1]))
    if not ckpts:
        return 0, None, None, []

    last_ckpt = ckpts[-1]
    ckpt = torch.load(last_ckpt, map_location='cpu')
    i_last = ckpt['i']

    # 恢复 steps_lb（关键！防止不一致）
    global steps_lb, steps_ub
    steps_lb = np.array(ckpt['steps_lb'])
    steps_ub = steps_lb

    # 恢复 nets[i_last] 的 state_dict（已在 GPU）
    nets[i_last].load_state_dict(ckpt['model_state_dict'])

    # 恢复 models 列表
    models = ckpt.get('models', [])

    # 将每个 model.net 移到 GPU
    for model in models:
        if hasattr(model, 'net'):
            model.net = model.net.cuda()  # 或 .to('cuda')

    h_pred_current = ckpt.get('h_pred_current')
    all_predictions = ckpt.get('all_predictions')

    print(f"Found checkpoint: {last_ckpt}, resuming from i = {i_last + 1}")
    return i_last + 1, h_pred_current, all_predictions, models


# main
if __name__ == '__main__':
    args = parser.parse_args()
    with open('args_output.txt', 'w') as f:
        args_dict = vars(args)
        for key, value in args_dict.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

    params_name = "_"

    layers = generate_layers(args.hid_neurons, args.hid_layers, args.q)

    Nx_scale = args.Nx_scale
    Ny_scale = args.Ny_scale
    Nt_scale = args.Nt_scale

    epsilon = args.epsilon
    L_d = args.L_d
    alpha_var = args.alpha_var
    theta_s = args.theta_s

    q = args.q

    M = args.M
    num_x = args.num_x
    num_y = args.num_y

    num_x_col = args.num_x_col
    num_y_col = args.num_y_col

    adam_iter, lbfgs_iter = args.adam_iter, args.lbfgs_iter
    adam_lr, lbfgs_lr = args.adam_lr, args.lbfgs_lr

    model_type = args.model_type  # 0:baseline  1:AM  2:AM_AW

    AM_type = args.AM_type  # 0:RAM  1:WAM
    AM_K = args.AM_K
    AM_count = args.AM_count

    move_type = args.move_type

    AW_lr = args.AW_lr
    t_final = args.t_final

    threshold = args.threshold

    debug_val = args.debug_val  # 0 代表 False （不画图，但是效率高），1 代表 True（画图）

    lb = np.array([-1.0 * Nx_scale, -1.0 * Ny_scale])  # np.array([0.0, 0.0])
    ub = np.array([1.0 * Nx_scale, 0 * Ny_scale])

    x = np.expand_dims(np.linspace(lb[0], ub[0], 129), axis=1)
    y = np.expand_dims(np.linspace(lb[1], ub[1], 129), axis=1)
    t = np.expand_dims(np.linspace(0, t_final, 1001), axis=1)

    # 计算网格间距
    dx_domain = ub[0] - lb[0]
    dy_domain = ub[1] - lb[1]

    # --------------------生成测试点和边界点-----------------------------------
    # 因为要取动态的内部点，根据interface来选，所以要把x_f_N移到时间块循环里面
    (x_test, xyt_b_bot_meshgrid_train, xyt_b_top_meshgrid_train, xyt_b_left_meshgrid_train,
     xyt_b_right_meshgrid_train) = data_generate(x, y, num_x, num_y)
    # --------------------生成固定的内部点（画选点图里的黑色点）-----------------------------------
    x_f_M = random_fun(M)

    # 把时间分块
    # 根据 delta_t 计算时间块数量
    num_time_blocks = max(int(t_final / args.delta_t), 1)

    # 计算每个时间块对应的索引步长
    step_range = max((len(t) - 1) // num_time_blocks, 1)

    # 确保步长至少为1
    step_range = max(step_range, 1)

    # 生成时间块的起始索引
    steps_lb = np.arange(0, len(t), step_range)

    # 如果最后一个索引没有达到数组末尾，添加末尾索引
    if steps_lb[-1] != len(t) - 1:
        steps_lb = np.append(steps_lb, len(t) - 1)

    steps_ub = steps_lb

    # 初始化每个时间块的net
    nets = [is_cuda(Net(layers, q)) for _ in range(len(steps_lb) - 1)]  # 确保数量正确

    # ------------------------ 恢复最新检查点 ------------------------
    start_i, h_pred_current, all_predictions, models = load_latest_checkpoint(nets)
    # 如果是从 i=0 开始，h_pred_current 应为 None，否则它已经是上一个块的预测
    # 注意：在 i=0 时 get_initial_interface 会用初始条件，所以没问题
    # 如果没有恢复，初始化为空
    if all_predictions is None:
        all_predictions = None
    if models is None:
        models = []

    # ------------------------ 时间块循环 ------------------------
    for i in range(start_i, steps_lb.size - 1):
        idx_t0 = steps_lb[i]
        idx_t1 = steps_ub[i + 1]
        tFinal = np_to_tensor_to_cuda(np.array([t[idx_t1]]))  # 将取出的值放入一个长度为1的数组中
        tStart = np_to_tensor_to_cuda(np.array([t[idx_t0]]))
        dt = tFinal - tStart

        print("From " + str(tStart.detach().cpu().numpy()) + " to " + str(tFinal.detach().cpu().numpy()))

        # ------------------------AS后的自适应内部点-------------------------------------
        # 从data_generate移到这里来，因为需要动态选interface。
        # --- 使用原有的 h_pred_current 作为上一个时间块的预测 ---
        x_f_N = get_initial_interface(lb, ub, h_pred_current, x_test, i)  # h_pred_current是上一个时间块最后预测出来的h

        model = Model(
            net=nets[i],
            xyt_b_bot_meshgrid_train=xyt_b_bot_meshgrid_train,
            xyt_b_top_meshgrid_train=xyt_b_top_meshgrid_train,
            xyt_b_left_meshgrid_train=xyt_b_left_meshgrid_train,
            xyt_b_right_meshgrid_train=xyt_b_right_meshgrid_train,
            x_f_loss_fun_h=x_f_loss_fun_h,
            x_f_loss_fun_mu=x_f_loss_fun_mu,
            x_test=x_test,
            dt=dt,
            tStart=tStart,
            tFinal=tFinal,
            q=args.q,
            x_f_N=x_f_N,
            x_f_M=x_f_M,
            current_t_num=i,
            debug_val=debug_val)

        model.train(i)
        models.append(model)

        draw_current_exact(i)  # 看结果和保存数值结果

        if debug_val == 1:
            draw_pred_q_current(i)
            draw_epoch_loss_current(i)
            draw_exact_points_current(i, model.x_f_M, N_points=model.x_f_N)

        h_pred_current = model.predict_H(x_test).cpu().detach().numpy()
        h_pred = np.reshape(h_pred_current, (x.shape[0], y.shape[0]), order='F')

        # 将预测值存储起来
        if i == 0:
            all_predictions = h_pred  # 这里是0.1的值
        else:
            h_pred_concat = h_pred  # 这里就只是0.2，0.3，。。。，1的值
            all_predictions = np.concatenate([all_predictions, h_pred_concat], axis=0)

        # --- 保存完整状态 ---
        save_checkpoint(i, nets[i], h_pred_current, x_test.shape, all_predictions, models)

        if i < steps_lb.size - 2:
            nets[i + 1].load_state_dict(nets[i].state_dict())  # 继承上一个时间块的神经网络参数

    # ------------------------ 最终绘图 ------------------------
    if 'all_predictions' in locals() and all_predictions is not None:
        draw_exact()
        draw_pred_q()
        draw_epoch_loss()
        draw_epoch_w()
