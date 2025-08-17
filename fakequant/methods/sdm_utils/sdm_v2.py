import torch
import sys
import matplotlib.pyplot as plt
import torch.fft as torch_fft
import numpy as np
from kmeans_utils import RowWiseKMeansQuantizerTorch,KMeansQuantizerTorch

# 定义窗口函数映射
_win_equiv = {
    'boxcar': torch.ones,
    'triang': lambda Nx: torch.from_numpy(np.bartlett(Nx)),
    'blackman': lambda Nx: torch.from_numpy(np.blackman(Nx)),
    'hamming': lambda Nx: torch.from_numpy(np.hamming(Nx)),
    'hann': lambda Nx: torch.from_numpy(np.hanning(Nx)),
    'bartlett': lambda Nx: torch.from_numpy(np.bartlett(Nx)),
    'flattop': lambda Nx: torch.from_numpy(np.flat(Nx)),
    'parzen': lambda Nx: torch.from_numpy(np.parzen(Nx)),
    'bohman': lambda Nx: torch.from_numpy(np.bohman(Nx)),
    'blackmanharris': lambda Nx: torch.from_numpy(np.blackmanharris(Nx)),
    'nuttall': lambda Nx: torch.from_numpy(np.nuttall(Nx)),
    'barthann': lambda Nx: torch.from_numpy(np.barthann(Nx)),
    'cosine': lambda Nx: torch.from_numpy(np.cosine(Nx)),
    'exponential': lambda Nx, tau: torch.from_numpy(np.exponential(Nx, tau)),
    'tukey': lambda Nx, alpha: torch.from_numpy(np.tukey(Nx, alpha)),
    'taylor': lambda Nx, nbar, sll: torch.from_numpy(np.taylor(Nx, nbar, sll)),
    'lanczos': lambda Nx: torch.from_numpy(np.sinc(np.linspace(-Nx/2, Nx/2, Nx))),
    'kaiser': lambda Nx, beta: torch.from_numpy(np.kaiser(Nx, beta)),
    'kaiser_bessel_derived': lambda Nx, beta: torch.from_numpy(np.kaiser_bessel_derived(Nx, beta)),
    'gaussian': lambda Nx, std: torch.from_numpy(np.gaussian(Nx, std)),
    'general_cosine': lambda Nx, a: torch.from_numpy(np.general_cosine(Nx, a)),
    'general_gaussian': lambda Nx, p, sig: torch.from_numpy(np.general_gaussian(Nx, p, sig)),
    'general_hamming': lambda Nx, alpha: torch.from_numpy(np.general_hamming(Nx, alpha)),
    'dpss': lambda Nx, NW: torch.from_numpy(np.dpss(Nx, NW)),
    'chebwin': lambda Nx, at: torch.from_numpy(np.chebwin(Nx, at))
}

_needs_param = {
    'kaiser', 'kaiser_bessel_derived', 'gaussian', 'general_cosine',
    'general_gaussian', 'general_hamming', 'dpss', 'chebwin', 'exponential',
    'tukey', 'taylor'
}
def get_window(window, Nx, fftbins=True):
    sym = not fftbins
    try:
        beta = float(window)
    except (TypeError, ValueError) as e:
        args = ()
        if isinstance(window, tuple):
            winstr = window[0]
            if len(window) > 1:
                args = window[1:]
        elif isinstance(window, str):
            if window in _needs_param:
                raise ValueError("The '" + window + "' window needs one or more parameters -- pass a tuple.") from e
            else:
                winstr = window
        else:
            raise ValueError(f"{type(window)} as window type is not supported.") from e

        try:
            winfunc = _win_equiv[winstr]
        except KeyError as e:
            raise ValueError("Unknown window type.") from e

        if winfunc is _win_equiv['dpss']:
            params = (Nx,) + args + (None,)
        else:
            params = (Nx,) + args
    else:
        winfunc = _win_equiv['kaiser']
        params = (Nx, beta)

    return winfunc(*params)

def resample(x, num, t=None, axis=0, window=None, domain='time'):
    # Check arguments
    if not isinstance(num, int):
        num = int(num)

    if domain not in ('time', 'freq'):
        raise ValueError("Acceptable domain flags are 'time' or"
                         f" 'freq', not domain={domain}")

    x = torch.as_tensor(x)
    Nx = x.shape[axis]

    # Check if we can use faster real FFT
    real_input = torch.isreal(x)
    real_input=True

    if domain == 'time':
        # Forward transform
        if real_input:
            X = torch_fft.rfft(x, dim=axis)
        else:  # Full complex FFT
            X = torch_fft.fft(x, dim=axis)
    else:  # domain == 'freq'
        X = x

    # Apply window to spectrum
    if window is not None: 
        if callable(window):
            W = window(torch_fft.fftfreq(Nx))
        elif isinstance(window, torch.Tensor):
            if window.shape != (Nx,):
                raise ValueError('window must have the same length as data')
            W = window
        else:
            W = torch.fft.ifftshift(torch.tensor(get_window(window, Nx)))

        newshape_W = [1] * x.ndim
        newshape_W[axis] = X.shape[axis]
        if real_input:
            # Fold the window back on itself to mimic complex behavior
            W_real = W.clone()
            W_real[1:] += W_real[-1:0:-1]
            W_real[1:] *= 0.5
            X *= W_real[:newshape_W[axis]].reshape(newshape_W)
        else:
            X *= W.reshape(newshape_W)
   
    # Placeholder array for output spectrum
    newshape = list(x.shape)
    if real_input:
        newshape[axis] = num // 2 + 1
       # print("newshape[axis]:",newshape[axis])
    else:
        newshape[axis] = num
    Y = torch.zeros(newshape, dtype=X.dtype, device=X.device)
    #print("Y.shape:",Y.shape)
    # Copy positive frequency components (and Nyquist, if present)
    N = min(num, Nx)
    nyq = N // 2 + 1  # Slice index that includes Nyquist if present
    sl = [slice(None)] * x.ndim
    sl[axis] = slice(0, nyq)
    Y[tuple(sl)] = X[tuple(sl)]
    if not real_input:
        # Copy negative frequency components
        if N > 2:  # (slice expression doesn't collapse to empty array)
            sl[axis] = slice(nyq - N, None)
            Y[tuple(sl)] = X[tuple(sl)]

    # Split/join Nyquist component(s) if present
    # So far we have set Y[+N/2]=X[+N/2]
    if N % 2 == 0:
        if num < Nx:  # downsampling
            if real_input:
                sl[axis] = slice(N//2, N//2 + 1)
                Y[tuple(sl)] *= 2.
            else:
                # select the component of Y at frequency +N/2,
                # add the component of X at -N/2
                sl[axis] = slice(-N//2, -N//2 + 1)
                Y[tuple(sl)] += X[tuple(sl)]
        elif Nx < num:  # upsampling
            # select the component at frequency +N/2 and halve it
            sl[axis] = slice(N//2, N//2 + 1)
            Y[tuple(sl)] *= 0.5
            if not real_input:
                temp = Y[tuple(sl)]
                # set the component at -N/2 equal to the component at +N/2
                sl[axis] = slice(num-N//2, num-N//2 + 1)
                Y[tuple(sl)] = temp
    #print("Y.shape:",Y.shape)
    # Inverse transform
    if real_input:
        y = torch_fft.irfft(Y, n=num, dim=axis)
    else:
        y = torch_fft.ifft(Y, dim=axis)

    y *= (float(num) / float(Nx))

    if t is None:
        return y
    else:
        new_t = torch.arange(0, num, device=x.device) * (t[1] - t[0]) * Nx / float(num) + t[0]
        return y, new_t


def extend(weight,OSR):
    return resample(weight, weight.shape[-1]*OSR,axis=-1)


class SigmaDeltaQuantizer:
    def __init__(self, first2=False, sd_scale=1, order=1):
        self.integrator1 = None
        self.integrator2 = None
        self.previous_output = None
        self.first2 = first2
        self.sd_scale = sd_scale
        self.order = order

    def quantize(self, input_matrix,percentile_1=0.95,percentile_2=0.55):
        rows, cols = input_matrix.shape
        dev=input_matrix.device
        output_matrix = torch.zeros_like(input_matrix).to(dev)
        self.integrator1 = torch.zeros(rows).to(dev)
        self.previous_output = torch.zeros(rows).to(dev)

        if self.order == 2:
            self.integrator2 = torch.zeros(rows).to(dev)


        #按每行
        # mean_vals = torch.mean(input_matrix, dim=-1, keepdim=True)
        # input_matrix = input_matrix - mean_vals
        # max_vals = torch.quantile(torch.abs(input_matrix), percentile_1, dim=-1, keepdim=True)
        # scale = max_vals * self.sd_scale
        # input_matrix = input_matrix /scale
        # cut_threshold=0
        # cut_threshold_first3=torch.quantile(torch.abs(input_matrix), percentile_2, dim=1)

        #按整块
        mean_vals = torch.mean(input_matrix)
        input_matrix = input_matrix - mean_vals
        max_vals = torch.quantile(torch.abs(input_matrix), percentile_1)
        scale = max_vals * self.sd_scale
        input_matrix = input_matrix /scale
        cut_threshold=0
        cut_threshold_first3 = torch.quantile(torch.abs(input_matrix), percentile_2)
        

        for j in range(cols):
            # 计算积分器输入
            integrator_input = input_matrix[:, j] - self.previous_output
            if self.order == 1:
                # 一阶 Sigma-Delta 调制
                self.integrator1 += integrator_input
                if not self.first2:  # 一阶三元量化
                    output_matrix[:, j][self.integrator1 >= cut_threshold_first3] = 1
                    output_matrix[:, j][self.integrator1 <= -cut_threshold_first3] = -1
                else:  # 一阶二元量化
                    output_matrix[:, j][self.integrator1 >= cut_threshold] = 1
                    output_matrix[:, j][self.integrator1 <= cut_threshold] = -1
            elif self.order == 2:#仅调试
                # 二阶 Sigma-Delta 调制
                self.integrator1 += integrator_input
                self.integrator2 += self.integrator1 - self.previous_output
                if not self.first2:  # 二阶三元量化
                    output_matrix[:, j][self.integrator2 >= 1 / 3] = 1
                    output_matrix[:, j][self.integrator2 <= -1 / 3] = -1
                else:  # 二阶二元量化
                    output_matrix[:, j][self.integrator2 >= 0] = 1
                    output_matrix[:, j][self.integrator2 <= 0] = -1

            # 更新前一时刻的输出
            self.previous_output = output_matrix[:, j]

        # 反归一化
        output_matrix = output_matrix *scale+mean_vals #(max_vals - min_vals)
        return output_matrix


def sdm_quant(weight,OSR=2,scale=0.9,percentile_scale=0.9,percentile_cut=0.5,first2=False):
    orgshape=weight.shape
    if len(weight.shape)==1:
        weight=weight.unsqueeze(0)
    elif len(weight.shape)>2:
        raise ValueError("weight shape should be 2")
    #temp=weight.clone()
    weight=extend(weight, OSR)
    sdm_quantizer = SigmaDeltaQuantizer(first2=first2, sd_scale=scale, order=1)
    weight=sdm_quantizer.quantize(weight,percentile_1=percentile_scale,percentile_2=percentile_cut)
    weight=resample(weight, orgshape[-1],axis=-1)
    #mse_loss = torch.mean((temp-weight)**2)
    return weight.reshape(orgshape)

