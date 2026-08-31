import warnings

warnings.filterwarnings("ignore")
import math

import torch
from torch import nn
from torch.nn import functional as F

from module import commons
from module import modules
from module import attentions
from f5_tts.model import DiT
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm
from module.commons import init_weights
from module.mrte_model import MRTE
from module.quantize import ResidualVectorQuantizer

# from text import symbols
from text import symbols as symbols_v1
from text import symbols2 as symbols_v2
from torch.cuda.amp import autocast
import contextlib
import random


class StochasticDurationPredictor(nn.Module):
    def __init__(
        self,
        in_channels,
        filter_channels,
        kernel_size,
        p_dropout,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()
        filter_channels = in_channels  # it needs to be removed from future version.
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.log_flow = modules.Log()
        self.flows = nn.ModuleList()
        self.flows.append(modules.ElementwiseAffine(2))
        for i in range(n_flows):
            self.flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.flows.append(modules.Flip())

        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        self.post_flows = nn.ModuleList()
        self.post_flows.append(modules.ElementwiseAffine(2))
        for i in range(4):
            self.post_flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.post_flows.append(modules.Flip())

        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

    def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
        x = torch.detach(x)
        x = self.pre(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.convs(x, x_mask)
        x = self.proj(x) * x_mask

        if not reverse:
            flows = self.flows
            assert w is not None

            logdet_tot_q = 0
            h_w = self.post_pre(w)
            h_w = self.post_convs(h_w, x_mask)
            h_w = self.post_proj(h_w) * x_mask
            e_q = torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype) * x_mask
            z_q = e_q
            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
                logdet_tot_q += logdet_q
            z_u, z1 = torch.split(z_q, [1, 1], 1)
            u = torch.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask
            logdet_tot_q += torch.sum((F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2])
            logq = torch.sum(-0.5 * (math.log(2 * math.pi) + (e_q**2)) * x_mask, [1, 2]) - logdet_tot_q

            logdet_tot = 0
            z0, logdet = self.log_flow(z0, x_mask)
            logdet_tot += logdet
            z = torch.cat([z0, z1], 1)
            for flow in flows:
                z, logdet = flow(z, x_mask, g=x, reverse=reverse)
                logdet_tot = logdet_tot + logdet
            nll = torch.sum(0.5 * (math.log(2 * math.pi) + (z**2)) * x_mask, [1, 2]) - logdet_tot
            return nll + logq  # [b]
        else:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
            z = torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype) * noise_scale
            for flow in flows:
                z = flow(z, x_mask, g=x, reverse=reverse)
            z0, z1 = torch.split(z, [1, 1], 1)
            logw = z0
            return logw


class DurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = modules.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = modules.LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    def forward(self, x, x_mask, g=None):
        x = torch.detach(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask


WINDOW = {}

class TextEncoder(nn.Module):
    def __init__(
        self,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        latent_channels=192,
        version="v2",
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.latent_channels = latent_channels
        self.version = version

        self.ssl_proj = nn.Conv1d(768, hidden_channels, 1)

        self.encoder_ssl = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers // 2,
            kernel_size,
            p_dropout,
        )

        self.encoder_text = attentions.Encoder(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )

        if self.version == "v1":
            symbols = symbols_v1.symbols
        else:
            symbols = symbols_v2.symbols
        self.text_embedding = nn.Embedding(len(symbols), hidden_channels)

        self.mrte = MRTE()

        self.encoder2 = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers // 2,
            kernel_size,
            p_dropout,
        )

        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, y, y_lengths, text, text_lengths, ge, speed=1, test=None, result_length:int=None, overlap_frames:torch.Tensor=None, padding_length:int=None):
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y.size(2)), 1).to(y.dtype)

        y = self.ssl_proj(y * y_mask) * y_mask

        y = self.encoder_ssl(y * y_mask, y_mask)

        text_mask = torch.unsqueeze(commons.sequence_mask(text_lengths, text.size(1)), 1).to(y.dtype)
        if test == 1:
            text[:, :] = 0
        text = self.text_embedding(text).transpose(1, 2)
        text = self.encoder_text(text * text_mask, text_mask)
        y = self.mrte(y, y_mask, text, text_mask, ge)

        if padding_length is not None and padding_length!=0:
            y = y[:, :, :-padding_length]
            y_mask = y_mask[:, :, :-padding_length]


        y = self.encoder2(y * y_mask, y_mask)

        if result_length is not None:
            y = y[:, :, -result_length:]
            y_mask = y_mask[:, :, -result_length:]

        if overlap_frames is not None:
            overlap_len = overlap_frames.shape[-1]
            window = WINDOW.get(overlap_len, None)
            if window is None:
                # WINDOW[overlap_len] = torch.hann_window(overlap_len*2, device=y.device, dtype=y.dtype)
                WINDOW[overlap_len] = torch.sin(torch.arange(overlap_len*2, device=y.device) * torch.pi / (overlap_len*2))
                window = WINDOW[overlap_len]


            window = window.to(y.device)
            y[:,:,:overlap_len] = (
                window[:overlap_len].view(1, 1, -1) * y[:,:,:overlap_len]
                + window[overlap_len:].view(1, 1, -1) * overlap_frames
            )
            
        y_ = y
        y_mask_ = y_mask



        if speed != 1:
            y = F.interpolate(y, size=int(y.shape[-1] / speed) + 1, mode="linear")
            y_mask = F.interpolate(y_mask, size=y.shape[-1], mode="nearest")
        stats = self.proj(y) * y_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return y, m, logs, y_mask, y_, y_mask_

    def extract_latent(self, x):
        x = self.ssl_proj(x)
        quantized, codes, commit_loss, quantized_list = self.quantizer(x)
        return codes.transpose(0, 1)

    def decode_latent(self, codes, y_mask, refer, refer_mask, ge):
        quantized = self.quantizer.decode(codes)

        y = self.vq_proj(quantized) * y_mask
        y = self.encoder_ssl(y * y_mask, y_mask)

        y = self.mrte(y, y_mask, refer, refer_mask, ge)

        y = self.encoder2(y * y_mask, y_mask)

        stats = self.proj(y) * y_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return y, m, logs, y_mask, quantized


class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        if g != None:
            g = g.detach()
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class Encoder(nn.Module):
    def __init__(
        self, in_channels, out_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x, x_lengths, g=None):
        if g != None:
            g = g.detach()
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        return stats, x_mask


class WNEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.norm = modules.LayerNorm(out_channels)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        out = self.proj(x) * x_mask
        out = self.norm(out)
        return out


class Generator(torch.nn.Module):
    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels=0,
        is_bias=False,
    ):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = modules.ResBlock1 if resblock == "1" else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=is_bias)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


v2pro_set = {"v2Pro", "v2ProPlus"}


class SynthesizerTrn(nn.Module):

    def __init__(
        self,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        n_speakers=0,
        gin_channels=0,
        use_sdp=True,
        semantic_frame_rate=None,
        freeze_quantizer=None,
        version="v2",
        **kwargs,
    ):
        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.version = version

        self.use_sdp = use_sdp
        self.enc_p = TextEncoder(
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            version=version,
        )
        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)

        # self.version=os.environ.get("version","v1")
        if self.version == "v1":
            self.ref_enc = modules.MelStyleEncoder(spec_channels, style_vector_dim=gin_channels)
        else:
            self.ref_enc = modules.MelStyleEncoder(704, style_vector_dim=gin_channels)

        ssl_dim = 768
        assert semantic_frame_rate in ["25hz", "50hz"]
        self.semantic_frame_rate = semantic_frame_rate
        if semantic_frame_rate == "25hz":
            self.ssl_proj = nn.Conv1d(ssl_dim, ssl_dim, 2, stride=2)
        else:
            self.ssl_proj = nn.Conv1d(ssl_dim, ssl_dim, 1, stride=1)

        self.quantizer = ResidualVectorQuantizer(dimension=ssl_dim, n_q=1, bins=1024)
        self.freeze_quantizer = freeze_quantizer

        self.is_v2pro = self.version in v2pro_set
        if self.is_v2pro:
            self.sv_emb = nn.Linear(20480, gin_channels)
            self.ge_to512 = nn.Linear(gin_channels, 512)
            self.prelu = nn.PReLU(num_parameters=gin_channels)

    def forward(self, ssl, y, y_lengths, text, text_lengths, sv_emb=None):
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y.size(2)), 1).to(y.dtype)
        if self.version == "v1":
            ge = self.ref_enc(y * y_mask, y_mask)
        else:
            ge = self.ref_enc(y[:, :704] * y_mask, y_mask)
        if self.is_v2pro:
            sv_emb = self.sv_emb(sv_emb)  # B*20480->B*512
            ge += sv_emb.unsqueeze(-1)
            ge = self.prelu(ge)
            ge512 = self.ge_to512(ge.transpose(2, 1)).transpose(2, 1)
        with autocast(enabled=False):
            maybe_no_grad = torch.no_grad() if self.freeze_quantizer else contextlib.nullcontext()
            with maybe_no_grad:
                if self.freeze_quantizer:
                    self.ssl_proj.eval()
                    self.quantizer.eval()
            ssl = self.ssl_proj(ssl)
            quantized, codes, commit_loss, quantized_list = self.quantizer(ssl, layers=[0])

        if self.semantic_frame_rate == "25hz":
            quantized = F.interpolate(quantized, size=int(quantized.shape[-1] * 2), mode="nearest")

        x, m_p, logs_p, y_mask, _, _ = self.enc_p(quantized, y_lengths, text, text_lengths, ge512 if self.is_v2pro else ge)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=ge)
        z_p = self.flow(z, y_mask, g=ge)

        z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths, self.segment_size)
        o = self.dec(z_slice, g=ge)
        return (
            o,
            commit_loss,
            ids_slice,
            y_mask,
            y_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
            quantized,
        )

    def infer(self, ssl, y, y_lengths, text, text_lengths, test=None, noise_scale=0.5):
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y.size(2)), 1).to(y.dtype)
        if self.version == "v1":
            ge = self.ref_enc(y * y_mask, y_mask)
        else:
            ge = self.ref_enc(y[:, :704] * y_mask, y_mask)

        ssl = self.ssl_proj(ssl)
        quantized, codes, commit_loss, _ = self.quantizer(ssl, layers=[0])
        if self.semantic_frame_rate == "25hz":
            quantized = F.interpolate(quantized, size=int(quantized.shape[-1] * 2), mode="nearest")

        x, m_p, logs_p, y_mask, _, _ = self.enc_p(quantized, y_lengths, text, text_lengths, ge, test=test)
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

        z = self.flow(z_p, y_mask, g=ge, reverse=True)

        o = self.dec((z * y_mask)[:, :, :], g=ge)
        return o, y_mask, (z, z_p, m_p, logs_p)


    @torch.no_grad()
    def decode(self, codes, text, refer, noise_scale=0.5, speed=1, sv_emb=None):
        def get_ge(refer, sv_emb):
            ge = None
            if refer is not None:
                refer_lengths = torch.LongTensor([refer.size(2)]).to(refer.device)
                refer_mask = torch.unsqueeze(commons.sequence_mask(refer_lengths, refer.size(2)), 1).to(refer.dtype)
                if self.version == "v1":
                    ge = self.ref_enc(refer * refer_mask, refer_mask)
                else:
                    ge = self.ref_enc(refer[:, :704] * refer_mask, refer_mask)
                if self.is_v2pro:
                    sv_emb = self.sv_emb(sv_emb)  # B*20480->B*512
                    ge += sv_emb.unsqueeze(-1)
                    ge = self.prelu(ge)
            return ge

        if type(refer) == list:
            ges = []
            for idx, _refer in enumerate(refer):
                ge = get_ge(_refer, sv_emb[idx] if self.is_v2pro else None)
                ges.append(ge)
            ge = torch.stack(ges, 0).mean(0)
        else:
            ge = get_ge(refer, sv_emb)

        y_lengths = torch.LongTensor([codes.size(2) * 2]).to(codes.device)
        text_lengths = torch.LongTensor([text.size(-1)]).to(text.device)

        quantized = self.quantizer.decode(codes)
        if self.semantic_frame_rate == "25hz":
            quantized = F.interpolate(quantized, size=int(quantized.shape[-1] * 2), mode="nearest")
        x, m_p, logs_p, y_mask, _, _ = self.enc_p(
            quantized,
            y_lengths,
            text,
            text_lengths,
            self.ge_to512(ge.transpose(2, 1)).transpose(2, 1) if self.is_v2pro else ge,
            speed,
        )
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

        z = self.flow(z_p, y_mask, g=ge, reverse=True)

        o = self.dec((z * y_mask)[:, :, :], g=ge)
        return o


    @torch.no_grad()
    def decode_streaming(self, codes, text, refer, noise_scale=0.5, speed=1, sv_emb=None, result_length:int=None, overlap_frames:torch.Tensor=None, padding_length:int=None):
        def get_ge(refer, sv_emb):
            ge = None
            if refer is not None:
                refer_lengths = torch.LongTensor([refer.size(2)]).to(refer.device)
                refer_mask = torch.unsqueeze(commons.sequence_mask(refer_lengths, refer.size(2)), 1).to(refer.dtype)
                if self.version == "v1":
                    ge = self.ref_enc(refer * refer_mask, refer_mask)
                else:
                    ge = self.ref_enc(refer[:, :704] * refer_mask, refer_mask)
                if self.is_v2pro:
                    sv_emb = self.sv_emb(sv_emb)  # B*20480->B*512
                    ge += sv_emb.unsqueeze(-1)
                    ge = self.prelu(ge)
            return ge

        if type(refer) == list:
            ges = []
            for idx, _refer in enumerate(refer):
                ge = get_ge(_refer, sv_emb[idx] if self.is_v2pro else None)
                ges.append(ge)
            ge = torch.stack(ges, 0).mean(0)
        else:
            ge = get_ge(refer, sv_emb)

        y_lengths = torch.LongTensor([codes.size(2) * 2]).to(codes.device)
        text_lengths = torch.LongTensor([text.size(-1)]).to(text.device)

        quantized = self.quantizer.decode(codes)
        if self.semantic_frame_rate == "25hz":
            quantized = F.interpolate(quantized, size=int(quantized.shape[-1] * 2), mode="nearest")
            result_length = (2*result_length) if result_length is not None else None
            padding_length = (2*padding_length) if padding_length is not None else None
        x, m_p, logs_p, y_mask, y_, y_mask_ = self.enc_p(
            quantized,
            y_lengths,
            text,
            text_lengths,
            self.ge_to512(ge.transpose(2, 1)).transpose(2, 1) if self.is_v2pro else ge,
            speed,
            result_length=result_length, 
            overlap_frames=overlap_frames, 
            padding_length=padding_length
            )
        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale

        z = self.flow(z_p, y_mask, g=ge, reverse=True)

        o = self.dec((z * y_mask)[:, :, :], g=ge)
        return o, y_, y_mask_

    def extract_latent(self, x):
        ssl = self.ssl_proj(x)
        quantized, codes, commit_loss, quantized_list = self.quantizer(ssl)
        return codes.transpose(0, 1)


class CFM(torch.nn.Module):
    def __init__(self, in_channels, dit):
        super().__init__()
        self.sigma_min = 1e-6

        self.estimator = dit

        self.in_channels = in_channels

        self.criterion = torch.nn.MSELoss()

        self.use_conditioner_cache = True

    @torch.inference_mode()
    def inference(self, mu, x_lens, prompt, n_timesteps, temperature=1.0, inference_cfg_rate=0):
        """Forward diffusion"""
        B, T = mu.size(0), mu.size(1)
        x = torch.randn([B, self.in_channels, T], device=mu.device, dtype=mu.dtype) * temperature
        prompt_len = prompt.size(-1)
        prompt_x = torch.zeros_like(x, dtype=mu.dtype)
        prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
        x[..., :prompt_len] = 0
        mu = mu.transpose(2, 1)
        t = 0
        d = 1 / n_timesteps
        text_cache = None
        text_cfg_cache = None
        dt_cache = None
        d_tensor = torch.ones(x.shape[0], device=x.device, dtype=mu.dtype) * d
        for j in range(n_timesteps):
            t_tensor = torch.ones(x.shape[0], device=x.device, dtype=mu.dtype) * t
            # v_pred = model(x, t_tensor, d_tensor, **extra_args)
            v_pred, text_emb, dt = self.estimator(
                x,
                prompt_x,
                x_lens,
                t_tensor,
                d_tensor,
                mu,
                use_grad_ckpt=False,
                drop_audio_cond=False,
                drop_text=False,
                infer=True,
                text_cache=text_cache,
                dt_cache=dt_cache,
            )
            v_pred = v_pred.transpose(2, 1)
            if self.use_conditioner_cache:
                text_cache = text_emb
                dt_cache = dt
            if inference_cfg_rate > 1e-5:
                neg, text_cfg_emb, _ = self.estimator(
                    x,
                    prompt_x,
                    x_lens,
                    t_tensor,
                    d_tensor,
                    mu,
                    use_grad_ckpt=False,
                    drop_audio_cond=True,
                    drop_text=True,
                    infer=True,
                    text_cache=text_cfg_cache,
                    dt_cache=dt_cache,
                )
                neg = neg.transpose(2, 1)
                if self.use_conditioner_cache:
                    text_cfg_cache = text_cfg_emb
                v_pred = v_pred + (v_pred - neg) * inference_cfg_rate
            x = x + d * v_pred
            t = t + d
            x[:, :, :prompt_len] = 0
        return x

    def forward(self, x1, x_lens, prompt_lens, mu, use_grad_ckpt):
        b, _, t = x1.shape
        t = torch.rand([b], device=mu.device, dtype=x1.dtype)
        x0 = torch.randn_like(x1, device=mu.device)
        vt = x1 - x0
        xt = x0 + t[:, None, None] * vt
        dt = torch.zeros_like(t, device=mu.device)
        prompt = torch.zeros_like(x1)
        for i in range(b):
            prompt[i, :, : prompt_lens[i]] = x1[i, :, : prompt_lens[i]]
            xt[i, :, : prompt_lens[i]] = 0
        gailv = 0.3  # if ttime()>1736250488 else 0.1
        if random.random() < gailv:
            base = torch.randint(2, 8, (t.shape[0],), device=mu.device)
            d = 1 / torch.pow(2, base)
            d_input = d.clone()
            d_input[d_input < 1e-2] = 0
            # with torch.no_grad():
            v_pred_1 = self.estimator(xt, prompt, x_lens, t, d_input, mu, use_grad_ckpt).transpose(2, 1).detach()
            # v_pred_1 = self.diffusion(xt, t, d_input, cond=conditioning).detach()
            x_mid = xt + d[:, None, None] * v_pred_1
            # v_pred_2 = self.diffusion(x_mid, t+d, d_input, cond=conditioning).detach()
            v_pred_2 = self.estimator(x_mid, prompt, x_lens, t + d, d_input, mu, use_grad_ckpt).transpose(2, 1).detach()
            vt = (v_pred_1 + v_pred_2) / 2
            vt = vt.detach()
            dt = 2 * d

        vt_pred = self.estimator(xt, prompt, x_lens, t, dt, mu, use_grad_ckpt).transpose(2, 1)
        loss = 0
        for i in range(b):
            loss += self.criterion(vt_pred[i, :, prompt_lens[i] : x_lens[i]], vt[i, :, prompt_lens[i] : x_lens[i]])
        loss /= b

        return loss


def set_no_grad(net_g):
    for name, param in net_g.named_parameters():
        param.requires_grad = False


class SynthesizerTrnV3(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        n_speakers=0,
        gin_channels=0,
        use_sdp=True,
        semantic_frame_rate=None,
        freeze_quantizer=None,
        version="v3",
        **kwargs,
    ):
        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.version = version

        self.model_dim = 512
        self.use_sdp = use_sdp
        self.enc_p = TextEncoder(
            inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )
        # self.ref_enc = modules.MelStyleEncoder(spec_channels, style_vector_dim=gin_channels)###Rollback
        self.ref_enc = modules.MelStyleEncoder(704, style_vector_dim=gin_channels)  ###Rollback
        # self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates,
        #                      upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels)
        # self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16,
        #                               gin_channels=gin_channels)
        # self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)

        ssl_dim = 768
        assert semantic_frame_rate in ["25hz", "50hz"]
        self.semantic_frame_rate = semantic_frame_rate
        if semantic_frame_rate == "25hz":
            self.ssl_proj = nn.Conv1d(ssl_dim, ssl_dim, 2, stride=2)
        else:
            self.ssl_proj = nn.Conv1d(ssl_dim, ssl_dim, 1, stride=1)

        self.quantizer = ResidualVectorQuantizer(dimension=ssl_dim, n_q=1, bins=1024)
        self.freeze_quantizer = freeze_quantizer
        inter_channels2 = 512
        self.bridge = nn.Sequential(nn.Conv1d(inter_channels, inter_channels2, 1, stride=1), nn.LeakyReLU())
        self.wns1 = Encoder(inter_channels2, inter_channels2, inter_channels2, 5, 1, 8, gin_channels=gin_channels)
        self.linear_mel = nn.Conv1d(inter_channels2, 100, 1, stride=1)
        self.cfm = CFM(
            100,
            DiT(**dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=inter_channels2, conv_layers=4)),
        )  # text_dim is condition feature dim
        if self.freeze_quantizer == True:
            set_no_grad(self.ssl_proj)
            set_no_grad(self.quantizer)
            set_no_grad(self.enc_p)

    def forward(
        self, ssl, y, mel, ssl_lengths, y_lengths, text, text_lengths, mel_lengths, use_grad_ckpt
    ):  # ssl_lengths no need now
        with autocast(enabled=False):
            y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y.size(2)), 1).to(y.dtype)
            ge = self.ref_enc(y[:, :704] * y_mask, y_mask)
            maybe_no_grad = torch.no_grad() if self.freeze_quantizer else contextlib.nullcontext()
            with maybe_no_grad:
                if self.freeze_quantizer:
                    self.ssl_proj.eval()  #
                    self.quantizer.eval()
                    self.enc_p.eval()
                ssl = self.ssl_proj(ssl)
                quantized, codes, commit_loss, quantized_list = self.quantizer(ssl, layers=[0])
                quantized = F.interpolate(quantized, scale_factor=2, mode="nearest")  ##BCT
                x, m_p, logs_p, y_mask, y_, y_mask_ = self.enc_p(quantized, y_lengths, text, text_lengths, ge)
        fea = self.bridge(x)
        fea = F.interpolate(fea, scale_factor=(1.875 if self.version == "v3" else 2), mode="nearest")  ##BCT
        fea, y_mask_ = self.wns1(
            fea, mel_lengths, ge
        )  ##If the 1-minute fine-tuning works fine, no need to manually adjust the learning rate.
        B = ssl.shape[0]
        prompt_len_max = mel_lengths * 2 / 3
        prompt_len = (torch.rand([B], device=fea.device) * prompt_len_max).floor().to(dtype=torch.long)
        minn = min(mel.shape[-1], fea.shape[-1])
        mel = mel[:, :, :minn]
        fea = fea[:, :, :minn]
        cfm_loss = self.cfm(mel, mel_lengths, prompt_len, fea, use_grad_ckpt)
        return cfm_loss

    @torch.no_grad()
    def decode_encp(self, codes, text, refer, ge=None, speed=1):
        # print(2333333,refer.shape)
        # ge=None
        if ge == None:
            refer_lengths = torch.LongTensor([refer.size(2)]).to(refer.device)
            refer_mask = torch.unsqueeze(commons.sequence_mask(refer_lengths, refer.size(2)), 1).to(refer.dtype)
            ge = self.ref_enc(refer[:, :704] * refer_mask, refer_mask)
        y_lengths = torch.LongTensor([int(codes.size(2) * 2)]).to(codes.device)
        if speed == 1:
            sizee = int(codes.size(2) * (3.875 if self.version == "v3" else 4))
        else:
            sizee = int(codes.size(2) * (3.875 if self.version == "v3" else 4) / speed) + 1
        y_lengths1 = torch.LongTensor([sizee]).to(codes.device)
        text_lengths = torch.LongTensor([text.size(-1)]).to(text.device)

        quantized = self.quantizer.decode(codes)
        if self.semantic_frame_rate == "25hz":
            quantized = F.interpolate(quantized, scale_factor=2, mode="nearest")  ##BCT
        x, m_p, logs_p, y_mask, _, _ = self.enc_p(quantized, y_lengths, text, text_lengths, ge, speed)
        fea = self.bridge(x)
        fea = F.interpolate(fea, scale_factor=(1.875 if self.version == "v3" else 2), mode="nearest")  ##BCT
        ####more wn paramter to learn mel
        fea, y_mask_ = self.wns1(fea, y_lengths1, ge)
        return fea, ge

    def extract_latent(self, x):
        ssl = self.ssl_proj(x)
        quantized, codes, commit_loss, quantized_list = self.quantizer(ssl)
        return codes.transpose(0, 1)
