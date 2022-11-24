import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict
from .utils import MetricRecorder
from .dataset import StockDataDict

class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.lnorm = nn.LayerNorm(hidden_size)
    
    def forward(self, x: torch.Tensor):
        # x: (B, T, I)
        o, (h, _) = self.lstm(x) # o: (B, T, H) / h: (1, B, H)
        normed_context = self.lnorm(h[-1, :, :])
        return normed_context

class LSTMAttention(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.lnorm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, rt_attn: bool=False):
        # x: (B, T, I)
        o, (h, _) = self.lstm(x) # o: (B, T, H) / h: (1, B, H)
        h = h[-1, :, :]  # (B, H)
        score = torch.bmm(o, h.unsqueeze(-1)) # (B, T, H) x (B, H, 1)
        attn = torch.softmax(score, 1).squeeze(-1)  # (B, T)
        context = torch.bmm(attn.unsqueeze(1), o).squeeze(1)  # (B, 1, T) x (B, T, H)
        normed_context = self.lnorm(context)  # (B, H)
        if rt_attn:
            return normed_context, attn
        else:
            return normed_context, None

class RelationNet(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.rn = nn.Sequential(
            nn.Linear(2*hidden_size, 2*hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(2*hidden_size, 2*hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(2*hidden_size, 2*hidden_size, bias=False),
            nn.ReLU()
        )

    def forward(self, encoded: torch.Tensor):
        # encoded: (B, N, K, H)
        B, N, K, _ = encoded.size()
        left = torch.repeat_interleave(encoded, K, dim=2)
        left = torch.repeat_interleave(left, N, dim=1)
        right = encoded.repeat((1, N, K, 1))
        x = torch.cat([left, right], dim=-1)  # x: (B, N^2, K^2, 2H) 
        o = self.rn(x)
        o = o.view(B, N, N*K*K, -1).mean(2)  # o: (B, N, 2H)
        return o  

class MetaModel(nn.Module):
    """Meta Model
    Structure Ref: 
    * LEO-Deepmind: https://github.com/deepmind/leo/blob/de9a0c2a77dd7a42c1986b1eef18d184a86e294a/model.py#L256
    * LEO-pytorch: https://github.com/timchen0618/pytorch-leo/blob/master/model.py
    """
    def __init__(
            self, 
            feature_size: int, 
            embed_size: int,
            n_classes: int,  # only should be number of classes
            num_layers: int, 
            drop_rate: float, 
            inner_lr_init: float,
            # inner_lr_schedular_gamma: float,
            param_l2_lambda: float,
            device: str
        ):
        super().__init__()
        self.embed_size = embed_size
        self.output_size = n_classes
        
        self.inner_lr = nn.Parameter(torch.FloatTensor([inner_lr_init])) # inner_lr_init
        # self.inner_lr_schedular_gamma = inner_lr_schedular_gamma
        self.param_l2_lambda = param_l2_lambda
        # Network
        self.dropout = nn.Dropout(drop_rate)
        self.lstm_encoder = LSTMAttention(input_size=feature_size, hidden_size=embed_size, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(embed_size)
        self.encoder = nn.Sequential(
            nn.Linear(embed_size, 2*embed_size),
            nn.ELU()
        )
        # self.relation_net = RelationNet(hidden_size)
        self.decoder = nn.Linear(embed_size, 2*embed_size, bias=False)

        # Loss
        # self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn = nn.NLLLoss()

        # Meta Mode Support / Query
        self._mode_query(False)

        # Recoder
        self.recorder = MetricRecorder().to(device)

    def meta_train(self):
        self.train()

    def meta_eval(self):
        self.manual_model_eval()

    def _mode_query(self, mode: bool=True):
        # check if is support of query
        self.is_query = mode

    def manual_model_eval(self, mode: bool=False):
        """
        [PyTorch Issue] RuntimeError: cudnn RNN backward can only be called in training mode
        cannot use `model.eval()`. 
        see https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch
        """
        for module in self.children():
            self.training = mode
            if isinstance(module, nn.Dropout) or isinstance(module, nn.LayerNorm):
                module.train(mode)

    def encode_lstm(self, inputs: torch.Tensor, rt_attn: bool=False):
        """forward data by each stock to avoid trained by other stocks
        - B: number of samples
        - N: number of classes=`output_size`
        - K: number of `n_support`/`n_query`
        - T: window size
        - I: input size
        - E: embedding size
        - M: M = N * K

        Args:
            inputs: (B, N*K, T, I). single stock data.
            - support: (B, N*K, T, I)
            - query: (B, 1, T, I)

        Returns:
            encoded: (B, N*K, E)
            attn: (B, N*K, T)
        """
        B, M, T, I = inputs.size()
        inputs = inputs.view(B*M, T, I)  # (B*M, T, I) 
        inputs = self.dropout(inputs)
        encoded, attn = self.lstm_encoder(inputs, rt_attn)  # (B*N*K, E), (B*N*K, T)
        encoded = self.layer_norm(encoded)
        encoded = encoded.view(B, M, -1)  # (B, N*K, E)
        if rt_attn:
            attn = attn.view(B, M, -1)  # (B, N*K, T)
        return encoded, attn

    def forward_encoder(self, inputs: torch.Tensor, rt_attn: bool=False):
        """Forward Encoder: from `inputs` to `z`
        - B: number of samples
        - N: number of classes=`output_size`
        - K: number of `n_support`/`n_query`
        - T: window size
        - E: embedding size
        - H: hidden size

        Args:
            inputs: (B, N*K, T, I). single stock data.
            - support: (B, N*K, T, I)
            - query: (B, 1, T, I)

        Returns:
            l: (B, N*K, E).
            z: (B, N, H). sampled parameters for each class.
            kld_loss: (1,). KL-Divergence Loss between N($\mu_n^e$, $\sigma_n^e$) and N(0, 1).
            attn: (B, N, K, T). attention weights for each inputs.
        
        """
        # support l: (B, N*K, E), attn: (B, N*K, T)
        # query l: (B, 1, E), attn: (B, 1, T)
        l, attn = self.encode_lstm(inputs, rt_attn=rt_attn)  
        B, M, _ = l.size()
        if M > 1:
            # Reshape the size
            N = self.output_size  # number of classes
            K = M // N
            # forward support
            if rt_attn:
                attn = attn.view(B, N, K, -1)  # attn: (B, N, K, T)
            l_reshape = l.view(B, N, K, -1)  # l_reshape: (B, N, K, E)
            # -----------------------------
            # Encoder-to-z
            e = self.encoder(l_reshape)  # e: (B, N, K, 2E)
            # hs = self.relation_net(e)  # hs: (B, N, 2H)
            # -----------------------------
            hs = e.mean(2)  # hs: (B, N, 2E)
            z, kld_loss = self.sample(hs, size=self.embed_size)  # z: (B, N, E)
            return l, z, kld_loss, attn
        else:
            # forward query
            return l, None, None, attn


    def cal_kl_div(self, dist: torch.distributions.Normal, z: torch.Tensor):
        normal = torch.distributions.Normal(torch.zeros_like(z), torch.ones_like(z))
        return torch.mean(dist.log_prob(z) - normal.log_prob(z))

    def sample(self, distribution_params: torch.Tensor, size: int, std_offset: float=0.0):
        """parameters of a probability distribution in a low-dimensional space `z` for each class"""
        mean, log_std = torch.split(distribution_params, split_size_or_sections=size, dim=-1)
        if self.is_query:
            return mean, torch.zeros((1,)).to(mean.device)
        std = torch.exp(log_std)
        std = std - (1. - std_offset)
        std = torch.maximum(std, torch.tensor([1e-10]).to(std.device))
        dist = torch.distributions.Normal(mean, std)
        z = dist.rsample()
        kld_loss = self.cal_kl_div(dist, z)
        return z, kld_loss

    def decode(self, z: torch.Tensor):
        """decode from `z` to `parameters`
        - B: number of samples
        - N: number of classes=`output_size`
        - E: embedding size
        - H: hidden size

        Args:
            z: (B, N, E). sampled parameters for each class.

        Returns:
            parameters: (B, N, E). $\theta$
        """
        param_hs = self.decoder(z)  # param_hs: (B, N, 2E)
        fan_in = self.embed_size  # E
        fan_out = self.output_size  # N
        std_offset = np.sqrt(2. / (fan_out + fan_in))
        parameters, _ = self.sample(param_hs, size=self.embed_size, std_offset=std_offset)  # (B, N, E)
        return parameters

    def predict(self, l: torch.Tensor, parameters: torch.Tensor, labels: torch.Tensor):
        """
        l: (B, N*K, E) for support, (B, 1, E) for query
        parameters: (B, N, E)
        """
        N = parameters.size(1)
        theta = parameters.permute((0, 2, 1))  # (B, N, E) -> (B, E, N)
        # support: (B, N*K, E) x (B, E, N) = (B, N*K, N) -> (B*N*K, N)
        # query: (B, 1, E) x (B, E, N) = (B, 1, N) -> (B, N)
        scores = l.bmm(theta).squeeze(1).view(-1, N)
        probs = torch.log_softmax(scores, -1)  # (B, N)
        loss = self.loss_fn(probs, labels)
        param_l2_loss = (parameters**2).mean()
        return loss, probs.argmax(1), param_l2_loss

    def forward_decoder(self, z: torch.Tensor, l: torch.Tensor, labels: torch.Tensor):
        """Decoder
        - B: number of samples
        - N: number of classes=`output_size`
        - E: embedding size
        - H: hidden size

        Args:
            l: (B, N*K, E) for support, (B, 1, E) for query
            z: (B, N, H). sampled parameters for each class.

        Returns:
            loss: (1,).
            preds: (B,).
            parameters: (B, N, E).
        """
        parameters = self.decode(z)  # parameters: (B, N, E)
        pred_loss, preds, param_l2_loss = self.predict(l, parameters, labels)
        return pred_loss, param_l2_loss, preds, parameters

    def inner_loop(
            self, data: Dict[str, torch.Tensor], 
            n_inner_step: int=5, 
            rt_attn: bool=False
        ):
        self._mode_query(False)
        s_inputs = data['support']
        s_labels = data['support_labels']

        # Forward Encoder
        s_l, s_z, kld_loss, s_attn = self.forward_encoder(s_inputs, rt_attn=rt_attn)

        # initialize z', Forward Decoder
        z_prime = s_z
        s_pred_loss, s_param_l2_loss, s_preds, parameters = self.forward_decoder(z=z_prime, l=s_l, labels=s_labels)
        s_loss = s_pred_loss + self.param_l2_lambda * s_param_l2_loss

        # Check the gradient flow: must set requires_grad to True to use optimizer
        # print(f's_z: {s_z.requires_grad} | z_prime = {z_prime.requires_grad}')
        # print(f's_z: {s_z.is_leaf} | z_prime = {z_prime.is_leaf}')
        # s_z: True | z_prime = True
        # s_z: False | z_prime = True

        # inner adaptation to z
        # inner_optimizer = torch.optim.SGD([z_prime], lr=float(self.inner_lr))
        # inner_scheduler = torch.optim.lr_scheduler.ExponentialLR(inner_optimizer, gamma=self.inner_lr_schedular_gamma)
        # inner_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     inner_optimizer, T_0=math.floor(n_inner_step/4), T_mult=2, eta_min=0.001)
        for i in range(n_inner_step):
            # inner_optimizer.zero_grad()
            # equal to: 
            z_prime.retain_grad()
            s_loss.backward(retain_graph=True)
            # inner_optimizer.step()
            # inner_scheduler.step()
            # similar to: z_prime = z_prime - scheduler(self.inner_lr) * z_prime.grad.data
            z_prime = z_prime - self.inner_lr * z_prime.grad.data
            s_pred_loss, s_param_l2_loss, s_preds, parameters = self.forward_decoder(z=z_prime, l=s_l, labels=s_labels)
            s_loss = s_pred_loss + self.param_l2_lambda * s_param_l2_loss

        # Stop Gradient: 
        z_prime = z_prime.detach()
        z_loss = torch.mean((z_prime - s_z)**2)
        # Check the gradient flow
        # print(f's_z: {s_z.requires_grad} | z_prime = {z_prime.requires_grad}')
        # print(f's_z: {s_z.is_leaf} | z_prime = {z_prime.is_leaf}')
        # s_z: True | z_prime = False
        # s_z: False | z_prime = True
        
        # Record
        self.recorder.update('Support_Loss', s_loss)
        self.recorder.update('Support_ParamL2Loss', s_param_l2_loss)
        self.recorder.update('Support_PredLoss', s_pred_loss)
        self.recorder.update('Support_Accuracy', s_preds, s_labels)
        self.recorder.update('Support_Precision', s_preds, s_labels)
        self.recorder.update('Support_Recall', s_preds, s_labels)
        self.recorder.update('InnerLearningRate', float(self.inner_lr))
        self.recorder.update('Z_Loss', z_loss)
        self.recorder.update('KLD_Loss', kld_loss)

        return parameters, kld_loss, z_loss, s_attn

    def query_validate(
            self, data: Dict[str, torch.Tensor], 
            parameters: torch.Tensor, 
            rt_attn: bool=False
        ):
        self._mode_query(True)
        q_inputs = data['query']
        q_labels = data['query_labels']
        
        q_l, *_, q_attn = self.forward_encoder(q_inputs, rt_attn=rt_attn)
        q_pred_loss, q_preds, q_param_l2_loss = self.predict(
            l=q_l, parameters=parameters, labels=q_labels
        )
        q_loss = q_pred_loss + self.param_l2_lambda * q_param_l2_loss
        # Record
        self.recorder.update('Query_Loss', q_loss)
        self.recorder.update('Query_ParamL2Loss', q_param_l2_loss)
        self.recorder.update('Query_PredLoss', q_pred_loss)
        self.recorder.update('Query_Accuracy', q_preds, q_labels)
        self.recorder.update('Query_Precision', q_preds, q_labels)
        self.recorder.update('Query_Recall', q_preds, q_labels)

        return q_loss, q_preds, q_attn

    def cal_total_loss(self, query_loss, kld_loss, z_loss, beta: float, gamma: float, lambda2: float):
        orthogonality_loss = self.orthgonality_constraint(list(self.decoder.parameters())[0])
        total_loss = query_loss + beta*kld_loss + gamma*z_loss + lambda2*orthogonality_loss
        
        # Record
        self.recorder.update('Orthogonality_Loss', orthogonality_loss)
        self.recorder.update('Total_Loss', total_loss)

        return total_loss

    def orthgonality_constraint(self, params: torch.Tensor):
        # purpose: encourages the dimensions of the latent code as well as the decoder network to be maximally expressive
        # params: (2E, H), 2E = (mean, std)
        p_dot = params.mm(params.transpose(0, 1))
        p_norm = torch.norm(params, dim=1, keepdim=True) + 1e-15
        corr = p_dot / p_norm.mm(p_norm.transpose(0, 1))
        corr.masked_fill_(corr >  1.0,  1.0)
        corr.masked_fill_(corr < -1.0, -1.0)
        I = torch.eye(corr.size(0)).to(corr.device)
        orthogonality_loss = torch.mean((corr - I)**2)
        return orthogonality_loss

    def forward(
            self, data: StockDataDict, # Dict[str, torch.Tensor], 
            beta: float=0.001, 
            gamma: float=1e-9, 
            lambda2: float=0.1, 
            n_inner_step: int=5, 
            rt_attn: bool=False
        ):
        parameters, kld_loss, z_loss, s_attn = self.inner_loop(
            data, n_inner_step, rt_attn
        )
        q_loss, q_preds, q_attn = self.query_validate(data, parameters, rt_attn)
        total_loss = self.cal_total_loss(q_loss, kld_loss, z_loss, beta, gamma, lambda2)
        
        return total_loss, q_preds, s_attn, q_attn

    def meta_predict(self):
        raise NotImplementedError('TODO')
