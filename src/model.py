import torch
import torch.nn as nn
from collections import defaultdict
from src.dataset import StockDataDict

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
    def __init__(self, hidden_size):
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

# class MappingNet(nn.Module):
#     def __init__(self, hidden_size):
#         super().__init__()
#         self.rn = nn.Sequential(
#             nn.Linear(hidden_size, 2*hidden_size, bias=True),
#             nn.ReLU(),
#             nn.Linear(2*hidden_size, 2*hidden_size, bias=True),
#         )

#     def forward(self, x: torch.Tensor):
#         # x: (B, H)
#         outputs = self.rn(x)
#         return outputs

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
            hidden_size: int, 
            output_size: int,  # only should be number of classes
            num_layers: int, 
            drop_rate: float, 
            inner_lr_init: float,
            finetuning_lr_init: float
        ):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        
        self.inner_lr = nn.Parameter(torch.FloatTensor([inner_lr_init]))
        self.finetuning_lr = nn.Parameter(torch.FloatTensor([finetuning_lr_init]))

        # Network
        self.dropout = nn.Dropout(drop_rate)
        self.lstm_encoder = LSTMAttention(feature_size, embed_size, num_layers)
        self.encoder = nn.Linear(embed_size, hidden_size)
        self.relation_net = RelationNet(hidden_size)
        self.decoder = nn.Linear(hidden_size, 2*embed_size, bias=False)

        # Loss
        self.loss_fn = nn.CrossEntropyLoss()

        # Meta Training Mode
        self.meta_train()

    def meta_train(self):
        self._meta_mode_change(True)
        self.train()

    def meta_valid(self):
        self._meta_mode_change(False)
        self.manual_model_eval()

    def _meta_mode_change(self, mode=True):
        self.is_meta_train = mode

    def manual_model_eval(self, mode=False):
        # [PyTorch Issue] RuntimeError: cudnn RNN backward can only be called in training mode
        # cannot use model.eval()
        # https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch
        for module in self.children():
            self.training = mode
            if isinstance(module, nn.Dropout) or isinstance(module, nn.LayerNorm):
                module.train(mode)

    def reset_records(self):        
        self.records = defaultdict(list)

    def encode_lstm(self, inputs, rt_attn: bool=False):
        """forward data by each stock to avoid trained by other stocks
        - B: number of samples
        - N: number of classes=`output_size`
        - K: number of `n_support`/`n_query`
        - T: window size
        - I: input size
        - E: embedding size
        - M: M = N * K

        Args:
            inputs: (B, N, T, I). single stock data.

        Returns:
            encoded: (B, N*K, E)
            attn: (B, N*K, T)
        """
        B, M, T, I = inputs.size()
        inputs = inputs.view(B*M, T, I)  # (B*M, T, I) 
        inputs = self.dropout(inputs)
        encoded, attn = self.lstm_encoder(inputs, rt_attn)  # (B*N*K, E), (B*N*K, T)
        return encoded.view(B, M, -1), attn.view(B, M, -1)  # (B, N*K, E), (B, N*K, T)

    def forward_encoder(self, inputs, rt_attn: bool=False):
        """Forward Encoder: from `inputs` to `z`
        - B: number of samples
        - N: number of classes=`output_size`
        - K: number of `n_support`/`n_query`
        - T: window size
        - E: embedding size
        - H: hidden size

        Args:
            inputs: (B, N, T, I). single stock data.

        Returns:
            x: (B, E). averaged lstm encoded for each example.
            z: (B, N, H). sampled parameters for each class.
            kld_loss: (1,). KL-Divergence Loss between N($\mu_n^e$, $\sigma_n^e$) and N(0, 1).
            attn: (B, N, K, T). attention weights for each inputs.
        
        """
        l, attn = self.encode_lstm(inputs, rt_attn=rt_attn)  # l: (B, N*K, E)
        # Reshape the size
        B = l.size(0)
        N = self.output_size
        K = l.size(1) // N
        if rt_attn:
            attn = attn.view(B, N, K, -1)  # attn: (B, N, K, T)
        l_reshape = l.view(B, N, K, -1)  # l_reshape: (B, N, K, E)
        # Encoder-to-z
        e = self.encoder(l_reshape)  # e: (B, N, K, H)
        hs = self.relation_net(e)  # hs: (B, N, 2H)
        z, kld_loss = self.sample(hs, size=self.hidden_size)  # z: (B, N, H)
        x = l.mean(1)  # x: (B, E)
        return x, z, kld_loss, attn

    def cal_kl_div(self, dist, z):
        normal = torch.distributions.Normal(torch.zeros_like(z), torch.ones_like(z))
        return torch.mean(dist.log_prob(z) - normal.log_prob(z))

    def sample(self, distribution_params, size):
        """parameters of a probability distribution in a low-dimensional space `z` for each class"""
        mean, log_std = torch.split(distribution_params, split_size_or_sections=size, dim=-1)
        if not self.is_meta_train:
            return mean, torch.zeros((1,)).to(mean.device)
        std = torch.exp(log_std)
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
            z: (B, N, H). sampled parameters for each class.

        Returns:
            parameters: (B, N, E). $\theta$
        """
        param_hs = self.decoder(z)  # param_hs: (B, N, 2H)
        parameters, _ = self.sample(param_hs, size=self.embed_size)  # (B, N, E)
        return parameters

    def predict(self, x, parameters, labels):
        theta = parameters.permute((0, 2, 1))  # (B, N, E) -> (B, E, N)
        scores = x.unsqueeze(1).bmm(theta).squeeze(1)  # (B, 1, E) x (B, E, N) = (B, N)
        loss = self.loss_fn(scores, labels)
        acc = self.cal_accuracy(scores, labels)
        return loss, acc

    def forward_decoder(self, z, x, labels):
        """Decoder
        - B: number of samples
        - N: number of classes=`output_size`
        - E: embedding size
        - H: hidden size

        Args:
            x: (B, E). averaged lstm encoded for each example.
            z: (B, N, H). sampled parameters for each class.

        Returns:
            loss: (1,).
            acc: (1,).
            parameters: (B, N, E).
        """
        parameters = self.decode(z)  # parameters: (B, N, E)
        loss, acc = self.predict(x, parameters, labels)
        return loss, acc, parameters

    def cal_accuracy(self, scores, target):
        pred = scores.argmax(1)
        correct = pred.eq(target).sum()
        acc = correct / len(target)
        return acc

    def inner_loop(self, data, n_inner_step: int=5, n_finetuning_step: int=5, rt_attn: bool=False):
        records = {}
        s_inputs = data['support']
        s_labels = data['support_labels']

        # Forward Encoder
        s_x, s_z, kld_loss, s_attn = self.forward_encoder(s_inputs, rt_attn=rt_attn)

        # initialize z', Forward Decoder
        z_prime = s_z
        s_loss, s_acc, _ = self.forward_decoder(z=z_prime, x=s_x, labels=s_labels)
        # inner adaptation to z
        for i in range(n_inner_step):
            z_prime.retain_grad()
            s_loss.backward(retain_graph=True)
            z_prime = z_prime - self.inner_lr * z_prime.grad.data
            s_loss, s_acc, parameters = self.forward_decoder(z=z_prime, x=s_x, labels=s_labels)

        # Stop Gradient: 
        # z_prime.requires_grad == False
        # s_z.requires_grad == True
        z_prime = z_prime.detach()  
        z_penalty = torch.mean((z_prime - s_z)**2)

        records['Support Loss'] = s_loss.item()
        records['Support Accuracy'] = s_acc.item()
        records['Inner LR'] = float(self.inner_lr)
        records['Finetuning LR'] = float(self.finetuning_lr)
        # self.records['Z Prime'] = z_prime.detach().cpu().numpy()
        # self.records['Z'] = s_z.detach().cpu().numpy()

        # finetuning adaptation to parameters
        if n_finetuning_step > 0:
            for i in range(n_finetuning_step):
                parameters.retain_grad()
                s_loss.backward(retain_graph=True)
                parameters = parameters - self.finetuning_lr * parameters.grad
                s_loss, s_acc = self.predict(
                    x=s_x, parameters=parameters, labels=s_labels
                )

            records['Finetune Loss'] = s_loss.item()
            records['Finetune Accuracy'] = s_acc.item()
        else:
            records['Finetune Loss'] = 0.0
            records['Finetune Accuracy'] = 0.0
            
        return parameters, kld_loss, z_penalty, s_attn, records

    def validate(self, data, parameters, rt_attn: bool=False):
        q_inputs = data['query']
        q_labels = data['query_labels']
        
        q_x, *_, q_attn = self.forward_encoder(q_inputs, rt_attn=rt_attn)
        q_loss, q_acc = self.predict(
            x=q_x, parameters=parameters, labels=q_labels
        )
        return q_loss, q_acc, q_attn

    def cal_total_loss(self, query_loss, kld_loss, z_penalty, beta, gamma, lambda2):
        orthogonality_penalty = self.orthgonality_constraint(list(self.decoder.parameters())[0])
        total_loss = query_loss + beta*kld_loss + gamma*z_penalty + lambda2*orthogonality_penalty
        # loggings
        self.records['KLD Loss'] = kld_loss.item()
        self.records['Z Penalty'] = z_penalty.item()
        self.records['Orthogonality Penalty'] = orthogonality_penalty.item()
        
        return total_loss

    def orthgonality_constraint(self, params):
        # purpose: encourages the dimensions of the latend code as well as the decoder network to be maximally expressive
        # number of class x hidden_size x 2(mean, std)
        p_dot = params.mm(params.transpose(0, 1))
        p_norm = torch.norm(params, dim=1, keepdim=True) + 1e-15
        corr = p_dot / p_norm.mm(p_norm.transpose(0, 1))
        corr.masked_fill_(corr>1.0, 1.0)
        corr.masked_fill_(corr<-1.0, -1.0)
        I = torch.eye(corr.size(0)).to(corr.device)
        orthogonality_penalty = torch.mean((corr - I)**2)
        return orthogonality_penalty

    def forward(
            self, data, 
            n_inner_step: int=5, 
            n_finetuning_step:int =5, 
            rt_attn: bool=False
        ):
        self.reset_records()
        parameters, kld_loss, z_penalty, s_attn, inner_records = self.inner_loop(
            data, n_inner_step, n_finetuning_step, rt_attn
        )
        q_loss, q_acc, q_attn = self.validate(data, parameters, rt_attn)
        return q_loss, q_acc, kld_loss, z_penalty, s_attn, q_attn

    def meta_run(self, stock_data: StockDataDict, 
            beta: float=0.001, 
            gamma: float=1e-9, 
            lambda2: float=0.1,
            n_inner_step: int=5, 
            n_finetuning_step:int =5,
            rt_attn: bool=False
        ):

        total_q_loss = 0.
        total_loss = 0.
        for data in stock_data:
            q_loss, q_acc, kld_loss, z_penalty, s_attn, q_attn = self(
                data, n_inner_step, n_finetuning_step, rt_attn
            )
            total_loss += self.cal_total_loss(q_loss, kld_loss, z_penalty, beta, gamma, lambda2)
            total_q_loss += q_loss
        
        # logging
        self.records['Query Loss'] = total_q_loss.item()
        self.records['Total Loss'] = total_loss.item()
        self.records['Query Accuracy'] = q_acc.item()
        if q_attn is not None:
            self.records['Query Attn'] = q_attn.detach().cpu().numpy()
        if s_attn is not None:
            self.records['Support Attn'] = s_attn.detach().cpu().numpy()

        return total_loss, self.records
