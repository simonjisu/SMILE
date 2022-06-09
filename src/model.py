import torch
import torch.nn as nn
import pytorch_lightning as pl

class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.lnorm = nn.LayerNorm(hidden_size)
    
    def forward(self, x: torch.tensor):
        # x: (B, T, I)
        o, (h, _) = self.lstm(x) # o: (B, T, H) / h: (1, B, H)
        normed_context = self.lnorm(h[-1, :, :])
        return normed_context

class LSTMAttention(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.lnorm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.tensor, rt_attn: bool=False):
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

class MappingNet(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.rn = nn.Sequential(
            nn.Linear(hidden_size, 2*hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(2*hidden_size, 2*hidden_size, bias=True),
        )

    def forward(self, x: torch.tensor):
        # x: (B, H)
        outputs = self.rn(x)
        return outputs

class MetaModel(nn.Module):
    """Meta Model
    Structure Ref: 
    * LEO-Deepmind: https://github.com/deepmind/leo/blob/de9a0c2a77dd7a42c1986b1eef18d184a86e294a/model.py#L256
    * LEO-pytorch: https://github.com/timchen0618/pytorch-leo/blob/master/model.py
    """
    def __init__(
            self, 
            feature_size: int, 
            hidden_size: int, 
            output_size: int, 
            num_layers: int, 
            drop_rate: float, 
            inner_lr_init: float,
            finetuning_lr_init: float
        ):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.parameter_size = hidden_size*output_size
        
        self.inner_lr = nn.Parameter(torch.FloatTensor([inner_lr_init]))
        self.finetuning_lr = nn.Parameter(torch.FloatTensor([finetuning_lr_init]))

        # Network
        self.dropout = nn.Dropout(drop_rate)
        self.feature_transform = nn.Linear(feature_size, hidden_size)  # (B, T, I) > (B, T, H)
        self.lstm = LSTMAttention(hidden_size, hidden_size, num_layers)  # (B, T, H) > (B, T, H)
        self.mapping_net = MappingNet(hidden_size)  # (B, H) > (B, 2H)
        self.decoder = nn.Linear(hidden_size, 2*self.parameter_size, bias=False)  # (B, 2H) > (B, H*O)

        # Loss
        self.loss_fn = nn.CrossEntropyLoss() if output_size >=2 else nn.BCEWithLogitsLoss()

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
            if isinstance(module, nn.Dropout): # or isinstance(module, nn.LayerNorm):
                module.train(mode)

    def reset_records(self):
        self.records = {}

    def encode(self, inputs, rt_attn: bool=False):
        inputs = self.feature_transform(inputs)  # (B, T, I) > (B, T, H)
        inputs = self.dropout(inputs)  # (B, T, H) > (B, T, H)
        encoded, attn = self.lstm(inputs, rt_attn)  # (B, H)
        return encoded, attn

    def forward_encoder(self, inputs, rt_attn: bool=False):
        # inputs: (B, T, I)
        # encoded: (B, H)
        encoded, attn = self.encode(inputs, rt_attn=rt_attn)
        hs = self.mapping_net(encoded) # (B, H) > (B, 2H)
        z, kld_loss = self.sample(hs, size=encoded.size(1))  # (B, 2H)
        return encoded, z, kld_loss, attn

    def cal_kl_div(self, dist, z):
        normal = torch.distributions.Normal(torch.zeros_like(z), torch.ones_like(z))
        return torch.mean(dist.log_prob(z) - normal.log_prob(z))

    def sample(self, distribution_params, size):
        mean, log_std = distribution_params[:, :size], distribution_params[:, size:]
        if not self.is_meta_train:
            return mean, torch.zeros((1,)).to(mean.device)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        z = dist.rsample()
        kld_loss = self.cal_kl_div(dist, z)
        return z, kld_loss

    def decode(self, z):
        # z: (B, H)
        # param_hs: (B, 2*P)
        param_hs = self.decoder(z)
        parameters, _ = self.sample(param_hs, size=self.parameter_size)
        return parameters

    def predict(self, encoded, parameters, labels):
        theta = parameters.view(-1, self.hidden_size, self.output_size)
        scores = encoded.unsqueeze(1).bmm(theta).squeeze()
        loss = self.loss_fn(scores, labels)
        acc = self.cal_accuracy(scores, labels)
        return loss, acc

    def forward_decoder(self, z, encoded, labels):
        parameters = self.decode(z)
        loss, acc = self.predict(encoded, parameters, labels)
        return loss, acc, parameters

    def cal_accuracy(self, scores, target):
        if self.output_size >= 2:
            pred = scores.argmax(1)
        else:
            pred = (torch.sigmoid(scores) >= 0.5).long()
        correct = pred.eq(target).sum()
        acc = correct / len(target)
        return acc

    def inner_loop(self, data, n_inner_step: int=5, n_finetuning_step: int=5, rt_attn: bool=False):
        support_X, support_y = data['support'], data['support_labels']
        support_y = support_y.float() if self.output_size == 1 else support_y
        support_encoded, support_z, kld_loss, support_attn = self.forward_encoder(support_X, rt_attn=rt_attn)

        # initialize z', 
        z_prime = support_z
        train_loss, train_acc, _ = self.forward_decoder(z=z_prime, encoded=support_encoded, labels=support_y)
        # inner adaptation to z
        for i in range(n_inner_step):
            z_prime.retain_grad()
            train_loss.backward(retain_graph=True)
            z_prime = z_prime - self.inner_lr * z_prime.grad.data

            train_loss, train_acc, parameters = self.forward_decoder(z=z_prime, encoded=support_encoded, labels=support_y)

        z_prime = z_prime.detach()  # Stop Gradient
        z_penalty = torch.mean((z_prime - support_z)**2)

        self.records['Support Loss'] = train_loss.item()
        self.records['Support Accuracy'] = train_acc.item()
        self.records['Inner LR'] = float(self.inner_lr)
        self.records['Finetuning LR'] = float(self.finetuning_lr)
        self.records['Z Prime'] = z_prime.detach().cpu().numpy()
        self.records['Z'] = support_z.detach().cpu().numpy()

        # finetuning adaptation to parameters
        if n_finetuning_step > 0:
            for i in range(n_finetuning_step):
                parameters.retain_grad()
                train_loss.backward(retain_graph=True)
                parameters = parameters - self.finetuning_lr * parameters.grad
                train_loss, finetune_train_acc = self.predict(
                    encoded=support_encoded, parameters=parameters, labels=support_y
                )

            self.records['Finetune Loss'] = train_loss.item()
            self.records['Finetune Accuracy'] = finetune_train_acc.item()
        else:
            self.records['Finetune Loss'] = 0.0
            self.records['Finetune Accuracy'] = 0.0
            
        return parameters, kld_loss, z_penalty, support_attn

    def validate(self, data, parameters, rt_attn: bool=False):
        self.manual_model_eval()
        query_X, query_y = data['query'], data['query_labels']
        query_y = query_y.float() if self.output_size == 1 else query_y
        
        query_encoded, *_, query_attn = self.forward_encoder(query_X, rt_attn=rt_attn)
        query_loss, query_acc = self.predict(
            encoded=query_encoded, parameters=parameters, labels=query_y
        )
        self.train()
        return query_loss, query_acc, query_attn


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
        parameters, kld_loss, z_penalty, support_attn = self.inner_loop(data, n_inner_step, n_finetuning_step, rt_attn)
        query_loss, query_acc, query_attn = self.validate(data, parameters, rt_attn)
        return query_loss, query_acc, kld_loss, z_penalty, support_attn, query_attn

    def meta_run(self, data, 
            beta: float=0.001, 
            gamma: float=1e-9, 
            lambda2: float=0.1,
            n_inner_step: int=5, 
            n_finetuning_step:int =5,
            rt_attn: bool=False
        ):
        query_loss, query_acc, kld_loss, z_penalty, support_attn, query_attn = self(
            data, n_inner_step, n_finetuning_step, rt_attn
        )
        total_loss = self.cal_total_loss(query_loss, kld_loss, z_penalty, beta, gamma, lambda2)
        # logging
        self.records['Query Loss'] = query_loss.item()
        self.records['Total Loss'] = total_loss.item()
        self.records['Query Accuracy'] = query_acc.item()
        if query_attn is not None:
            self.records['Query Attn'] = query_attn.detach().cpu().numpy()
        if support_attn is not None:
            self.records['Support Attn'] = support_attn.detach().cpu().numpy()

        return total_loss, self.records
