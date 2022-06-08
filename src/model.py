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
        normed_context = self.lnorm(h)
        return normed_context

class LSTMAttention(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.lnorm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.tensor, rt_attn: bool=False):
        # x: (B, T, I)
        o, (h, _) = self.lstm(x) # o: (B, T, H) / h: (1, B, H)
        score = torch.bmm(o, h.permute(1, 2, 0)) # (B, T, H) x (B, H, 1)
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
            nn.Linear(hidden_size, 2*hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(2*hidden_size, 2*hidden_size, bias=False),
        )

    def forward(self, x: torch.tensor):
        # x: (B, H)
        outputs = self.rn(x)
        return outputs

class MetaModel(nn.Module):
    def __init__(
            self, 
            feature_size: int, 
            hidden_size: int, 
            output_size: int, 
            num_layers: int, 
            drop_rate: float, 
            n_sample: int,
            inner_lr_init: float,
            finetuning_lr_init: float
        ):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.parameter_size = hidden_size*output_size
        self.n_sample = n_sample
        
        self.inner_lr = nn.Parameter(torch.FloatTensor([inner_lr_init]))
        self.finetuning_lr = nn.Parameter(torch.FloatTensor([finetuning_lr_init]))

        # Network
        self.dropout = nn.Dropout(drop_rate)
        self.feature_transform = nn.Linear(feature_size, hidden_size)
        self.lstm = LSTMAttention(hidden_size, hidden_size, num_layers)  # encode
        self.mapping_net = MappingNet(hidden_size)  # to generate z(latent)
        self.decoder = nn.Linear(hidden_size, 2*self.parameter_size, bias=False)
        self.prob_layer = nn.LogSoftmax(dim=1) if output_size >=2 else nn.LogSigmoid()

        # Loss
        self.loss_fn = nn.NLLLoss()

    def encode(self, inputs, rt_attn: bool=False):
        # inputs: (B, T, I)
        inputs = self.dropout(inputs)
        inputs = self.feature_transform(inputs)
        encoded, attn = self.lstm(inputs, rt_attn)  # B, H
        return encoded, attn

    def get_z(self, inputs, rt_attn: bool=False):
        # inputs: (B, T, I)
        encoded, attn = self.encode(inputs, rt_attn=rt_attn)
        hs = self.mapping_net(encoded)

        z, dist = self.sample(hs, size=self.hidden_size)
        return encoded, z, dist, attn

    def sample(self, distribution_params, size):
        mean, log_std = distribution_params[:, :size], distribution_params[:, size:]
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(std))
        z = dist.rsample()
        return mean + std*z, dist

    def decode(self, z):
        param_hs = self.decoder(z)  # check the distribution in params_hs?
        parameters, _ = self.sample(param_hs, size=self.parameter_size)
        return parameters

    def predict(self, encoded, parameters):
        theta = parameters.view(-1, self.hidden_size, self.output_size)
        scores = encoded.unsqueeze(1).bmm(theta).squeeze()
        probs = self.prob_layer(scores)
        return probs
        
    def cal_accuracy(self, log_probs, target):
        if self.output_size >= 2:
            pred = log_probs.argmax(1)
        else:
            pred = (torch.exp(log_probs) >= 0.5).long()
        correct = pred.eq(target).sum()
        acc = correct / len(target)
        return acc
    
    def reset_records(self):
        self.records = {}

    def inner_loop(self, data, n_inner_step: int=5, rt_attn: bool=False):
        support_X, support_y = data['support'], data['support_labels']
        support_encoded, support_z, support_dist, support_attn = self.get_z(support_X, rt_attn=rt_attn)
        kld_loss = self.cal_kl_div(support_dist, support_z)
        
        z_init = support_z.clone().detach()

        # inner adaptation to z
        for i in range(n_inner_step):
            support_z.retain_grad()
            parameters = self.decode(support_z)
            support_log_probs = self.predict(support_encoded, parameters)
            train_loss = self.loss_fn(support_log_probs, support_y)
            train_loss.backward(retain_graph=True)

            support_z = support_z - self.inner_lr * support_z.grad.data

        z_penalty = torch.mean((z_init - support_z)**2)
        return support_encoded, support_z, kld_loss, z_penalty, support_attn

    def outer_loop(self, data, support_z, support_encoded, n_finetuning_step: int=5, rt_attn: bool=False):
        # finetuning inner + validation
        
        # inner loop prediction
        support_y, query_X, query_y = data['support_labels'], data['query'], data['query_labels']
        parameters = self.decode(support_z)
        parameters.retain_grad()
        support_log_probs = self.predict(support_encoded, parameters)
        train_loss = self.loss_fn(support_log_probs, support_y)
        train_acc = self.cal_accuracy(support_log_probs, support_y)

        # logging
        self.records['Support Loss'] = train_loss.item()
        self.records['Support Accuracy'] = train_acc.item()
        self.records['Inner LR'] = float(self.inner_lr)
        self.records['Finetuning LR'] = float(self.finetuning_lr)
        self.records['Latents'] = support_z.clone().detach().cpu().numpy()

        # finetuning adaptation to parameters
        for i in range(n_finetuning_step):
            train_loss.backward(retain_graph=True)
            parameters = parameters - self.finetuning_lr * parameters.grad
            parameters.retain_grad()
            support_log_probs = self.predict(support_encoded, parameters)
            train_loss = self.loss_fn(support_log_probs, support_y)


        # meta validation     
        query_encoded, *_, query_attn = self.encode(query_X, rt_attn=rt_attn)
        query_log_probs = self.predict(query_encoded, parameters)
        query_loss = self.loss_fn(query_log_probs, query_y)
        query_acc = self.cal_accuracy(query_log_probs, query_y)

        return query_loss, query_acc, query_attn

    def cal_kl_div(self, dist, z):
        normal = torch.distributions.Normal(torch.zeros_like(z), torch.ones_like(z))
        return torch.mean(dist.log_prob(z) - normal.log_prob(z))

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
        support_encoded, support_z, kld_loss, z_penalty, support_attn = self.inner_loop(data, n_inner_step, rt_attn)
        query_loss, query_acc, query_attn = self.outer_loop(data, support_z, support_encoded, n_finetuning_step, rt_attn)
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
        self.records['Query Loss'] = total_loss.item()
        self.records['Query Accuracy'] = query_acc.item()
        if query_attn is not None:
            self.records['Query Attn'] = query_attn.detach().cpu().numpy()
        if support_attn is not None:
            self.records['Support Attn'] = support_attn.detach().cpu().numpy()

        return total_loss, self.records