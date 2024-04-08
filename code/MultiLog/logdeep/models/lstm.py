import torch
import torch.nn as nn
from torch.autograd import Variable

if torch.cuda.is_available():
    device_robustlog = torch.device("cuda:1")
    device_loganomaly = torch.device("cuda:3")
    device_dislog = torch.device("cuda:3")
    device = torch.device("cuda:3")
else:
    device_robustlog = torch.device("cpu")
    device_loganomaly = torch.device("cpu")
    device_dislog = torch.device("cpu")
    device = torch.device("cpu")


class deeplog(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(deeplog, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, features):
        input0 = features[0]
        h0 = torch.zeros(self.num_layers, input0.size(0),
                         self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, input0.size(0),
                         self.hidden_size).to(device)
        out, _ = self.lstm(input0, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class loganomaly(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(loganomaly, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm0 = nn.LSTM(input_size,
                             hidden_size,
                             num_layers,
                             batch_first=True)
        self.lstm1 = nn.LSTM(input_size,
                             hidden_size,
                             num_layers,
                             batch_first=True)
        self.fc = nn.Linear(2 * hidden_size, num_keys)
        self.attention_size = self.hidden_size

        self.w_omega = Variable(
            torch.zeros(self.hidden_size, self.attention_size))
        self.u_omega = Variable(torch.zeros(self.attention_size))

        self.sequence_length = 28

    def attention_net(self, lstm_output):
        output_reshape = torch.Tensor.reshape(lstm_output,
                                              [-1, self.hidden_size])
        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega.to(device_loganomaly)))
        attn_hidden_layer = torch.mm(
            attn_tanh, torch.Tensor.reshape(self.u_omega.to(device_loganomaly), [-1, 1]))
        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer),
                                    [-1, self.sequence_length])
        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        alphas_reshape = torch.Tensor.reshape(alphas,
                                              [-1, self.sequence_length, 1])
        state = lstm_output
        attn_output = torch.sum(state * alphas_reshape, 1)
        return attn_output

    def forward(self, features):
        input0, input1 = features[0], features[1]

        h0_0 = torch.zeros(self.num_layers, input0.size(0),
                           self.hidden_size).to(device_loganomaly)
        c0_0 = torch.zeros(self.num_layers, input0.size(0),
                           self.hidden_size).to(device_loganomaly)

        out0, _ = self.lstm0(input0, (h0_0, c0_0))

        h0_1 = torch.zeros(self.num_layers, input1.size(0),
                           self.hidden_size).to(device_loganomaly)
        c0_1 = torch.zeros(self.num_layers, input1.size(0),
                           self.hidden_size).to(device_loganomaly)

        out1, _ = self.lstm1(input1, (h0_1, c0_1))
        multi_out = torch.cat((out0[:, -1, :], out1[:, -1, :]), -1)
        out = self.fc(multi_out)
        return out


class robustlog(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(robustlog, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers,
                            batch_first=True).to(device_robustlog)
        self.fc = nn.Linear(hidden_size, num_keys).to(device_robustlog)

    def forward(self, features):
        input0 = features[0]
        h0 = torch.zeros(self.num_layers, input0.size(0),
                         self.hidden_size).to(device_robustlog)
        c0 = torch.zeros(self.num_layers, input0.size(0),
                         self.hidden_size).to(device_robustlog)
        out, _ = self.lstm(input0, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, attention_size):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attention_size = attention_size

        # 定义权重
        self.W_q = nn.Linear(hidden_size, attention_size, bias=False)
        self.W_k = nn.Linear(hidden_size, attention_size, bias=False)
        self.W_v = nn.Linear(hidden_size, attention_size, bias=False)

    def forward(self, x):
        # x 的维度为 [batch_size, seq_len, hidden_size]

        # 生成 Q, K, V
        Q = self.W_q(x)  # [batch_size, seq_len, attention_size]
        K = self.W_k(x)  # [batch_size, seq_len, attention_size]
        V = self.W_v(x)  # [batch_size, seq_len, attention_size]

        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.attention_size, dtype=torch.float32))
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 应用注意力权重
        output = torch.matmul(attention_weights, V)  # [batch_size, seq_len, attention_size]

        return output


class dislog(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(dislog, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm0 = nn.LSTM(input_size,
                             hidden_size,
                             num_layers,
                             batch_first=True)
        self.lstm1 = nn.LSTM(input_size,
                             hidden_size,
                             num_layers,
                             batch_first=True)
        self.lstm2 = nn.LSTM(input_size,
                             hidden_size,
                             num_layers,
                             batch_first=True)
        self.fc = nn.Linear(3 * hidden_size, num_keys)
        self.attention_size = self.hidden_size
        self.self_attention = SelfAttention(self.hidden_size, self.attention_size)

    def forward(self, features):
        input0, input1, input2 = features[0], features[1], features[2]

        h0_0 = torch.zeros(self.num_layers, input0.size(0),
                           self.hidden_size).to(device_dislog)
        c0_0 = torch.zeros(self.num_layers, input0.size(0),
                           self.hidden_size).to(device_dislog)

        out0, _ = self.lstm0(input0, (h0_0, c0_0))
        attn_out0 = self.self_attention(out0)

        h0_1 = torch.zeros(self.num_layers, input1.size(0),
                           self.hidden_size).to(device_dislog)
        c0_1 = torch.zeros(self.num_layers, input1.size(0),
                           self.hidden_size).to(device_dislog)

        out1, _ = self.lstm1(input1, (h0_1, c0_1))
        attn_out1 = self.self_attention(out1)

        h0_2 = torch.zeros(self.num_layers, input2.size(0),
                           self.hidden_size).to(device_dislog)
        c0_2 = torch.zeros(self.num_layers, input2.size(0),
                           self.hidden_size).to(device_dislog)

        out2, _ = self.lstm1(input2, (h0_2, c0_2))
        attn_out2 = self.self_attention(out2)

        multi_out = torch.cat((attn_out0[:, -1, :], attn_out1[:, -1, :], attn_out2[:, -1, :]), -1)
        out = self.fc(multi_out)
        return out
