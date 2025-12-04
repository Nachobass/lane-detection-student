import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias
        )

        self.hidden_dim = hidden_dim

    def forward(self, x, state):
        h_prev, c_prev = state
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)

        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)

        return h, (h, c)

    def init_hidden(self, batch_size, H, W):
        h = torch.zeros(batch_size, self.hidden_dim, H, W).cuda()
        c = torch.zeros(batch_size, self.hidden_dim, H, W).cuda()
        return (h, c)


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size)

    def forward(self, seq_tensor):
        # seq_tensor: [B, T, C, H, W]
        B, T, C, H, W = seq_tensor.shape

        state = self.cell.init_hidden(B, H, W)

        for t in range(T):
            frame = seq_tensor[:, t]
            out, state = self.cell(frame, state)

        return out  # el último estado
