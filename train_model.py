import torch
from torch import nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
import numpy as np

# 数据集
num_sample = 1000

class sindata(Dataset):
    def __init__(self, num_sample, x_len):
        super().__init__()
        self.num_sample = num_sample
        self.sample_dim = x_len
        self.base_x = torch.linspace(-5,5,self.sample_dim)
        self.samples = self.get_samples()

    def get_samples(self,):  # 做n个随机的正弦曲线
        freq = torch.randn(self.num_sample).reshape(-1,1)  # freq
        phase = torch.randn(self.num_sample).reshape(-1,1) # phase
        samples = torch.sin(self.base_x*freq + phase)
        return torch.concat([samples, freq, phase], dim=-1)

    def plot_samples(self):
        for i in range(10):
            plt.plot(self.samples[100-i])
        plt.show()

    def __getitem__(self, item):
        return self.samples[item]

    def __len__(self):
        return self.num_sample


if __name__ == '__main__':
    from diffusion_model import Net
    torch.manual_seed(0)
    device = torch.device('cuda:0')
    time_step = 1000
    beta = torch.linspace(0.0001, 0.002, time_step)
    alpha_t = 1 - beta
    alpha_t_hat = torch.cumprod(alpha_t, dim=-1)
    alpha_t_hat_s = (torch.sqrt(alpha_t_hat)).to(device=device)
    one_minus_alpha_t_sqrt = torch.sqrt(1 - alpha_t_hat).to(device=device)

    num_sample = 30000
    x_len = 128
    time_dim = x_len
    dataset = sindata(num_sample, x_len)
    loader = DataLoader(dataset=dataset, shuffle=True, batch_size=2048)
    model = Net([x_len,256,512,1024,2048,4096], time_dim, steps=time_step).to(device)
    # model.load_state_dict(torch.load('model.pt'))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # scheduler = StepLR(optimizer,step_size=8, gamma=0.9, verbose=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10, min_lr=5e-5)
    epoches = 5000
    loss_on_epochs = []
    loss_on = None
    with tqdm(range(epoches)) as tq:
        for epoch in tq:
            loss_on_epoch = []
            for x in loader:
                x = x[:, :-2].to(device)
                target = x[:, -2:].to(device)
                optimizer.zero_grad()
                t = torch.randint(0, time_step, (x.size(0),)).to(device)  #
                noise = torch.randn_like(x).to(device)
                x = alpha_t_hat_s[t].reshape(-1,1) * x + one_minus_alpha_t_sqrt[t].reshape(-1,1) * noise
                pred = model(x, t, target)
                loss = torch.mean(torch.pow(pred - noise, 2))
                loss.backward()
                clip_norm = 1.0  # 设置梯度的最大范数
                # torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
                loss_on_epoch.append(loss.item())
                tq.set_postfix({'loss_on_batch': loss.item(),'loss_on_epoch': loss_on,'lr':scheduler.get_last_lr()[0]})
            loss_on = np.mean(np.array(loss_on_epoch))
            loss_on_epochs.append(loss_on)
            torch.save(model.state_dict(),'model.pt')
            scheduler.step(loss_on)

            # 更新学习率调度器
            if len(loss_on_epochs) > 6:
                if np.array(loss_on_epochs[-6:-3]).mean()<=np.array(loss_on_epochs[-3:]).mean():
                    print('step')





