import torch
import snntorch as snn

lif = snn.Leaky(beta=0.9)
mem = torch.zeros(1)

x = torch.ones(5)

for t in range(5):
    spk, mem = lif(x[t:t+1], mem)
    print(f"t={t}: spike={spk.item()}, membrane={mem.item():.4f}")
