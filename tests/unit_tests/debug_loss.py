import torch

# data = torch.load("/mnt/public/peihong/codes/RLinf_support/examples/embodiment/loss_data_debug_0.pt")

def test(path):
    data = torch.load(path)
    print(data.keys())

    for k, v in data.items():
        print(k, data[k].shape if hasattr(data[k], "shape") else data[k])

    clipped_ratio = data["clipped_ratio"]
    ratio = data["ratio"]
    advantages = data["advantages"]

    loss1 = -advantages * ratio
    loss2 = -advantages * clipped_ratio

    loss = torch.max(loss1, loss2)
    print(f"loss.shape: {loss.shape}")
    print(loss.mean())


    loss_mask = data["loss_mask"]
    loss_mask_ratio = data["loss_mask_ratio"]
    loss = (loss / loss_mask_ratio * loss_mask).mean()
    print("final loss:", loss)

'''
print(data["clipped_ratio"])
print(data["ratio"])
print(f"data['clipped_ratio'].shape: {data['clipped_ratio'].shape}")
print(f"data['ratio'].shape: {data['ratio'].shape}")
is_close = torch.isclose(data["clipped_ratio"], data["ratio"], atol=1e-4, rtol=0)
print(is_close)
'''
# test("/mnt/public/peihong/codes/RLinf_support/examples/embodiment/loss_data_debug_0.pt")
# print("##############")
# test("/mnt/public/peihong/codes/RLinf_support/examples/embodiment/loss_data_debug_1.pt")
# print("##############")
# test("/mnt/public/peihong/codes/RLinf_support/examples/embodiment/loss_data_debug_2.pt")
# print("##############")
# test("/mnt/public/peihong/codes/RLinf_support/examples/embodiment/loss_data_debug_3.pt")

# test("/mnt/public/peihong/codes/RLinf_support/examples/embodiment/loss_data_ttttt_0.pt")
datas = torch.load("/mnt/public/peihong/codes/RLinf_support/examples/embodiment/debug_datas_actor_rank0.pt")["res"]
print(datas[0].keys())
print(len(datas))

res = []
for i in range(len(datas)):
    data = datas[i]
    def func(data):
        for k, v in data.items():
            print(k, data[k].shape if hasattr(data[k], "shape") else data[k])

        clipped_ratio = data["clipped_ratio"]
        ratio = data["ratio"]
        advantages = data["advantages"]

        loss1 = -advantages * ratio
        loss2 = -advantages * clipped_ratio

        loss = torch.max(loss1, loss2)
        print(f"loss.shape: {loss.shape}")
        print(loss.mean())

        loss_mask = data["loss_mask"]
        # loss_mask_ratio = data["loss_mask_ratio"]
        # loss = (loss / loss_mask_ratio * loss_mask).mean()
        # print("final loss:", loss)
        
        loss_mask = loss_mask.expand_as(loss)
        loss = (loss * loss_mask).sum(axis=None) / loss_mask.sum(axis=None)

        return loss
        
    res.append(func(data))
    print("##############")

print("loss sum:", sum(res))

print("loss mean:", sum(res)/len(res))
