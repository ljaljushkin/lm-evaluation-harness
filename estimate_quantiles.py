from bitsandbytes import functional as F
import torch


# values = torch.randn(1000, 1000, device='cuda')
# print(values[:5,:5])

def create_emp_code(values):
    bits = 4
    emp_code = F.create_quantile_map(values, bits).cuda()
    # torch.set_printoptions(precision=20)
    n = len(emp_code)
    zero_ids = [i for i in range(n) if i not in torch.nonzero(emp_code)]
    non_zero_except_one = [i for i in range(n) if i not in zero_ids[:-1]]
    emp_code = emp_code[non_zero_except_one]
    assert len(emp_code) == 16
    assert len(torch.nonzero(emp_code)) == 15
    # print(emp_code)
    return emp_code


nf4_code = torch.Tensor([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
                -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
                0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
                0.7229568362236023, 1.0]).to('cuda')
nf4_code_fn = lambda _: nf4_code

assert len(nf4_code) == 16
# values = torch.randn(1, 32, device='cuda')
# values /= values.abs().max()
# values[values.abs() < 1e-6] += 1e-5

for code_fn in [nf4_code_fn, create_emp_code]:

    abserrs = []
    relerrs = []
    for i in range(10):
        values = torch.randn(1024, 1024, device='cuda')
        values /= values.abs().max()
        values[values.abs() < 1e-6] += 1e-5

        code = code_fn(values)

        q1 = []
        v1 = []
        for v in values[0]:
            idx = torch.abs(v-code).argmin()
            q1.append(idx.item())
            v1.append(code[idx].item())

        q1 = torch.Tensor(q1).cuda()
        v1 = torch.Tensor(v1).cuda()

        q2, S2 = F.quantize_blockwise(values, code=code, blocksize=1024)
        v2 = F.dequantize_blockwise(q2, S2, blocksize=1024)
        # print(v2)
        idx = torch.isclose(q1.int(), q2.int())
        err2 = torch.abs(v2-values)
        abserr = err2.mean().item()
        print(abserr)
        abserrs.append(abserr)
        relerrs.append((err2/(1e-10+values).abs()).mean().item())
        if idx.sum():
            # some weird cases
            err1 = torch.abs(v1-values).mean()
            assert err2.mean() <= err1
        else:
            torch.testing.assert_close(q1, q2)
    print('abserr:', sum(abserrs)/len(abserrs), 'relerr:', sum(relerrs)/len(relerrs))