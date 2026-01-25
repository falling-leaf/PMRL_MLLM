import torch
import torch.nn as nn


class MMD_loss(nn.Module):
    def __init__(
        self,
        kernel_type="rbf",
        kernel_mul=2.0,
        kernel_num=5,
        eps=1e-6,
        debug=False,
        auto_fallback=True,   # 新增：RBF 失效自动退化
    ):
        super().__init__()
        self.kernel_type = kernel_type
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.eps = eps
        self.debug = debug
        self.auto_fallback = auto_fallback

    # ---------- debug helpers ----------
    def _stat(self, name, x):
        print(
            f"[MMD][STAT] {name}: "
            f"shape={tuple(x.shape)}, "
            f"min={x.min().item():.3e}, "
            f"max={x.max().item():.3e}, "
            f"mean={x.mean().item():.3e}, "
            f"nan={torch.isnan(x).any().item()}, "
            f"inf={torch.isinf(x).any().item()}"
        )

    def _quantile(self, name, x):
        qs = torch.quantile(
            x.flatten(),
            torch.tensor([0.0, 0.25, 0.5, 0.75, 0.95, 0.99], device=x.device)
        )
        print(
            f"[MMD][Q] {name}: "
            + ", ".join([f"{q.item():.3e}" for q in qs])
        )

    # ---------- kernels ----------
    def gaussian_kernel(self, source, target):
        total = torch.cat([source, target], dim=0)

        # 强制 finite（不影响梯度方向，只防数值污染）
        total = torch.nan_to_num(
            total,
            nan=0.0,
            posinf=1e4,
            neginf=-1e4,
        )

        diff = total.unsqueeze(0) - total.unsqueeze(1)
        L2 = (diff * diff).sum(dim=2)

        if self.debug:
            self._stat("L2_distance", L2)
            self._quantile("L2_distance", L2)

        # bandwidth 统计
        bw = torch.mean(L2).detach()
        bw = torch.clamp(bw, min=self.eps)

        if self.debug:
            print(f"[MMD][STAT] bandwidth_base = {bw.item():.3e}")

        kernels = 0.0
        for i in range(self.kernel_num):
            bw_i = bw * (self.kernel_mul ** i)
            bw_i = torch.clamp(bw_i, min=self.eps)

            scaled = L2 / bw_i
            scaled = torch.clamp(scaled, max=50.0)

            k = torch.exp(-scaled)
            kernels = kernels + k

            if self.debug:
                self._stat(f"kernel_{i}", k)

        return kernels

    # ---------- linear fallback ----------
    def linear_mmd2(self, x, y):
        delta = x.mean(dim=0) - y.mean(dim=0)
        return delta.dot(delta)

    def cosine_kernel(self, x, y, eps=1e-8):
        """
        x: [N, D]
        y: [M, D]
        return: [N, M]
        """
        x = x / (x.norm(dim=1, keepdim=True) + eps)
        y = y / (y.norm(dim=1, keepdim=True) + eps)
        return torch.mm(x, y.t())

    def cosine_mmd(self, x, y, eps=1e-8):
        """
        x: [N, D]
        y: [M, D]
        """
        K_xx = self.cosine_kernel(x, x, eps)
        K_yy = self.cosine_kernel(y, y, eps)
        K_xy = self.cosine_kernel(x, y, eps)

        m = x.size(0)
        n = y.size(0)

        # 去掉对角线，避免 self-similarity 偏置
        K_xx = K_xx - torch.diag_embed(torch.diagonal(K_xx))
        K_yy = K_yy - torch.diag_embed(torch.diagonal(K_yy))

        mmd = (
            K_xx.sum() / (m * (m - 1) + eps)
            + K_yy.sum() / (n * (n - 1) + eps)
            - 2 * K_xy.mean()
        )

        return mmd


    # ---------- forward ----------
    def forward(self, source, target):
        if self.kernel_type == "cosine":
            loss = self.cosine_mmd(source, target)
            return torch.nan_to_num(loss, nan=0.0)
        if self.kernel_type == "linear":
            loss = self.linear_mmd2(source, target)
            return torch.nan_to_num(loss, nan=0.0)

        if self.kernel_type == "rbf":
            B = source.size(0)
            K = self.gaussian_kernel(source, target)

            XX = K[:B, :B].mean()
            YY = K[B:, B:].mean()
            XY = K[:B, B:].mean()
            YX = K[B:, :B].mean()

            loss = XX + YY - XY - YX

            if self.debug:
                print(
                    f"[MMD][STAT] "
                    f"XX={XX.item():.3e}, "
                    f"YY={YY.item():.3e}, "
                    f"XY={XY.item():.3e}, "
                    f"YX={YX.item():.3e}, "
                    f"MMD={loss.item():.3e}"
                )

            # -------- 核心诊断逻辑 --------
            if not torch.isfinite(loss):
                print("[MMD][WARNING] RBF-MMD numerical failure")

                if self.auto_fallback:
                    print("[MMD][FALLBACK] Switching to linear MMD for this batch")
                    return self.linear_mmd2(source, target).detach()

                return torch.zeros_like(loss)

            return loss

        raise ValueError(f"Unknown kernel type: {self.kernel_type}")
