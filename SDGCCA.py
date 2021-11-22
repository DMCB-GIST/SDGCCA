import torch
import torch.nn as nn
import torch.nn.init as init

# Weight & Bias Initialization
def initialization(net):
    if isinstance(net, nn.Linear):
        init.xavier_uniform(net.weight)
        init.zeros_(net.bias)

# SDGCCA with 3 Modality
class SDGCCA_3_M(nn.Module):
    def __init__(self, m1_embedding_list, m2_embedding_list, m3_embedding_list, top_k):
        super(SDGCCA_3_M, self).__init__()
        """
        m1_embedding_list = [d_1, ..., \bar{d_1}]
        m1_embedding_list = [d_2, ..., \bar{d_2}]
        m1_embedding_list = [d_3, ..., \bar{d_3}]
        top_k = k
        """

        # Embedding List of each modality
        m1_du0, m1_du1, m1_du2, m1_du3 = m1_embedding_list
        m2_du0, m2_du1, m2_du2, m2_du3 = m2_embedding_list
        m3_du0, m3_du1, m3_du2 = m3_embedding_list

        # Deep neural network of each modality
        self.model1 = nn.Sequential(
            nn.Linear(m1_du0, m1_du1), nn.Tanh(),
            nn.Linear(m1_du1, m1_du2), nn.Tanh(),
            nn.Linear(m1_du2, m1_du3), nn.Tanh())

        self.model2 = nn.Sequential(
            nn.Linear(m2_du0, m2_du1), nn.Tanh(),
            nn.Linear(m2_du1, m2_du2), nn.Tanh(),
            nn.Linear(m2_du2, m2_du3), nn.Tanh())

        self.model3 = nn.Sequential(
            nn.Linear(m3_du0, m3_du1), nn.Tanh(),
            nn.Linear(m3_du1, m3_du2), nn.Tanh())

        # Weight & Bias Initialization
        self.model1.apply(initialization)
        self.model2.apply(initialization)
        self.model3.apply(initialization)

        self.top_k = top_k

        # Projection matrix
        self.U = None

        # Softmax Function
        self.softmax = nn.Softmax(dim=1)

    # Input: Each modality
    # Output: Deep neural network output of the each modality = [H_1, H_2, H_3]
    def forward(self, x1, x2, x3):
        output1 = self.model1(x1)
        output2 = self.model2(x2)
        output3 = self.model3(x3)

        return output1, output2, output3

    # Calculate correlation loss
    def cal_loss(self, H_list, train=True):
        eps = 1e-8
        AT_list = []

        for H in H_list:
            assert torch.isnan(H).sum().item() == 0
            m = H.size(1)  # out_dim
            Hbar = H - H.mean(dim=1).repeat(m, 1).view(-1, m)
            assert torch.isnan(Hbar).sum().item() == 0

            A, S, B = Hbar.svd(some=True, compute_uv=True)
            A = A[:, :self.top_k]
            assert torch.isnan(A).sum().item() == 0

            S_thin = S[:self.top_k]
            S2_inv = 1. / (torch.mul(S_thin, S_thin) + eps)
            assert torch.isnan(S2_inv).sum().item() == 0

            T2 = torch.mul(torch.mul(S_thin, S2_inv), S_thin)
            assert torch.isnan(T2).sum().item() == 0

            T2 = torch.where(T2 > eps, T2, (torch.ones(T2.shape) * eps).to(H.device))
            T = torch.diag(torch.sqrt(T2))
            assert torch.isnan(T).sum().item() == 0

            T_unnorm = torch.diag(S_thin + eps)
            assert torch.isnan(T_unnorm).sum().item() == 0

            AT = torch.mm(A, T)
            AT_list.append(AT)

        M_tilde = torch.cat(AT_list, dim=1)
        assert torch.isnan(M_tilde).sum().item() == 0

        Q, R = M_tilde.qr()
        assert torch.isnan(R).sum().item() == 0
        assert torch.isnan(Q).sum().item() == 0

        U, lbda, _ = R.svd(some=False, compute_uv=True)
        assert torch.isnan(U).sum().item() == 0
        assert torch.isnan(lbda).sum().item() == 0

        G = Q.mm(U[:, :self.top_k])
        assert torch.isnan(G).sum().item() == 0

        U = []  # Projection Matrix

        # Get mapping to shared space
        views = H_list
        F = [H.shape[0] for H in H_list]  # features per view
        for idx, (f, view) in enumerate(zip(F, views)):
            _, R = torch.qr(view)
            Cjj_inv = torch.inverse((R.T.mm(R) + eps * torch.eye(view.shape[1], device=view.device)))
            assert torch.isnan(Cjj_inv).sum().item() == 0
            pinv = Cjj_inv.mm(view.T)

            U.append(pinv.mm(G))

        # If model training -> Change projection matrix
        # Else -> Using projection matrix for calculate correlation loss
        if train:
            self.U = U
            for i in range(len(self.U)):
                self.U[i] = nn.Parameter(torch.tensor(self.U[i]))

        _, S, _ = M_tilde.svd(some=True)

        assert torch.isnan(S).sum().item() == 0
        use_all_singular_values = False
        if not use_all_singular_values:
            S = S.topk(self.top_k)[0]
        corr = torch.sum(S)
        assert torch.isnan(corr).item() == 0

        loss = - corr
        return loss

    # SDGCCA prediction
    # Input: Each modality
    # Output: Soft voting of the label presentation of each modality
    def predict(self, x1, x2, x3):
        out1 = self.model1(x1)
        out2 = self.model2(x2)
        out3 = self.model3(x3)

        t1 = torch.matmul(out1, self.U[0])
        t2 = torch.matmul(out2, self.U[1])
        t3 = torch.matmul(out3, self.U[2])

        y_hat1 = torch.matmul(t1, torch.pinverse(self.U[3]))
        y_hat2 = torch.matmul(t2, torch.pinverse(self.U[3]))
        y_hat3 = torch.matmul(t3, torch.pinverse(self.U[3]))
        y_ensemble = (y_hat1+y_hat2+y_hat3)/3

        y_hat1 = self.softmax(y_hat1)
        y_hat2 = self.softmax(y_hat2)
        y_hat3 = self.softmax(y_hat3)
        y_ensemble = self.softmax(y_ensemble)

        return y_hat1, y_hat2, y_hat3, y_ensemble