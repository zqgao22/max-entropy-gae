import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
class GCN(nn.Module):

    def __init__(self, dim_in, max_num_nodes,c,d,e):
        super(GCN, self).__init__()
        # self.A = A
        c = 29
        d = 29
        e = 20
        self.max_num_nodes = max_num_nodes
        self.encoder_0 = nn.Linear(dim_in * 29, dim_in * d, bias = False)
        self.encoder_1 = nn.Linear(dim_in * 29, dim_in * d, bias = False)
        self.encoder_2 = nn.Linear(dim_in * 29, dim_in * d, bias = False)
        self.encoder_3 = nn.Linear(dim_in * 29, dim_in * d, bias = False)
        self.encoder_4 = nn.Linear(dim_in * 29, dim_in * d, bias=False)

        self.encoder_00 = nn.Linear(dim_in * d, dim_in * c, bias = True)
        self.encoder_11 = nn.Linear(dim_in * d, dim_in * c, bias = True)
        self.encoder_22 = nn.Linear(dim_in * d, dim_in * c, bias = True)
        self.encoder_33 = nn.Linear(dim_in * d, dim_in * c, bias = True)
        self.encoder_44 = nn.Linear(dim_in * d, dim_in * c, bias=True)

        self.decoder_0 = nn.Linear(dim_in * c, dim_in * e, bias=True)
        self.decoder_1 = nn.Linear(dim_in * c, dim_in * e, bias=True)
        self.decoder_2 = nn.Linear(dim_in * c, dim_in * e, bias=True)
        self.decoder_3 = nn.Linear(dim_in * c, dim_in * e, bias=True)
        self.decoder_4 = nn.Linear(dim_in * c, dim_in * e, bias=True)

        self.decoder_00 = nn.Linear(dim_in * e, dim_in * 29, bias=True)
        self.decoder_11 = nn.Linear(dim_in * e, dim_in * 29, bias=True)
        self.decoder_22 = nn.Linear(dim_in * e, dim_in * 29, bias=True)
        self.decoder_33 = nn.Linear(dim_in * e, dim_in * 29, bias=True)
        self.decoder_44 = nn.Linear(dim_in * e, dim_in * 29, bias=True)
        self.concat = nn.Linear(dim_in * 29 * 5, dim_in * 29, bias=True)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.lrelu = F.leaky_relu
        self.bn11 = nn.BatchNorm1d(c)
        self.bn12 = nn.BatchNorm1d(c)
        self.bn13 = nn.BatchNorm1d(c)
        self.bn14 = nn.BatchNorm1d(c)
        self.bn10 = nn.BatchNorm1d(c)

        self.bn2 = nn.BatchNorm1d(29)
        self.a = nn.Parameter(torch.FloatTensor([1]))
        self.b = nn.Parameter(torch.FloatTensor([1]))
        self.c = nn.Parameter(torch.FloatTensor([1]))
        self.d = nn.Parameter(torch.FloatTensor([1]))
        # self.softmax = F.softmax()

    def reset_parameters(self):
        if self.linear.weight is not None:
            init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)

    def forward(self, A_ori, X, A_WL1, A_WL2, A_WL3, A_WL4, A_WL_I_1, A_WL_I_2, A_WL_I_3, A_WL_I_4, A_WL_I_0):
        # A_inverse = torch.squeeze(A_inverse, 0)
        # X = self.sigmoid(X)
        ori_num = A_ori.size(0)
        C1 = self.lrelu(self.encoder_1(A_WL1.mm(X)))
        C2 = self.lrelu(self.encoder_2(A_WL2.mm(X)))
        C3 = self.lrelu(self.encoder_3(A_WL3.mm(X)))
        C0 = self.lrelu(self.encoder_0(A_ori.mm(X)))
        C4 = self.lrelu(self.encoder_4(A_WL4.mm(X)))

        # C0 = self.bn10(self.relu(self.encoder_00(C0)))
        # C1 = self.bn10(self.relu(self.encoder_11(C1)))
        # C2 = self.bn10(self.relu(self.encoder_22(C2)))
        # C3 = self.bn10(self.relu(self.encoder_33(C3)))
        # C4 = self.bn10(self.relu(self.encoder_33(C4)))
        C0 = self.lrelu(self.encoder_00(C0))
        C1 = self.lrelu(self.encoder_11(C1))
        C2 = self.lrelu(self.encoder_22(C2))
        C3 = self.lrelu(self.encoder_33(C3))
        C4 = self.lrelu(self.encoder_33(C4))
        # Z = F.dropout(Z, p=0.7)
        Z0 = self.lrelu(self.decoder_00(self.lrelu(self.decoder_0(A_WL_I_0.mm(C0)))))
        Z1 = self.lrelu(self.decoder_11(self.lrelu(self.decoder_1(A_WL_I_1.mm(C1)))))
        Z2 = self.lrelu(self.decoder_22(self.lrelu(self.decoder_2(A_WL_I_2.mm(C2)))))
        Z3 = self.lrelu(self.decoder_33(self.lrelu(self.decoder_3(A_WL_I_3.mm(C3)))))
        Z4 = self.lrelu(self.decoder_44(self.lrelu(self.decoder_4(A_WL_I_4.mm(C4)))))

        # Z0 = self.lrelu(self.decoder_00(self.lrelu(self.decoder_0(C0))))
        # Z1 = self.lrelu(self.decoder_11(self.lrelu(self.decoder_1(C1))))
        # Z2 = self.lrelu(self.decoder_22(self.lrelu(self.decoder_2(C2))))
        # Z3 = self.lrelu(self.decoder_33(self.lrelu(self.decoder_3(C3))))

        # Z0 = self.lrelu(self.decoder_00(self.lrelu(self.decoder_0(A_ori.mm(C0)))))
        # Z1 = self.lrelu(self.decoder_11(self.lrelu(self.decoder_1(A_ori.mm(C1)))))
        # Z2 = self.lrelu(self.decoder_22(self.lrelu(self.decoder_2(A_ori.mm(C2)))))
        # Z3 = self.lrelu(self.decoder_33(self.lrelu(self.decoder_3(A_ori.mm(C3)))))

        Z = torch.cat((Z0,Z1,Z2,Z3,Z4), 1)
        Z = self.concat(Z)
        # Z = F.dropout(Z, p=0.7)
        return self.lrelu(Z), C0, C1, C2, C3, C4
