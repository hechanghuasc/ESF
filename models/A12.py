import torch
from torch import nn
import torch.nn.functional as F



class AIfmlayer(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super(AIfmlayer, self).__init__()

        self.atten_embedding = nn.Linear(in_dim, out_dim)

        self.self_attns = nn.MultiheadAttention(out_dim, 2, dropout=0.5)
        self.attn_fc = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        atten_x = self.atten_embedding(x)
        # cross_term = atten_x.transpose(0, 1).unsqueeze(dim=2)
        cross_term = atten_x.unsqueeze(dim=0)
        cross_term, _ = self.self_attns(cross_term, cross_term, cross_term)
        # cross_term = cross_term.transpose(0, 1).squeeze(dim=2)
        cross_term = cross_term.squeeze(dim=0)
        embed_x = self.attn_fc(cross_term)

        return embed_x



class DAnet(nn.Module):
    def __init__(self, args):
        super(DAnet, self).__init__()
        # 初始化参数
        self.args = args

        # 第一模块
        self.fm_layer_1 = AIfmlayer(self.args.dim_data, self.args.dim_embed)
        self.residual_layer_1 = nn.Sequential(
            nn.Linear(self.args.dim_data, self.args.dim_embed),
            nn.BatchNorm1d(self.args.dim_embed),
            nn.ELU(),
            nn.Linear(self.args.dim_embed, self.args.dim_embed),
            # nn.BatchNorm1d(out_dim),
            nn.ELU(),
        )
        # self.dropout_1 = nn.Dropout(0.5)
        # self.bn_1 = nn.BatchNorm1d(self.args.dim_data + self.args.dim_embed * 2)
        self.mlp_1 = nn.Linear(self.args.dim_data + self.args.dim_embed * 2, self.args.dim_embed)

        # 第二模块
        self.fm_layer_2 = AIfmlayer(self.args.dim_embed, self.args.dim_embed)
        self.residual_layer_2 = nn.Sequential(
            nn.Linear(self.args.dim_embed, self.args.dim_embed),
            nn.BatchNorm1d(self.args.dim_embed),
            nn.ELU(),
            nn.Linear(self.args.dim_embed, self.args.dim_embed),
            # nn.BatchNorm1d(out_dim),
            nn.ELU(),
        )
        # self.dropout_2 = nn.Dropout(0.5)
        # self.bn_2 = nn.BatchNorm1d(self.args.dim_data + self.args.dim_embed * 3)
        self.mlp_2 = nn.Linear(self.args.dim_data + self.args.dim_embed * 3, self.args.dim_embed)

        # 第三模块
        self.fm_layer_3 = AIfmlayer(self.args.dim_embed, self.args.dim_embed)
        self.residual_layer_3 = nn.Sequential(
            nn.Linear(self.args.dim_embed, self.args.dim_embed),
            nn.BatchNorm1d(self.args.dim_embed),
            nn.ELU(),
            nn.Linear(self.args.dim_embed, self.args.dim_embed),
            # nn.BatchNorm1d(out_dim),
            nn.ELU(),
        )
        self.mlp_3 = nn.Linear(self.args.dim_data + self.args.dim_embed * 4, self.args.dim_embed)
        # self._3 = nn.Dropout(0.5)
        # self.bn_3 = nn.BatchNorm1d(self.args.dim_data + self.args.dim_embed * 4)

        self.logr = nn.Linear(self.args.dim_embed, 1)

    def forward(self, x):

        fm_1 = self.fm_layer_1(x)
        residual_layer_1 = self.residual_layer_1(x)
        cat_1 = torch.cat((x, fm_1, residual_layer_1), dim=1)
        # cat_1 = self.bn_1(cat_1)
        # cat_1 = self.dropout_1(cat_1)
        mlp_1 = self.mlp_1(cat_1)

        fm_2 = self.fm_layer_2(mlp_1)
        residual_layer_2 = self.residual_layer_2(mlp_1)
        cat_2 = torch.cat((x, mlp_1, fm_2, residual_layer_2), dim=1)
        # cat_2 = self.bn_2(cat_2)
        # cat_2 = self.dropout_2(cat_2)
        mlp_2 = self.mlp_2(cat_2)

        fm_3 = self.fm_layer_3(mlp_2)
        residual_layer_3 = self.residual_layer_3(mlp_2)
        cat_3 = torch.cat((x, mlp_1, mlp_2, fm_3, residual_layer_3), dim=1)
        # cat_3 = self.bn_3(cat_3)
        # cat_3 = self.dropout_3(cat_3)
        mlp_3 = self.mlp_3(cat_3)

        out = torch.sigmoid(self.logr(mlp_3))

        return out


class DAnet_wo_fm(nn.Module):
    def __init__(self, args):
        super(DAnet_wo_fm, self).__init__()
        # 初始化参数
        self.args = args

        self.residual_layer_1 = nn.Sequential(
            nn.Linear(self.args.dim_data, self.args.dim_embed),
            nn.BatchNorm1d(self.args.dim_embed),
            nn.ELU(),
            nn.Linear(self.args.dim_embed, self.args.dim_embed),
            # nn.BatchNorm1d(out_dim),
            nn.ELU(),
        )
        self.dropout_1 = nn.Dropout(0.5)
        self.bn_1 = nn.BatchNorm1d(self.args.dim_data + self.args.dim_embed * 1)
        self.mlp_1 = nn.Linear(self.args.dim_data + self.args.dim_embed * 1, self.args.dim_embed)

        self.residual_layer_2 = nn.Sequential(
            nn.Linear(self.args.dim_embed, self.args.dim_embed),
            nn.BatchNorm1d(self.args.dim_embed),
            nn.ELU(),
            nn.Linear(self.args.dim_embed, self.args.dim_embed),
            # nn.BatchNorm1d(out_dim),
            nn.ELU(),
        )
        self.dropout_2 = nn.Dropout(0.5)
        self.bn_2 = nn.BatchNorm1d(self.args.dim_data + self.args.dim_embed * 2)
        self.mlp_2 = nn.Linear(self.args.dim_data + self.args.dim_embed * 2, self.args.dim_embed)

        self.residual_layer_3 = nn.Sequential(
            nn.Linear(self.args.dim_embed, self.args.dim_embed),
            nn.BatchNorm1d(self.args.dim_embed),
            nn.ELU(),
            nn.Linear(self.args.dim_embed, self.args.dim_embed),
            # nn.BatchNorm1d(out_dim),
            nn.ELU(),
        )
        # self.mlp_3 = nn.Linear(self.args.dim_data + self.args.dim_embed * 4, out_dim)
        self._3 = nn.Dropout(0.5)
        self.bn_3 = nn.BatchNorm1d(self.args.dim_data + self.args.dim_embed * 3)
        self.logr = nn.Linear(self.args.dim_data + self.args.dim_embed * 3, 1)

    def forward(self, x):

        residual_layer_1 = self.residual_layer_1(x)
        cat_1 = torch.cat((x, residual_layer_1), dim=1)
        # cat_1 = self.bn_1(cat_1)
        # cat_1 = self.dropout_1(cat_1)
        mlp_1 = self.mlp_1(cat_1)

        residual_layer_2 = self.residual_layer_2(mlp_1)
        cat_2 = torch.cat((x, mlp_1, residual_layer_2), dim=1)
        # cat_2 = self.bn_2(cat_2)
        # cat_2 = self.dropout_2(cat_2)
        mlp_2 = self.mlp_2(cat_2)

        residual_layer_3 = self.residual_layer_3(mlp_2)
        cat_3 = torch.cat((x, mlp_1, mlp_2, residual_layer_3), dim=1)
        # cat_3 = self.bn_3(cat_3)
        # cat_3 = self.dropout_3(cat_3)

        out = torch.sigmoid(self.logr(cat_3))

        return out


class DAnet_wo_dense(nn.Module):
    def __init__(self, args):
        super(DAnet_wo_dense, self).__init__()
        # 初始化参数
        self.args = args

        self.fm_layer_1 = AIfmlayer(self.args.dim_data, self.args.dim_embed)
        self.residual_layer_1 = nn.Sequential(
            nn.Linear(self.args.dim_data, self.args.dim_embed),
            nn.BatchNorm1d(self.args.dim_embed),
            nn.ELU(),
            nn.Linear(self.args.dim_embed, self.args.dim_embed),
            # nn.BatchNorm1d(out_dim),
            nn.ELU(),
        )
        self.dropout_1 = nn.Dropout(0.5)
        self.bn_1 = nn.BatchNorm1d(self.args.dim_embed * 2)
        self.mlp_1 = nn.Linear(self.args.dim_embed * 2, self.args.dim_embed)

        self.fm_layer_2 = AIfmlayer(self.args.dim_embed, self.args.dim_embed)
        self.residual_layer_2 = nn.Sequential(
            nn.Linear(self.args.dim_embed, self.args.dim_embed),
            nn.BatchNorm1d(self.args.dim_embed),
            nn.ELU(),
            nn.Linear(self.args.dim_embed, self.args.dim_embed),
            # nn.BatchNorm1d(out_dim),
            nn.ELU(),
        )
        self.dropout_2 = nn.Dropout(0.5)
        self.bn_2 = nn.BatchNorm1d(self.args.dim_embed * 2)
        self.mlp_2 = nn.Linear(self.args.dim_embed * 2, self.args.dim_embed)

        self.fm_layer_3 = AIfmlayer(self.args.dim_embed, self.args.dim_embed)
        self.residual_layer_3 = nn.Sequential(
            nn.Linear(self.args.dim_embed, self.args.dim_embed),
            nn.BatchNorm1d(self.args.dim_embed),
            nn.ELU(),
            nn.Linear(self.args.dim_embed, self.args.dim_embed),
            # nn.BatchNorm1d(out_dim),
            nn.ELU(),
        )
        # self.mlp_3 = nn.Linear(self.args.dim_data + self.args.dim_embed * 4, out_dim)
        self._3 = nn.Dropout(0.5)
        self.bn_3 = nn.BatchNorm1d(self.args.dim_embed * 2)
        self.logr = nn.Linear(self.args.dim_embed * 2, 1)

    def forward(self, x):

        fm_1 = self.fm_layer_1(x)
        residual_layer_1 = self.residual_layer_1(x)
        cat_1 = torch.cat((fm_1, residual_layer_1), dim=1)
        # cat_1 = self.bn_1(cat_1)
        # cat_1 = self.dropout_1(cat_1)
        mlp_1 = self.mlp_1(cat_1)

        fm_2 = self.fm_layer_2(mlp_1)
        residual_layer_2 = self.residual_layer_2(mlp_1)
        cat_2 = torch.cat((fm_2, residual_layer_2), dim=1)
        # cat_2 = self.bn_2(cat_2)
        # cat_2 = self.dropout_2(cat_2)
        mlp_2 = self.mlp_2(cat_2)

        fm_3 = self.fm_layer_3(mlp_2)
        residual_layer_3 = self.residual_layer_3(mlp_2)
        cat_3 = torch.cat((fm_3, residual_layer_3), dim=1)
        # cat_3 = self.bn_3(cat_3)
        # cat_3 = self.dropout_3(cat_3)

        out = torch.sigmoid(self.logr(cat_3))

        return out



class DAnet_wo_fm_dense(nn.Module):
    def __init__(self, args):
        super(DAnet_wo_fm_dense, self).__init__()
        # 初始化参数
        self.args = args

        self.encoder = nn.Linear(self.args.dim_data, self.args.dim_embed)
        self.out = nn.Linear(self.args.dim_embed, 1)

    def forward(self, x):
        x_embed = F.relu(self.encoder(x))
        out = torch.sigmoid(self.out(x_embed))

        return out


class DAnet_rubust_test(nn.Module):
    def __init__(self, args):
        super(DAnet_rubust_test, self).__init__()
        # 初始化参数
        self.args = args

        # 第一模块
        self.fm_layer_1 = AIfmlayer(self.args.dim_data, self.args.dim_embed)
        self.residual_layer_1 = nn.Sequential(
            nn.Linear(self.args.dim_data, self.args.dim_embed),
            nn.BatchNorm1d(self.args.dim_embed),
            nn.ELU(),
            nn.Linear(self.args.dim_embed, self.args.dim_embed),
            # nn.BatchNorm1d(out_dim),
            nn.ELU(),
        )
        self.dropout_1 = nn.Dropout(0.5)
        self.bn_1 = nn.BatchNorm1d(self.args.dim_data + self.args.dim_embed * 2)
        self.mlp_1 = nn.Linear(self.args.dim_data + self.args.dim_embed * 2, self.args.dim_embed)

        # 第二模块
        self.fm_layer_2 = AIfmlayer(self.args.dim_embed, self.args.dim_embed)
        self.residual_layer_2 = nn.Sequential(
            nn.Linear(self.args.dim_embed, self.args.dim_embed),
            nn.BatchNorm1d(self.args.dim_embed),
            nn.ELU(),
            nn.Linear(self.args.dim_embed, self.args.dim_embed),
            # nn.BatchNorm1d(out_dim),
            nn.ELU(),
        )
        # self.dropout_2 = nn.Dropout(0.5)
        # self.bn_2 = nn.BatchNorm1d(self.args.dim_data + self.args.dim_embed * 3)
        self.mlp_2 = nn.Linear(self.args.dim_data + self.args.dim_embed * 3, self.args.dim_embed)


        # 第三模块
        self.fm_layer_3 = AIfmlayer(self.args.dim_embed, self.args.dim_embed)
        self.residual_layer_3 = nn.Sequential(
            nn.Linear(self.args.dim_embed, self.args.dim_embed),
            nn.BatchNorm1d(self.args.dim_embed),
            nn.ELU(),
            nn.Linear(self.args.dim_embed, self.args.dim_embed),
            # nn.BatchNorm1d(out_dim),
            nn.ELU(),
        )
        self.mlp_3 = nn.Linear(self.args.dim_data + self.args.dim_embed * 4, self.args.dim_embed)

        # self.dropout_3 = nn.Dropout(0.5)
        # self.bn_3 = nn.BatchNorm1d(self.args.dim_data + self.args.dim_embed * 4)

        # 第4模块
        self.fm_layer_4 = AIfmlayer(self.args.dim_embed, self.args.dim_embed)
        self.residual_layer_4 = nn.Sequential(
            nn.Linear(self.args.dim_embed, self.args.dim_embed),
            nn.BatchNorm1d(self.args.dim_embed),
            nn.ELU(),
            nn.Linear(self.args.dim_embed, self.args.dim_embed),
            # nn.BatchNorm1d(out_dim),
            nn.ELU(),
        )
        self.mlp_4 = nn.Linear(self.args.dim_data + self.args.dim_embed * 5, self.args.dim_embed)

        # 第5模块
        self.fm_layer_5 = AIfmlayer(self.args.dim_embed, self.args.dim_embed)
        self.residual_layer_5 = nn.Sequential(
            nn.Linear(self.args.dim_embed, self.args.dim_embed),
            nn.BatchNorm1d(self.args.dim_embed),
            nn.ELU(),
            nn.Linear(self.args.dim_embed, self.args.dim_embed),
            # nn.BatchNorm1d(out_dim),
            nn.ELU(),
        )
        self.mlp_5 = nn.Linear(self.args.dim_data + self.args.dim_embed * 6, self.args.dim_embed)

        # 第6模块
        self.fm_layer_6 = AIfmlayer(self.args.dim_embed, self.args.dim_embed)
        self.residual_layer_6 = nn.Sequential(
            nn.Linear(self.args.dim_embed, self.args.dim_embed),
            nn.BatchNorm1d(self.args.dim_embed),
            nn.ELU(),
            nn.Linear(self.args.dim_embed, self.args.dim_embed),
            # nn.BatchNorm1d(out_dim),
            nn.ELU(),
        )
        self.mlp_6 = nn.Linear(self.args.dim_data + self.args.dim_embed * 7, self.args.dim_embed)

        # 第7模块
        self.fm_layer_7 = AIfmlayer(self.args.dim_embed, self.args.dim_embed)
        self.residual_layer_7 = nn.Sequential(
            nn.Linear(self.args.dim_embed, self.args.dim_embed),
            nn.BatchNorm1d(self.args.dim_embed),
            nn.ELU(),
            nn.Linear(self.args.dim_embed, self.args.dim_embed),
            # nn.BatchNorm1d(out_dim),
            nn.ELU(),
        )
        self.mlp_7 = nn.Linear(self.args.dim_data + self.args.dim_embed * 8, self.args.dim_embed)

        # 第8模块
        self.fm_layer_8 = AIfmlayer(self.args.dim_embed, self.args.dim_embed)
        self.residual_layer_8 = nn.Sequential(
            nn.Linear(self.args.dim_embed, self.args.dim_embed),
            nn.BatchNorm1d(self.args.dim_embed),
            nn.ELU(),
            nn.Linear(self.args.dim_embed, self.args.dim_embed),
            # nn.BatchNorm1d(out_dim),
            nn.ELU(),
        )
        self.mlp_8 = nn.Linear(self.args.dim_data + self.args.dim_embed * 9, self.args.dim_embed)



        self.logr = nn.Linear(self.args.dim_embed, 1)

    def forward(self, x):
        fm_1 = self.fm_layer_1(x)
        residual_layer_1 = self.residual_layer_1(x)
        cat_1 = torch.cat((x, fm_1, residual_layer_1), dim=1)
        mlp_1 = self.mlp_1(cat_1)

        fm_2 = self.fm_layer_2(mlp_1)
        residual_layer_2 = self.residual_layer_2(mlp_1)
        cat_2 = torch.cat((x, mlp_1, fm_2, residual_layer_2), dim=1)
        mlp_2 = self.mlp_2(cat_2)

        fm_3 = self.fm_layer_3(mlp_2)
        residual_layer_3 = self.residual_layer_3(mlp_2)
        cat_3 = torch.cat((x, mlp_1, mlp_2, fm_3, residual_layer_3), dim=1)
        mlp_3 = self.mlp_3(cat_3)

        fm_4 = self.fm_layer_4(mlp_3)
        residual_layer_4 = self.residual_layer_4(mlp_3)
        cat_4 = torch.cat((x, mlp_1, mlp_2, mlp_3, fm_4, residual_layer_4), dim=1)
        mlp_4 = self.mlp_4(cat_4)

        fm_5 = self.fm_layer_5(mlp_4)
        residual_layer_5 = self.residual_layer_5(mlp_4)
        cat_5 = torch.cat((x, mlp_1, mlp_2, mlp_3, mlp_4, fm_5, residual_layer_5), dim=1)
        mlp_5 = self.mlp_5(cat_5)

        fm_6 = self.fm_layer_6(mlp_5)
        residual_layer_6 = self.residual_layer_6(mlp_5)
        cat_6 = torch.cat((x, mlp_1, mlp_2, mlp_3, mlp_4, mlp_5, fm_6, residual_layer_6), dim=1)
        mlp_6 = self.mlp_6(cat_6)

        fm_7 = self.fm_layer_7(mlp_6)
        residual_layer_7 = self.residual_layer_6(mlp_6)
        cat_7 = torch.cat((x, mlp_1, mlp_2, mlp_3, mlp_4, mlp_5, mlp_6, fm_7, residual_layer_7), dim=1)
        mlp_7 = self.mlp_7(cat_7)

        fm_8 = self.fm_layer_8(mlp_7)
        residual_layer_8 = self.residual_layer_8(mlp_7)
        cat_8 = torch.cat((x, mlp_1, mlp_2, mlp_3, mlp_4, mlp_5, mlp_6, mlp_7, fm_8, residual_layer_8), dim=1)
        mlp_8 = self.mlp_8(cat_8)




        out = torch.sigmoid(self.logr(mlp_8))

        return out