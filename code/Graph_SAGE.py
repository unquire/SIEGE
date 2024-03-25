import sys
sys.path.append('/home/YMT/SIEGE/')

from utils import prepare_folder, EarlyStopping, GNNDataset
import argparse
import torch
from torch_geometric.loader import ImbalancedSampler, NeighborLoader  # , NeighborSampler, DataLoader,
from models import SAGE_NeighSampler
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def get_args():
    # 添加命令行字段
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='sage', help='mlp, sage, gat, rgcn, multignn,gat')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16384)  # 16384
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')

    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--node_attr_channels', type=int, default=32)
    parser.add_argument('--time_channels', type=int, default=10)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--out_channels', type=int, default=2)
    parser.add_argument('--num_msg_passing_layers', type=int, default=1)
    parser.add_argument('--num_out_layers', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=5)
    args = parser.parse_args(args=[])  # 把括号里面的去掉就可以按照输入命令行计算了
    return args


# --------------------------- Train Function Defination -------------------------
def train(epoch, train_loader, model, data, optimizer, device, no_conv=False, ):
    model.train()
    model = model.to(device)
    total_loss = 0
    i = 1
    for batch in train_loader:
        # Data(x=[2261039, 54], edge_index=[2, 3881529], contract_cluter=[2261039],
        # next_features=[2261039, 27], n_id=[2261039], e_id=[3881529], input_id=[16384], batch_size=16384)
        optimizer.zero_grad()
        recon_x1, recon_x2, recon_x3 = model(batch.x.to(device), batch.edge_index.to(device))
        loss = model.self_supervised_loss(batch.x[:batch.batch_size].to(device), recon_x1[:batch.batch_size], recon_x2[:batch.batch_size], recon_x3[:batch.batch_size],  batch.next_features[:batch.batch_size].to(device), batch.contract_cluter[:batch.batch_size].to(device))
        # loss = F.nll_loss(out, batch.y[:batch.batch_size].to(device))
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新模型参数
        total_loss += float(loss)
        torch.cuda.empty_cache()
        print("第", epoch, "轮   第", i, "批训练结束,loss:", float(loss))
        i = i+1
    loss = total_loss / len(train_loader)
    return loss


# --------------------------- Parameters Defination -------------------------
gcn_parameters = {'lr': 0.001
    , 'num_layers': 2
    , 'out_class': 6
    , 'hidden_channels': 64
    , 'dropout': 0.5
    , 'batchnorm': False
    , 'l2': 5e-7
                  }


def main():
    # -----------------------------get parameters-----------------------------
    args = get_args()
    print("程序开始运行")
    no_conv = False
    if args.model in ['mlp']: no_conv = True

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    # 20230825
    device = torch.device(device)

    # --------------------------- Data Load and Process -------------------------
    dataset = GNNDataset(root='../graph_data/')  # , transform=T.ToSparseTensor())
    # 加载地址转换数据
    print("接收到数据")
    data = dataset[0]
    print(data)
    # 输出维度
    nlabels = len(data.x[0])//2

    # --------------------------- Feature Process -------------------------
    x = data.x
    # print("origin_x:",x)
    # 计算每个特征的均值和标准差
    mean = x.mean(0)
    std = x.std(0)
    # 检查标准差是否为0
    zero_std_mask = std == 0
    # 将标准差为0的特征的标准差改为1
    std[zero_std_mask] = 1
    # 对特征进行标准化
    x = (x - mean) / std
    # x = (x - x.mean(0)) / x.std(0)  # 特征标准化
    data.x = x.float()
    #print("new_x:",x)

    next_features = data.next_features
    #print("origin_next_features:", next_features)
    mean = next_features.mean(0)
    std = next_features.std(0)
    # 检查标准差是否为0
    zero_std_mask = std == 0
    # 将标准差为0的特征的标准差改为1
    std[zero_std_mask] = 1
    # 对特征进行标准化
    next_features = (next_features - mean) / std
    # x = (x - x.mean(0)) / x.std(0)  # 特征标准化
    data.next_features = next_features.float()
    #print("new_next_features:", next_features)
    model_dir = prepare_folder('../save_model', args.model)
    print('model_dir:', model_dir)

    # --------------------------- DataLoader Defination -------------------------

    train_loader = NeighborLoader(data, num_neighbors=[-1], batch_size=args.batch_size, num_workers=args.num_workers)
    print("训练批次:",len(train_loader))
    # --------------------------- Model Defination -------------------------
    if args.model in ['sage']:
        para_dict = gcn_parameters
        model_para = gcn_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')
        if args.model == 'sage':
            model = SAGE_NeighSampler(in_channels=len(data.x[0]), out_channels=nlabels, **model_para).to(device)
            # print("model:",model)
    print(f'Model {args.model} initialized')
    # print(model)
    patience = args.patience
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=para_dict['lr'], weight_decay=para_dict['l2'])
    min_loss = 1e8
    # print("epoch:", args.epochs, " train_loader:", train_loader, " model:", args.model, " data:", data, " train_idx:",
    #       train_idx, " optimizer:", optimizer, " device:", device, " no_conv:", no_conv)

    # --------------------------- Train and Test -------------------------
    for epoch in range(1, args.epochs + 1):
        print("\n\n第",epoch,"轮开始训练")
        loss = train(epoch, train_loader, model, data,optimizer, device, no_conv)
        if loss < min_loss:
            min_loss = loss
            torch.save(model.state_dict(), model_dir + 'model.pt')
        if early_stopping.early_stop:
            torch.save(model.state_dict(), model_dir + 'model.pt')
            break

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  )

    # 这里写一个推断的模型


if __name__ == "__main__":
    main()
