import torch.nn as nn

class surv_model(nn.Module):
    def __init__(self,hidden_dim1,hidden_dim2):
        super(surv_model,self).__init__()
        self.fc0=nn.Linear(hidden_dim1, hidden_dim2)
        self.fc1=nn.Linear(hidden_dim2,1)
        self.fc2=nn.Linear(hidden_dim1,1)
        self.bn = nn.BatchNorm1d(hidden_dim2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self,x):
        a=self.fc2(x.float())
        x=self.fc0(x.float())
        x=self.bn(x)
        x=self.relu(x)
        x=self.dropout(x)
        x=self.fc1(x)
        return x,a


