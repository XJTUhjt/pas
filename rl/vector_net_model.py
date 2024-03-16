import math
import copy
import numpy as np
import torch
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
import configparser
import logging
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from .pas_rnn_model import *
#common utils
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

def reshapeT(T, seq_length, nenv):
    shape = T.size()[1:]
    return T.unsqueeze(0).reshape((seq_length, nenv, *shape))

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
    
class Mlp(nn.Module):
    def __init__(self,
                 embed_dim,
                 mlp_ratio,
                 dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, int(embed_dim * mlp_ratio))
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        self.fc2 = nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x

class Embedding(nn.Module):
    def __init__(self, initial_dim, embed_dim=128, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(initial_dim, embed_dim)
        nn.init.xavier_normal_(self.embedding.weight)
        nn.init.constant_(self.embedding.bias, 0.0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        return x

class MultiheadSelfAttention(nn.Module):
    def __init__(self, feature_dim=128, num_heads=8, dropout=0.1, attention_dropout=0.1):
        super(MultiheadSelfAttention, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.attn_head_size = feature_dim // num_heads
        self.scales = self.attn_head_size ** -0.5

        # 为每个注意力头定义线性映射
        self.linear_query = nn.Linear(feature_dim, feature_dim)
        nn.init.xavier_normal_(self.linear_query.weight)
        nn.init.constant_(self.linear_query.bias, 0.0)
        self.linear_key = nn.Linear(feature_dim, feature_dim)
        nn.init.xavier_normal_(self.linear_key.weight)
        nn.init.constant_(self.linear_key.bias, 0.0)
        self.linear_value = nn.Linear(feature_dim, feature_dim)
        nn.init.xavier_normal_(self.linear_value.weight)
        nn.init.constant_(self.linear_value.bias, 0.0)

        self.linear_projection = nn.Linear(feature_dim, feature_dim)
        nn.init.xavier_normal_(self.linear_projection.weight)
        nn.init.constant_(self.linear_projection.bias, 0.0)

        self.attn_dropout = nn.Dropout(attention_dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # 线性映射得到查询、键和值
        q = self.linear_query(x).view(batch_size, seq_len, self.num_heads, -1)
        k = self.linear_key(x).view(batch_size, seq_len, self.num_heads, -1)
        v = self.linear_value(x).view(batch_size, seq_len, self.num_heads, -1)

        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scales
        attention_weights = F.softmax(scores, dim=-1)

        attention_weights = self.attn_dropout(attention_weights)

        # 加权求和得到每个头的输出
        weighted_sum = torch.matmul(attention_weights, v)

        # 将多头的输出进行连接并重塑形状（投影或者直接加和多头注意力）
        concatenated_representation = weighted_sum.view(batch_size, seq_len, -1)

        # 对连接后的表示进行投影
        projected_representation = self.linear_projection(concatenated_representation)
        aggregated_representation = self.proj_dropout(projected_representation)

        return aggregated_representation

class EncoderLayer(nn.Module):
    def __init__(self,
                 embed_dim=128,
                 num_heads=8,
                 mlp_ratio=4.,
                 dropout=0.1,
                 attention_dropout=0.1,
                 drop_path_ratio=0.1):
        super().__init__()
        self.attn_norm = nn.LayerNorm(embed_dim)

        self.attn = MultiheadSelfAttention(embed_dim,
                              num_heads,
                              dropout,
                              attention_dropout)

        self.mlp_norm = nn.LayerNorm(embed_dim)

        self.mlp = Mlp(embed_dim, mlp_ratio, dropout)

        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()

    def forward(self, x):

        # x = x + self.drop_path(self.attn(self.attn_norm(x)))
        # x = x + self.drop_path(self.mlp(self.mlp_norm(x)))

        h = x
        x = self.attn_norm(x)
        x = self.attn(x)

        x = x + h

        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = x + h

        return x   # [nenv, sqe, feature_dim]
    
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


#transformer encoder
class Trans_encoder(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 depth,
                 mlp_ratio=4.0,
                 dropout=0.,
                 attention_dropout=0.):
        super().__init__()
        layer_list = []
        for i in range(depth):
            encoder_layer = EncoderLayer(embed_dim,
                                         num_heads,
                                         mlp_ratio=mlp_ratio,
                                         dropout=dropout,
                                         attention_dropout=attention_dropout,)
            layer_list.append(copy.deepcopy(encoder_layer))
        self.layers = nn.ModuleList(layer_list)
        self.encoder_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        out = self.encoder_norm(x)
        return out
    
class Trans_encoder_withposition(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 depth,
                 mlp_ratio=4.0,
                 dropout=0.,
                 attention_dropout=0.):
        super().__init__()
        self.pos_encoder = PositionalEncoding(embed_dim)
        layer_list = []
        for i in range(depth):
            encoder_layer = EncoderLayer(embed_dim,
                                         num_heads,
                                         mlp_ratio=mlp_ratio,
                                         dropout=dropout,
                                         attention_dropout=attention_dropout,)
            layer_list.append(copy.deepcopy(encoder_layer))
        self.layers = nn.ModuleList(layer_list)
        self.encoder_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x)
        out = self.encoder_norm(x)
        return out

# polyline_encoder
#三种不同向量维度的维度映射到64，便于polyline encoder处理
class Occ_vector_embedding(Embedding):
    def __init__(self, initial_dim=11, embed_dim=64, dropout=0.1):
        super().__init__(initial_dim, embed_dim, dropout)
    
    def forward(self, x):
        x = super().forward(x)
        return x
    
class FOV_vector_embedding(Embedding):
    def __init__(self, initial_dim=2, embed_dim=64, dropout=0.1):
        super().__init__(initial_dim, embed_dim, dropout)
    
    def forward(self, x):
        x = super().forward(x)
        return x
    
class Humans_state_vector_embedding(Embedding):
    def __init__(self, initial_dim=8, embed_dim=64, dropout=0.1):
        super().__init__(initial_dim, embed_dim, dropout)

    def forward(self, x):
        x = super().forward(x)
        return x
    
class Robot_state_vector_embedding(Embedding):
    def __init__(self, initial_dim=10, embed_dim=64, dropout=0.1):
        super().__init__(initial_dim, embed_dim, dropout)
    
    def forward(self, x):
        x = super().forward(x)
        return x

class With_type_embedding(Embedding):
    def __init__(self, initial_dim=70, embed_dim=128, dropout=0.1):
        super().__init__(initial_dim, embed_dim, dropout)
    
    def forward(self, x):
        x = super().forward(x)
        return x

#polyline_transformer_encoder
class Polyline_trans_encoder(nn.Module):
    def __init__(self,
                 human_num,
                 embed_dim=64,
                 temporal_depth=1,
                 num_heads=8,
                 mlp_ratio=4,
                 dropout=0.1,
                 attention_dropout=0.1):
        super().__init__()
        self.human_num = human_num

        # create multi head self-attention layers
        self.polyline_encoder = Trans_encoder(embed_dim,
                                        num_heads,
                                        temporal_depth,
                                        mlp_ratio,
                                        dropout,
                                        attention_dropout)
        
        self.polyline_encoder_with_position = Trans_encoder_withposition(embed_dim,
                                        num_heads,
                                        temporal_depth,
                                        mlp_ratio,
                                        dropout,
                                        attention_dropout)

        
        self.mlp1 = nn.Linear(12*64, 64)
        nn.init.xavier_normal_(self.mlp1.weight)
        nn.init.constant_(self.mlp1.bias, 0.0)

        self.mlp2 = nn.Linear(8*64, 64)
        nn.init.xavier_normal_(self.mlp2.weight)
        nn.init.constant_(self.mlp2.bias, 0.0)

        self.mlp3 = nn.Linear(self.human_num*64, 64)
        nn.init.xavier_normal_(self.mlp3.weight)
        nn.init.constant_(self.mlp3.bias, 0.0)


    def forward(self, x, type):
        #每次传入的数据格式均为(batch_size(nenv), seq_len(saved_len), feature_dim)
        # batch_size, seq_len, feature_dim = x.shape

        if type == 'robot&humans':
            state = self.polyline_encoder_with_position(x)   # [nenv, seq_len, all_head_size]
            nenv = state.shape[0]
            state = state.view(nenv, 8*64)
            state = self.mlp2(state)
        elif type == 'FOV':
            state = self.polyline_encoder(x)   # [nenv, seq_len, all_head_size]
            nenv = state.shape[0]
            state = state.view(nenv, 12*64)
            state = self.mlp1(state)
        elif type == 'Occ':
            state = self.polyline_encoder_with_position(x)
            nenv = state.shape[0]
            state = state.view(nenv, 1, -1)
            state = self.mlp3(state)

        # state = torch.mean(state, dim=1, keepdim=True)  #池化seq_len

        # state = state[:, 0, :].unsqueeze(1)  #只取第一个（时间为t的)

        return state.unsqueeze(1)            # [nenv, 1, feature_dim]

#interaction_transformer_encoder
class Interaction_trans_encoder(nn.Module):
    def __init__(self,
                 view_dim,
                 embed_dim=128,
                 interaction_depth=1,
                 num_heads=8,
                 mlp_ratio=4,
                 dropout=0.1,
                 attention_dropout=0.1,
                 ):
        self.view_dim = view_dim
        super().__init__()

        # create multi head self-attention layers
        self.interaction_encoder = Trans_encoder(embed_dim,
                                        num_heads,
                                        interaction_depth,
                                        mlp_ratio,
                                        dropout,
                                        attention_dropout)
        
        self.mlp1 = nn.Linear(7*128, 7*128)      #(num_humans + robot_num + 1(fov) + num_occ,feature_dim)
        nn.init.xavier_normal_(self.mlp1.weight)
        nn.init.constant_(self.mlp1.bias, 0.0)

        self.mlp2 = nn.Linear(view_dim*128, 128)     #view_dim = (num_humans + robot_num + 1(fov) + num_occ,feature_dim)所有元素的个数
        nn.init.xavier_normal_(self.mlp2.weight)
        nn.init.constant_(self.mlp2.bias, 0.0)

        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):

        state = self.interaction_encoder(x)   # [nenv, seq_len, all_head_size]

        # state = torch.mean(state, dim=1, keepdim=True)  #池化seq_len

        # state = state[:, 0, :].unsqueeze(1)  #只取第一个（时间为t的)
        nenv = state.shape[0]

        state = state.view(nenv, self.view_dim*128)

        # state = self.mlp1(state)

        # state = self.avgpool(state).flatten(1)

        state = self.mlp2(state)


        return state.unsqueeze(1)            # [nenv, 1, feature_dim]


#total_pipeline_class
class Vector_net(nn.Module):
    """
    Class representing the vector-net model
    """
    def __init__(self, args, config, infer=False):
        """
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        """
        super(Vector_net, self).__init__()
        self.args=args
        self.config = config

        # self.seq_length = config.pas.sequence
        self.num_steps = args.num_steps
        self.nenv = args.num_processes
        self.nminibatch = args.num_mini_batch

        self.env_ebedding_dim = config.vector_net.env_dim
        self.output_size = 128
        self.human_num = config.sim.human_num
        self.is_occ = config.vector_net.is_occ
        self.is_fov = config.vector_net.is_fov

        self.Occ_vector_embedding = Occ_vector_embedding()
        self.FOV_vector_embedding = FOV_vector_embedding()
        self.Humans_state_vector_embedding = Humans_state_vector_embedding()
        self.Robot_state_vector_embedding = Robot_state_vector_embedding()

        self.Occ_vector_embedding_with_type = With_type_embedding()
        self.FOV_vector_embedding_with_type = With_type_embedding()
        self.Humans_state_vector_embedding_with_type = With_type_embedding()
        self.Robot_state_vector_embedding_with_type = With_type_embedding()

        self.env_feature_mlp = Mlp(self.env_ebedding_dim, 2)

        self.polyline_encoder = Polyline_trans_encoder(human_num=self.human_num)
        self.interaction_encoder = Interaction_trans_encoder(view_dim=(self.human_num + self.is_fov + self.is_occ + 1))

        
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(self.env_ebedding_dim, self.env_ebedding_dim)), nn.Tanh(),
            init_(nn.Linear(self.env_ebedding_dim, self.output_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(self.env_ebedding_dim, self.env_ebedding_dim)), nn.Tanh(),
            init_(nn.Linear(self.env_ebedding_dim, 32)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(32, 1))

        # PAS_RNN
        if self.config.pas.gridtype == 'global':
            robot_state_length = 9
        
        elif self.config.pas.gridtype == 'local':
            if self.config.action_space.kinematics == 'holonomic':    
                robot_state_length = 4 # robot state vector : rel_px, rel_py, vx, vy
            else:
                robot_state_length = 5 # robot state vector : rel_px, rel_py, theta, v, w
                
        vector_feat_dim = 128
        #把机器人向量映射到featuredim维度
        self.vector_linear = init_(nn.Linear(robot_state_length, vector_feat_dim)) 
        embed_input_size = args.rnn_output_size + vector_feat_dim
        
            
        if config.pas.encoder_type == 'vae':
            self.Sensor_VAE = Sensor_VAE(args, config)

            self.Label_VAE = Label_VAE(args)
        else:
            if config.pas.seq_flag:
                self.Sensor_CNN_seq = Sensor_CNN_seq(args, config)
            else:
                self.Sensor_CNN = Sensor_CNN(args)

        #最后被RNN编码的特征维度为之前pasRNN的输出维度和车辆映射的特征维度相加
        self.FeatureRNN = FeatureRNN(args, embed_input_size)
        #pas_rnn
        
        # self.FeatureRNN = FeatureRNN(args, embed_input_size)

        # self.rnd_module = RND(128, 256, 1)

    def forward(self, inputs, rnn_hxs, masks, infer=False):
        """[summary]
        Args:
            inputs  ['vector': (1*nenv, 1, vec_length) , 'grid':(nenv, seq_len, *grid_shape) or (seq_len*nenv, S, *grid_shape)
            
                        'FOV_points': (nenv , 1 , 12, 2 )]
                        'poly_occ_points': (nenv , 1, human_num, 11)
                        'ratated_humans_all_states': (nenv, 1, human_num * savedlength, 8)
                        
                        [human_id, self.px, self.py, self.vx, self.vy, self.theta, self.radius, timestamp]
                        'ratated_robot_all_states': (nenv, 1, savedlength, 8)

                        [100, self.px, self.py, self.vx, self.vy, self.theta, self.radius, timestamp]
                        'visible_id' : (nenv, 1, human_num)]
                        
            rnn_hxs ([type]): ['vector': (1*nenv, 1, hidden_size) , 'grid':(1*nenv, 1, hidden_size)]
            masks ([type]): [description] (seq_len*nenv, 1) or  (seq_len*nenv,seq_length)             
            infer (bool, optional): [description]. Defaults to False.
        """

        vector = inputs['vector'] 
        grid = inputs['grid'] 
        
        FOV_points = inputs['FOV_points']
        poly_occ_points = inputs['poly_occ_points']  #(nenv, 1, human_num, 11)
        rotated_humans_all_states = inputs['rotated_humans_all_states']
        rotated_robot_all_states = inputs['rotated_robot_all_states']
        visible_id = inputs['visible_id']
        
        assert not torch.isnan(FOV_points).any(), "NaN values found in FOV_points!"
        assert not torch.isnan(poly_occ_points).any(), "NaN values found in poly_occ_points!"
        assert not torch.isnan(rotated_humans_all_states).any(), "NaN values found in rotated_humans_all_states!"
        assert not torch.isnan(rotated_robot_all_states).any(), "NaN values found in rotated_robot_all_states!"

        #对输入的数据做向量化处理
        # vectorized_humans_states = self.vector_humans_states(rotated_humans_all_states)
        # vectorized_robot_states = self.vector_robot_states(rotated_robot_all_states)
        # vectorized_FOV_points = self.vector_FOV_points(FOV_points)
        # vectorized_poly_occ_poihnts = self.vector_poly_occ_points(poly_occ_points)

        #mlp embedding
        embemdded_FOV_points = self.FOV_vector_embedding(FOV_points)                                                            #(nenv, saved_len(1), 12poly ,64)

        embemdded_poly_occ_points = self.Occ_vector_embedding(poly_occ_points)                                                  #(nenv, 1, human_num, 11)->(nenv, 1, human_num, 64)                 
        # embemdded_poly_occ_points_list = embemdded_poly_occ_points.chunk(self.config.sim.human_num, dim=2)                      #human_num * (nenv, 1 , 1, 64)

        embemdded_rotated_humans_all_states = self.Humans_state_vector_embedding(rotated_humans_all_states)                     #(nenv, 1, human_num*saved_len, 64)
        embemdded_rotated_humans_all_states_list = embemdded_rotated_humans_all_states.chunk(self.config.sim.human_num, dim=2)  #human_num * (nenv, 1 , saved_len, 64)

        embemdded_rotated_robot_all_states = self.Robot_state_vector_embedding(rotated_robot_all_states)                        #(nenv, 1 ,saved_len, 64)


        #polyline transformer encoder，把输入映射到(batch_size, seq_len, feature_dim)维度, 获得(batch_size(nenv), 1, feature_dim)
        poly_encoded_FOV_vector = self.polyline_encoder(embemdded_FOV_points.squeeze(1),'FOV')

        # print(poly_encoded_FOV_vector[:, :, -10:])

        poly_encoded_robot_states_vector = self.polyline_encoder(embemdded_rotated_robot_all_states.squeeze(1), 'robot&humans')

        #对每一个行人状态进行编码
        poly_encoded_humans_states_list = []
        for item in embemdded_rotated_humans_all_states_list:
            poly_encoded_humans_states_list.append(self.polyline_encoder(item.squeeze(1), 'robot&humans'))
            
        #对每一个遮挡区域进行编码 
        # poly_encoded_occ_list = []
        # for item in embemdded_poly_occ_points_list:
        #     poly_encoded_occ_list.append(self.polyline_encoder(item.squeeze(1), 'Occ'))

        poly_encoded_occ_vector = self.polyline_encoder(embemdded_poly_occ_points.squeeze(1), 'Occ')                   #(12, 1, 1, 64)
        
        #polyline type encoder  把64维度拼接成70维度 机器人：0 行人：1 fov：2 遮挡区域：3  拼接长度为手动设置，会影响下面的mlp参数
        type_encodeed_poly_encoded_robot_states_vector = torch.cat((poly_encoded_robot_states_vector, torch.zeros(poly_encoded_robot_states_vector.shape[0], 1, 6).to('cuda')), dim=-1)
        type_encodeed_poly_encoded_FOV_vector = torch.cat((poly_encoded_FOV_vector, 2*torch.ones(poly_encoded_FOV_vector.shape[0], 1, 6).to('cuda')), dim=-1)

        # print(type_encodeed_poly_encoded_FOV_vector[:, :, -10:])

        for i in range(len(poly_encoded_humans_states_list)):
            poly_encoded_humans_states_list[i] = torch.cat((poly_encoded_humans_states_list[i], torch.ones(poly_encoded_humans_states_list[i].shape[0], 1, 6).to('cuda')), dim=-1)


        # for i in range(len(poly_encoded_occ_list)):
        #     poly_encoded_occ_list[i] = torch.cat((poly_encoded_occ_list[i], 3*torch.ones(poly_encoded_occ_list[i].shape[0], 1, 6).to('cuda')), dim=-1)
        
        poly_encoded_occ_vector_with_type = torch.cat((poly_encoded_occ_vector, 3*torch.ones(poly_encoded_occ_vector.shape[0], 1, 1, 6).to('cuda')), dim=-1)
        
        #把所有向量元素映射到128维
        Robot_states_vector = self.Robot_state_vector_embedding_with_type(type_encodeed_poly_encoded_robot_states_vector)
        FOV_vector = self.FOV_vector_embedding_with_type(type_encodeed_poly_encoded_FOV_vector)

        Humans_states_list = []
        for item in poly_encoded_humans_states_list:
            Humans_states_list.append(self.Humans_state_vector_embedding_with_type(item))

        # Occ_list = []
        # for item in poly_encoded_occ_list:
        #     Occ_list.append(self.Occ_vector_embedding_with_type(item))
        poly_encoded_occ_vector_with_typeembeding = self.Occ_vector_embedding_with_type(poly_encoded_occ_vector_with_type)

        
        #interaction transformer encoder  对于每个环境里的agent，输入的均是(nenv, 1，feature_dim_with_type)
        #做成seq,每个环境有自己的agent，每个agent应该只跟自己环境中的agent做trans，把tensor变换成(batch_size(nenv), seq_len(每个环境中所有元素的数量), feature_dim), 最后输出为(nenv,env_feature_dim)
        humans_seq_tensor = torch.cat(Humans_states_list,dim=1)
        Occ_seq_tensor = torch.squeeze(poly_encoded_occ_vector_with_typeembeding, dim=1)
        # env_element_vector_seq_tensor = torch.cat([Robot_states_vector, humans_seq_tensor, Occ_seq_tensor, FOV_vector],dim=1)
        # env_element_vector_seq_tensor = torch.cat([Robot_states_vector, humans_seq_tensor, FOV_vector],dim=1)
        env_element_vector_seq_tensor = torch.cat([humans_seq_tensor, FOV_vector, Robot_states_vector, Occ_seq_tensor],dim=1)


        #interaction transformer
        trans_output = self.interaction_encoder(env_element_vector_seq_tensor) #（nenv, humans_num + robot_num + 1(fov) + 1(occ), 128) -> (12,1,128)

        if self.config.pas.encoder_type == 'vae':
            mu, logvar, z = self.Sensor_VAE.encode(grid)
            
        #最后拼接机器人和其他的环境状态
        # if self.config.pas.encoder_type == 'vae':
        #     mu, logvar, z = self.Sensor_VAE.encode(grid)
        
        robot_vector = self.vector_linear(vector)
        env_add_robot_state = torch.cat((robot_vector, trans_output, z), dim=2)



        #mlp  get the vector for total env (nenv, 1, env_dim)   (12,1,256)
        env_feature_vector = self.env_feature_mlp(env_add_robot_state)

        #rl
        hidden_critic = self.critic(env_feature_vector)  
        hidden_actor = self.actor(env_feature_vector)


        return self.critic_linear(hidden_critic).squeeze(1), hidden_actor.squeeze(1), rnn_hxs, env_feature_vector.squeeze() #(12,1)  (12, 128)  (12,256)


    def vector_humans_states(self, ratated_humans_all_states):
        raise NotImplementedError
        
    def vector_robot_states(self, ratated_robot_all_states):
        raise NotImplementedError
        
    def vector_FOV_points(self, FOV_points):
        raise NotImplementedError
        
    def vector_poly_occ_points(self, poly_occ_points):
        raise NotImplementedError