
from .ssra import *
from .base_block import *


class Encoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=14,
                 local_range = 1,region_num = 8,cuda=0,device=1):
        super(Encoder, self).__init__()

        self.hidden_dim = hidden_dim
        cuda_ = not cuda
        print(device)
        self.device = 'cuda:'+str(device) if cuda_ else 'cpu'


        self.corr = HFComp(dim=hidden_dim)

        self.ssra = SSRA_stage(hidden_dim = hidden_dim,region_num = region_num)

        self.encoder_layer = ConvGRU(hidden_dim=hidden_dim, input_dim=input_dim)

        self.encoder_layer1_b = ConvGRU(hidden_dim=hidden_dim, input_dim=input_dim, is_backward=True)

        self.conv_inp_1 = nn.Conv2d(1, self.hidden_dim, 3, 1, 1, bias=True)
        self.conv_inp_2 = nn.Conv2d(self.hidden_dim, hidden_dim, 3, 1, 1, bias=True)
       
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def in2fea(self, x):
        out = self.lrelu(self.conv_inp_1(x))
        out = self.lrelu(self.conv_inp_2(out))
        out = out + x
        return out

    def forward(self, x):
        
        if len(x.shape) == 4:
            x = torch.unsqueeze(x, dim=2)
        n, b, c, h, w = x.shape


        hidden_states_f = []
        hidden_state = x.new_zeros(n, self.hidden_dim, h, w)
        in_fea = []
        for i in range(b - 1, -1, -1):
            in_fea_i = self.in2fea(x[:,i])
            in_fea.append(in_fea_i)

            hidden_state = self.corr(in_fea_i,hidden_state)
            hidden_state = self.encoder_layer(hidden_state,in_fea_i)

            hidden_states_f.append(hidden_state)


        hidden_states_b = []
        hidden_states_f = hidden_states_f[::-1]
        in_fea = in_fea[::-1]
        hidden_state = x.new_zeros(n, self.hidden_dim, h, w).to(self.device)

        hidden_state = hidden_states_f[0]

        for i in range(0, b):
            hidden_state_f = hidden_states_f[i]

            hidden_state = self.encoder_layer1_b(hidden_state, in_fea[i], hidden_state_f)
            hidden_states_b.append(hidden_state)

        inputs = hidden_states_b


        inputs = self.ssra(inputs)
        return inputs


class SSRA_stage(nn.Module):
    def __init__(self, hidden_dim=14,region_num = 8):
        super(SSRA_stage, self).__init__()
    
        self.ssra = SSRA(in_channels=hidden_dim, out_channels=hidden_dim, 
                                    kernel_size=3, region_num=region_num,
                            guide_input_channel=True, padding=1)

    def forward(self, hidden_states_b):      
        hidden_states_enhance = []
    
        b = len(hidden_states_b)
        for i in range(0, b): 
            hidden_state =  hidden_states_b[i]      
            hidden_state = self.ssra(hidden_state, hidden_state)
            hidden_states_enhance.append(hidden_state)  

        return hidden_states_enhance


class Decoder(nn.Module):
    def __init__(self, hidden_dim=14,
                 local_range = 1,region_num = 8):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.range = local_range

        self.corr = HFComp(dim=hidden_dim)
        self.convgru_layers = ConvGRU(hidden_dim=hidden_dim,input_dim=hidden_dim)

        self.decoder_layer1_b = ConvGRU(hidden_dim=hidden_dim, input_dim=hidden_dim, is_backward=True)


        self.attention = AAM(dec_hid_dim = 2*self.range+1)

        self.conv_out_1 = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=True)
        self.conv_out_2 = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, bias=True)
        self.conv_out_3 = nn.Conv2d(hidden_dim, 1, 3, 1, 1, bias=True)
            
        self.conv_inp_1 = nn.Conv2d(1, self.hidden_dim, 3, 1, 1, bias=True)
        self.conv_inp_2 = nn.Conv2d(self.hidden_dim, hidden_dim, 3, 1, 1, bias=True)
        
         
        self.fution_layers = nn.Conv2d(hidden_dim*2, hidden_dim, 3, 1, 1, bias=True)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def hidden2out(self, x_fea, base):
        out = self.lrelu(self.conv_out_1(x_fea))
        out = self.lrelu(self.conv_out_2(out))
        out = self.conv_out_3(out)
        return out + base

    def out2inp(self, x):
        out = self.lrelu(self.conv_inp_1(x))
        out = self.lrelu(self.conv_inp_2(out))
        out = out + x
        return out
    def forward(self, x, encoder_out):
        if len(x.shape) == 4:
            x = torch.unsqueeze(x, dim=2)
        n, b, c, h, w = x.shape
        base = x


        hidden_state = encoder_out[-1]
        hidden_states_f = []
        outputs = []
        input = encoder_out[-1]
        input_list = []
        for i in range(b - 1, -1, -1):
            encoder_out_m = torch.unsqueeze(encoder_out[i], dim=1)
            if self.range == 0:
                adjacent_encoder_out = encoder_out_m
            else:
                if i == b - 1:
                    encoder_out_l = get_back(encoder_out, i, self.range)
                    adjacent_encoder_out = torch.cat([encoder_out_l, encoder_out_m], dim=1)
                elif i == 0:
                    encoder_out_r = get_forth(encoder_out, i, self.range)
                    adjacent_encoder_out = torch.cat([encoder_out_m, encoder_out_r], dim=1)
                else:
                    encoder_out_l = get_back(encoder_out, i, self.range)
                    encoder_out_r = get_forth(encoder_out, i, self.range)
                    adjacent_encoder_out = torch.cat([encoder_out_l, encoder_out_m, encoder_out_r], dim=1)


            context = self.attention(hidden_state,adjacent_encoder_out)

            input = context
            input_list.append(input)

            hidden_state = self.corr(input,hidden_state)

            hidden_state = self.convgru_layers(hidden_state, input)
            hidden_states_f.append(hidden_state)

   
        input_list = input_list[::-1]
        hidden_states_f = hidden_states_f[::-1]
        hidden_state = x.new_zeros(n, self.hidden_dim, h, w)
        input = torch.zeros_like(hidden_state)

        hidden_state = hidden_states_f[0]
        input = input_list[0]
        for i in range(0 , b):
            hidden_state_f = hidden_states_f[i]

            input = self.lrelu(self.fution_layers(torch.cat((input, input_list[i]), dim=1)))
            input_list[i] = input
            hidden_state = self.decoder_layer1_b(hidden_state,input_list[i],hidden_state_f)  
            output = self.hidden2out(hidden_state, base[:,i])
            input = self.out2inp(output)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=1).squeeze(dim=2)

        return outputs

