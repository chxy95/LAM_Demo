import os
import torch

MODEL_DIR = './ModelZoo/models'


NN_LIST = [
    'RCAN',
    'CARN',
    'RRDBNet',
    'RNAN', 
    'SAN',
    'SwinIR',
    'ECCV',
    'ECCV2',
    'Restormer'
]


MODEL_LIST = {
    'RCAN': {
        'Base': 'RCAN.pt',
    },
    'CARN': {
        'Base': 'CARN_7400.pth',
    },
    'RRDBNet': {
        'Base': 'RRDBNet_PSNR_SRx4_DF2K_official-150ff491.pth',
    },
    'SAN': {
        'Base': 'SAN_BI4X.pt',
    },
    'RNAN': {
        'Base': 'RNAN_SR_F64G10P48BIX4.pt',
    },
    'SwinIR':{
        'SRx4_win8': 'SwinIR_W8_SRx4_DF2K.pth',
        'SRx4_win16': 'SwinIR_W16_SRx4_DF2K.pth',
        'SRx4_win32': 'SwinIR_W32_SRx4_DF2K.pth',
    },
    'ECCV':{
        'Baseline': '213.pth',
        'OCAB': '396.pth',
        'CAB': '418.pth',
        'Ours': '416.pth'
    },
    'ECCV2':{
        'Ours': '435.pth',
        'SwinIR': 'SwinIR_W8_SRx4_DF2K.pth',
        'RCAN': 'RCAN.pt',
        'EDSR': 'EDSR-64-16_15000.pth'
    },
    'Restormer': {
        'ICCV-013': 'restormer_iccv_013.pth',
    },
}

def print_network(model, model_name):
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('Network [%s] was created. Total number of parameters: %.1f kelo. '
          'To see the architecture, do print(network).'
          % (model_name, num_params / 1000))


def get_model(model_name, training_name=None, factor=4, num_channels=3):
    """
    All the models are defaulted to be X4 models, the Channels is defaulted to be RGB 3 channels.
    :param model_name:
    :param factor:
    :param num_channels:
    :return:
    """
    print(f'Getting SR Network {model_name}')
    if model_name.split('-')[0] in NN_LIST:

        if model_name == 'RCAN':
            from .NN.rcan import RCAN
            net = RCAN(factor=factor, num_channels=num_channels)

        elif model_name == 'CARN':
            from .CARN.carn import CARNet
            net = CARNet(factor=factor, num_channels=num_channels)

        elif model_name == 'RRDBNet':
            from .NN.rrdbnet import RRDBNet
            net = RRDBNet(num_in_ch=num_channels, num_out_ch=num_channels)

        elif model_name == 'SAN':
            from .NN.san import SAN
            net = SAN(factor=factor, num_channels=num_channels)

        elif model_name == 'RNAN':
            from .NN.rnan import RNAN
            net = RNAN(factor=factor, num_channels=num_channels)

        elif model_name == 'SwinIR':
            from .NN.swinir import SwinIR_wConv
            if training_name == 'SRx4_win8':
                net = SwinIR_wConv(img_size=64, window_size=8, upscale=factor)
            elif training_name == 'SRx4_win16':
                net = SwinIR_wConv(img_size=64, window_size=16, upscale=factor)
            elif training_name == 'SRx4_win32':
                net = SwinIR_wConv(img_size=64, window_size=32, upscale=factor)

        elif model_name == 'ECCV':
            if training_name == 'Baseline':
                from .NN.Baseline import SwinIR_wConv
                net = SwinIR_wConv()
            elif training_name == 'OCAB':
                from .NN.OCAB import SwinIR_Cascade_Overlap_Win
                net = SwinIR_Cascade_Overlap_Win()
            elif training_name == 'CAB':
                from .NN.CAB import SwinIR_Parallel_Compress_CAB
                net = SwinIR_Parallel_Compress_CAB()
            elif training_name == 'Ours':
                from .NN.Ours import Final_Model
                net = Final_Model()
        
        elif model_name == 'ECCV2':
            if training_name == 'Ours':
                from .NN.Ours import Final_Model
                net = Final_Model()
            elif training_name == 'SwinIR':
                from .NN.swinir import SwinIR_wConv
                net = SwinIR_wConv(img_size=64, window_size=8, upscale=factor)
            elif training_name == 'RCAN':
                from .NN.rcan import RCAN
                net = RCAN(factor=4, num_channels=3)
            elif training_name == 'EDSR':
                from .NN.edsr import EDSR
                net = EDSR(factor=4, num_channels=3)
        
        elif model_name == 'Restormer':
            from .NN.restormer import RestormerSR
            net = RestormerSR(scale=4)

        else:
            raise NotImplementedError()

        print_network(net, model_name)
        return net
    else:
        raise NotImplementedError()


def load_model(model_loading_name):
    """
    :param model_loading_name: model_name-training_name
    :return:
    """
    splitting = model_loading_name.split('@')
    if len(splitting) == 1:
        model_name = splitting[0]
        training_name = 'Base'
    elif len(splitting) == 2:
        model_name = splitting[0]
        training_name = splitting[1]
    else:
        raise NotImplementedError()
    assert model_name in NN_LIST or model_name in MODEL_LIST.keys(), 'check your model name before @'
    net = get_model(model_name, training_name)
    state_dict_path = os.path.join(MODEL_DIR, MODEL_LIST[model_name][training_name])
    print(f'Loading model {state_dict_path} for {model_name} network.')

    state_dict = torch.load(state_dict_path)
    if 'params_ema' in state_dict:
        net.load_state_dict(state_dict['params_ema'], strict=True)
    elif 'params_ema' in state_dict:
        net.load_state_dict(state_dict['params_ema'], strict=True)
    elif 'params' in state_dict:
        net.load_state_dict(state_dict['params'], strict=True)
    else:
        net.load_state_dict(state_dict)
    return net




