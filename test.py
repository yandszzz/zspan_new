# GPL License
# Copyright (C) 2023 , UESTC
# All Rights Reserved
#
# @Time    : 2023/10/1 20:29
# @Author  : Qi Cao
import argparse
import h5py
from Toolbox.model_RSP import FusionNet
from Toolbox.indexes import *
import scipy.io as sio
from Toolbox.Pancollection.indexes_evaluation_FS import indexes_evaluation_FS
from Toolbox.Pancollection.evaluation import analysis_accu
# ================== Pre-Define =================== #
parser = argparse.ArgumentParser()
parser.add_argument("--satellite", type=str, default='wv3/', help="Satellite type")
parser.add_argument("--name", type=int, default=0, help="Data ID (0-19)")
args = parser.parse_args()

satellite = args.satellite
name = args.name
ckpt = f'model_FUG/{satellite}{name}'

model = FusionNet()
weight = torch.load(ckpt)
model.load_state_dict(weight)


###################################################################
# ------------------- Main Test (Run second)----------------------------------
###################################################################


def test():
    print("Test-Full...")
    # -------------------------------- Test-Full ------------------------------------
    file_path_full = '/mnt/disk/zds/Zspan Dataset/WV-3/Testing Dataset (FullData, H5 Format)/test_wv3_OrigScale_multiExm1.h5'
    dataset = h5py.File(file_path_full, 'r')
    ms = np.array(dataset['ms'][name], dtype=np.float32) / 2047.0
    lms = np.array(dataset['lms'][name], dtype=np.float32) / 2047.0
    pan = np.array(dataset['pan'][name], dtype=np.float32) / 2047.0

    ms = torch.from_numpy(ms).float()
    lms = torch.from_numpy(lms).float()
    pan = torch.from_numpy(pan).float()

    ms = torch.unsqueeze(ms.float(), dim=0)
    lms = torch.unsqueeze(lms.float(), dim=0)
    pan = torch.unsqueeze(pan.float(), dim=0)

    res = model(lms, pan)
    out = res + lms
    sr = torch.squeeze(out * 2047).permute(1, 2, 0).cpu().detach().numpy()
    I_pan = torch.squeeze(pan * 2047).cpu().detach().numpy()
    I_ms = torch.squeeze(lms * 2047).permute(1, 2, 0).cpu().detach().numpy()
    I_ms_lr = torch.squeeze(ms * 2047).permute(1, 2, 0).cpu().detach().numpy()

    I_SR = torch.squeeze(out*2047).permute(1, 2, 0).cpu().detach().numpy()  # HxWxC
    I_MS_LR = torch.squeeze(ms*2047).permute(1, 2, 0).cpu().detach().numpy()  # HxWxC
    I_MS = torch.squeeze(lms*2047).permute(1, 2, 0).cpu().detach().numpy()  # HxWxC
    I_PAN = torch.squeeze(pan*2047).cpu().detach().numpy()  # HxWxC
    sio.savemat('result/' + satellite + str(name) + 'full.mat', {'I_SR': I_SR, 'I_MS_LR': I_MS_LR, 'I_MS': I_MS, 'I_PAN': I_PAN})

    # -------------------------------- Test-Low ------------------------------------
    print("Test-low...")
    file_path_low = '/mnt/disk/zds/Zspan Dataset/WV-3/Testing Dataset (FullData, H5 Format)/test_wv3_OrigScale_multiExm1.h5'
    dataset = h5py.File(file_path_low, 'r')
    ms = np.array(dataset['ms'][name], dtype=np.float32) / 2047.0
    lms = np.array(dataset['lms'][name], dtype=np.float32) / 2047.0
    pan = np.array(dataset['pan'][name], dtype=np.float32) / 2047.0

    ms = torch.from_numpy(ms).float()
    lms = torch.from_numpy(lms).float()
    pan = torch.from_numpy(pan).float()

    ms = torch.unsqueeze(ms.float(), dim=0)
    lms = torch.unsqueeze(lms.float(), dim=0)
    pan = torch.unsqueeze(pan.float(), dim=0)

    res = model(lms, pan)
    out = res + lms
    sr = torch.squeeze(out * 2047).permute(1, 2, 0).cpu().detach().numpy()
    I_pan = torch.squeeze(pan * 2047).cpu().detach().numpy()
    I_ms = torch.squeeze(lms * 2047).permute(1, 2, 0).cpu().detach().numpy()
    I_ms_lr = torch.squeeze(ms * 2047).permute(1, 2, 0).cpu().detach().numpy()

    I_SR = torch.squeeze(out*2047).permute(1, 2, 0).cpu().detach().numpy()  # HxWxC
    I_MS_LR = torch.squeeze(ms*2047).permute(1, 2, 0).cpu().detach().numpy()  # HxWxC
    I_MS = torch.squeeze(lms*2047).permute(1, 2, 0).cpu().detach().numpy()  # HxWxC
    I_PAN = torch.squeeze(pan*2047).cpu().detach().numpy()  # HxWxC
    sio.savemat('result/' + satellite + str(name) + 'low.mat', {'I_SR': I_SR, 'I_MS_LR': I_MS_LR, 'I_MS': I_MS, 'I_PAN': I_PAN})
    print("Test...Finish!")


    ################################################################################################################
    #     |= == D_lambda_avg(0) == = |= == == D_s_avg(0) == == = |= == == =QNR(1) == == == =
    #     FusionNet 0.0254 0.0000 0.0291 0.0000 0.9462 0.0000
    ################################################################################################################
    
    file_name_full = 'result/' + satellite + str(name) + 'low.mat'
    # I_fusionnet = sio.loadmat("/home/zds/CODE3070/U2Net-main/output/full_test/output_mulExm_0.mat")['sr']
    I_fusionnet = sio.loadmat(file_name_full)['I_SR']
    result_our = I_fusionnet / 2047.0
    # print('I_SR:', result_our.shape)
    # result_our = torch.from_numpy(I_fusionnet).float().cuda()

    data = h5py.File(r"/mnt/disk/zds/Zspan Dataset/WV-3/Testing Dataset (FullData, H5 Format)/test_wv3_OrigScale_multiExm1.h5".replace('\\', '/'))
    lms = np.asarray(data['lms']) / 2047.0
    ms = np.asarray(data['ms']) / 2047.0
    pan = np.asarray(data['pan']) / 2047.0
    # print('LMS:', lms.shape, 'MS:', ms.shape, 'PAN:', pan.shape)

    L = 11
    dim_cut = 21
    # thvalues = 0
    # pan = torch.from_numpy(pan).float().cuda()
    # lms = torch.from_numpy(lms).float().cuda()
    # ms = torch.from_numpy(ms).float().cuda()
    full_metrics = indexes_evaluation_FS(I_F=result_our, I_MS_LR=ms[name].transpose(1, 2, 0), I_MS=lms[name].transpose(1, 2, 0), I_PAN=pan[name].transpose(1, 2, 0),
                                         L=L, th_values=1, sensor='wv3', ratio=4, Qblocks_size=32, flagQNR=1)
    print("---full_evaluate---")
    print(full_metrics)



    ################################################################################################################
    #  |====PSNR(Inf)====|====SSIM(1)====|====Q(1)====|===Q_avg(1)===|=====SAM(0)=====|======ERGAS(0)=======|=======CC(1)=======|=======SCC(1)=======|=======RMSE(0)=======
    #  FusionNet 39.2423 0.0000 0.9859 0.0000 0.6548 0.0000 0.5854 0.0000 2.0508 0.0000 4.2014 0.0000 0.9800 0.0000 0.9803 0.0000 0.0122 0.0000
    ################################################################################################################
    
    file_name_low = 'result/' + satellite + str(name) + 'low.mat'
    # I_fusionnet = sio.loadmat("/home/zds/CODE3070/U2Net-main/output/reduce_test/output_mulExm_0.mat")['sr']
    I_fusionnet = sio.loadmat(file_name_low)['I_SR']
    I_fusionnet /= 2047.0
    # print('I_SR:',I_fusionnet.shape)
    result_our = torch.from_numpy(I_fusionnet).float().cuda().clip(0, 1)

    data = h5py.File(r"/mnt/disk/zds/Zspan Dataset/WV-3/Testing Dataset (ReducedData, H5 Format)/test_wv3_multiExm1.h5".replace('\\', '/'))
    gt = np.asarray(data['gt']) / 2047.0
    lms = np.asarray(data['lms']) / 2047.0
    ms = np.asarray(data['ms']) / 2047.0
    pan = np.asarray(data['pan']) / 2047.0
    # print('GT:', gt.shape, 'LMS:', lms.shape, 'MS:', ms.shape, 'PAN:', pan.shape)

    gt = torch.from_numpy(gt).float().cuda()

    Qblocks_size = 32;
    dim_cut = 30
    reduce_metrics = analysis_accu(gt[name].permute(1, 2, 0), result_our, 4, dim_cut=dim_cut)
    print("---low_evaluate----")
    print(reduce_metrics)


###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################


if __name__ == "__main__":
    test()
