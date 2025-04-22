import torch
import torch.optim as optim
import models
import os,logging
from os.path import join
from utility import *
from copy import deepcopy
from .misc import set_random_seed
from .loss import get_loss
from .logger import Logger
import time

class build_model(object):
    def __init__(self,opt):
        self.opt = None
        self.net = None
        self.optimizer = None
        self.criterion = None
        self.basedir = None
        self.trainDir = None
        self.logger = None
        
        self.epoch = None
        self.best_psnr = None
        self.best_epoch = None
        self.best_loss = None

        self.__setup(opt)

    def __setup(self,opt):
        
        self.epoch = 1
        self.best_psnr = 0
        self.best_epoch = 0
        self.best_loss = 1e6
        self.opt = deepcopy(opt)
        self.basedir = self.opt['train']['save_path']
        self.device = 'cpu' if self.opt['cpu'] else 'cuda:' + str(self.opt['gpu_ids'])[0] 
        
        # checkpoint and log save_path setting
        opt.save_dir = os.path.join(opt['train']['save_path'], opt['arch'])
        opt.log_dir = os.path.join(opt['train']['save_path'], opt['arch'],'logs')
        if not os.path.exists(opt.save_dir):
            os.makedirs(opt.save_dir)
        if not os.path.exists(opt.log_dir):
            os.makedirs(opt.log_dir)
        self.logger = Logger(os.path.join(self.opt['train']['save_path'],self.opt['arch']) + '/logs/'+ self.opt['prefix'] + '_' +str(time.strftime("%Y-%m-%d", time.localtime())) + '.log',logging.ERROR,logging.DEBUG)
        set_random_seed(self.opt['manual_seed'])
        
        # creat model
        if self.opt['arch'] == 's2s':
            self.net = models.__dict__[self.opt['arch']](self.opt)
        else:
            self.net = models.__dict__[self.opt['arch']]()
            
        self.criterion = get_loss(self.opt['train']['loss_opt']['type'])
        self.optimizer  = optim.Adam(self.net.parameters(), lr=self.opt['lr'], betas=self.opt['train']['optim_g']['betas'],weight_decay=self.opt['train']['optim_g']['weight_decay'])
        
        if self.opt['train']['resume_opt']['resume'] or self.opt['mode']=='test':
            # print(torch.load(self.opt['train']['resume_opt']['resumePath']))
            self.load(self.opt['train']['resume_opt']['resumePath'])
            
            print("resume model '{}' from '{}' with device '{}'".format(self.opt['arch'],self.opt['train']['resume_opt']['resumePath'],self.device))
            self.logger.info("resume model '{}' from '{}' with device '{}'".format(self.opt['arch'],self.opt['train']['resume_opt']['resumePath'],self.device))
        else:
            print("creating model '{}' with device '{}'".format(self.opt['arch'],self.device))
            self.logger.info("creating model '{}' with device '{}'".format(self.opt['arch'],self.device))

        if self.device != 'cpu' :
            self.net.to(self.device)
            self.criterion = self.criterion.to(self.device)
        
            if len(str(self.opt['gpu_ids'])) > 1:
                from models.sync_batchnorm import DataParallelWithCallback
                self.net = DataParallelWithCallback(self.net, device_ids=self.opt['gpu_ids'])
                
        total = sum([param.nelement() for param in self.net.parameters()])    
        print("Number of parameter: %.2fM" % (total/1e6))
        self.logger.info("Number of parameter: %.2fM" % (total/1e6))

    def forward(self, inputs):        
        
        return self.net(inputs)


    def __step(self, train, inputs, targets):  
              
        if train:
            self.optimizer.zero_grad()
            
        loss_data = 0
        total_norm = None
        
        if train:
            self.net.train()
        else:
            self.net.eval()


        outputs = self.net(inputs)
        loss = self.criterion(outputs[...], targets)

        if train:
            loss.backward()
            
        loss_data += loss.item()
        
        if train:
            total_norm = nn.utils.clip_grad_norm_(self.net.parameters(), self.opt['train']['clip'])
            self.optimizer.step()

        return outputs, loss_data, total_norm

    def load(self, resumePath=None):
        
        checkpoint = torch.load(resumePath)
        self.get_net().load_state_dict(checkpoint['net'])
        
    def update_learning_rate(self,optimizer, lr):   
         
        print('Adjust Learning Rate => %.4e' %lr)
        self.logger.info('Adjust Learning Rate => %.4e' %lr)
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr



    def save_checkpoint(self, model_out_path=None, **kwargs):
        if not model_out_path:
            model_out_path = join(self.basedir, self.opt['arch'], self.opt['prefix'], "model_epoch_%d.pth" % (
                self.epoch))

        state = {
            'net': self.get_net().state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch,
        }
        
        state.update(kwargs)
        
        if not os.path.isdir(join(self.basedir, self.opt['arch'], self.opt['prefix'])):
            os.makedirs(join(self.basedir, self.opt['arch'], self.opt['prefix']))

        torch.save(state, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def get_net(self):
        
        return self.net.module if len(str(self.opt['gpu_ids'])) > 1 else self.net 
        
    def train(self,train_loader):
        
        self.net.train()
        train_loss = 0
        train_psnr = 0
        
        print("Training Eopch: %s/%s" %(self.epoch,self.opt['train']['total_epochs']))
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if batch_idx == 0:
                train_h,train_w = inputs.shape[2],inputs.shape[3]
            
            if not self.opt['cpu']:
                inputs, targets = inputs.to(self.device), targets.to(self.device)  
                          
            outputs, loss_data, total_norm = self.__step(True, inputs, targets)
            train_loss += loss_data
            avg_loss = train_loss / (batch_idx+1)
            psnr = np.mean(cal_bwpsnr(outputs, targets))
            train_psnr += psnr
            avg_psnr = train_psnr/ (batch_idx+1)
            
            
            progress_bar(batch_idx, len(train_loader), 'AvgLoss: %.4e | Loss: %.4e | Norm: %.4e | Psnr: %.4f' 
                         % (avg_loss, loss_data, total_norm,psnr))
            
        self.logger.info("trainepoch: %s | AvgLoss: %.4f | Loss: %.4f | Norm: %.4f | Psnr: %.2f'"%(self.epoch,avg_loss, loss_data, total_norm,avg_psnr))
        self.epoch = self.epoch + 1

    def validate(self, valid_loader):
        
        self.net.eval()
        validate_loss = 0
        total_psnr = 0
        total_sam = 0
        RMSE = []
        SSIM = []
        SAM = []
        ERGAS = []
        PSNR = []
        macs = 0
        params = 0
 
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valid_loader):
                 
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                if batch_idx == 0:
                    train_h,train_w = inputs.shape[2],inputs.shape[3]
                
                if batch_idx == -1:

                    # from thop import profile
                    # input_data = torch.randn(size=(31,512,512))
                    # macs, params = profile(self.net,inputs=(input_data,))

                    from ptflops import get_model_complexity_info
                    B,C,H,W = inputs.shape
                    macs, params = get_model_complexity_info(self.net, (C,H,W),as_strings=True,
                                        print_per_layer_stat=False, verbose=False)

                    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
                    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
                    self.logger.info('{:<30}  {:<8}'.format('Computational complexity: ', macs))
                    self.logger.info('{:<30}  {:<8}'.format('Number of parameters: ', params))  

                outputs, loss_data, _ = self.__step(False, inputs, targets)
                psnr = np.mean(cal_bwpsnr(outputs, targets))
                sam = cal_sam(outputs, targets)
                validate_loss += loss_data
                total_sam += sam
                avg_loss = validate_loss / (batch_idx+1)
                avg_sam = total_sam / (batch_idx+1)

                total_psnr += psnr
                avg_psnr = total_psnr / (batch_idx+1)

                progress_bar(batch_idx, len(valid_loader), 'Loss: %.4e | PSNR: %.4f | AVGPSNR: %.4f '% (avg_loss, psnr, avg_psnr))
                
                psnr = []
                c,h,w=inputs.shape[-3:]
                
                result = outputs.squeeze().cpu().detach().numpy()
            
                img = targets.squeeze().cpu().numpy()
                for k in range(c):
                    psnr.append(10*np.log10((h*w)/sum(sum((result[k]-img[k])**2))))
                PSNR.append(sum(psnr)/len(psnr))
                
                mse = sum(sum(sum((result-img)**2)))
                mse /= c*h*w
                mse *= 255*255
                rmse = np.sqrt(mse)
                RMSE.append(rmse)

                ssim = []
                k1 = 0.01
                k2 = 0.03
                for k in range(c):
                    ssim.append((2*np.mean(result[k])*np.mean(img[k])+k1**2) \
                        *(2*np.cov(result[k].reshape(h*w), img[k].reshape(h*w))[0,1]+k2**2) \
                        /(np.mean(result[k])**2+np.mean(img[k])**2+k1**2) \
                        /(np.var(result[k])+np.var(img[k])+k2**2))
                SSIM.append(sum(ssim)/len(ssim))

                temp = (np.sum(result*img, 0) + np.spacing(1)) \
                    /(np.sqrt(np.sum(result**2, 0) + np.spacing(1))) \
                    /(np.sqrt(np.sum(img**2, 0) + np.spacing(1)))
                    
                sam = np.mean(np.arccos(temp))*180/np.pi
                SAM.append(sam)

                ergas = 0.
                for k in range(c):
                    ergas += np.mean((img[k]-result[k])**2)/np.mean(img[k])**2
                ergas = 100*np.sqrt(ergas/c)
                ERGAS.append(ergas)
                
        print("validate: PSNR: %.2f dB | SSIM: %.4f | SAM: %.4f" % (avg_psnr,sum(SSIM)/len(SSIM),avg_sam))

        if avg_psnr > self.best_psnr :
            self.best_epoch = self.epoch
            self.best_psnr = avg_psnr
            model_best_path = os.path.join(self.basedir, self.opt['arch'], self.opt['prefix'], 'model_best.pth')
            self.save_checkpoint(
                model_out_path=model_best_path
            )
             
        self.logger.info("validate: PSNR: %.2f | SSIM: %.4f | SAM: %.4f | BestPSNR: %.2f | bestEpoch: %s" % (avg_psnr,sum(SSIM)/len(SSIM),avg_sam,self.best_psnr,self.best_epoch) )
 

    def test(self, valid_loader, filen):
        self.net.eval()
        validate_loss = 0
        total_psnr = 0
        total_sam = 0
        total_cost = 0
        RMSE = []
        SSIM = []
        SAM = []
        ERGAS = []
        PSNR = []
        
        # filenames = [
        #         fn
        #         for fn in os.listdir(filen)
        #         if fn.endswith('.mat')
        #     ]

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valid_loader):
                
                if batch_idx == 0:
                    train_h,train_w = inputs.shape[2],inputs.shape[3]
                    
                if not self.opt['cpu']:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                if batch_idx == -1:

                    # from thop import profile
                    # input_data = torch.randn(size=(31,512,512))
                    # macs, params = profile(self.net,inputs=(input_data,))

                    from ptflops import get_model_complexity_info
                    B,C,H,W = inputs.shape
                    macs, params = get_model_complexity_info(self.net, (C,H,W),as_strings=True,
                                        print_per_layer_stat=False, verbose=False)

                    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
                    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

                strart_time = time.time()
                inputs.requires_grad = True
                outputs, loss_data, _ = self.__step(False, inputs, targets)

                end_time = time.time()
                test_time = end_time-strart_time
                total_cost += test_time
                # print('cost-time 55555: ',(test_time))

                psnr = np.mean(cal_bwpsnr(outputs, targets))
                sam = cal_sam(outputs, targets)
                
                validate_loss += loss_data
                total_sam += sam
                avg_loss = validate_loss / (batch_idx+1)
                avg_sam = total_sam / (batch_idx+1)

                total_psnr += psnr
                avg_psnr = total_psnr / (batch_idx+1)


                progress_bar(batch_idx, len(valid_loader), 'Loss: %.4e | PSNR: %.4f | AVGPSNR: %.4f '
                          % (avg_loss, psnr, avg_psnr))
                
                psnr = []
                c,h,w=inputs.shape[-3:]
                result = outputs.squeeze().cpu().detach().numpy()
            
                img = targets.squeeze().cpu().numpy()
                
                for k in range(c):
                    psnr.append(10*np.log10((h*w)/sum(sum((result[k]-img[k])**2))))
                PSNR.append(sum(psnr)/len(psnr))
                
                mse = sum(sum(sum((result-img)**2)))
                mse /= c*h*w
                mse *= 255*255
                rmse = np.sqrt(mse)
                RMSE.append(rmse)

                ssim = []
                k1 = 0.01
                k2 = 0.03
                for k in range(c):
                    ssim.append((2*np.mean(result[k])*np.mean(img[k])+k1**2) \
                        *(2*np.cov(result[k].reshape(h*w), img[k].reshape(h*w))[0,1]+k2**2) \
                        /(np.mean(result[k])**2+np.mean(img[k])**2+k1**2) \
                        /(np.var(result[k])+np.var(img[k])+k2**2))
                SSIM.append(sum(ssim)/len(ssim))

                temp = (np.sum(result*img, 0) + np.spacing(1)) \
                    /(np.sqrt(np.sum(result**2, 0) + np.spacing(1))) \
                    /(np.sqrt(np.sum(img**2, 0) + np.spacing(1)))
                sam = np.mean(np.arccos(temp))*180/np.pi
                SAM.append(sam)

                ergas = 0.
                for k in range(c):
                    ergas += np.mean((img[k]-result[k])**2)/np.mean(img[k])**2
                ergas = 100*np.sqrt(ergas/c)
                ERGAS.append(ergas)
                
                #########################   draw results, examples as below ###############

                # save_path = '/data1/jiahua/result/wdc/'
                # # test = minmax_normalize(result)
                # test = result.transpose((2,1,0))
                # test = test.transpose((1,0,2))
                # print(test.shape)
                # savemat(save_path+"s2s.mat", {'data': test})
                
                # save_dir = save_path + str(self.opt.arch)
                # savemat(save_dir+".mat", {'data': test})

                # kaist
                # save_path = '/data1/jiahua/result/supplyment_data/kaist/'
                # # noisy
                # inputs = inputs.squeeze().cpu().numpy()
                # color_img = np.concatenate([inputs[9][np.newaxis,:],inputs[19][np.newaxis,:],inputs[29][np.newaxis,:]],0)
                # color_img = color_img.transpose((1,2,0))*255
                # cv2.imwrite(os.path.join(save_path, 'noise_'+filenames[batch_idx][:-4]+'.png'),color_img)
                # # gt
                # targets = targets.squeeze().cpu().numpy()
                # color_img = np.concatenate([targets[9][np.newaxis,:],targets[19][np.newaxis,:],targets[29][np.newaxis,:]],0)
                # color_img = color_img.transpose((1,2,0))*255
                # cv2.imwrite(os.path.join(save_path, 'gt_'+filenames[batch_idx][:-4]+'.png'),color_img)
                # output
                # color_img = np.concatenate([result[9][np.newaxis,:],result[19][np.newaxis,:],result[29][np.newaxis,:]],0)
                # color_img = color_img.transpose((1,2,0))*255
                # cv2.imwrite(os.path.join(save_path, self.opt.arch+'_'+filenames[batch_idx][:-4]+'.png'),color_img)

                # result = result.squeeze().cpu().detach().numpy()
                # for band in range(31):
                #     img = result[band]*255#
                #     cv2.imwrite(os.path.join(save_path, filenames[batch_idx][:-4] +'_band_'+str(band)+'.jpg'),cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))

                # color_img = np.concatenate([result[9][np.newaxis,:],result[15][np.newaxis,:],result[28][np.newaxis,:]],0)
                # color_img = color_img.transpose((1,2,0))*255
                # cv2.imwrite(os.path.join(save_path, filenames[batch_idx][:-4] +'color.png'),cv2.cvtColor(color_img.astype(np.uint8),cv2.COLOR_RGB2BGR))

                # wdc
                # savemat(save_path+"/fidnet_wdc.mat", {'output': result})
                # # 1-31 代表 0-30 channel

                # icvl
                # color_img = np.concatenate([result[9][np.newaxis,:],result[19][np.newaxis,:],result[29][np.newaxis,:]],0)
                # color_img = color_img.transpose((1,2,0))*255
                # cv2.imwrite(os.path.join(save_path, self.opt.arch +'.png'),color_img)

        # print(psnr)
                # result_psnr = result_psnr[:batch_idx] + [a + b for a, b in zip(result_psnr[batch_idx:], psnr)]

            print(sum(PSNR)/len(PSNR), sum(RMSE)/len(RMSE), sum(SSIM)/len(SSIM), sum(SAM)/len(SAM), sum(ERGAS)/len(ERGAS))
            # print("avg_psnr: "+str(avg_psnr)+" avg_SSIM: "+str(sum(SSIM)/len(SSIM))+" avg_sam: "+str(avg_sam))
            print("PSNR: %.2f dB | SSIM: %.4f | SAM: %.4f" % (avg_psnr,sum(SSIM)/len(SSIM),avg_sam) )
 


                  
