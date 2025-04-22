import os
from utility import *
from utils.options import  parse_options
from utils.logger import Logger
from utils.config import build_model
from utils.dataloader import get_test_loader,get_single_train_loader,get_multi_train_loader,get_val_loader

def test(opt):
    
    torch.backends.cudnn.benchmark = True
    
    model = build_model(opt)
    test_loader,len_test = get_test_loader(opt,use_2dconv=model.get_net().use_2dconv)
    # print(len_test)

    strart_time = time.time()
    # engine.judge(mat_loaders[0], basefolder)
    model.test(test_loader[0], opt['testDir'])
    end_time = time.time()
    test_time = end_time-strart_time
    print('totol_cost: ',(test_time/len_test))
    
def train(opt,HSI2Tensor):
    
    start_epoch  = 1
    current_lr_stage = 0
    train_stage = 0
    
    torch.backends.cudnn.benchmark = True
    
    
    model = build_model(opt)
    
    if opt['train']['multiDatasets']['type'] == False:
        train_loaders = get_single_train_loader(opt,HSI2Tensor,use_2dconv=model.get_net().use_2dconv)
        print("%s noise added \n"%(opt['train']['noiseType'])) 
        model.logger.info("%s noise added \n"%(opt['train']['multiDatasets']['noiseType'][train_stage])) 
    else:
        train_loaders = get_multi_train_loader(opt,HSI2Tensor,use_2dconv=model.get_net().use_2dconv)
        print("%s noise added \n"%(opt['train']['multiDatasets']['noiseType'][train_stage])) 
        model.logger.info("%s noise added \n"%(opt['train']['multiDatasets']['noiseType'][train_stage])) 

    val_loader = get_val_loader(opt,use_2dconv=model.get_net().use_2dconv)
    
    model.logger.info('opt: %s' % (opt))
    
    for epoch in range(start_epoch, opt['train']['total_epochs'] + 1):
        np.random.seed()

        if opt['train']['multiDatasets']['type'] == True and current_lr_stage + 1 > len(opt['train']['scheduler']['milestones']) or epoch > opt['train']['scheduler']['milestones'][current_lr_stage]:
            model.update_learning_rate(model.optimizer, opt['lr'] * opt['train']['scheduler']['gammas'][current_lr_stage])
            current_lr_stage = current_lr_stage + 1
        
        if train_stage + 1 > len(opt['train']['multiDatasets']['stones']) or epoch > opt['train']['multiDatasets']['stones'][train_stage]:
            train_stage = train_stage + 1
            print("change to %s noise \n"%(opt['train']['multiDatasets']['noiseType'][train_stage])) 

        model.validate(val_loader[0])
        
        if opt['train']['multiDatasets']['type'] == False:
            train_loader = train_loaders
        else:
            train_loader = train_loaders[train_stage]
            
        model.train(train_loader)
        
        print('Latest Result Saving...')
        if model.epoch % opt['train']['checkpoints_per_save'] == 0 or model.epoch >= opt['train']['total_epochs'] - 10:
            model.save_checkpoint()

        model.save_checkpoint(
            model_out_path=os.path.join(model.basedir,opt['arch'], opt['prefix'], 'model_latest.pth')
        )

        display_learning_rate(model.optimizer)
            

if __name__ == '__main__':
 
    opt = parse_options()
    if opt['mode'] == 'train':
        HSI2Tensor = partial(HSI2Tensor, use_2dconv=True)
        train(opt,HSI2Tensor) 
    else:
        test(opt) 




