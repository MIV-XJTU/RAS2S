from utility import *

class SequentialSelect(object):
    def __pos(self, n):
        i = 0
        while True:
            # print(i)
            yield i
            i = (i + 1) % n

    def __init__(self, transforms):
        self.transforms = transforms
        self.pos = LockedIterator(self.__pos(len(transforms)))

    def __call__(self, img):
        out = self.transforms[next(self.pos)](img)
        return out

def get_train_transform(noiseType,HSI2Tensor):
    if noiseType == "gaussian":
        train_transform = Compose([
            AddNoise(50),
            HSI2Tensor()
        ])
        # print("gaussian noise added\n")  
    elif noiseType == "blind":
        train_transform = Compose([
            AddNoiseBlindv1(10, 70),
            HSI2Tensor()
        ])
        # print("blind noise added\n")     
    elif noiseType == "complex":
        sigmas = [10, 30, 50, 70]
        train_transform =  Compose([
            AddNoiseNoniid(sigmas),
            SequentialSelect(
                transforms=[
                    lambda x: x,
                    AddNoiseImpulse(),
                    AddNoiseStripe(),
                    AddNoiseDeadline()
                ]
            ),
            HSI2Tensor()
        ])
        # print("complex noise added\n")
    elif noiseType == "noniid":
        train_transform = Compose([
            AddNoiseNoniid_v2(0,55),
            HSI2Tensor()
        ]) 
        # print("noniid noise added\n")       
    return train_transform

def get_single_train_loader(opt,HSI2Tensor,use_2dconv=True):

    target_transform = HSI2Tensor()

    if opt['train']['Datasets']['type'] == 'icvl':
        train_datasets = LMDBDataset(opt['train']['Datasets']['trainDir'])
    elif opt['train']['Datasets']['type']  == 'wdc' or opt['train']['Datasets']['type']  == 'urban':
        train_datasets = LMDBDataset(opt['train']['Datasets']['trainDir'],repeat=10)
    
    target_transform = HSI2Tensor()
    train_dataset = ImageTransformDataset(train_datasets, get_train_transform(opt['train']['noiseType'],HSI2Tensor),target_transform)
    
    train_loader = DataLoader(train_dataset,
                              batch_size=opt['batchSize'], shuffle=True,
                              num_workers=opt['threads'], pin_memory=not opt['cpu'], worker_init_fn=worker_init_fn)
    
    return train_loader

def get_multi_train_loader(opt,HSI2Tensor,use_2dconv=True):

    target_transform = HSI2Tensor()
    train_loaders = []

    if opt['train']['Datasets']['type'] == 'icvl':
        train_datasets = LMDBDataset(opt['train']['Datasets']['trainDir'])
    elif opt['train']['Datasets']['trainDataset'] == 'wdc' or opt['train']['Datasets']['trainDataset'] == 'urban':
        train_datasets = LMDBDataset(opt['train']['Datasets']['trainDir'],repeat=10)
    
    target_transform = HSI2Tensor()
    
    for noiseType in opt['train']['multiDatasets']['noiseType']:
        train_dataset = ImageTransformDataset(train_datasets, get_train_transform(noiseType,HSI2Tensor),target_transform)
    
        train_loader = DataLoader(train_dataset,
                              batch_size=opt['batchSize'], shuffle=True,
                              num_workers=opt['threads'], pin_memory=not opt['cpu'], worker_init_fn=worker_init_fn)
        
        train_loaders.append(train_loader)
    
    return train_loaders

def get_val_loader(opt,use_2dconv=True):
    
    mat_datasets = [MatDataFromFolder(
        opt['train']['Datasets']['valDir'], size=opt['train']['Datasets']['val_matSize'])]
    
    if opt['train']['Datasets']['type'] == 'urban': 
        mat_datasets = [MatDataFromFolder(
        opt['train']['Datasets']['valDir'], size=opt['train']['Datasets']['val_matSize'],fns=['Urban.mat'])]

    if not use_2dconv:
        mat_transform = Compose([
            LoadMatHSI(input_key='input', gt_key='gt',
                    transform=lambda x:x[ ...][None], needsigma=False),
        ])
    else:
        mat_transform = Compose([
            LoadMatHSI(input_key='input', gt_key='gt', needsigma=False),
        ])

    mat_datasets = [TransformDataset(mat_dataset, mat_transform)
                    for mat_dataset in mat_datasets]
    
    val_loader = [DataLoader(
        mat_dataset,
        batch_size=1, shuffle=False,
        num_workers=1, pin_memory=opt['cpu']
    ) for mat_dataset in mat_datasets]       
    
    return val_loader

def get_test_loader(opt,use_2dconv=True):
    
    mat_datasets = [MatDataFromFolder(
        opt['testDir']) ]

    if not use_2dconv:
        mat_transform = Compose([
            LoadMatHSI(input_key='input', gt_key='gt',
                    transform=lambda x:x[ ...][None], needsigma=False),
        ])
    else:
        mat_transform = Compose([
            # LoadMatHSI(input_key='input', gt_key='gt', needsigma=False),
            LoadMatHSI(input_key='input', gt_key='gt', needsigma=False),
        ])

    mat_datasets = [TransformDataset(mat_dataset, mat_transform)
                    for mat_dataset in mat_datasets]

 
    mat_loaders = [DataLoader(
        mat_dataset,
        batch_size=1, shuffle=False,
        num_workers=1, pin_memory=opt['cpu']
        
    ) for mat_dataset in mat_datasets]        

    count = 0
    for _, _, files in os.walk(opt['testDir']):
        count += len(files)
    
    return mat_loaders,count

def get_test_data(testDir):
    
    mat_datasets = [MatDataFromFolder(
        testDir) ]


    mat_transform = Compose([
        # LoadMatHSI(input_key='input', gt_key='gt', needsigma=False),
        LoadMatHSI(input_key='input', gt_key='gt', needsigma=False),
    ])

    mat_datasets = [TransformDataset(mat_dataset, mat_transform)
                    for mat_dataset in mat_datasets]

 
    mat_loaders = [DataLoader(
        mat_dataset,
        batch_size=1, shuffle=False,
        num_workers=1
        
    ) for mat_dataset in mat_datasets]        

    
    return mat_loaders[0]


def make_dataset(opt, train_transform, target_transform, common_transform, batch_size=None, repeat=1):
    dataset = LMDBDataset(opt.dataroot, repeat=repeat)
    # dataset.length -= 1000
    # dataset.length = size or dataset.length

    """Split patches dataset into training, validation parts"""
    dataset = TransformDataset(dataset, common_transform)

    train_dataset = ImageTransformDataset(dataset, train_transform, target_transform)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size or opt.batchSize, shuffle=True,
                              num_workers=opt.threads, pin_memory=opt['cpu'], worker_init_fn=worker_init_fn)

    return train_loader