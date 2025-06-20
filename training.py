
import os
import shutil
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
import numpy as np

from models.grid_proto_fewshot import FewShotSeg
from dataloaders.dev_customized_med import med_fewshot, update_loader_dset, med_fewshot_val 
from dataloaders.GenericSuperDatasetv2 import SuperpixelDataset
from dataloaders.dataset_utils import DATASET_INFO
import dataloaders.augutils as myaug

from util.utils import set_seed, t2n, to01, compose_wt_simple, get_tversky_loss
from util.metric import Metric

from config_ssl_upload import ex
import tqdm
import json, copy, ast
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# config pre-trained model caching path
os.environ['TORCH_HOME'] = "./pretrained_model"



@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        os.makedirs(f'{_run.observers[0].dir}/trainsnaps', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')
        with open("config.json", "w") as f:
            json.dump(_config, f, indent=4)


    set_seed(_config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)

    _log.info('###### Create model ######')

    model = FewShotSeg(pretrained_path=None, cfg=_config['model'])

    model = model.cuda()
    model.train()

    _log.info('###### Load data ######')
    ### Training set
    data_name = _config['dataset']
    if data_name == 'CHAOST2_Superpix':
        baseset_name = 'CHAOST2'
    elif data_name == 'C0_Superpix':
        raise NotImplementedError
        baseset_name = 'C0'
    elif data_name == 'SABS':  # ADD THIS FOR SUPERVISED LEARNING
        baseset_name = 'SABS'
    elif data_name == 'SABS_Superpix':
        baseset_name = 'SABS'
    elif data_name=='FLARE22Train_Superpix':
        baseset_name='FLARE22Train'
    else:
        raise ValueError(f'Dataset: {data_name} not found')
    

    ###================== Transforms for data augmentation =============================== ###
    tr_transforms = myaug.transform_with_label({'aug': myaug.augs[_config['which_aug']]})
    ### ================================================================================== ###
    assert _config['scan_per_load'] < 0 # by default we load the entire dataset directly

    test_labels = DATASET_INFO[baseset_name]['LABEL_GROUP']['pa_all'] - DATASET_INFO[baseset_name]['LABEL_GROUP'][_config["label_sets"]]
    _log.info(f'###### Labels excluded in training : {[lb for lb in _config["exclude_cls_list"]]} ######')
    _log.info(f'###### Unseen labels evaluated in testing: {[lb for lb in test_labels]} ######')

    

    

    tr_parent = SuperpixelDataset( # base dataset
        which_dataset = baseset_name,
        base_dir=_config['path'][data_name]['data_dir'],
        idx_split = _config['eval_fold'],
        mode='train',
        min_fg=str(_config["min_fg_data"]), # dummy entry for superpixel dataset
        transforms =tr_transforms, #
        transform_param_limits = myaug.augs[_config['which_aug']],#tr_transforms, #
        nsup = _config['task']['n_shots'],
        scan_per_load = _config['scan_per_load'],
        exclude_list = _config["exclude_cls_list"],
        superpix_scale = _config["superpix_scale"],
        fix_length = _config["max_iters_per_load"] if (data_name == 'C0_Superpix') or (data_name == 'CHAOST2_Superpix') else None,
        dataset_config = _config['DATASET_CONFIG']
    )

    ### dataloaders
    trainloader = DataLoader(
        tr_parent,
        batch_size=_config['batch_size'],
        shuffle=True,
        #num_workers=_config['num_workers'],
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    _log.info('###### Set optimizer ######')
    if _config['optim_type'] == 'sgd':
        print("Using SGD optimizer")
        optimizer = torch.optim.SGD(model.parameters(), **_config['optim'])
    elif _config['optim_type'] == 'adam':
        print("Using ADAM optimizer")
        optimizer = torch.optim.Adam(model.parameters(), **_config['optim'])
    else:
        raise NotImplementedError

    # optimizer.load_state_dict(torch.load(_config['reload_model_path'])['opt'])

    scheduler = MultiStepLR(optimizer, milestones=_config['lr_milestones'],  gamma = _config['lr_step_gamma'])
    # scheduler.load_state_dict(torch.load(_config['reload_model_path'])['sch'])

    my_weight = compose_wt_simple(_config["use_wce"], data_name)
    criterion = nn.CrossEntropyLoss(ignore_index=_config['ignore_label'], weight = my_weight)

    i_iter = 0 # total number of iteration
    n_sub_epoches = _config['n_steps'] // _config['max_iters_per_load'] # number of times for reloading

    log_loss = {'loss': 0, 'align_loss': 0}

    _log.info('###### Training ######')
    stime = time.time()
    for sub_epoch in range(n_sub_epoches):
        _log.info(f'###### This is epoch {sub_epoch} of {n_sub_epoches} epoches ######')
        for _, sample_batched in enumerate(trainloader):
            # Prepare input
            i_iter += 1
            # add writers
            support_images = [[shot.float().cuda() for shot in way]
                              for way in sample_batched['support_images']]
            support_fg_mask = [[shot[f'fg_mask'].float().cuda() for shot in way]
                               for way in sample_batched['support_mask']]
            support_bg_mask = [[shot[f'bg_mask'].float().cuda() for shot in way]
                               for way in sample_batched['support_mask']]

            query_images = [query_image.float().cuda()
                            for query_image in sample_batched['query_images']]
            query_labels = torch.cat(
                [query_label.long().cuda() for query_label in sample_batched['query_labels']], dim=0)

            optimizer.zero_grad()
            # FIXME: in the model definition, filter out the failure case where 
            # pseudolabel falls outside of image or too small to calculate a prototype
            # try:
            

            mean = [m for m in sample_batched["mean"]]
            std = [s for s in sample_batched["std"]]
            ########################################################################
            query_pred, align_loss, debug_vis, assign_mats = model(support_images,
                                                                    support_fg_mask, 
                                                                    support_bg_mask, 
                                                                    query_images, 
                                                                    isval = False, val_wsize = None)
            ########################################################################
            # except:
                # print('Faulty batch detected, skip')
                # continue

            query_loss = criterion(query_pred, query_labels) + get_tversky_loss(query_pred.argmax(dim = 1, keepdim = True), query_labels[None, ...], 0.3, 0.7 ,1.0)
            loss = query_loss + align_loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Log loss
            query_loss = query_loss.detach().data.cpu().numpy()
            align_loss = align_loss.detach().data.cpu().numpy() if align_loss != 0 else 0

            _run.log_scalar('loss', query_loss)
            _run.log_scalar('align_loss', align_loss)
            log_loss['loss'] += query_loss
            log_loss['align_loss'] += align_loss

            # print loss and take snapshots
            if (i_iter + 1) % _config['print_interval'] == 0:

                nt = time.time()

                loss = log_loss['loss'] / _config['print_interval']
                align_loss = log_loss['align_loss'] / _config['print_interval']

                log_loss['loss'] = 0
                log_loss['align_loss'] = 0

                fig, ax = plt.subplots(1,2)
                # print(mean.shape, std.shape, mean, std)            
                # print(support_images[0][0][0].min(), support_images[0][0][0].max())
                si = (support_images[0][0][0].cpu()*std[0]+mean[0]).numpy().transpose((1,2,0))
                si = (si - si.min())/(si.max() - si.min() + 1e-6)
                # print(si.shape)
                ax[0].imshow(si)
                # print(support_fg_mask[0][0][0].shape)
                sm = support_fg_mask[0][0][0].cpu().numpy()
                # print(sm.shape)
                ax[0].imshow(np.stack([np.zeros(sm.shape),sm,np.zeros(sm.shape)], axis = 2), alpha = 0.3)

                # print(query_pred[0].shape)
                qi = (query_images[0][0].cpu()*std[0]+mean[0]).numpy().transpose((1,2,0))
                qi = (qi - qi.min())/(qi.max() - qi.min() + 1e-6)
                ax[1].imshow(qi)
                qm = query_labels[0].cpu().numpy()
                # print(sm.shape, qm.shape)
                ax[1].imshow(np.stack([np.zeros(qm.shape),qm,np.zeros(qm.shape)], axis = 2), alpha = 0.3)
                qp = query_pred.argmax(dim = 1).float().cpu().numpy()[0] #(query_pred > 0.5)[0][0].cpu().numpy() #.transpose((1,2,0))
                # print(qp.shape)
                ax[1].imshow(np.stack([qp,np.zeros(qp.shape),np.zeros(qp.shape)], axis = 2), alpha = 0.2)

                plt.savefig(os.path.join(f'{_run.observers[0].dir}/trainsnaps', f'{i_iter + 1}.png'), bbox_inches='tight')
                plt.close(fig)

                print(f'step {i_iter+1}: loss: {loss}, align_loss: {align_loss}, time: {(nt-stime)/60} mins')

            if (i_iter + 1) % _config['save_snapshot_every'] == 0:
                _log.info('###### Taking snapshot ######')
                torch.save({'model':model.state_dict(),'opt':optimizer.state_dict(),'sch':scheduler.state_dict()},
                           os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))
                

            if data_name == 'C0_Superpix' or data_name == 'CHAOST2_Superpix':
                if (i_iter + 1) % _config['max_iters_per_load'] == 0:
                    _log.info('###### Reloading dataset ######')
                    #trainloader.dataset.reload_buffer()
                    update_loader_dset(trainloader, tr_parent)
                    print(f'###### New dataset with {len(trainloader.dataset)} slices has been loaded ######')

            if (i_iter - 2) > _config['n_steps']:
                return 1 # finish up
            
