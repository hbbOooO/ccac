from distutils.log import Log
from torch.utils.data.dataloader import DataLoader
from torch.optim import SGD
import torch
from tqdm import tqdm

from track1.util.losses import BaseLoss
from common.logger import Logger
from common.timer import Timer
from common.checkpoint import CheckPoint

class Trainer:
    def __init__(self, config):
        self.config = config
        self.dataset_config = config['dataset']
        self.model_config = config['model']
        self.run_param = config['run_param']
        self.train_param = self.run_param['train_param']
        self.val_param = self.run_param['val_param']
        self.inference_param = self.run_param['inference_param']
        self.loss_param = self.train_param['loss']

        Logger.get_logger().info('the config is as following: \n' + str(self.config))

        self._import_classes()
        self._init_dataset()
        self._init_model()
        # self._init_model()
        self._init_dataloader()
        self._init_optimizer()
        self._init_loss()
        self._init_extra()

    def _import_classes(self):
        Logger.get_logger().info('----- import dataset class -----')
        dataset_name = self.dataset_config['class_name']
        dataset_module_path = self.dataset_config['module_path']
        dataset_module = __import__(dataset_module_path, fromlist=[dataset_module_path.split('.')[-1]])
        dataset_class = getattr(dataset_module, dataset_name)
        assert dataset_class
        self.dataset_class = dataset_class

        Logger.get_logger().info('----- import model class -----')
        model_name = self.model_config['class_name']
        model_module_path = self.model_config['module_path']
        model_module = __import__(model_module_path, fromlist=[model_module_path.split('.')[-1]])
        model_class = getattr(model_module, model_name)
        assert model_class
        self.model_class = model_class

    def _init_dataset(self):
        Logger.get_logger().info('----- init dataset -----')
        dataset_configs = self.dataset_config['datasets']
        datasets = {}
        for dataset_config in dataset_configs:
            datasets[dataset_config['dataset_type']] = self.dataset_class(dataset_config)
        self.datasets = datasets

    def _init_model(self):
        Logger.get_logger().info('----- init model -----')
        model_config = self.model_config['model']
        model = self.model_class(model_config)
        self.model = model
        Logger.get_logger().info(model)

    def _init_dataloader(self):
        Logger.get_logger().info('----- init dataloader -----')
        dataloaders = dict()
        for dataset_type, dataset in self.datasets.items():
            if dataset_type == 'train': batch_size = self.train_param['batch_size']
            elif dataset_type == 'val': batch_size = self.val_param['batch_size']
            elif dataset_type == 'inference': batch_size =- self.inference_param['batch_size']
            dataloaders[dataset_type] = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=True)
        self.dataloaders = dataloaders

    def _init_optimizer(self):
        Logger.get_logger().info('----- init optimizer -----')
        optimizer = SGD(
            self.model.parameters(),
            lr=self.train_param['lr']
        )
        self.optimizer = optimizer

    def _init_extra(self):
        Logger.get_logger().info('----- init extra -----')
        # only adapt one GPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        Timer.set_up(self.run_param['timer_type'])
        self.ckpter = CheckPoint(self.train_param['checkpoint'])
        

    def _init_loss(self):
        Logger.get_logger().info('----- init loss -----')
        self.criterion = BaseLoss(self.loss_param)

    def _to_cuda(self, batch):
        device = self.device
        fields = batch.keys()
        for field in fields:
            if isinstance(batch[field], torch.Tensor):
                batch[field] = batch[field].to(device)
            elif isinstance(batch[field], dict):
                self._to_cuda(batch[field])
        return batch

    def __call__(self):
        train_type = self.run_param['run_type']
        if 'train' in train_type:
            self.train()
        if 'val' in train_type:
            self.val()
        if 'inference' in train_type:
            self.inference()

    def train(self):
        Logger.get_logger().info('start training')
        self.max_epoch = self.train_param['max_epoch']
        self.max_iteration = len(self.dataloaders['train'])
        self.curr_epoch = 0
        self.epoch_val_metric = []
        self.best_metric_epoch = 0
        self.best_metric = {'f1': 0}
        
        self.model.to(self.device)
        self.model.train()

        for epoch_index in range(self.max_epoch):
            self.curr_epoch += 1

            self.curr_iteration = 0
            self.train_prediction = {}
            self.train_losses = []
            self.train_metric = {}

            for batch in self.dataloaders['train']:
                self.curr_iteration += 1
                prepared_batch = self._to_cuda(batch)
                pred_prob, pred_w_label = self.model(prepared_batch)
                loss = self.criterion(pred_prob, batch['gt_label'])
                self._update_train_meter(pred_w_label, loss)
                self._backward(loss)
                self._report()
                # test
                # self._epoch_summary()

            self._epoch_summary()
            


                
    def _update_train_meter(self, pred_w_label, loss):
        # update train prediction
        pred_w_label = pred_w_label.cpu().numpy()
        pred_w_label = {item[0]: item[1] for item in pred_w_label}
        self.train_prediction.update(pred_w_label)

        # update train loss
        self.train_losses.append(loss.item())

        # update metric
        self.metric = self.datasets['train'].evaluate(self.train_prediction)


    def _backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _report(self):
        log_interval = self.train_param['log_interval']
        # ckpt_internal = self.train_param['ckpt_internal']
        # save_epoch_ckpt = self.train_param['save_epoch_ckpt']
        if self.curr_iteration % log_interval == 0:
            Logger.get_logger().info(
                'epoch: {}/{}, iteration: {}/{}, loss(avg): {:6f}({:.6f}), f1: {:.4f}, precision: {:.4f}, recall: {:.4f}, acc: {:.4f}, lr: {}, cost time: {}, reamin time: {}'.format(
                    self.curr_epoch, self.max_epoch, 
                    self.curr_iteration, self.max_iteration,
                    self.train_losses[-1], sum(self.train_losses)/len(self.train_losses),
                    self.metric['f1'],
                    self.metric['precision'],
                    self.metric['recall'],
                    self.metric['acc'],
                    self.optimizer.state_dict()['param_groups'][0]['lr'],
                    Timer.calculate_spend(),
                    Timer.calculate_remain(self.curr_epoch, self.curr_iteration, self.max_epoch, self.max_iteration)
                )
            )
    
    def _epoch_summary(self):
        # evaluate on val dataset
        self.val()
        self.model.train()

        # compare val result with previous (based on f1 score)
        self.epoch_val_metric.append(self.val_metric)
        if self.best_metric['f1'] < self.val_metric['f1']: 
            self.best_metric = self.val_metric
            self.best_metric_epoch = self.curr_epoch

        # save checkpoint
        self.ckpter.save(self.curr_epoch, self.val_metric, self.best_metric_epoch, self.best_metric, self.model, self.optimizer, self.config)

        # report
        Logger.get_logger().info('Epoch {} finished. loss: {:.4f}, f1: {:.4f}, precision: {:.4f}, recall: {:.4f}, acc: {:.4f}, bset epoch: {}, best f1: {:.4f}, cost time: {}, reamin time: {}'.format(
            self.curr_epoch,
            sum(self.train_losses)/len(self.train_losses),
            self.val_metric['f1'],
            self.val_metric['precision'],
            self.val_metric['recall'],
            self.val_metric['acc'],
            self.best_metric_epoch,
            self.best_metric['f1'],
            Timer.calculate_spend(),
            Timer.calculate_remain(self.curr_epoch, self.curr_iteration, self.max_epoch, self.max_iteration)
        ))




    def val(self):
        self.val_prediction = {}
        self.val_metric = {}
        with torch.no_grad():
            self.model.eval()
            for batch in tqdm(self.dataloaders['val']):
                prepared_batch = self._to_cuda(batch)
                pred_prob, pred_w_label = self.model(prepared_batch)
                self._update_val_meter(pred_w_label)
            metric = self.datasets['val'].evaluate(self.val_prediction)
            self.val_metric.update(metric)
            Logger.get_logger().info('full result in val: f1: {:.4f}, precision: {:.4f}, recall: {:.4f}, acc: {:.4f}'.format(
                self.val_metric['f1'],
                self.val_metric['precision'],
                self.val_metric['recall'],
                self.val_metric['acc']
            ))
            
            

    def _update_val_meter(self, pred_w_label):
        # update val results
        pred_w_label = pred_w_label.cpu().numpy()
        pred_w_label = {item[0]: item[1] for item in pred_w_label}
        self.val_prediction.update(pred_w_label)



    def inference(self):
        pass








