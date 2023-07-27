import torch
from torch.autograd import variable 
from tensorboardX import SummaryWriter
from metrics import Eval_metrics
from utils import save_checkpoint, use_optimizer


class Engine(object):
    """Meta Engine for Training and Evaluating the model"""

    def __init__(self, config):
        self.config = config
        self._evaluate = Eval_metrics()
        self._writer = SummaryWriter(log_dir = 'runs/{}'.format(config['alias'])) 
        self._writer.add_text('config', str(config), 0)
        self.opt = use_optimizer(self.model, config)
        self.crit = torch.nn.BCELoss()
        

    def train_single_batch(self, users, items, rel_int, interest):
        
        if self.config['use_cuda'] is True:
            users, items, rel_int, interest = users.cuda(), items.cuda(), rel_int.cuda(), interest.cuda()
        self.opt.zero_grad()
        rating_pred = self.model(users, items, rel_int)
        loss = self.crit(rating_pred.view(-1),interest)


        loss.backward()

        #clip gradient 
        clip_grad = 3.0
        for _, p in self.model.named_parameters():
            #print('one')
            if p.grad is not None:
                #print('two')
                p.grad.data = torch.nan_to_num(p.grad.data)
                param_norm = p.grad.data.norm(2)
                clip_coef = clip_grad / (param_norm + 1e-6)
                if clip_coef < 1:
                    p.grad.data.mul_(clip_coef)
            
        self.opt.step()
        loss = loss.item()
        return loss
    
    def train_an_epoch(self, train_loader, epoch_id):
        self.model.train()
        total_loss = 0
        for batch_id, batch in enumerate(train_loader):
            assert isinstance(batch[0], torch.LongTensor)
            user, item, rel_int, interest = batch[0], batch[1], batch[2], batch[3]
            interest = interest.float()
            loss = self.train_single_batch(user, item, rel_int, interest)
            print('[epoch {}] batch {}, loss: {}'.format(epoch_id, batch_id, loss))
            total_loss += loss
        self._writer.add_scalar('model/loss', total_loss, epoch_id)


    def evaluate(self, eval_data, epoch_id):
        self.model.eval()
        with torch.no_grad():
            test_users, test_items, test_rel_int, test_y = eval_data[0], eval_data[1], eval_data[2], eval_data[3]
            if self.config['use_cuda'] is True:
                test_users = test_users.cuda()
                test_items = test_items.cuda()
                test_rel_int = test_rel_int.cuda()
                test_y = test_y.cuda() 
            test_scores = self.model(test_users, test_items, test_rel_int)
            
            if self.config['use_cuda'] is False:
                test_users = test_users.cpu()
                test_items = test_items.cpu()
                test_rel_int = test_rel_int.cpu()
                test_scores = test_scores.cpu()
                test_y = test_y.cpu() 
           
            self._evaluate.subjects = [test_users.data.view(-1).tolist(),
                                       test_items.data.view(-1).tolist(),
                                       test_scores.data.view(-1).tolist(),
                                       test_y.data.view(-1).tolist()]
            
            accuracy, recall, f1 = self._evaluate.cal_acc(), self._evaluate.cal_recall(), self._evaluate.cal_f1()
            self._writer.add_scalar('performace/ACC', accuracy, epoch_id)
            self._writer.add_scalar('performace/RECALL', recall, epoch_id)
            self._writer.add_scalar('performace/F1', f1, epoch_id)

            print('[Evaluating Epoch {}] ACC = {:.4f}, RECALL = {:.4f}, F1 = {:.4f}'.format(epoch_id, accuracy, recall, f1))
            return accuracy, recall, f1
        
    def save(self, alias, epoch_id, f1):
        if epoch_id in [20, 50, 99]:
            model_dir = self.config['model_dir'].format(alias, epoch_id, f1)
            save_checkpoint(self.model, model_dir)
