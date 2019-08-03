import torch as t
import torch.nn as nn
import torch.nn.functional as f
from config import config
from torch.optim import Adam, SGD, Adagrad
from torch.autograd import Variable
from data_utils import batch_by_num
from base_model import BaseModel, BaseModule
import logging
import os

class RotatEModule(BaseModule):
    def __init__(self, n_ent, n_rel, config):
        super(RotatEModule, self).__init__()
        sigma = 0.2
        self.gamma = nn.Parameter(
            t.Tensor([12.0]), 
            requires_grad=False
        )		
        self.rel_re_embed = nn.Embedding(n_rel, config.dim)
        self.rel_im_embed = nn.Embedding(n_rel, config.dim)
        self.ent_re_embed = nn.Embedding(n_ent, config.dim)
        self.ent_im_embed = nn.Embedding(n_ent, config.dim)
        for param in self.parameters():
            param.data.div_((config.dim / sigma ** 2) ** (1 / 6))

    def forward(self, src, rel, dst):
        head_ie = self.ent_im_embed(src)
        head_re = self.ent_re_embed(src)
        relation_ie = self.rel_im_embed(rel)
        relation_re = self.rel_re_embed(rel)
        tail_ie = self.ent_im_embed(dst)
        tail_re = self.ent_re_embed(dst)

        re_score = head_re * relation_re - head_ie * relation_ie#*就是点积，哈达玛积
        im_score = head_re * relation_ie + head_ie * relation_re#这两行就是复数乘积的公式（a+bj）*(c+dj)=(ac-bd)+(bc+ad)j
        re_score = re_score - tail_re
        im_score = im_score - tail_ie
        score = t.stack([re_score, im_score], dim = 0)#list中的每个元素是结果中第dim维的每个元素
        #score=(x,x)
        score = score.norm(dim = 0)#每一dim上求一个L1范数（平方和开根）
        #dim=0,就是其他维的index不变，dim这一维从0到size求一个L1范数，最后的个数是除了dim这一维以外其他维size的乘积
        score = self.gamma.item() - score.sum(dim = 2)#a number minus matrix
        return score
		
    def score(self, src, rel, dst):
        return -self.forward(src, rel, dst)

    def dist(self, src, rel, dst):
        return -self.forward(src, rel, dst)

    def prob_logit(self, src, rel, dst):
        return self.forward(src, rel, dst)

class RotatE(BaseModel):
    def __init__(self, n_ent, n_rel, config):
        super(RotatE, self).__init__()
        self.mdl = RotatEModule(n_ent, n_rel, config)
        self.mdl#.cuda()
        self.config = config
        self.weight_decay = config.lam / config.n_batch

    def pretrain(self, train_data, corrupter, tester):
        src, rel, dst = train_data
        n_train = len(src)
        n_epoch = self.config.n_epoch
        n_batch = self.config.n_batch
        optimizer = Adam(self.mdl.parameters(), weight_decay=self.weight_decay)
        best_perf = 0
        for epoch in range(n_epoch):
            epoch_loss = 0
            if epoch % self.config.sample_freq == 0:
                rand_idx = t.randperm(n_train)
                src = src[rand_idx]
                rel = rel[rand_idx]
                dst = dst[rand_idx]
                src_corrupted, rel_corrupted, dst_corrupted = corrupter.corrupt(src, rel, dst)
                src_corrupted = src_corrupted#.cuda()
                rel_corrupted = rel_corrupted#.cuda()
                dst_corrupted = dst_corrupted#.cuda()
            for ss, rs, ts in batch_by_num(n_batch, src_corrupted, rel_corrupted, dst_corrupted, n_sample=n_train):
                self.mdl.zero_grad()
                label = t.zeros(len(ss)).type(t.LongTensor)#.cuda()
                loss = t.sum(self.mdl.softmax_loss(Variable(ss), Variable(rs), Variable(ts), label))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.data[0]
            logging.info('Epoch %d/%d, Loss=%f', epoch + 1, n_epoch, epoch_loss / n_train)
            if (epoch + 1) % self.config.epoch_per_test == 0:
                test_perf = tester()
                if test_perf > best_perf:
                    self.save(os.path.join(config().task.dir, self.config.model_file))
                    best_perf = test_perf
        return best_perf