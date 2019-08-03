import torch
import torch.nn as nn
import torch.nn.functional as nnf
from config import config
from torch.autograd import Variable
from torch.optim import Adam
from metrics import mrr_mr_hitk
from data_utils import batch_by_size
import logging


class BaseModule(nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()

    def score(self, src, rel, dst):
        raise NotImplementedError

    def dist(self, src, rel, dst):
        raise NotImplementedError

    def prob_logit(self, src, rel, dst):
        raise NotImplementedError

    def prob(self, src, rel, dst):
        return nnf.softmax(self.prob_logit(src, rel, dst))

    def constraint(self):
        pass

    def pair_loss(self, src, rel, dst, src_bad, dst_bad):
        d_good = self.dist(src, rel, dst)
        d_bad = self.dist(src_bad, rel, dst_bad)
        return nnf.relu(self.margin + d_good - d_bad)

    def softmax_loss(self, src, rel, dst, truth):
        probs = self.prob(src, rel, dst)
        n = probs.size(0)
        truth_probs = torch.log(probs[torch.arange(0, n).type(torch.LongTensor), truth] + 1e-30)
        return -truth_probs


class BaseModel(object):
    def __init__(self):
        self.mdl = None # type: BaseModule
        self.weight_decay = 0

    def save(self, filename):
        torch.save(self.mdl.state_dict(), filename)

    def load(self, filename):
        self.mdl.load_state_dict(torch.load(filename, map_location=lambda storage, location: storage))

    def gen_step(self, src, rel, dst, n_sample=1, temperature=1.0, train=True):
        if not hasattr(self, 'opt'):
            self.opt = Adam(self.mdl.parameters(), weight_decay=self.weight_decay)
        n, m = dst.size()
        rel_var = Variable(rel)
        src_var = Variable(src)
        dst_var = Variable(dst)

        logits = self.mdl.prob_logit(src_var, rel_var, dst_var) / temperature
        probs = nnf.softmax(logits)
        row_idx = torch.arange(0, n).type(torch.LongTensor).unsqueeze(1).expand(n, n_sample)
        sample_idx = torch.multinomial(probs, n_sample, replacement=True)
        sample_srcs = src[row_idx, sample_idx.data.cpu()]
        sample_dsts = dst[row_idx, sample_idx.data.cpu()]
        rewards = yield sample_srcs, sample_dsts
        if train:
            self.mdl.zero_grad()
            log_probs = nnf.log_softmax(logits)
            reinforce_loss = -torch.sum(Variable(rewards) * log_probs[row_idx, sample_idx.data])
            reinforce_loss.backward()
            self.opt.step()
            self.mdl.constraint()
        yield None

    def dis_step(self, src, rel, dst, src_fake, dst_fake, train=True):
        if not hasattr(self, 'opt'):
            self.opt = Adam(self.mdl.parameters(), weight_decay=self.weight_decay)
        src_var = Variable(src)
        rel_var = Variable(rel)
        dst_var = Variable(dst)
        src_fake_var = Variable(src_fake)
        dst_fake_var = Variable(dst_fake)
        losses = self.mdl.pair_loss(src_var, rel_var, dst_var, src_fake_var, dst_fake_var)
        fake_scores = self.mdl.score(src_fake_var, rel_var, dst_fake_var)
        if train:
            self.mdl.zero_grad()
            torch.sum(losses).backward()
            self.opt.step()
            self.mdl.constraint()
        return losses.data, -fake_scores.data

    def test_link(self, test_data, n_ent, heads, tails, filt=False):
        mrr_tot = 0
        mr_tot = 0
        hit10_tot = 0
        count = 0
        for batch_s, batch_r, batch_t in batch_by_size(config().test_batch_size, *test_data):
            batch_size = batch_s.size(0)
            rel_var = batch_r.unsqueeze(1).expand(batch_size, n_ent)#.cuda()
            src_var = batch_s.unsqueeze(1).expand(batch_size, n_ent)#.cuda()
            dst_var = batch_t.unsqueeze(1).expand(batch_size, n_ent)#.cuda()
            all_var = torch.arange(0, n_ent).unsqueeze(0).expand(batch_size, n_ent).type(torch.LongTensor)
            batch_dst_scores = self.mdl.score(src_var, rel_var, all_var).data
            batch_src_scores = self.mdl.score(all_var, rel_var, dst_var).data
            for s, r, t, dst_scores, src_scores in zip(batch_s, batch_r, batch_t, batch_dst_scores, batch_src_scores):
                if filt:
                    if tails[(s.item(), r.item())]._nnz() > 1:
                        tmp = dst_scores[t].item()
                        dst_scores += tails[(s.item(), r.item())] * 1e30
                        dst_scores[t] = tmp
                    if heads[(t.item(), r.item())]._nnz() > 1:
                        tmp = src_scores[s].item()
                        src_scores += heads[(t.item(), r.item())] * 1e30
                        src_scores[s] = tmp
                mrr, mr, hit10 = mrr_mr_hitk(dst_scores, t)
                mrr_tot += mrr
                mr_tot += mr
                hit10_tot += hit10
                mrr, mr, hit10 = mrr_mr_hitk(src_scores, s)
                mrr_tot += mrr
                mr_tot += mr
                hit10_tot += hit10
                count += 2
        logging.info('Test_MRR=%f, Test_MR=%f, Test_H@10=%f', mrr_tot / count, mr_tot / count, hit10_tot / count)
        return mrr_tot / count
