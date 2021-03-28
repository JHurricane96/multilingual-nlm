import torch
import torch.nn as nn
import numpy as np
import time
import os
import utils.CLR as CLR
from utils.train_base import PAD_Sentences
from torch.nn.utils import clip_grad_norm_
from torch import optim

class Langage_Model_Class(nn.Module):

    def __init__(self, lm, lang_size, vocab2id_input, vocab2id_output):
        super().__init__()
        self.lm = lm
        self.lang_size = lang_size
        self.ignore_index = vocab2id_output["<ignore_idx>"]
        self.PAD_index = vocab2id_input["<PAD>"] #PAD Embedding index
        self.BOS_fwd_index = vocab2id_input["<BOS_fwd>"]
        self.BOS_bkw_index = vocab2id_input["<BOS_bkw>"]
        self.EOS_index = vocab2id_output["<EOS>"]
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='none')
    def set_device(self, gpuid):
        is_cuda = gpuid >= 0
        self.lm.set_device(is_cuda)
        if is_cuda:
            self.torch = torch.cuda
        else:
            self.torch = torch

    def __call__(self, input_id, s_lengths, *args):
        softmax_score = self.lm(self.torch.LongTensor(input_id), s_lengths, *args)
        return softmax_score

    def Calc_loss(self,softmax_score, output_id):
        #softmax_score: bs, s_len, tgtV
        #t_id_EOS: bs, s_len
        batch_size, s_len, tgtV = softmax_score.size()
        loss = self.cross_entropy(softmax_score.view(batch_size*s_len, tgtV), self.torch.LongTensor(output_id).view(-1))  # (bs * maxlen_t,)
        loss = torch.sum(loss) / batch_size
        return loss

    def Register_vocab(self,vocab2id_input, vocab2id_output,id2vocab_input,id2vocab_output):
        self.vocab2id_input = vocab2id_input
        self.vocab2id_output = vocab2id_output
        self.id2vocab_input = id2vocab_input
        self.id2vocab_output = id2vocab_output


class Trainer_base():
    def __init__(self, dataset, file_name, lr_finding):
        self.dataset = dataset
        self.file_name = file_name
        self.cumloss_old = np.inf
        self.cumloss_new = np.inf
        self.lr_finding = lr_finding

    def Update_params(self, model, dataset, optimizer, index, *args):

        return NotImplementedError

    def set_optimiser(self, model, opt_type, lr_rate):
        self.lr_rate = lr_rate
        self.scheduler = None
        if opt_type == "SGD":
            self.optimizer = optim.SGD(model.parameters(), lr=lr_rate, momentum=0.95, weight_decay=1e-4)
        elif opt_type == "ASGD":
            self.optimizer = optim.ASGD(model.parameters(), lr=lr_rate)
            self.scheduler = 1
        elif opt_type == "AdamW":
            self.optimizer = optim.AdamW(model.parameters(), lr=lr_rate, betas=(0.95, 0.99), weight_decay=0.4)

    def update_lr(self, optimizer, lr):
        for g in optimizer.param_groups:
            g['lr'] = lr

    def main(self, model, epoch_size, stop_threshold,remove_models =False):
        print ("epoch start")
        old_model_name = None
        model.train()
        for epoch in range(1, epoch_size+1): #for each epoch
            print("epoch: ",epoch)
            self.cumloss_old = self.cumloss_new
            cumloss = 0
            batch_idx_list = np.random.permutation(self.dataset.batch_idx_list) # shuffle batch order
            if self.scheduler == None and not self.lr_finding:
                steps_per_epoch = len(batch_idx_list) * 2 * model.lang_size
                self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, self.lr_rate, steps_per_epoch=steps_per_epoch, epochs=epoch_size)
            clr = None
            if self.lr_finding:
                clr = CLR.CLR(self.optimizer, len(batch_idx_list), max_lr=0.3, base_lr=self.lr_rate)
            start = time.time()
            for bt_num, bt_idx in enumerate(batch_idx_list):
                loss = self.Update_params(model, self.dataset, self.optimizer, bt_idx)
                if self.lr_finding:
                    lr = clr.calc_lr(loss)
                    if lr == -1 :
                        break
                    self.update_lr(self.optimizer, lr)
                cumloss = cumloss + loss
                if (bt_num + 1) % 100 == 0:
                    print(bt_num + 1, "mini-batches finished")
            if self.lr_finding:
                clr.plot()
            print("All mini-batches finished")
            #end of epoch

            self.cumloss_new = cumloss/len(batch_idx_list)
            elapsed_time = time.time() - start
            print("Train elapsed_time:{0}".format(elapsed_time) + "[sec]")
            print("loss: ", self.cumloss_new)
            print(self.file_name +"_epoch" + str(epoch))
            new_model_name = self.file_name + "_epoch" + str(epoch) +'.model'
            torch.save(model.state_dict(), new_model_name)
            if (remove_models and epoch != 1):
                print("remove the previous model")
                os.remove(old_model_name)
            old_model_name = new_model_name
            improvement_rate = self.cumloss_new / self.cumloss_old
            print("loss improvement rate:", 1 - improvement_rate)
            if (improvement_rate > stop_threshold):
                break
        print("finish training")
        return model


class Trainer(Trainer_base):
    def __init__(self, dataset, file_name, lr_finding):
        super().__init__(dataset, file_name, lr_finding)

    def Update_params_base(self, model, optimizer, s_id, s_id_EOS, s_lengths, *args):
        model.zero_grad()
        softmax_score = model(s_id, s_lengths, *args)
        loss = model.Calc_loss(softmax_score, s_id_EOS)
        loss.backward()
        clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if not self.lr_finding:
            self.scheduler.step()
        return loss.data.tolist()

    def Update_params(self, model, dataset, optimizer, index, *args):
        loss_all = 0
        for lang in range(model.lang_size):
            model.lm.Switch_Lang(lang)
            s_lengths, BOS_lines_id_input, lines_id_output_EOS, BOS_lines_id_input_bkw, lines_id_output_EOS_bkw = \
                PAD_Sentences(model, dataset.lengths[lang], dataset.lines_id_input[lang],
                              dataset.lines_id_output[lang], index)

            model.lm.Switch_fwdbkw("fwd")
            loss_all += self.Update_params_base(model, optimizer, BOS_lines_id_input, lines_id_output_EOS, s_lengths)

            model.lm.Switch_fwdbkw("bkw")
            loss_all += self.Update_params_base(model, optimizer, BOS_lines_id_input_bkw, lines_id_output_EOS_bkw,
                                                s_lengths)
        return loss_all
