from pathlib import Path
from pyfaidx import Fasta
import pandas as pd
import torch
from random import randrange, random
import numpy as np

string_complement_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'a': 't', 'c': 'g', 'g': 'c', 't': 'a'}

def string_reverse_complement(seq):
    rev_comp = ''
    for base in seq[::-1]:
        if base in string_complement_map:
            rev_comp += string_complement_map[base]
        # if bp not complement map, use the same bp
        else:
            rev_comp += base
    return rev_comp

class FastaInterval():
    def __init__(
        self,
        fasta_file,
        pad_interval=True,
        add_cls=True,
    ):
        fasta_file = Path(fasta_file)
        assert fasta_file.exists(), 'path to fasta file must exist'
        self.seqs = Fasta(str(fasta_file))

        self.pad_interval = pad_interval
        self.add_cls = add_cls
            
        # calc len of each chromosome in fasta file, store in dict
        self.chr_lens = {}
        for chr_name in self.seqs.keys():
            self.chr_lens[chr_name] = len(self.seqs[chr_name])

        print(self.chr_lens)
    def __call__(self, chr_name, start, end, cls_index, rc=False):
        
        chromosome = self.seqs[chr_name]
        # chromosome_length = len(chromosome)
        chromosome_length = self.chr_lens[chr_name]

        left_padding = right_padding = 0
        if start < 0:
            left_padding = -start
            start = 0
        if end > chromosome_length:
            right_padding = end - chromosome_length
            end = chromosome_length

        seq = str(chromosome[start:end])
        if self.pad_interval:
            seq = ('.' * left_padding) + seq + ('.' * right_padding)
        if rc:
            seq = string_reverse_complement(seq)
        if self.add_cls:
            seq[:cls_index] + '[CLS]' + seq[cls_index:]
        return seq

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class ChangeSeqDataset(Dataset):
    def __init__(self, df, genome, dna_sequence_length=1000, task='regression', threshold=0):
        super().__init__()
        self.df = df.reset_index(drop=True).query('chrom!="chrY"').query('chrom!="chrM"')
        self.genome = genome
        self.dna_sequence_length = dna_sequence_length
        
        self.length = 23 # length of target dna in dataset
        self.added_context = int((self.dna_sequence_length - self.length)/2)
        self.roi_start = self.added_context - 1
        self.roi_end = self.added_context + self.length - 1
        self.roi = (self.roi_start, self.roi_end)
        
        self.task = task
        self.threshold = threshold
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        chrom, strand, _, end = row[['chrom', 'Align.strand', 'Align.chromStart', 'Align.chromEnd']].values
        start = end - self.length - self.added_context # select start based on end since bulges modify length
        end = end + self.added_context
        sgRNA = row['sgRNA']

        dna = self.genome(chrom, start, end, rc=strand=='-', cls_index=self.roi_end if strand=='+' else self.roi_start)
        #print(strand, dna, row['Align.off-target'], end-start, len(dna))
        counts = torch.tensor([row['reads']])
        if self.task=='regression':
            y = torch.log(1+counts)
        elif self.task == 'classification':
            y = (counts > self.threshold).int()
        return sgRNA, dna, y

class ChangeSeqDataModule(pl.LightningDataModule):
    def __init__(self, 
                 data_path: str,
                 fasta_path: str, 
                 dna_sequence_length: int = 1000,
                 task: str = 'classification',
                 threshold: int = 0,
                 batch_size: int = 32, 
                 num_workers: int = 0):
        super().__init__()
        self.df = pd.read_csv(data_path)
        self.genome = FastaInterval(fasta_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dna_sequence_length = dna_sequence_length
        self.task = task
        self.threshold = threshold

        self.length = 23 # length of target dna in dataset
        self.added_context = int((self.dna_sequence_length - self.length)/2)
        self.roi_start = self.added_context - 1
        self.roi_end = self.added_context + self.length - 1
        self.roi = (self.roi_start, self.roi_end)
        
        train_sgRNA, test_sgRNA = torch.utils.data.random_split(pd.unique(self.df['sgRNA']), [0.8, 0.2])

        split = []
        for sgRNA in self.df.sgRNA:
            if sgRNA in test_sgRNA:
                split.append('val')
            else:
                split.append('train')

        self.df['split'] = split

    def setup(self, stage: str = None):
        """
        Sets up datasets for different stages: 'fit', 'test', 'predict'.
        """
        if stage == "fit" or stage is None:
            self.train_dataset = ChangeSeqDataset(
                self.df.query('split=="train"'), 
                self.genome, 
                dna_sequence_length=self.dna_sequence_length,
                task = self.task,
                threshold = self.threshold,
            )
            self.val_dataset = ChangeSeqDataset(
                self.df.query('split=="val"'), 
                self.genome, 
                dna_sequence_length=self.dna_sequence_length,
                task = self.task,
                threshold = self.threshold,
            )
        if stage == "test" or stage is None:
            self.test_dataset = ChangeSeqDataset(
                self.df.query('split=="test"'), 
                self.genome, 
                dna_sequence_length=self.dna_sequence_length,
                task = self.task,
                threshold = self.threshold,
            )
        if stage == "predict":
            self.test_dataset = ChangeSeqDataset(
                self.df, 
                self.genome, 
                dna_sequence_length=self.dna_sequence_length,
                task = self.task,
                threshold = self.threshold,
            )

    def dataloader(self, dataset, shuffle=True):
        return DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=shuffle,
        )
    def train_dataloader(self):
        return self.dataloader(self.train_dataset, shuffle=True)
    def val_dataloader(self):
        return self.dataloader(self.val_dataset, shuffle=True)
    def test_dataloader(self):
        return self.dataloader(self.test_dataset, shuffle=True)
    def predict_dataloader(self):
        return self.dataloader(self.test_dataset, shuffle=False)

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba2
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import AveragePrecision, AUROC

class BiMambaEncoder(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, bidirectional_method='add'):
        super(BiMambaEncoder, self).__init__()
        self.d_model = d_model
        self.bidirectional_method = bidirectional_method
        
        self.mamba = Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

        # Norm and feed-forward network layer
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        if self.bidirectional_method=='concat':
            self.reducer = nn.Linear(d_model * 2, d_model)
            
    def forward(self, x):
        # Residual connection of the original input
        residual = x
        
        # Forward Mamba
        x_norm = self.norm1(x)
        mamba_out_forward = self.mamba(x_norm)

        # Backward Mamba
        x_flip = torch.flip(x_norm, dims=[1])  # Flip Sequence
        mamba_out_backward = self.mamba(x_flip)
        mamba_out_backward = torch.flip(mamba_out_backward, dims=[1])  # Flip back

        # Combining forward and backward
        if self.bidirectional_method=='add':
            mamba_out = mamba_out_forward + mamba_out_backward
        elif self.bidirectional_method=='concat':
            mamba_out = torch.cat([mamba_out_forward, mamba_out_backward], dim=-1)
            mamba_out = self.reducer(mamba_out)
        else:
            raise NotImplementedError(f'bidirectional_method {self.bidirectional_method} is not implemented.')
        
        mamba_out = self.norm2(mamba_out)
        ff_out = self.feed_forward(mamba_out)

        output = ff_out + residual
        return output

class BiMambaStack(nn.Module):
    def __init__(self, n_layers, dim, d_state, d_conv, expand, bidirectional_method='add'):
        super().__init__()
        self.backbone = nn.Sequential(
            *[BiMambaEncoder(dim, d_state, d_conv, expand, bidirectional_method=bidirectional_method) for _ in range(n_layers)]
        )
    def forward(self, seq):
        return self.backbone(seq)

class BaseModule(pl.LightningModule): 
    def __init__(self):
        super().__init__()

        
    def step(self, batch):
        rna, dna, y = batch
        y_hat = self(rna, dna)
        loss = self.criterion(y_hat, y.float())
        return loss, y, y_hat
        
    def training_step(self, batch, batch_idx):
        loss, y, y_hat = self.step(batch)
        # Log training metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_auprc(y_hat, y)
        self.log("train_auprc", self.train_auprc, on_step=False, on_epoch=True, prog_bar=True)
        self.train_auc(y_hat, y)
        self.log("train_auc", self.train_auc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y, y_hat = self.step(batch)
        # Log validation metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_auprc(y_hat, y)
        self.log("val_auprc", self.val_auprc, on_step=False, on_epoch=True, prog_bar=True)
        self.val_auc(y_hat, y)
        self.log("val_auc", self.val_auc, on_step=False, on_epoch=True, prog_bar=True)


    def test_step(self, batch, batch_idx):
        loss, y, y_hat = self.step(batch)
        # Log test metrics
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.test_auprc(y_hat, y)
        self.log("test_auprc", self.test_auprc, on_step=False, on_epoch=True, prog_bar=True)
        self.test_auc(y_hat, y)
        self.log("test_auc", self.test_auc, on_step=False, on_epoch=True, prog_bar=True)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

class ChangeSeqModel(BaseModule):
    def __init__(self, tokenizer, n_layers, dim, d_state, d_conv, expand, bidirectional_method='add', pooling='mean', task='regression'):
        super().__init__()

        # Model
        self.tokenizer = tokenizer
        self.sgrna_embedding = nn.Embedding(tokenizer.vocab_size, dim)
        self.dna_embedding   = nn.Embedding(tokenizer.vocab_size, dim)
    
        self.backbone  = BiMambaStack(n_layers, dim, d_state, d_conv, expand, bidirectional_method=bidirectional_method)
        
        self.decoder = nn.Sequential(
            nn.Linear(dim, 1), 
            nn.Sigmoid() if task=='classification' else nn.Identity()
        )

        self.pooling = pooling
        self.task = task
        
        # Define a loss function
        if self.task=='classification':
            self.criterion = torch.nn.BCELoss()
        elif self.task=='regression':
            self.criterion = torch.nn.L1Loss()

        # Define metrics
        self.train_auprc = AveragePrecision(task="binary")
        self.val_auprc = AveragePrecision(task="binary")
        self.test_auprc = AveragePrecision(task="binary")

        self.train_auc = AUROC(task="binary")
        self.val_auc = AUROC(task="binary")
        self.test_auc = AUROC(task="binary")

    def forward(self, rna, dna):
        rna, dna = self.tokenizer(rna, return_tensors='pt')['input_ids'].to(self.device), self.tokenizer(dna, return_tensors='pt')['input_ids'].to(self.device)
        rna, dna = self.sgrna_embedding(rna), self.dna_embedding(dna)
        seq = torch.cat([rna, dna, rna], dim=-2)
        seq = self.backbone(seq)
        pooled = self.pool(seq)
        return self.decoder(pooled)
    
    def on_start(self):
        self.roi_start, self.roi_end = self.trainer.datamodule.roi
    def on_fit_start(self):
        self.on_start()
    def on_predict_start(self):
        self.on_start()

    def pool(self, X):
        if self.pooling=='mean':
            return torch.mean(X[:,self.roi_start+24:self.roi_end+24], dim=-2) 
        elif self.pooling=='CLS':
            return X[:,self.roi_end+24]

change_seq_path = 'files/datasets/CHANGEseq/include_on_targets/CHANGEseq_CR_Lazzarotto_2020_dataset.csv'
fasta_path = 'hg38/hg38.ml.fa'

tot_len = 512
datamodule = ChangeSeqDataModule(change_seq_path, fasta_path, dna_sequence_length=tot_len-24*2-1, task='classification', batch_size=128)

model_name = "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = ChangeSeqModel(tokenizer, 4, 256, 256, 3, 2, bidirectional_method='concat', pooling='mean', task='classification')

from lightning.pytorch.loggers import WandbLogger
wandb_logger = WandbLogger(project="Mamba-CRISPR")

from pytorch_lightning import Trainer
trainer = Trainer(
    max_epochs=10, 
    accelerator='gpu',
    limit_train_batches=0.1,
    limit_val_batches=0.1,
    logger=wandb_logger,
)

trainer.fit(model, datamodule)

