import argparse
import os
import re
from library.datamodules.sen2cloud import Sen2CloudDataModule
from terratorch.tasks import SemanticSegmentationTask
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
import lightning.pytorch as pl


def get_command_line_options():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--dataroot", "-d", type=str, default='./dataset')

    parser.add_argument("--backbone", "-bc", type=str, default='resnet50')
    parser.add_argument("--decoder", "-d", type=str, default='FCNDecoder')

    parser.add_argument("--epoch", "-e", type=int, default=50)
    parser.add_argument("--batch", "-b", type=int, default=8)

    parser.add_argument("--lr", '-lr', type=float, default=1e-3)

    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--gpu", "-g", type=int, default=0)

    parser.add_argument("--verbose", "-v", type=bool, default=True)

    return parser.parse_args()

def find_best_ckpt(rootdir:str):

    regex = re.compile('best*')

    for root, dirs, files in os.walk(rootdir):
        for file in files:
            if regex.match(file):
                break
    
    return file

def get_datamodule(data_root:'str'='./dataset', batch_size:int = 8, verbose:bool = True):

    datamodule = Sen2CloudDataModule(
        data_root = data_root,
        batch_size = batch_size,
        means = [2672.63818359375, 2678.138671875, 2587.265380859375, 3507.404052734375],
        stds = [3047.300537109375, 2805.623779296875, 2705.935791015625, 2409.601318359375],
    )

    if verbose:
        datamodule.setup("fit")
        datamodule.setup("test")

        train_dataset = datamodule.train_dataset
        val_dataset = datamodule.val_dataset
        test_dataset = datamodule.test_dataset

        print('# of Train samples: ', len(train_dataset))
        print('# of Val samples: ', len(val_dataset))
        print('# of Test samples: ', len(test_dataset))
    
    return datamodule

def get_model_args(backbone='prithvi_eo_v2_300', decoder='UNetDecoder', pretrained:bool = True, verbose:bool = True):

    if backbone=='prithvi_eo_v2_300':
        backbone_args = {
            "backbone": "prithvi_eo_v2_300",
            "backbone_kwargs": {
                'pretrained': pretrained,
                'bands': ["BLUE", "GREEN", "RED", "NIR_NARROW"],
                'img_size':512,
            },

            # Necks
            "necks": [
                {
                    "name": "SelectIndices",
                    "indices": [5, 11, 17, 23] # indices for prithvi_eo_v2_300
                },
                {"name": "ReshapeTokensToImage",},
                {"name": "LearnedInterpolateToPyramidal"}
            ],
        }
    elif backbone=='resnet50':
        backbone_args = {
            "backbone": "resnet50",
            "backbone_kwargs": {
                'pretrained': pretrained,
                'in_chans': 4,
            },
        }
    else:
        assert False, f"{backbone} is not recognizable."

    if decoder == 'UNetDecoder':
        decoder_args = {
            "decoder": "UNetDecoder",
            "decoder_kwargs": {
                'channels': [512, 256, 128, 64],
            },
        }
    elif decoder == 'FCNDecoder':
        decoder_args = {
            "decoder": "FCNDecoder",
            "decoder_kwargs": {
                'channels' : 256,
                'num_convs' : 4,
            },
        }
    else:
        assert False, f"{backbone} is not recognizable."

    head_args = {
        "head_dropout": 0.1,
        "num_classes": 2,
    }

    model_args = {}
    model_args.update(backbone_args)
    model_args.update(decoder_args)
    model_args.update(head_args)

    if verbose:
        print('Model configuration...')
        print(f"{backbone} (Pre-training: {pretrained}) + {decoder}")

    return model_args

def get_SegTask(model_args:dict, 
             loss:str = 'ce', lr = 1e-3, 
             opt:str = 'AdamW', opt_hyparam:dict = {"weight_decay": 0.05},
             freeze_backbone:bool = False, 
             freeze_decoder:bool = False, 
             class_name:list = ['No', 'Cloud']):

    task = SemanticSegmentationTask(
        model_args=model_args,
        model_factory="EncoderDecoderFactory",
        loss=loss,
        lr=lr,
        optimizer=opt,
        optimizer_hparams=opt_hyparam,
        freeze_backbone=freeze_backbone, # True. Only to speed up fine-tuning
        freeze_decoder=freeze_decoder,
        class_names=class_name,  # optionally define class names
        plot_on_val=0,
    )

    return task

if __name__ == "__main__":

    args = get_command_line_options()

    pl.seed_everything(args.seed)

    datamodule = get_datamodule(
        data_root = args.dataroot,
        batch_size = args.batch,
        verbose = args.verbose,
    )
    model_args = get_model_args(
        backbone = args.backbone,
        decoder = args.decoder,
        verbose = args.verbose,
    )
    task = get_SegTask(
        model_args = model_args,
        lr = args.lr,
    )

    logger = TensorBoardLogger(
        save_dir='output',
        version=f"E{args.epoch}_B{args.batch}_ce_LR{args.lr}",
        name=f"{args.backbone}_{args.decoder}"
        )

    trainer = Trainer(
        devices=[args.gpu],
        precision="16-mixed",
        logger=logger,
        max_epochs=args.epoch,
        default_root_dir='output',
    )

    ckpt_root = f"./output/L_{args.backbone}_{args.decoder}/E{args.epoch}_B{args.batch}_ce_LR{args.lr}/checkpoints/"
    ckpt_name = find_best_ckpt(ckpt_root)

    test_results = trainer.test(model=task, datamodule=datamodule, ckpt_path=os.path.join(ckpt_root, ckpt_name))
    
    print("+++++++++++++++++++++++++++++++++++++++++")
    print(test_results)
    print("+++++++++++++++++++++++++++++++++++++++++")

    print("\n\Test completed!")

