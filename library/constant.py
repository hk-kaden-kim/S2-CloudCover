from .analysis import TensorboardLogReader

# Small models
r34_fcn = {
    'name' : 'r34_fcn',
    'train' : TensorboardLogReader('output/resnet34_FCNDecoder/FT_Enc_Dec_E50_B8_ce_LR0.001/events.out.tfevents.1742341287.ip-172-16-79-209.ec2.internal.1919.1'),
    'test' : TensorboardLogReader('output/resnet34_FCNDecoder/FT_Enc_Dec_E50_B8_ce_LR0.001/events.out.tfevents.1742343899.ip-172-16-79-209.ec2.internal.1919.2'),
    'param_M' : 25
}
r34_unet = {
    'name' : 'r34_unet',
    'train' : TensorboardLogReader('output/resnet34_UNetDecoder/FT_Enc_Dec_E50_B8_ce_LR0.001/events.out.tfevents.1742385267.ip-172-16-179-221.ec2.internal.21939.0'),
    'test' : TensorboardLogReader('output/resnet34_UNetDecoder/FT_Enc_Dec_E50_B8_ce_LR0.001/events.out.tfevents.1742387590.ip-172-16-179-221.ec2.internal.21939.1'),
    'param_M' : 29.9
}
p100_fcn = {
    'name' : 'p100_fcn',
    'train' : TensorboardLogReader('output/prithvi_eo_v1_100_FCNDecoder/FT_Enc_Dec_E50_B8_ce_LR1e-06/events.out.tfevents.1742675956.ip-172-16-72-143.ec2.internal.21407.0'),
    'test' : TensorboardLogReader('output/prithvi_eo_v1_100_FCNDecoder/FT_Enc_Dec_E50_B8_ce_LR1e-06/events.out.tfevents.1742680339.ip-172-16-72-143.ec2.internal.21407.1'),
    'param_M' : 92.4
}
p100_unet = {
    'name' : 'p100_unet',
    'train' : TensorboardLogReader('output/prithvi_eo_v1_100_UNetDecoder/FT_Enc_Dec_E50_B8_ce_LR1e-06/events.out.tfevents.1742676091.ip-172-16-72-143.ec2.internal.23267.0'),
    'test' : TensorboardLogReader('output/prithvi_eo_v1_100_UNetDecoder/FT_Enc_Dec_E50_B8_ce_LR1e-06/events.out.tfevents.1742680115.ip-172-16-72-143.ec2.internal.23267.1'),
    'param_M' : 101
}

# Large models
r50_fcn = {
    'name' : 'r50_fcn',
    'train' : TensorboardLogReader('output/resnet50_FCNDecoder/FT_Enc_Dec_E50_B8_ce_LR0.001/events.out.tfevents.1742422637.ip-172-16-14-62.ec2.internal.30329.0'),
    'test' : TensorboardLogReader('output/resnet50_FCNDecoder/FT_Enc_Dec_E50_B8_ce_LR0.001/events.out.tfevents.1742425301.ip-172-16-14-62.ec2.internal.30329.2'),
    'param_M' : 28.8
}
r50_unet = {
    'name' : 'r50_unet',
    'train' : TensorboardLogReader('output/resnet50_UNetDecoder/FT_Enc_Dec_E50_B8_ce_LR0.001/events.out.tfevents.1742422803.ip-172-16-14-62.ec2.internal.21549.0'),
    'test' : TensorboardLogReader('output/resnet50_UNetDecoder/FT_Enc_Dec_E50_B8_ce_LR0.001/events.out.tfevents.1742425403.ip-172-16-14-62.ec2.internal.21549.1'),
    'param_M' : 43.9
}
p300_fcn ={
    'name' : 'p300_fcn',
    'train' : TensorboardLogReader('output/prithvi_eo_v2_300_FCNDecoder/FT_Enc_Dec_E50_B8_ce_LR1e-06/events.out.tfevents.1742676512.ip-172-16-72-143.ec2.internal.28420.0'),
    'test' : TensorboardLogReader('output/prithvi_eo_v2_300_FCNDecoder/FT_Enc_Dec_E50_B8_ce_LR1e-06/events.out.tfevents.1742683120.ip-172-16-72-143.ec2.internal.28420.1'),
    'param_M' : 312
}
p300_unet = {
    'name' : 'p300_unet',
    'train' : TensorboardLogReader('output/prithvi_eo_v2_300_UNetDecoder/FT_Enc_Dec_E50_B8_ce_LR1e-06/events.out.tfevents.1742683899.ip-172-16-1-59.ec2.internal.17679.0'),
    'test' : TensorboardLogReader('output/prithvi_eo_v2_300_UNetDecoder/FT_Enc_Dec_E50_B8_ce_LR1e-06/events.out.tfevents.1742690052.ip-172-16-1-59.ec2.internal.17679.1'),
    'param_M' : 323
}

# Final models
l_r50_unet = {
    'name' : 'l_r50_unet',
    'train' : TensorboardLogReader('output/L_resnet50_UNetDecoder/E30_B8_ce_LR0.001/events.out.tfevents.1743591068.ip-172-31-91-190.ec2.internal'),
    'test' : TensorboardLogReader('output/L_resnet50_UNetDecoder/E30_B8_ce_LR0.001/events.out.tfevents.1743891972.ip-172-31-91-190.ec2.internal'),
    'param_M' : 43.9
}
l_p300_unet = {
    'name' : 'l_p300_unet',
    'train' : TensorboardLogReader('output/L_prithvi_eo_v2_300_UNetDecoder/E30_B8_ce_LR1e-06/events.out.tfevents.1743633467.ip-172-31-91-190.ec2.internal'),
    'test' : TensorboardLogReader('output/L_prithvi_eo_v2_300_UNetDecoder/E30_B8_ce_LR1e-06/events.out.tfevents.1743891966.ip-172-31-91-190.ec2.internal'),
    'param_M' : 323
}