# monuseg

python main_monuseg.py --model_type='SCU_Net' 
python main_monuseg.py --model_type='AttU_Net'
python main_monuseg.py --model_type='BAMU_Net' --att_mode='bam' --reduction_ratio 8
python main_monuseg.py --model_type='CBAMU_Net' --att_mode='cbam' --reduction_ratio 8
python main_monuseg.py --model_type='SKU_Net' --conv_type='sk'
python main_monuseg.py --model_type='MHU_Net' --n_head 2

python test_monuseg.py --model_type='SSU_Net' --n_head 2
python test_monuseg.py --model_type='SSU_Net' --loss_type='nll+ssim' --n_head 2
python test_monuseg.py --model_type='SEU_Net' --att_mode='se' --reduction_ratio 8
python test_monuseg.py --model_type='UNet'


# weights on Cam

