import torch.nn as nn
import cv2
SETTINGS = {
### ------------------------------------------------ NSA ------------------------------------------------ ###
    'Shift' : {
        'fname': 'shift.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True,
        'final_activation': 'sigmoid',
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True,
                           'shift':True, 'same':False, 'mode':cv2.NORMAL_CLONE, 'label_mode':'binary'}
    },
    'Shift-923874273' : {
        'fname': 'shift_923874273.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True,
        'final_activation': 'sigmoid',
        'seed': 923874273,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True,
                           'shift':True, 'same':False, 'mode':cv2.NORMAL_CLONE, 'label_mode':'binary'}
    },
    'Shift-2388222932' : {
        'fname': 'shift_2388222932.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True,
        'final_activation': 'sigmoid',
        'seed': 2388222932,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True,
                           'shift':True, 'same':False, 'mode':cv2.NORMAL_CLONE, 'label_mode':'binary'}
    },
    'Shift-676346783' : {
        'fname': 'shift_676346783.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True,
        'final_activation': 'sigmoid',
        'seed': 676346783,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True,
                           'shift':True, 'same':False, 'mode':cv2.NORMAL_CLONE, 'label_mode':'binary'}
    },
    'Shift-123425' : {
        'fname': 'shift_123425.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True,
        'final_activation': 'sigmoid',
        'seed': 123425,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True,
                           'shift':True, 'same':False, 'mode':cv2.NORMAL_CLONE, 'label_mode':'binary'}
    },
    'Shift-Intensity' : {
        'fname': 'shift_intensity.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True,
        'final_activation': 'sigmoid',
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True,
                           'shift':True, 'same':False, 'mode':cv2.NORMAL_CLONE, 'label_mode':'logistic-intensity'}
    },
    'Shift-Intensity-923874273' : {
        'fname': 'shift_intensity_923874273.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True,
        'final_activation': 'sigmoid',
        'seed': 923874273,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True,
                           'shift':True, 'same':False, 'mode':cv2.NORMAL_CLONE, 'label_mode':'logistic-intensity'}
    },
    'Shift-Intensity-2388222932' : {
        'fname': 'shift_intensity_2388222932.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True,
        'final_activation': 'sigmoid',
        'seed': 2388222932,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True,
                           'shift':True, 'same':False, 'mode':cv2.NORMAL_CLONE, 'label_mode':'logistic-intensity'}
    },
    'Shift-Intensity-676346783' : {
        'fname': 'shift_intensity_676346783.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True,
        'final_activation': 'sigmoid',
        'seed': 676346783,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True,
                           'shift':True, 'same':False, 'mode':cv2.NORMAL_CLONE, 'label_mode':'logistic-intensity'}
    },
    'Shift-Intensity-123425' : {
        'fname': 'shift_intensity_123425.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True,
        'final_activation': 'sigmoid',
        'seed': 123425,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True,
                           'shift':True, 'same':False, 'mode':cv2.NORMAL_CLONE, 'label_mode':'logistic-intensity'}
    },
    'Shift-Raw-Intensity' : {
        'fname': 'shift_raw_intensity.pt',
        'out_dir': 'shift/',
        'loss': nn.MSELoss,
        'skip_background': True,
        'final_activation': 'relu',
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True,
                           'shift':True, 'same':False, 'mode':cv2.NORMAL_CLONE, 'label_mode':'intensity'}
    },
    'Shift-Raw-Intensity-923874273' : {
        'fname': 'shift_raw_intensity_923874273.pt',
        'out_dir': 'shift/',
        'loss': nn.MSELoss,
        'skip_background': True,
        'final_activation': 'sigmoid',
        'seed': 923874273,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True,
                           'shift':True, 'same':False, 'mode':cv2.NORMAL_CLONE, 'label_mode':'intensity'}
    },
    'Shift-Raw-Intensity-2388222932' : {
        'fname': 'shift_raw_intensity_2388222932.pt',
        'out_dir': 'shift/',
        'loss': nn.MSELoss,
        'skip_background': True,
        'final_activation': 'sigmoid',
        'seed': 2388222932,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True,
                           'shift':True, 'same':False, 'mode':cv2.NORMAL_CLONE, 'label_mode':'intensity'}
    },
    'Shift-Raw-Intensity-676346783' : {
        'fname': 'shift_raw_intensity_676346783.pt',
        'out_dir': 'shift/',
        'loss': nn.MSELoss,
        'skip_background': True,
        'final_activation': 'sigmoid',
        'seed': 676346783,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True,
                           'shift':True, 'same':False, 'mode':cv2.NORMAL_CLONE, 'label_mode':'intensity'}
    },
    'Shift-Raw-Intensity-123425' : {
        'fname': 'shift_raw_intensity_123425.pt',
        'out_dir': 'shift/',
        'loss': nn.MSELoss,
        'skip_background': True,
        'final_activation': 'sigmoid',
        'seed': 123425,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True,
                           'shift':True, 'same':False, 'mode':cv2.NORMAL_CLONE, 'label_mode':'intensity'}
    },
    'Shift-M' : {
        'fname': 'shift_m.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True,
        'final_activation': 'sigmoid',
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True,
                           'shift':True, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'binary'}
    },
    'Shift-M-923874273' : {
        'fname': 'shift_m_923874273.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True,
        'final_activation': 'sigmoid',
        'seed': 923874273,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True,
                           'shift':True, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'binary'}
    },
    'Shift-M-2388222932' : {
        'fname': 'shift_m_2388222932.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True,
        'final_activation': 'sigmoid',
        'seed': 2388222932,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True,
                           'shift':True, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'binary'}
    },
    'Shift-M-676346783' : {
        'fname': 'shift_m_676346783.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True,
        'final_activation': 'sigmoid',
        'seed': 676346783,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True,
                           'shift':True, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'binary'}
    },
    'Shift-M-123425' : {
        'fname': 'shift_m_123425.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True,
        'final_activation': 'sigmoid',
        'seed': 123425,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True,
                           'shift':True, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'binary'}
    },
    'Shift-Intensity-M' : {
        'fname': 'shift_intensity_m.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True,
        'final_activation': 'sigmoid',
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True,
                           'shift':True, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'logistic-intensity'}
    },
    'Shift-Intensity-M-923874273' : {
        'fname': 'shift_intensity_m_923874273.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True,
        'final_activation': 'sigmoid',
        'seed': 923874273,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True,
                           'shift':True, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'logistic-intensity'}
    },
    'Shift-Intensity-M-2388222932' : {
        'fname': 'shift_intensity_m_2388222932.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True,
        'final_activation': 'sigmoid',
        'seed': 2388222932,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True,
                           'shift':True, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'logistic-intensity'}
    },
    'Shift-Intensity-M-676346783' : {
        'fname': 'shift_intensity_m_676346783.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True,
        'final_activation': 'sigmoid',
        'seed': 676346783,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True,
                           'shift':True, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'logistic-intensity'}
    },
    'Shift-Intensity-M-123425' : {
        'fname': 'shift_intensity_m_123425.pt',
        'out_dir': 'shift/',
        'loss': nn.BCELoss,
        'skip_background': True,
        'final_activation': 'sigmoid',
        'seed': 123425,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True,
                           'shift':True, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'logistic-intensity'}
    },
    'Shift-Raw-Intensity-M' : {
        'fname': 'shift_raw_intensity_m.pt',
        'out_dir': 'shift/',
        'loss': nn.MSELoss,
        'skip_background': True,
        'final_activation': 'relu',
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True,
                           'shift':True, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'intensity'}
    },
    'Shift-Raw-Intensity-M-923874273' : {
        'fname': 'shift_raw_intensity_m_923874273.pt',
        'out_dir': 'shift/',
        'loss': nn.MSELoss,
        'skip_background': True,
        'final_activation': 'sigmoid',
        'seed': 923874273,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True,
                           'shift':True, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'intensity'}
    },
    'Shift-Raw-Intensity-M-2388222932' : {
        'fname': 'shift_raw_intensity_m_2388222932.pt',
        'out_dir': 'shift/',
        'loss': nn.MSELoss,
        'skip_background': True,
        'final_activation': 'sigmoid',
        'seed': 2388222932,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True,
                           'shift':True, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'intensity'}
    },
    'Shift-Raw-Intensity-M-676346783' : {
        'fname': 'shift_raw_intensity_m_676346783.pt',
        'out_dir': 'shift/',
        'loss': nn.MSELoss,
        'skip_background': True,
        'final_activation': 'sigmoid',
        'seed': 676346783,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True,
                           'shift':True, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'intensity'}
    },
    'Shift-Raw-Intensity-M-123425' : {
        'fname': 'shift_raw_intensity_m_123425.pt',
        'out_dir': 'shift/',
        'loss': nn.MSELoss,
        'skip_background': True,
        'final_activation': 'sigmoid',
        'seed': 123425,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True,
                           'shift':True, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'intensity'}
    },
### ------------------------------------ Foreign patch poisson blending / interpolation ------------------------------------ ###
'FPI-Poisson' : {
        'fname': 'fpi_poisson.pt',
        'out_dir': 'fpi/',
        'loss': nn.BCELoss,
        'skip_background': True,
        'final_activation': 'sigmoid',
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':False,
                           'shift':False, 'same':False, 'mode':cv2.MIXED_CLONE, 'label_mode':'continuous'}
    },
'FPI' : {
        'fname': 'fpi.pt',
        'out_dir': 'fpi/',
        'loss': nn.BCELoss,
        'skip_background': True,
        'final_activation': 'sigmoid',
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':False,
                           'shift':False, 'same':False, 'mode':'uniform', 'label_mode':'continuous'}
    },
### ------------------------------------ Shifted patch pasting ------------------------------------ ###
'CutPaste' : {
        'fname': 'cut_paste.pt',
        'out_dir': 'cut_paste/',
        'loss': nn.BCELoss,
        'skip_background': True,
        'final_activation': 'sigmoid',
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':False,
                           'shift':True, 'same':True, 'mode':'swap', 'label_mode':'binary'}
    },

    'Shift-Intensity-Swap' : {
        'fname': 'shift_intensity_swap.pt',
        'out_dir': 'swap/',
        'loss': nn.BCELoss,
        'skip_background': True,
        'final_activation': 'sigmoid',
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True,
                           'shift':True, 'same':False, 'mode':'swap', 'label_mode':'logistic-intensity'}
    },
    'Shift-Intensity-Swap-923874273' : {
        'fname': 'shift_intensity_swap_923874273.pt',
        'out_dir': 'swap/',
        'loss': nn.BCELoss,
        'skip_background': True,
        'final_activation': 'sigmoid',
        'seed': 923874273,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True,
                           'shift':True, 'same':False, 'mode':'swap', 'label_mode':'logistic-intensity'}
    },
    'Shift-Intensity-Swap-2388222932' : {
        'fname': 'shift_intensity_swap_2388222932.pt',
        'out_dir': 'swap/',
        'loss': nn.BCELoss,
        'skip_background': True,
        'final_activation': 'sigmoid',
        'seed': 2388222932,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True,
                           'shift':True, 'same':False, 'mode':'swap', 'label_mode':'logistic-intensity'}
    },
    'Shift-Intensity-Swap-676346783' : {
        'fname': 'shift_intensity_swap_676346783.pt',
        'out_dir': 'swap/',
        'loss': nn.BCELoss,
        'skip_background': True,
        'final_activation': 'sigmoid',
        'seed': 676346783,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True,
                           'shift':True, 'same':False, 'mode':'swap', 'label_mode':'logistic-intensity'}
    },
    'Shift-Intensity-Swap-123425' : {
        'fname': 'shift_intensity_swap_123425.pt',
        'out_dir': 'swap/',
        'loss': nn.BCELoss,
        'skip_background': True,
        'final_activation': 'sigmoid',
        'seed': 123425,
        'self_sup_args' : {'gamma_params':(2, 0.05, 0.03), 'resize':True,
                           'shift':True, 'same':False, 'mode':'swap', 'label_mode':'logistic-intensity'}
    },
}

# note: these are half-widths in [0, 0.5]
# ((h_min, h_max), (w_min, w_max))
WIDTH_BOUNDS_PCT = {'bottle':((0.03, 0.4), (0.03, 0.4)), 'cable':((0.05, 0.4), (0.05, 0.4)), 'capsule':((0.03, 0.15), (0.03, 0.4)),
                    'hazelnut':((0.03, 0.35), (0.03, 0.35)), 'metal_nut':((0.03, 0.4), (0.03, 0.4)), 'pill':((0.03, 0.2), (0.03, 0.4)),
                    'screw':((0.03, 0.12), (0.03, 0.12)), 'toothbrush':((0.03, 0.4), (0.03, 0.2)), 'transistor':((0.03, 0.4), (0.03, 0.4)),
                    'zipper':((0.03, 0.4), (0.03, 0.2)),
                    'carpet':((0.03, 0.4), (0.03, 0.4)), 'grid':((0.03, 0.4), (0.03, 0.4)),
                    'leather':((0.03, 0.4), (0.03, 0.4)), 'tile':((0.03, 0.4), (0.03, 0.4)), 'wood':((0.03, 0.4), (0.03, 0.4))}

MIN_OVERLAP_PCT = {'bottle': 0.25,  'capsule':0.25,
                   'hazelnut':0.25, 'metal_nut':0.25, 'pill':0.25,
                   'screw':0.25, 'toothbrush':0.25,
                   'zipper':0.25}

MIN_OBJECT_PCT = {'bottle': 0.7,  'capsule':0.7,
                  'hazelnut':0.7, 'metal_nut':0.5, 'pill':0.7,
                  'screw':.5, 'toothbrush':0.25,
                  'zipper':0.7}

NUM_PATCHES = {'bottle':3, 'cable':3, 'capsule':3, 'hazelnut':3, 'metal_nut':3,
               'pill':3, 'screw':4, 'toothbrush':3, 'transistor':3, 'zipper':4,
               'carpet':4, 'grid':4, 'leather':4, 'tile':4, 'wood':4}

# k, x0 pairs
INTENSITY_LOGISTIC_PARAMS = {'bottle':(1/12, 24), 'cable':(1/12, 24), 'capsule':(1/2, 4), 'hazelnut':(1/12, 24), 'metal_nut':(1/3, 7),
            'pill':(1/3, 7), 'screw':(1, 3), 'toothbrush':(1/6, 15), 'transistor':(1/6, 15), 'zipper':(1/6, 15),
            'carpet':(1/3, 7), 'grid':(1/3, 7), 'leather':(1/3, 7), 'tile':(1/3, 7), 'wood':(1/6, 15)}

# bottle is aligned but it's symmetric under rotation
UNALIGNED_OBJECTS = ['bottle', 'hazelnut', 'metal_nut', 'screw']

# non-aligned objects get extra time
EPOCHS = {'bottle':320, 'cable':320, 'capsule':320, 'hazelnut':560, 'metal_nut':560,
          'pill':320, 'screw':560, 'toothbrush':320, 'transistor':320, 'zipper':320,
          'carpet':320, 'grid':320, 'leather':320, 'tile':320, 'wood':320}

# brightness, threshold pairs
BACKGROUND = {'bottle':(200, 60), 'screw':(200, 60), 'capsule':(200, 60), 'zipper':(200, 60),
              'hazelnut':(20, 20), 'pill':(20, 20), 'toothbrush':(20, 20), 'metal_nut':(20, 20)}
