import numpy as np
import os, glob
import os.path as osp

import matplotlib.pyplot as plt

from PIL import Image

from scipy.stats import ttest_ind

from mmaction.datasets.pipelines import Compose
import mmcv

# ..........torch imports............
import torch
import torchvision

from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms

#.... Captum imports..................
from captum.attr import LayerGradientXActivation, LayerIntegratedGradients

from captum.concept import TCAV
from captum.concept import Concept

from captum.concept._utils.data_iterator import dataset_to_dataloader, CustomIterableDataset
from captum.concept._utils.common import concepts_to_str

from mmaction.apis import init_recognizer, inference_recognizer

config_file = 'configs/recognition/swin/swin_tiny_patch244_window877_kinetics400_1k.py'
config = mmcv.Config.fromfile(config_file)
device = 'cpu' # or 'cpu'
device = torch.device(device)

model = init_recognizer(config_file, device=device, checkpoint='checkpoints/swin_tiny_patch244_window877_kinetics400_1k.pth')

# Method to normalize a video to Kinetics-400 mean and standard deviation
def transform(video, cfg=config):
    # build the data pipeline
    test_pipeline = cfg.data.test.pipeline
    data = dict(filename=video, label=-1, start_index=0, modality='RGB')
    if 'Init' not in test_pipeline[0]['type']:
        test_pipeline = [dict(type='OpenCVInit')] + test_pipeline
    else:
        test_pipeline[0] = dict(type='OpenCVInit')
    for i in range(len(test_pipeline)):
        if 'Decode' in test_pipeline[i]['type']:
            test_pipeline[i] = dict(type='OpenCVDecode')
    test_pipeline = Compose(test_pipeline)
    data = test_pipeline(data)
    data = data['imgs'].to(device)
    return data

def assemble_concept(name, id, concepts_path="../../shashank/kinetics-dataset/k400/avinab/interest_classes/"):
    concept_path = os.path.join(concepts_path, name) + "/"
    dataset = CustomIterableDataset(transform, concept_path)
    concept_iter = dataset_to_dataloader(dataset)

    return Concept(id=id, name=name, data_iter=concept_iter)

concepts_path = "../data/sravan_concepts/k400_concept_videos/"

sample1_concept = assemble_concept("fork", 0, concepts_path=concepts_path)
sample2_concept = assemble_concept("random", 2, concepts_path=concepts_path)

layers=['cls_head.avg_pool']

mytcav = TCAV(model=model,
              layers=layers)
              #layer_attr_method = LayerIntegratedGradients(
              #  model, None, multiply_by_inputs=False))

experimental_set_rand = [[sample1_concept, sample2_concept]]
experimental_set_rand

# Load sample images from folder
input_tensors = torch.stack([transform(video) for video in glob.glob("../../shashank/kinetics-dataset/k400/avinab/interest_classes/din")[:8]])
#input_tensors = torch.randn([1, 24, 3, 224, 224])

target_ind = 91

tcav_scores_w_random = mytcav.interpret(inputs=input_tensors,
                                        experimental_sets=experimental_set_rand,
                                        target=target_ind,
                                        additional_forward_args={'return_loss': False}
                                       )
print(tcav_scores_w_random)

def format_float(f):
    return float('{:.3f}'.format(f) if abs(f) >= 0.0005 else '{:.3e}'.format(f))

def plot_tcav_scores(experimental_sets, tcav_scores):
    import pdb
    pdb.set_trace()
    fig, ax = plt.subplots(1, len(experimental_sets), figsize = (25, 7))

    barWidth = 1 / (len(experimental_sets[0]) + 1)

    for idx_es, concepts in enumerate(experimental_sets):

        concepts = experimental_sets[idx_es]
        concepts_key = concepts_to_str(concepts)

        pos = [np.arange(len(layers))]
        for i in range(1, len(concepts)):
            pos.append([(x + barWidth) for x in pos[i-1]])
        _ax = (ax[idx_es] if len(experimental_sets) > 1 else ax)
        for i in range(len(concepts)):
            val = [format_float(scores['sign_count'][i]) for layer, scores in tcav_scores[concepts_key].items()]
            _ax.bar(pos[i], val, width=barWidth, edgecolor='white', label=concepts[i].name)

        # Add xticks on the middle of the group bars
        _ax.set_xlabel('Set {}'.format(str(idx_es)), fontweight='bold', fontsize=16)
        _ax.set_xticks([r + 0.3 * barWidth for r in range(len(layers))])
        _ax.set_xticklabels(layers, fontsize=16)

        # Create legend & Show graphic
        _ax.legend(fontsize=16)

    plt.show()
    plt.savefig('test_captum_1.jpg')

plot_tcav_scores(experimental_set_rand, tcav_scores_w_random)