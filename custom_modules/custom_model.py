# import re
# from turtle import back
from ultralytics.nn.tasks import BaseModel, DetectionModel, yaml_model_load, parse_model
# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics.nn.modules import (
    AIFI,
    C1,
    C2,
    C3,
    C3TR,
    ELAN1,
    OBB,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    AConv,
    ADown,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C3Ghost,
    C3x,
    CBFuse,
    CBLinear,
    Classify,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    Detect,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Pose,
    RepC3,
    RepConv,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    RTDETRDecoder,
    SCDown,
    Segment,
    WorldDetect,
    v10Detect,
)
from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, colorstr, emojis, yaml_load
from ultralytics.utils.checks import check_requirements, check_suffix, check_yaml
from ultralytics.utils.loss import (
    E2EDetectLoss,
    v8ClassificationLoss,
    v8DetectionLoss,
    v8OBBLoss,
    v8PoseLoss,
    v8SegmentationLoss,
)
from ultralytics.utils.ops import make_divisible
from ultralytics.utils.plotting import feature_visualization
from ultralytics.utils.torch_utils import (
    fuse_conv_and_bn,
    fuse_deconv_and_bn,
    initialize_weights,
    intersect_dicts,
    model_info,
    scale_img,
    time_sync,
)

try:
    import thop
except ImportError:
    thop = None

from ultralytics.engine.exporter import Exporter


class multihead(BaseModel):
    def __init__(self, cfg_list: list, ch=3, verbose=True, pt_list: list= []):  # model, input channels, number of classes
        """Initialize the YOLOv8 detection model with the given config and parameters."""
        super().__init__()
        # 如果cfg是list, 则加载多检测头模型
        self.yaml = None #正常检测
        self.yaml2 = None #黑点检测
        for cfg in cfg_list:
            cfg_yaml = yaml_model_load(cfg)  # cfg dict
            if cfg_yaml["nc"] == 1:
                self.yaml2 = cfg_yaml
            else : self.yaml = cfg_yaml

        # Define model
        # ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        # if nc and nc != self.yaml["nc"]:
        #     LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
        #     self.yaml["nc"] = nc  # override YAML value
        
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.model2, self.save2 = parse_model(deepcopy(self.yaml2), ch=ch, verbose=verbose)
        print(self.save, self.save2)

        # Check if the two models are identical
        self.backbone = compare_sequential_modules(self.model[:-1], self.model2[:-1])
        self.head1 = self.model[-1:]
        self.head2 = self.model2[-1:]


        self.names1 = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.names2 = {i: f"{i}" for i in range(self.yaml2["nc"])}  # default names dict

        self.inplace = self.yaml.get("inplace", True)
        self.end2end = getattr(self.model[-1], "end2end", False)

        # Build strides(导出时需要)
        m = self.head1[-1]  # Detect()
        if isinstance(m, Detect):  # includes all Detect subclasses like Segment, Pose, OBB, WorldDetect
            s = 256  # 2x min stride
            m.inplace = self.inplace

            def _forward(x):
                """Performs a forward pass through the model, handling different Detect subclass types accordingly."""
                if self.end2end:
                    return self.forward(x)["one2many"]
                return self.forward(x)[0] if isinstance(m, (Segment, Pose, OBB)) else self.forward(x)

            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))[0]])  # forward
            self.stride = m.stride
            m.bias_init()  # only run once
        else:
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR

        # 手动添加task属性
        self.task = "detect"


        # 加载每个子模块的预训练权重
        if len(pt_list) != 3:
            raise ValueError("pt_list模块数量不正确")
        else:
            self.load(weights=pt_list[0], submodule=self.backbone)
            self.load(weights=pt_list[1], submodule=self.head1)
            self.load(weights=pt_list[2], submodule=self.head2)


        # # Init weights, biases
        # initialize_weights(self)
        # if verbose:
        #     self.info()
        #     LOGGER.info("")
    
    def load(self, weights, submodule=None, verbose=True):
        if not weights:
            raise ValueError("weights参数不能为空")
        if str(weights).endswith('.pt'):
            weights = torch.load(weights, map_location='cpu')
        submodel = weights['model'].model if isinstance(weights, dict) else weights
        csd = submodel.float().state_dict()
        csd = intersect_dicts(csd, submodule.state_dict(), exclude=[])  # intersect
        submodule.load_state_dict(csd, strict=False)
        # compare_sequential_modules(submodel, self.backbone, True)
        if verbose:
            LOGGER.info(f"Transferred {len(csd)}/{len(submodule.state_dict())} items from pretrained weights")

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.
            embed (list, optional): A list of feature vectors/embeddings to return.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        y, dt, embeddings = [], [], []  # outputs
        for m in self.backbone:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save or m.i in self.save2 else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        for m1, m2 in zip(self.head1, self.head2):
            if m1.f != -1 or m2.f != -1:  # if not from previous layer
                x1 = y[m1.f] if isinstance(m1.f, int) else [x if j == -1 else y[j] for j in m1.f]  # from earlier layers
                x2 = y[m2.f] if isinstance(m2.f, int) else [x if j == -1 else y[j] for j in m2.f]  # from earlier layers
            # if profile:
            #     self._profile_one_layer(m, x, dt)
            # x = m(x) 
            x1, x2 = m1(x1), m2(x2)
        return [x1, x2] #x, 

def parse_model(d, ch, verbose=True):  # model_dict, input_channels(3)
    """Parse a YOLO model.yaml dictionary into a PyTorch model."""
    import ast

    # Args
    max_channels = float("inf")
    nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    if scales:
        scale = d.get("scale")
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m = getattr(torch.nn, m[3:]) if "nn." in m else globals()[m]  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in {
            Classify,
            Conv,
            ConvTranspose,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            DWConv,
            Focus,
            BottleneckCSP,
            C1,
            C2,
            C2f,
            RepNCSPELAN4,
            ELAN1,
            ADown,
            AConv,
            SPPELAN,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
            RepC3,
            PSA,
            SCDown,
            C2fCIB,
        }:
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is C2fAttn:
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)  # embed channels
                args[2] = int(
                    max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2]
                )  # num heads

            args = [c1, c2, *args[1:]]
            if m in {BottleneckCSP, C1, C2, C2f, C2fAttn, C3, C3TR, C3Ghost, C3x, RepC3, C2fCIB}:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in {HGStem, HGBlock}:
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)  # number of repeats
                n = 1
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in {Detect, WorldDetect, Segment, Pose, OBB, ImagePoolingAttn, v10Detect}:
            args.append([ch[x] for x in f])
            if m is Segment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
        elif m is RTDETRDecoder:  # special case, channels arg must be passed in index 1
            args.insert(1, [ch[x] for x in f])
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        m.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}")  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)

def compare_sequential_modules(seq1: nn.Sequential, seq2: nn.Sequential, compare_weights: bool = False) -> nn.Sequential:
    # 检查两个模块的层数是否相同
    if len(seq1) != len(seq2):
        raise ValueError("The two nn.Sequential modules have different number of layers.")
    
    # 逐层比较两个模块的参数
    for layer1, layer2 in zip(seq1, seq2):
        # 比较每一层的参数
        for param1, param2 in zip(layer1.parameters(), layer2.parameters()):
            if not param1.shape==param2.shape:
                raise ValueError("不是相同的backbone")
            if compare_weights and not torch.equal(param1, param2):
                print("模块的参数不相同")
    
    print("是相同结构的backbone.")
    return seq1

def export(
    model,# self,
    **kwargs,
) -> str:
    """
    Exports the model to a different format suitable for deployment.

    This method facilitates the export of the model to various formats (e.g., ONNX, TorchScript) for deployment
    purposes. It uses the 'Exporter' class for the export process, combining model-specific overrides, method
    defaults, and any additional arguments provided.

    Args:
        **kwargs (Dict): Arbitrary keyword arguments to customize the export process. These are combined with
            the model's overrides and method defaults. Common arguments include:
            format (str): Export format (e.g., 'onnx', 'engine', 'coreml').
            half (bool): Export model in half-precision.
            int8 (bool): Export model in int8 precision.
            device (str): Device to run the export on.
            workspace (int): Maximum memory workspace size for TensorRT engines.
            nms (bool): Add Non-Maximum Suppression (NMS) module to model.
            simplify (bool): Simplify ONNX model.

    Returns:
        (str): The path to the exported model file.

    Raises:
        AssertionError: If the model is not a PyTorch model.
        ValueError: If an unsupported export format is specified.
        RuntimeError: If the export process fails due to errors.

    Examples:
        >>> model = YOLO("yolov8n.pt")
        >>> model.export(format="onnx", dynamic=True, simplify=True)
        'path/to/exported/model.onnx'
    """
    # self._check_is_pytorch_model()
    # from .exporter import Exporter

    custom = {
        "imgsz": 1024,
        "batch": 1,
        "data": None,
        "device": None,  # reset to avoid multi-GPU errors
        "verbose": False,
    }  # method defaults
    args = {**custom, **kwargs, "mode": "export"}  # highest priority args on the right
    return Exporter(overrides=args, _callbacks=None)(model=model)

if __name__ == "__main__":
    cfg1 = '/workspace/飞行检测/custom_model_config/yolov8s-ghost-onlyp2.yaml'
    cfg2 = '/workspace/飞行检测/custom_model_config/yolov8s-ghost-p3-p5.yaml'
    pt1 = '/workspace/飞行检测/results/v8s_p2_new/0910_5cls_1024/weights/backbone_module_epoch99.pt'
    pt2 = '/workspace/飞行检测/results/v8s_p2_new/0910_5cls_1024/weights/detect_module_epoch99_nc5.pt'
    pt3 = '/workspace/飞行检测/results/fine_tune/0910_1cls_1024/weights/detect_module_nc1.pt'
    model = multihead(cfg_list=[cfg1, cfg2], ch=3, pt_list=[pt1, pt2, pt3])

    # 加载预训练权重
    # model.load(weights='/home/tyjt/桌面/ultralytics/runs/detect/train8/weights/backbone_module_epoch0.pt', submodule=model.backbone)

    # 不同的模式产生的不同的输出(训练，推理，导出)
    # model.eval()
    # x = torch.randn(1, 3, 640, 640)
    # head_opt = model(x)
    # print(type(head_opt), len(head_opt))
    # print(type(head_opt[0]),len(head_opt[0]))
    # print(type(head_opt[1]),len(head_opt[1]))
    # print(head_opt[0][0].shape, head_opt[1][0].shape)
    # print([x.shape for x in head_opt[0][1]])

    export(model, format="onnx")