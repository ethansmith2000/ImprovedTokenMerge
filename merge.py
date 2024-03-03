import torch
import math
import torch.nn.functional as F


def up_or_downsample(item, cur_w, cur_h, new_w, new_h, method="nearest"):
    batch_size, l, d = item.shape

    item = item.reshape(batch_size, cur_h, cur_w, -1)
    item = item.permute(0, 3, 1, 2)
    df = cur_h // new_h
    item = F.interpolate(item, size=(new_h, new_w), mode=method)
    item = item.permute(0, 2, 3, 1)
    item = item.reshape(batch_size, new_h * new_w, -1)

    return item


class ToDoPatch:

    def __init__(self, token_merge_args):
        self.unet_input_hook = None
        self.timestep = None
        self.size = None
        self.attn_hooks = []

        self.settings = {
            "size": None,
            "timestep": None,
            "hooks": [],
            "args": {
                "downsample_method": token_merge_args.get("downsample_method", "nearest"),
                # native torch interpolation methods ["nearest", "linear", "bilinear", "bicubic", "nearest-exact"]
                "downsample_factor": token_merge_args.get("downsample_factor", 2),  # amount to downsample by
                "timestep_threshold_stop": token_merge_args.get("timestep_threshold_stop", 0.0),
                # timestep to stop merging, 0.0 means stop at 0 steps remaining

                "downsample_factor_level_2": token_merge_args.get("downsample_factor_level_2", 1),
                # amount to downsample by at the 2nd down block of unet
            }
        }

    def hook_model(self, unet):
        def hook(module, args):
            self.timestep = (args[0].shape[2], args[0].shape[3])
            self.size = args[1].item()
            return None

        self.unet_input_hook = unet.register_forward_pre_hook(hook)

        def input_downsample_hook(module, args):
            # we need to edit args and it comes in as a tuple
            args = list(args)

            original_h, original_w = self.size
            original_tokens = original_h * original_w
            downsample = int(math.ceil(math.sqrt(original_tokens // args[0].shape[1])))
            if downsample == 1:
                downsample_factor = self.settings['downsample_factor']
            elif downsample == 2:
                downsample_factor = self.settings['downsample_factor_level_2']
            else:
                downsample_factor = 1

            cur_h, cur_w = original_h // downsample, original_w // downsample
            new_h, new_w = cur_h // downsample_factor, cur_w // downsample_factor

            if downsample <= 2 and self.timestep / 1000 > self.settings['timestep_threshold_stop']:
                if downsample_factor > 1:
                    args[0] = up_or_downsample(args[0], cur_w, cur_h, new_w, new_h, self.settings["downsample_method"])

            return tuple(args)

        # gather attention blocks
        attn_modules = [module for name, module in unet.named_modules() if
                        module.__class__.__name__ == 'BasicTransformerBlock']


        for attn_module in attn_modules:
            self.attn_hooks.append(attn_module.attn1.to_k.register_forward_pre_hook(input_downsample_hook))
            self.attn_hooks.append(attn_module.attn1.to_v.register_forward_pre_hook(input_downsample_hook))


    def remove_patch(self):
        self.unet_input_hook.remove()
        for hook in self.attn_hooks:
            hook.remove()
        self.attn_hooks = []












