import torch

class AttentionExtractor:
    """A class to attach hooks and extract attention maps from a transformer model"""
    def __init__(self, model, layers_to_hook):
        self.model = model
        self.layers_to_hook = layers_to_hook
        self.attention_maps = []
        self.hooks = []

    def _hook_fn(self, module, input, output):
        self.attention_maps.append(output[1].detach().cpu())
    
    def register_hooks(self):
        for name, module in self.model.named_modules():
            if any(layer_name in name for layer_name in self.layers_to_hook):
                self.hooks.append(module.register_forward_hook(self._hook_fn))
    
    def unregister_hooks(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
    
    def get_and_clear_attention(self):
        attentions = list(self._attention_maps)
        self._attention_maps = []
        return attentions

def get_attention_layers(model):
    """Helper to find the names of the multi head attention layers in the transformer encoder"""
    attention_layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            attention_layer_names.append(name)
    print(f"Found attention layers: {attention_layer_names}")
    return attention_layer_names
