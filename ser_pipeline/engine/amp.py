"""Mixed-precision helpers, compatible across the torch.amp API move."""
try:
    from torch.amp import autocast as _ac, GradScaler as _gs

    def make_autocast():
        return _ac(device_type='cuda')

    def make_scaler():
        return _gs(device='cuda')
except (ImportError, TypeError):
    from torch.cuda.amp import autocast as _ac, GradScaler as _gs

    def make_autocast():
        return _ac()

    def make_scaler():
        return _gs()
