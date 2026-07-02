from backend_utils import VendorDescriptor

vendor_info = VendorDescriptor(
    vendor_name="spacemit",
    device_name="cpu",
    device_query_cmd="spacemit-tcm-smi",
    fp64_enabled=False,
    bf16_enabled=False,
    int64_enabled=False,
)

CUSTOMIZED_UNUSED_OPS = ()

__all__ = ["*"]
