from backend_utils import VendorDescriptor

vendor_info = VendorDescriptor(
    vendor_name="thead",
    device_name="cuda",
    device_query_cmd="ppu-smi",
)

CUSTOMIZED_UNUSED_OPS = ()

__all__ = ["*"]
