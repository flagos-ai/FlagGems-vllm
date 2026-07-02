from backend_utils import VendorDescriptor

vendor_info = VendorDescriptor(
    vendor_name="kunlunxin",
    device_name="cuda",
    device_query_cmd="xpu-smi",
    triton_extra_name="xpu",
    fp64_enabled=False,
)

CUSTOMIZED_UNUSED_OPS = ()

__all__ = ["*"]
