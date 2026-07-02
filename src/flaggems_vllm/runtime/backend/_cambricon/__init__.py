from backend_utils import VendorDescriptor

vendor_info = VendorDescriptor(
    vendor_name="cambricon",
    device_name="mlu",
    device_query_cmd="cnmon",
    dispatch_key="PrivateUse1",
    fp64_enabled=False,
)

CUSTOMIZED_UNUSED_OPS = ()

__all__ = ["*"]
